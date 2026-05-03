use crate::model::{AppState, AppStateB, generate_chunks};
use crate::protocol::{TtsReply, TtsRequest, error_codes};
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use ptts::tts_model::TTSState;
use ptts::wav::Sample;
use std::sync::Arc;
use xn::Unquantized;

pub async fn ws_handler(State(app): State<AppState>, ws: WebSocketUpgrade) -> Response {
    async fn handle_socket(socket: WebSocket, app: AppState) {
        let result = match app {
            AppState::Cpu(s) => serve_b(socket, s).await,
            #[cfg(feature = "cuda")]
            AppState::Cuda(s) => serve_b(socket, s).await,
        };
        if let Err(e) = result {
            tracing::error!(error = %e, "ws session terminated");
        }
    }
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

async fn serve_b<B>(socket: WebSocket, app: Arc<AppStateB<B>>) -> Result<()>
where
    B: xn::Backend,
{
    let (mut tx, mut rx) = socket.split();
    let (reply_tx, mut reply_rx) = tokio::sync::mpsc::unbounded_channel();

    let forwarder = tokio::spawn(async move {
        while let Some(reply) = reply_rx.recv().await {
            let json = serde_json::to_string(&reply)?;
            if tx.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
        Ok::<_, anyhow::Error>(())
    });

    let outcome = run_session(app, &mut rx, &reply_tx).await;
    drop(reply_tx);
    let _ = forwarder.await;
    outcome
}

enum SessionState<B: xn::Backend> {
    Awaiting,
    Ready { base_state: TTSState<Unquantized<f32, B>>, text_buffer: String, stream_id: u32 },
}

async fn run_session<B>(
    app: Arc<AppStateB<B>>,
    stream: &mut futures_util::stream::SplitStream<WebSocket>,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<()>
where
    B: xn::Backend,
{
    let mut sess: SessionState<B> = SessionState::Awaiting;

    while let Some(msg) = stream.next().await {
        let msg = msg?;
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => return Ok(()),
            Message::Binary(_) | Message::Ping(_) | Message::Pong(_) => continue,
        };
        let req: TtsRequest = match serde_json::from_str(text.as_str()) {
            Ok(r) => r,
            Err(e) => {
                send_error(reply_tx, error_codes::BAD_REQUEST, format!("invalid request: {e}"))?;
                continue;
            }
        };
        match (&mut sess, req) {
            (
                SessionState::Awaiting,
                TtsRequest::Setup { model_name, output_format, voice, voice_id, voice_emb, .. },
            ) => match handle_setup(
                &app,
                model_name,
                output_format,
                voice,
                voice_id,
                voice_emb,
                reply_tx,
            )
            .await?
            {
                Some(new_state) => sess = new_state,
                None => continue,
            },
            (SessionState::Awaiting, _) => {
                send_error(
                    reply_tx,
                    error_codes::BAD_REQUEST,
                    "expected setup as first message".into(),
                )?;
            }
            (SessionState::Ready { .. }, TtsRequest::Setup { .. }) => {
                send_error(
                    reply_tx,
                    error_codes::BAD_REQUEST,
                    "session already initialized".into(),
                )?;
            }
            (SessionState::Ready { text_buffer, .. }, TtsRequest::Text { text }) => {
                text_buffer.push_str(&text);
            }
            (
                SessionState::Ready { base_state, text_buffer, stream_id },
                TtsRequest::Flush { flush_id },
            ) => {
                flush_buffer(&app, base_state, text_buffer, stream_id, reply_tx).await?;
                let _ = reply_tx.send(TtsReply::Flushed { flush_id });
            }
            (
                SessionState::Ready { base_state, text_buffer, stream_id },
                TtsRequest::EndOfStream,
            ) => {
                flush_buffer(&app, base_state, text_buffer, stream_id, reply_tx).await?;
                let _ = reply_tx.send(TtsReply::EndOfStream);
                return Ok(());
            }
        }
    }
    Ok(())
}

async fn flush_buffer<B>(
    app: &Arc<AppStateB<B>>,
    base_state: &TTSState<Unquantized<f32, B>>,
    text_buffer: &mut String,
    stream_id: &mut u32,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<()>
where
    B: xn::Backend + 'static,
{
    if text_buffer.is_empty() {
        return Ok(());
    }
    let stream_id_now = *stream_id;
    *stream_id = stream_id.saturating_add(1);
    let text = std::mem::take(text_buffer);
    if let Err(e) = generate_one(app, base_state, &text, stream_id_now, reply_tx).await {
        tracing::warn!(error = %e, stream_id = stream_id_now, "generation failed");
        send_error(reply_tx, error_codes::INTERNAL, format!("generation failed: {e}"))?;
    }
    Ok(())
}

async fn handle_setup<B>(
    app: &Arc<AppStateB<B>>,
    model_name: String,
    output_format: String,
    voice: Option<String>,
    voice_id: Option<String>,
    voice_emb: Option<String>,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<Option<SessionState<B>>>
where
    B: xn::Backend + 'static,
{
    if voice_emb.as_deref().is_some_and(|s| !s.is_empty()) {
        send_error(
            reply_tx,
            error_codes::NOT_IMPLEMENTED,
            "voice_emb prompts are not yet supported".into(),
        )?;
        return Ok(None);
    }
    if output_format != "pcm_s16_le" {
        send_error(
            reply_tx,
            error_codes::BAD_REQUEST,
            format!("unsupported output_format '{output_format}'; only 'pcm_s16_le' is accepted"),
        )?;
        return Ok(None);
    }
    let voice_name = voice_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .or(voice.as_deref().filter(|s| !s.is_empty()))
        .unwrap_or("alba");
    let voice_name = if voice_name == "default" { "alba" } else { voice_name }.to_string();
    let voice_emb_t = match app.voices.get(&voice_name) {
        Some(v) => v,
        None => {
            send_error(reply_tx, error_codes::NOT_FOUND, format!("unknown voice '{voice_name}'"))?;
            return Ok(None);
        }
    };
    let mut base_state = match app.model.init_flow_lm_state(1, app.max_seq_len) {
        Ok(s) => s,
        Err(e) => {
            send_error(reply_tx, error_codes::INTERNAL, format!("init_flow_lm_state failed: {e}"))?;
            return Ok(None);
        }
    };
    if let Err(e) = app.model.prompt_audio(&mut base_state, voice_emb_t) {
        send_error(reply_tx, error_codes::INTERNAL, format!("prompt_audio failed: {e}"))?;
        return Ok(None);
    }
    let request_id = uuid::Uuid::new_v4().to_string();
    let model_name =
        if model_name.is_empty() { "kyutai/pocket-tts".to_string() } else { model_name };
    let ready = TtsReply::Ready {
        model_name,
        sample_rate: app.sample_rate,
        frame_size: app.frame_size,
        audio_stream_names: vec![],
        text_stream_names: vec![],
        request_id,
    };
    if reply_tx.send(ready).is_err() {
        anyhow::bail!("reply channel closed before ready");
    }
    Ok(Some(SessionState::Ready { base_state, text_buffer: String::new(), stream_id: 0 }))
}

async fn generate_one<B>(
    app: &Arc<AppStateB<B>>,
    base_state: &TTSState<Unquantized<f32, B>>,
    text: &str,
    stream_id: u32,
    reply_tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
) -> Result<()>
where
    B: xn::Backend + 'static,
{
    let (prepared, frames_after_eos) = ptts::tts_model::prepare_text_prompt(text);
    let tokens = app.model.flow_lm.conditioner.tokenize(&prepared)?;
    let state = base_state.clone();
    let model = Arc::clone(&app.model);
    let temperature = app.temperature;
    let seed = app.seed_base ^ (stream_id as u64).wrapping_mul(0x9E3779B97F4A7C15);

    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
    let join = tokio::task::spawn_blocking(move || {
        generate_chunks(model, state, tokens, temperature, seed, frames_after_eos, audio_tx)
    });

    let sample_rate = app.sample_rate as f64;
    let mut samples_emitted: usize = 0;
    while let Some(pcm) = audio_rx.recv().await {
        let n = pcm.len();
        let pcm_i16: Vec<i16> = pcm.iter().map(|s| s.to_i16()).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&pcm_i16);
        let audio = base64::engine::general_purpose::STANDARD.encode(bytes);
        let start_s = samples_emitted as f64 / sample_rate;
        let stop_s = (samples_emitted + n) as f64 / sample_rate;
        samples_emitted += n;
        if reply_tx.send(TtsReply::Audio { audio, start_s, stop_s, stream_id }).is_err() {
            break;
        }
    }
    drop(audio_rx);
    join.await??;
    Ok(())
}

fn send_error(
    tx: &tokio::sync::mpsc::UnboundedSender<TtsReply>,
    code: u32,
    message: String,
) -> Result<()> {
    tx.send(TtsReply::Error { message, code })
        .map_err(|_| anyhow::anyhow!("reply channel closed"))?;
    Ok(())
}
