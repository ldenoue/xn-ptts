mod handler;
mod model;
mod protocol;

use anyhow::Result;
use axum::Router;
use axum::routing::any;
use clap::Parser;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "ptts-ws-server")]
#[command(about = "WebSocket server for Pocket TTS")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8080")]
    addr: String,

    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    #[arg(long, default_value_t = 4242424242424242)]
    seed: u64,

    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,

    /// Use the CUDA backend (requires building with --features cuda).
    #[arg(long, default_value_t = false)]
    cuda: bool,
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::Layer::new().with_target(false))
        .with(filter)
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();

    let app_state = build_app_state(&args)?;

    let app = Router::new()
        .route("/speech/tts", any(handler::ws_handler))
        .with_state(app_state)
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    tracing::info!(addr = %args.addr, "listening on /speech/tts");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutdown requested");
        })
        .await?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn build_app_state(args: &Args) -> Result<model::AppState> {
    if args.cuda {
        let dev = xn::CudaDevice::new(0)?;
        unsafe {
            dev.disable_event_tracking();
        }
        let s = model::load_pocket_tts(args.temperature, args.seed, args.max_seq_len, dev)?;
        Ok(model::AppState::Cuda(Arc::new(s)))
    } else {
        let s = model::load_pocket_tts(args.temperature, args.seed, args.max_seq_len, xn::CPU)?;
        Ok(model::AppState::Cpu(Arc::new(s)))
    }
}

#[cfg(not(feature = "cuda"))]
fn build_app_state(args: &Args) -> Result<model::AppState> {
    if args.cuda {
        anyhow::bail!("--cuda requested but binary was not built with --features cuda");
    }
    let s = model::load_pocket_tts(args.temperature, args.seed, args.max_seq_len, xn::CPU)?;
    Ok(model::AppState::Cpu(Arc::new(s)))
}
