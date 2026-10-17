# xn-ptts

## 🚀 [Try the live browser demo →](https://ldenoue.github.io/xn-ptts/)

No installation or server required. Pick an engine, choose a voice and language,
load the weights, and generate speech entirely in your browser.

`xn-ptts` is a Rust implementation of
[Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts) built on
[`xn`](https://github.com/LaurentMazare/xn). The web demo also makes it easy to
compare Pocket TTS backends with Supertonic 3 and Kokoro 82M.

## Why try the demo?

- Runs client-side with WebAssembly: text and generated audio stay in the browser.
- Streams Pocket TTS audio as it is generated.
- Offers multiple engines and quantization choices in one page.
- Caches downloaded models and voices for faster subsequent sessions.
- Lets you replay and save the generated WAV audio.
- Includes multilingual Pocket TTS through the Candle Q8_0 engine.

## Engines

| Picker option | Languages | Approximate first download | Notes |
| --- | --- | ---: | --- |
| **xn GGUF q8/f32** | English | 192 MB (q8) or 286 MB (f32), including voices | This repository's native `xn` Pocket TTS implementation |
| **Candle Q8_0 multilingual** | English, French, German, Spanish, Portuguese, Italian | ~142 MB per language, including the selected voice | Based on [`idle-intelligence/tts-web`](https://github.com/idle-intelligence/tts-web); 26 voices available on demand |
| **Supertonic 3 ORT** | Arabic, English, French, German, Portuguese, Italian, Spanish | ~401 MB | Multilingual ONNX Runtime Web engine |
| **Kokoro 82M ORT** | English (US and UK voices) | ~93 MB | `kokoro-js` with q8, q4f16, and q4 choices |

For the multilingual Candle engine, changing the language automatically selects
a suitable default voice and inserts example text in that language. Each
language has its own model; switching languages may therefore trigger another
download. Assets are retained in the browser Cache API until **Clear Cache** is
used.

## Run the web demo locally

Prebuilt WebAssembly files are checked in, so the quickest option is:

```bash
cd ptts-wasm/www
python3 -m http.server 8080
```

Then open [http://localhost:8080](http://localhost:8080).

To rebuild this repository's `xn` WebAssembly engine first:

```bash
cargo install wasm-pack
cd ptts-wasm
make build
cd pkg
python3 -m http.server 8080
```

More WebAssembly build and browser compatibility details are in
[`ptts-wasm/README.md`](ptts-wasm/README.md).

## Native command-line example

The Rust example downloads the required Pocket TTS assets from Hugging Face on
first use:

```bash
cargo run --release -p ptts --features sp --example pocket_tts -- \
  "Hello from Pocket TTS running with xn." \
  --voice alba \
  --quant q8 \
  --output output.wav
```

Run it with `--help` to see the available sampling, voice, weight, and device
options.

## Workspace layout

- `ptts/` — Pocket TTS model implementation and native examples.
- `ptts-wasm/` — browser bindings, demo UI, workers, and checked-in WASM builds.
- `ptts-ws-server/` — streaming WebSocket TTS server.
- `ptts-pyo3/` — Python bindings built with Maturin.
- `vendor/xn/` — the tensor and inference runtime used by the native engine.

## Credits and licenses

This project builds on
[Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts),
[`xn`](https://github.com/LaurentMazare/xn), and
[`idle-intelligence/tts-web`](https://github.com/idle-intelligence/tts-web).
The repository is available under MIT or Apache 2.0. The bundled Idle
Intelligence WASM license is preserved in
[`ptts-wasm/IDLE_TTS_WEB_LICENSE`](ptts-wasm/IDLE_TTS_WEB_LICENSE).
