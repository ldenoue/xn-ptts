# wasm-pocket-tts

WebAssembly build of [Pocket TTS](../ptts/) — run text-to-speech directly in the browser.

Try it online [here](https://ldenoue.github.io/xn-ptts/).

## Prerequisites

Install [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/):

```bash
cargo install wasm-pack
```

## Build

From the `ptts-wasm/` directory:

```bash
make build
```

This runs `wasm-pack build` and copies `index.html` into `pkg/`.

The browser build enables fixed-width WebAssembly SIMD (`simd128`) but not
relaxed SIMD. Relaxed SIMD can work in Chrome, but Safari rejects those opcodes
while parsing the module.

## Run

Serve the `pkg/` directory with any HTTP server, for example:

```bash
cd ptts-wasm/pkg
python3 -m http.server 8080
```

Then open http://localhost:8080 in your browser. The page will download the
model weights from HuggingFace on first use (~240 MB for f32, ~146 MB for q8).
The demo loads built-in voices from `embeddings_v3` (~46 MB total for the voices
shown in the UI). The worker stores the tokenizer, selected model weights, and
built-in voice files in the browser Cache API, so later loads can reuse those
blobs without re-downloading them. Use the page's Clear Cache button to remove
those cached assets.

The page also includes optional ONNX Runtime engines for Supertonic 3 and
Kokoro 82M. Kokoro runs through `kokoro-js` on the WASM backend and defaults to
the q8 ONNX weights (~92 MB plus the selected voice). Additional Kokoro
quantized choices are exposed in the UI for q4f16 and q4. The Kokoro browser
path currently exposes the English US/UK voices supported by `kokoro-js`; use
Supertonic for the multilingual language selector.

## Todo

- Handle long prompts, see `split_into_best_sentences` in
  [tts_model.py](https://github.com/kyutai-labs/pocket-tts/blob/aca7dc8db698e5885fe9dd4850bacfa757b429b1/pocket_tts/models/tts_model.py#L893).
- Voice cloning.
