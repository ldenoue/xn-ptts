import { KaleTTS, toWav } from 'https://kalecomputer-kale-tts-1.static.hf.space/js/kale-tts.mjs';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.wasm.min.mjs';

const KALE_ROOT = 'https://kalecomputer-kale-tts-1.static.hf.space';
const MODEL_ROOT = `${KALE_ROOT}/models/kale-tts-1-nano/`;
const FRONTEND_ROOT = `${KALE_ROOT}/frontend/`;
const ORT_ROOT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
const CACHE_NAME = 'kale-tts-1-nano-v1';

ort.env.wasm.wasmPaths = ORT_ROOT;
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(4, self.navigator?.hardwareConcurrency || 1)
  : 1;
ort.env.logLevel = 'error';

let tts = null;
const canceledRequests = new Set();

function post(status, extra = {}) {
  self.postMessage({ status, ...extra });
}

// The Space signs large-file redirects with no-store. Cache the stable public
// URLs explicitly so the graph, front end, and voices survive a page reload.
const nativeFetch = self.fetch.bind(self);
self.fetch = async (input, init) => {
  const url = String(input?.url || input);
  if (!url.startsWith(KALE_ROOT) || !('caches' in self)) {
    return nativeFetch(input, init);
  }

  const key = url.split('?')[0];
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(key);
  if (cached) return cached;

  const response = await nativeFetch(input, init);
  if (!response.ok) return response;

  if (key.endsWith('/model.onnx') && response.body) {
    const total = Number(response.headers.get('content-length')) || 0;
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      post('progress', {
        message: 'Loading Kale TTS nano model',
        pct: total ? Math.round(loaded * 100 / total) : -1,
        detail: total
          ? `${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`
          : `${(loaded / 1e6).toFixed(1)} MB`,
      });
    }

    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    const stored = new Response(bytes, {
      headers: {
        'content-type': response.headers.get('content-type') || 'application/octet-stream',
        'content-length': String(loaded),
      },
    });
    await cache.put(key, stored.clone());
    return stored;
  }

  await cache.put(key, response.clone());
  return response;
};

async function loadModel(voice = 's057b') {
  if (!tts) {
    post('progress', { message: 'Loading Kale TTS nano (INT8, WASM)...' });
    tts = await KaleTTS.load(MODEL_ROOT, {
      ort,
      frontend: FRONTEND_ROOT,
      graph: 'int8',
      executionProviders: ['wasm'],
      sessionOptions: { graphOptimizationLevel: 'all' },
    });
  }

  post('progress', { message: `Loading Kale voice ${voice}...` });
  await tts.preload(voice);
}

async function clearCache() {
  tts = null;
  if ('caches' in self) await caches.delete(CACHE_NAME);
}

self.addEventListener('message', async (event) => {
  const data = event.data || {};

  if (data.command === 'cancel') {
    if (data.requestId != null) canceledRequests.add(data.requestId);
    return;
  }

  if (data.command === 'clear-cache') {
    await clearCache();
    post('ready');
    return;
  }

  if (data.command === 'load') {
    try {
      await loadModel(data.voice || 's057b');
      post('loaded', { backend: 'wasm', dtype: 'int8', sampleRate: 24000 });
    } catch (error) {
      tts = null;
      post('error', { error: error?.message || String(error) });
    }
    return;
  }

  if (data.command !== 'tts') return;

  const requestId = data.requestId;
  try {
    const text = data.text?.trim();
    if (!text) throw new Error('No text provided');
    if (canceledRequests.has(requestId)) throw new DOMException('Canceled', 'AbortError');

    const voice = data.voice || 's057b';
    await loadModel(voice);
    post('progress', { message: `Generating Kale voice ${voice}...` });
    const result = await tts.speak(text, { voice });

    if (canceledRequests.has(requestId)) throw new DOMException('Canceled', 'AbortError');
    const wav = toWav(result.audio, result.sampleRate);
    post('complete', {
      requestId,
      sampleRate: result.sampleRate,
      durationSec: result.audio.length / result.sampleRate,
      audio: new Blob([wav], { type: 'audio/wav' }),
    });
  } catch (error) {
    const canceled = error?.name === 'AbortError' || canceledRequests.has(requestId);
    post(canceled ? 'canceled' : 'error', {
      requestId,
      error: canceled ? 'Canceled' : error?.message || String(error),
    });
  } finally {
    if (requestId != null) canceledRequests.delete(requestId);
  }
});

post('ready');
