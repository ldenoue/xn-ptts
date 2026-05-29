import { KokoroTTS, env } from 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/dist/kokoro.web.js';

const MODEL_ID = 'onnx-community/Kokoro-82M-v1.0-ONNX';
const HF_ROOT = `https://huggingface.co/${MODEL_ID}/resolve/main`;
const ORT_WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/';
const LEGACY_CACHE_NAMES = ['kokoro-voices'];
const MODEL_CACHE_MATCHERS = ['kokoro', 'transformers', 'huggingface'];

env.wasmPaths = ORT_WASM_ROOT;

let tts = null;
let loadedDtype = null;
let loadedBackend = null;
const canceledRequests = new Set();

function post(status, extra = {}) {
  self.postMessage({ status, ...extra });
}

function isCanceled(requestId) {
  return requestId != null && canceledRequests.has(requestId);
}

function throwIfCanceled(requestId) {
  if (isCanceled(requestId)) throw new DOMException('Canceled', 'AbortError');
}

function progressMessage(progress) {
  const file = progress.file || progress.name || progress.url || 'Kokoro asset';
  if (progress.status === 'progress') {
    const pct = Number.isFinite(progress.progress) ? Math.round(progress.progress) : -1;
    const loaded = Number.isFinite(progress.loaded) ? progress.loaded : 0;
    const total = Number.isFinite(progress.total) ? progress.total : 0;
    const detail = total > 0
      ? `${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`
      : loaded > 0 ? `${(loaded / 1e6).toFixed(1)} MB` : '';
    post('progress', { message: `Loading ${file}`, pct, detail });
    return;
  }

  if (progress.status === 'ready') {
    post('progress', { message: `${file} ready` });
  } else if (progress.status === 'done') {
    post('progress', { message: `${file} loaded` });
  } else if (progress.status === 'initiate') {
    post('progress', { message: `Fetching ${file}` });
  }
}

async function loadModel(dtype = 'q8', backend = 'wasm') {
  if (tts && loadedDtype === dtype && loadedBackend === backend) return;

  post('progress', { message: `Loading Kokoro 82M (${dtype}, ${backend.toUpperCase()})...` });
  tts = await KokoroTTS.from_pretrained(MODEL_ID, {
    dtype,
    device: backend,
    progress_callback: progressMessage,
  });
  loadedDtype = dtype;
  loadedBackend = backend;
}

async function ensureVoiceCached(voice) {
  if (!voice || !('caches' in self)) return;

  const url = `${HF_ROOT}/voices/${voice}.bin`;
  const cache = await caches.open('kokoro-voices');
  const cached = await cache.match(url);
  if (cached) return;

  post('progress', { message: `Loading Kokoro voice ${voice}...` });
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load Kokoro voice ${voice}: HTTP ${response.status}`);
  await cache.put(url, response);
}

async function deleteKokoroCaches() {
  if (!('caches' in self)) return;

  const names = await caches.keys();
  const targets = new Set(LEGACY_CACHE_NAMES);
  for (const name of names) {
    const lower = name.toLowerCase();
    if (MODEL_CACHE_MATCHERS.some((matcher) => lower.includes(matcher))) {
      targets.add(name);
    }
  }
  await Promise.all([...targets].map((cacheName) => caches.delete(cacheName)));
}

async function speak(data) {
  const requestId = data.requestId;
  const text = data.text?.trim();
  if (!text) throw new Error('No text provided');

  const dtype = data.dtype || 'q8';
  const backend = data.backend || 'wasm';
  const voice = data.voice || 'af_heart';
  const speed = Number(data.speed || 1);

  throwIfCanceled(requestId);
  await loadModel(dtype, backend);
  throwIfCanceled(requestId);

  post('progress', { message: `Generating Kokoro voice ${voice}...` });
  const audio = await tts.generate(text, { voice, speed });
  throwIfCanceled(requestId);

  const blob = audio.toBlob();
  return {
    blob,
    sampleRate: audio.sampling_rate || 24000,
    durationSec: audio.audio?.length ? audio.audio.length / (audio.sampling_rate || 24000) : undefined,
  };
}

self.addEventListener('message', async (event) => {
  const data = event.data || {};

  if (data.command === 'cancel') {
    if (data.requestId != null) canceledRequests.add(data.requestId);
    return;
  }

  if (data.command === 'clear-cache') {
    tts = null;
    loadedDtype = null;
    loadedBackend = null;
    await deleteKokoroCaches();
    post('ready');
    return;
  }

  if (data.command === 'load') {
    try {
      await loadModel(data.dtype || 'q8', data.backend || 'wasm');
      await ensureVoiceCached(data.voice || 'af_heart');
      post('loaded', {
        backend: loadedBackend,
        dtype: loadedDtype,
        sampleRate: 24000,
      });
    } catch (err) {
      post('error', { error: err?.message || String(err), id: data.id });
    }
    return;
  }

  if (data.command !== 'tts') return;

  try {
    const result = await speak(data);
    post('complete', {
      text: data.text,
      id: data.id,
      requestId: data.requestId,
      reason: data.reason,
      sampleRate: result.sampleRate,
      durationSec: result.durationSec,
      audio: result.blob,
    });
  } catch (err) {
    const canceled = err?.name === 'AbortError' || isCanceled(data.requestId);
    post(canceled ? 'canceled' : 'error', {
      error: canceled ? 'Canceled' : err?.message || String(err),
      id: data.id,
      requestId: data.requestId,
      reason: data.reason,
    });
  } finally {
    if (data.requestId != null) canceledRequests.delete(data.requestId);
  }
});

post('ready');
