function isIOS() {
  const ua = self.navigator?.userAgent || '';
  const platform = self.navigator?.platform || '';
  return /iPhone|iPad|iPod/i.test(ua) || (platform === 'MacIntel' && self.navigator?.maxTouchPoints > 1);
}

const REQUESTED_BACKEND = new URL(self.location.href).searchParams.get('backend') === 'webgpu' && !isIOS() ? 'webgpu' : 'wasm';
const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

console.log(
  `[supertonic-ort-worker] device=${REQUESTED_BACKEND}, ios=${isIOS()}` +
  (REQUESTED_BACKEND === 'wasm' ? ', wasmThreads=1' : '')
);

importScripts(`${ORT_DIST}${REQUESTED_BACKEND === 'webgpu' ? 'ort.all.min.js' : 'ort.wasm.min.js'}`);

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = ORT_DIST;
ort.env.logLevel = 'warning';

const MODEL_LABEL = 'Supertonic 3';
const MODEL_VERSION = 'supertonic-3';
const HF_ASSET_ROOT = 'https://huggingface.co/Supertone/supertonic-3/resolve/main';
const PROXY_ASSET_ROOT = `${self.location.origin}/supertonic-model/${MODEL_VERSION}`;
const CACHE_NAME = 'supertonic-ort-worker-v2';
const LEGACY_CACHE_NAMES = ['supertonic-ort-worker-v1'];
let currentLang = 'en';
const MODEL_ASSET_PATHS = [
  'onnx/tts.json',
  'onnx/unicode_indexer.json',
  'onnx/duration_predictor.onnx',
  'onnx/text_encoder.onnx',
  'onnx/vector_estimator.onnx',
  'onnx/vocoder.onnx'
];
const VOICE_NAMES = ['F1', 'F2', 'F3', 'F4', 'F5', 'M1', 'M2', 'M3', 'M4', 'M5'];
const pendingFetches = new Map();

const state = {
  cfgs: null,
  textProcessor: null,
  sessions: null,
  backend: null,
  styleName: null,
  style: null,
  sampleRate: 24000
};
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

function modelAssetUrl(assetPath) {
  return `${PROXY_ASSET_ROOT}/${assetPath}`;
}

function hfAssetUrl(assetPath) {
  return `${HF_ASSET_ROOT}/${assetPath}`;
}

function assertOk(response, assetPath) {
  if (!response.ok) throw new Error(`Failed to load ${assetPath}: HTTP ${response.status}`);
}

async function fetchModelAsset(assetPath, cache, cacheKey) {
  let response = null;
  try {
    response = await fetch(modelAssetUrl(assetPath));
  } catch (err) {
    console.warn(`[supertonic-ort-worker] model proxy failed for ${assetPath}; falling back to Hugging Face`, err);
  }

  if (!response?.ok) {
    console.warn(`[supertonic-ort-worker] model proxy miss for ${assetPath}; falling back to Hugging Face`);
    response = await fetch(hfAssetUrl(assetPath));
  }
  assertOk(response, assetPath);

  if (cache) {
    try {
      await cache.put(cacheKey, response.clone());
    } catch (err) {
      console.warn(`[supertonic-ort-worker] failed to cache ${assetPath}`, err);
    }
  }

  return response;
}

async function cachedFetch(assetPath) {
  const cacheKey = modelAssetUrl(assetPath);
  if (!('caches' in self)) return fetchModelAsset(assetPath, null, cacheKey);

  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(cacheKey);
  if (cached) return cached;

  const pending = pendingFetches.get(cacheKey);
  if (pending) return (await pending).clone();

  const pendingFetch = fetchModelAsset(assetPath, cache, cacheKey);
  pendingFetches.set(cacheKey, pendingFetch);
  try {
    return (await pendingFetch).clone();
  } finally {
    pendingFetches.delete(cacheKey);
  }
}

async function cacheModelAssets(voices = VOICE_NAMES) {
  const voiceAssets = voices.map((voice) => `voice_styles/${voice}.json`);
  await Promise.allSettled([...MODEL_ASSET_PATHS, ...voiceAssets].map((assetPath) => cachedFetch(assetPath)));
}

async function deleteModelCaches() {
  if (!('caches' in self)) return;
  await Promise.all([CACHE_NAME, ...LEGACY_CACHE_NAMES].map((cacheName) => caches.delete(cacheName)));
}

function warmVoiceStyleCache() {
  cacheModelAssets().catch((err) => {
    console.warn('[supertonic-ort-worker] failed to warm voice style cache', err);
  });
}

async function fetchJson(assetPath) {
  return (await cachedFetch(assetPath)).json();
}

async function fetchBytes(assetPath) {
  return new Uint8Array(await (await cachedFetch(assetPath)).arrayBuffer());
}

function preprocessText(text) {
  let normalized = text.normalize('NFKD');
  normalized = normalized.replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]+/gu, '');
  normalized = normalized.replace(/[–‑—]/g, '-');
  normalized = normalized.replace(/[“”]/g, '"');
  normalized = normalized.replace(/[‘’´`]/g, "'");
  normalized = normalized.replace(/[\[\]|/#\\♥☆♡©]/g, ' ');
  normalized = normalized.replace(/\s+/g, ' ').trim();
  if (!/[.!?;:,'"')\]}…。」』】〉》›»]$/.test(normalized)) normalized += '.';
  return `<${currentLang}>${normalized}</${currentLang}>`;
}

class UnicodeProcessor {
  constructor(indexer) {
    this.indexer = indexer;
  }

  encode(text) {
    const processed = preprocessText(text);
    const ids = new BigInt64Array(processed.length);
    const mask = new Float32Array(processed.length);
    for (let i = 0; i < processed.length; i++) {
      const codePoint = processed.codePointAt(i);
      ids[i] = BigInt(codePoint < this.indexer.length ? this.indexer[codePoint] : -1);
      mask[i] = 1;
    }
    return { ids, mask, length: processed.length };
  }
}

function flattenNumbers(source, out, cursor) {
  if (Array.isArray(source)) {
    for (const item of source) flattenNumbers(item, out, cursor);
    return;
  }
  out[cursor.index++] = source;
}

async function loadVoiceStyle(voice) {
  const styleJson = await fetchJson(`voice_styles/${voice}.json`);
  const ttlDims = styleJson.style_ttl.dims;
  const dpDims = styleJson.style_dp.dims;
  const ttl = new Float32Array(ttlDims.reduce((a, b) => a * b, 1));
  const dp = new Float32Array(dpDims.reduce((a, b) => a * b, 1));

  flattenNumbers(styleJson.style_ttl.data, ttl, { index: 0 });
  flattenNumbers(styleJson.style_dp.data, dp, { index: 0 });

  return {
    ttl: new ort.Tensor('float32', ttl, ttlDims),
    dp: new ort.Tensor('float32', dp, dpDims)
  };
}

async function createSession(name, file) {
  const backend = state.backend || 'wasm';
  post('progress', { message: `Loading ${name}...` });
  const bytes = await fetchBytes(`onnx/${file}`);
  post('progress', { message: `Opening ${name} on ${backend.toUpperCase()}...` });
  return ort.InferenceSession.create(bytes, {
    executionProviders: [backend],
    executionMode: 'sequential',
    graphOptimizationLevel: 'basic',
    enableMemPattern: false,
    enableCpuMemArena: false
  });
}

async function loadModel(backend = 'wasm') {
  if (backend !== REQUESTED_BACKEND) {
    throw new Error(`Worker was opened for ${REQUESTED_BACKEND.toUpperCase()}, not ${backend.toUpperCase()}.`);
  }

  if (state.sessions && state.backend === backend) return;

  releaseSessions();
  state.backend = backend;
  state.cfgs = null;
  state.textProcessor = null;
  state.sessions = null;
  state.styleName = null;
  state.style = null;

  post('progress', { message: `Loading ${MODEL_LABEL} configuration...` });
  console.log(`[supertonic-ort-worker] loading ${MODEL_LABEL} on ${backend.toUpperCase()}${isIOS() ? ' single-thread' : ''}`);
  state.cfgs = await fetchJson('onnx/tts.json');
  state.sampleRate = state.cfgs.ae.sample_rate;
  state.textProcessor = new UnicodeProcessor(await fetchJson('onnx/unicode_indexer.json'));

  const duration = await createSession('duration predictor', 'duration_predictor.onnx');
  const textEncoder = await createSession('text encoder', 'text_encoder.onnx');
  const vectorEstimator = await createSession('vector estimator', 'vector_estimator.onnx');
  const vocoder = await createSession('vocoder', 'vocoder.onnx');
  state.sessions = { duration, textEncoder, vectorEstimator, vocoder };
  warmVoiceStyleCache();
}

async function ensureStyle(voice) {
  if (state.styleName === voice && state.style) return;
  post('progress', { message: `Loading voice ${voice}...` });
  state.style = await loadVoiceStyle(voice);
  state.styleName = voice;
}

function releaseSessions() {
  if (!state.sessions) return;
  for (const session of Object.values(state.sessions)) {
    if (typeof session.release === 'function') session.release();
  }
}

function sampleNoisyLatent(duration) {
  const bsz = 1;
  const wavLenMax = Math.floor(duration * state.sampleRate);
  const chunkSize = state.cfgs.ae.base_chunk_size * state.cfgs.ttl.chunk_compress_factor;
  const latentLen = Math.floor((wavLenMax + chunkSize - 1) / chunkSize);
  const latentDim = state.cfgs.ttl.latent_dim * state.cfgs.ttl.chunk_compress_factor;
  const xt = new Float32Array(bsz * latentDim * latentLen);
  const latentMask = new Float32Array(bsz * latentLen);
  const latentLength = Math.floor((wavLenMax + chunkSize - 1) / chunkSize);

  for (let t = 0; t < latentLen; t++) latentMask[t] = t < latentLength ? 1 : 0;

  let offset = 0;
  for (let d = 0; d < latentDim; d++) {
    for (let t = 0; t < latentLen; t++) {
      const u1 = Math.max(0.0001, Math.random());
      const u2 = Math.random();
      xt[offset++] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * latentMask[t];
    }
  }

  return { xt, latentMask, xtShape: [bsz, latentDim, latentLen], latentMaskShape: [bsz, 1, latentLen] };
}

async function synthesize(text, steps, speed, requestId) {
  throwIfCanceled(requestId);
  const encoded = state.textProcessor.encode(text);
  const textIdsTensor = new ort.Tensor('int64', encoded.ids, [1, encoded.length]);
  const textMaskTensor = new ort.Tensor('float32', encoded.mask, [1, 1, encoded.length]);

  throwIfCanceled(requestId);
  const durationOutputs = await state.sessions.duration.run({
    text_ids: textIdsTensor,
    style_dp: state.style.dp,
    text_mask: textMaskTensor
  });
  throwIfCanceled(requestId);
  const duration = durationOutputs.duration.data[0] / speed;

  const textEncoderOutputs = await state.sessions.textEncoder.run({
    text_ids: textIdsTensor,
    style_ttl: state.style.ttl,
    text_mask: textMaskTensor
  });
  throwIfCanceled(requestId);

  let { xt, latentMask, xtShape, latentMaskShape } = sampleNoisyLatent(duration);
  const latentMaskTensor = new ort.Tensor('float32', latentMask, latentMaskShape);
  const totalStepTensor = new ort.Tensor('float32', new Float32Array([steps]), [1]);

  for (let step = 0; step < steps; step++) {
    throwIfCanceled(requestId);
    post('progress', { message: `Denoising ${step + 1}/${steps}...` });
    const vectorOutputs = await state.sessions.vectorEstimator.run({
      noisy_latent: new ort.Tensor('float32', xt, xtShape),
      text_emb: textEncoderOutputs.text_emb,
      style_ttl: state.style.ttl,
      latent_mask: latentMaskTensor,
      text_mask: textMaskTensor,
      current_step: new ort.Tensor('float32', new Float32Array([step]), [1]),
      total_step: totalStepTensor
    });
    xt = vectorOutputs.denoised_latent.data;
  }

  throwIfCanceled(requestId);
  post('progress', { message: 'Running vocoder...' });
  const vocoderOutputs = await state.sessions.vocoder.run({
    latent: new ort.Tensor('float32', xt, xtShape)
  });

  throwIfCanceled(requestId);
  const wavLen = Math.floor(state.sampleRate * duration);
  return vocoderOutputs.wav_tts.data.subarray(0, wavLen);
}

function writeWavFile(audioData, sampleRate) {
  const bitsPerSample = 16;
  const dataSize = audioData.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bitsPerSample / 8, true);
  view.setUint16(32, bitsPerSample / 8, true);
  view.setUint16(34, bitsPerSample, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  const pcm = new Int16Array(buffer, 44);
  for (let i = 0; i < audioData.length; i++) {
    const clamped = Math.max(-1, Math.min(1, audioData[i]));
    pcm[i] = Math.floor(clamped * 32767);
  }
  return buffer;
}

async function speak(data) {
  const text = data.text.trim();
  const voice = data.voice || 'F5';
  const backend = data.backend || REQUESTED_BACKEND;
  currentLang = data.lang || data.language || currentLang;
  const steps = Number(data.num_inference_steps ?? data.steps ?? 5);
  const speed = Number(data.speed ?? 1.1);
  const requestId = data.requestId;

  throwIfCanceled(requestId);
  await loadModel(backend);
  throwIfCanceled(requestId);
  await ensureStyle(voice);
  const wav = await synthesize(text, steps, speed, requestId);
  return new Blob([writeWavFile(wav, state.sampleRate)], { type: 'audio/wav' });
}

self.addEventListener('message', async (event) => {
  const data = event.data || {};

  if (data.command === 'cancel') {
    if (data.requestId != null) canceledRequests.add(data.requestId);
    return;
  }

  if (data.command === 'clear-cache') {
    releaseSessions();
    await deleteModelCaches();
    state.sessions = null;
    state.backend = null;
    state.style = null;
    state.styleName = null;
    post('ready');
    return;
  }

  if (data.command === 'cache-models') {
    try {
      await cacheModelAssets(Array.isArray(data.voices) && data.voices.length ? data.voices : VOICE_NAMES);
      post('cached');
    } catch (err) {
      post('error', { error: err?.message || String(err), id: data.id });
    }
    return;
  }

  if (data.command === 'load') {
    try {
      const backend = data.backend || REQUESTED_BACKEND;
      currentLang = data.lang || data.language || currentLang;
      await loadModel(backend);
      await ensureStyle(data.voice || 'F5');
      post('loaded', { backend, sampleRate: state.sampleRate, lang: currentLang });
    } catch (err) {
      post('error', { error: err?.message || String(err), id: data.id });
    }
    return;
  }

  if (data.command !== 'tts') return;
  if (!data.text?.trim()) {
    post('error', { error: 'No text provided', id: data.id });
    return;
  }

  try {
    const blob = await speak(data);
    post('complete', { text: data.text, id: data.id, requestId: data.requestId, reason: data.reason, audio: blob });
  } catch (err) {
    const canceled = err?.name === 'AbortError' || isCanceled(data.requestId);
    post(canceled ? 'canceled' : 'error', {
      error: canceled ? 'Canceled' : err?.message || String(err),
      id: data.id,
      requestId: data.requestId,
      reason: data.reason
    });
  } finally {
    if (data.requestId != null) canceledRequests.delete(data.requestId);
  }
});

post('ready');
