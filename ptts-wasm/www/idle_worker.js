import init, { Model, Tokenizer } from './idle_tts_wasm.js';

const HF_BASE = 'https://huggingface.co/idle-intelligence/pocket-tts-gguf/resolve/main';
// Voice KV caches must match the exact Pocket TTS revision used for the GGUF files.
const VOICE_BASE = 'https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/4e1e0a3e611c51c0b4ed8174fc10f32a54644303';
const SUPPORTED_LANGUAGES = new Set(['english', 'french', 'german', 'spanish', 'portuguese', 'italian']);
const ASSET_CACHE = 'pocket-tts-idle-assets-v2';
const CACHE_KEY_PREFIX = '/__pocket_tts_idle_asset__?url=';

let model = null;
let tokenizer = null;
let language = 'english';
let voiceIndexMap = {};

function modelUrl() {
  return `${HF_BASE}/languages/${language}/pocket-tts-q8_0.gguf`;
}

function tokenizerUrl() {
  return `${HF_BASE}/languages/${language}/tokenizer.model`;
}

function voiceUrl(name) {
  return `${VOICE_BASE}/languages/${language}/embeddings/${name}.safetensors`;
}

function post(type, data = {}, transferables = []) {
  self.postMessage({ type, ...data }, transferables);
}

function cacheRequest(url) {
  return new Request(CACHE_KEY_PREFIX + encodeURIComponent(url));
}

async function fetchWithProgress(url, label) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const contentType = resp.headers.get('content-type') || 'application/octet-stream';
  const total = parseInt(resp.headers.get('content-length') || '0', 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) {
      const pct = Math.round(received / total * 100);
      post('progress', {
        label,
        pct,
        detail: `${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`,
      });
    } else {
      post('progress', { label, pct: -1, detail: `${(received / 1e6).toFixed(1)} MB` });
    }
  }
  post('progress_done');

  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return { bytes, contentType };
}

async function fetchCachedBytes(url, label) {
  if (!('caches' in self)) return (await fetchWithProgress(url, label)).bytes;

  let cache = null;
  let request = null;
  try {
    cache = await caches.open(ASSET_CACHE);
    request = cacheRequest(url);
    const cached = await cache.match(request);
    if (cached) {
      post('status', { message: `${label}: loaded from browser cache` });
      return new Uint8Array(await cached.arrayBuffer());
    }
  } catch (err) {
    console.warn(`Could not read browser cache for ${label}`, err);
  }

  const { bytes, contentType } = await fetchWithProgress(url, label);
  if (cache && request) {
    try {
      await cache.put(request, new Response(bytes.slice(), {
        headers: {
          'content-length': String(bytes.byteLength),
          'content-type': contentType,
        },
      }));
      post('status', { message: `${label}: saved to browser cache` });
    } catch (err) {
      console.warn(`Could not cache ${label}`, err);
      post('status', { message: `${label}: downloaded, but browser cache storage was unavailable or full` });
    }
  }
  return bytes;
}

async function clearAssetCache() {
  if (!('caches' in self)) {
    post('cache_cleared', { message: 'Browser Cache API is not available.' });
    return;
  }
  const deleted = (await Promise.all([
    caches.delete(ASSET_CACHE),
    caches.delete('pocket-tts-idle-assets-v1'),
  ])).some(Boolean);
  post('cache_cleared', {
    message: deleted
      ? 'Cached Idle Intelligence Pocket TTS assets cleared.'
      : 'No cached Idle Intelligence Pocket TTS assets were found.',
  });
}

async function loadVoice(name) {
  if (voiceIndexMap[name] != null) return voiceIndexMap[name];
  const voiceData = await fetchCachedBytes(voiceUrl(name), `Idle ${language} voice: ${name}`);
  const voiceIndex = model.add_voice(voiceData);
  voiceIndexMap[name] = voiceIndex;
  return voiceIndex;
}

async function handleLoad(selectedLanguage, voiceName) {
  language = selectedLanguage || 'english';
  if (!SUPPORTED_LANGUAGES.has(language)) throw new Error(`Unsupported Idle language: ${language}`);
  voiceIndexMap = {};

  await init('./idle_tts_wasm_bg.wasm');
  post('status', { message: 'Candle WASM initialized. Loading tokenizer and model...' });

  const tokData = await fetchCachedBytes(tokenizerUrl(), `Idle ${language} tokenizer`);
  tokenizer = new Tokenizer(tokData);
  post('status', { message: `Idle ${language} tokenizer loaded (${tokenizer.vocab_size()} pieces)` });

  const modelWeights = await fetchCachedBytes(modelUrl(), `Idle ${language} Q8_0 model weights`);
  post('status', { message: `Initializing Idle Intelligence ${language} model...` });
  model = new Model(modelWeights, language);
  await loadVoice(voiceName);

  post('loaded', {
    sampleRate: model.sample_rate(),
    voiceSet: `${language} / ${voiceName}`,
    engineLabel: 'Idle Intelligence Candle Q8_0 multilingual',
  });
}

async function handleGenerate(text, voiceName, temperature) {
  const voiceIndex = await loadVoice(voiceName);
  const [processedText, framesAfterEos] = model.prepare_text(text);
  const tokenIds = tokenizer.encode(processedText);
  post('gen_start', { numTokens: tokenIds.length });

  const promptT0 = performance.now();
  model.start_generation(voiceIndex, tokenIds, framesAfterEos, temperature);
  const promptMs = performance.now() - promptT0;

  let step = 0;
  let stepMsTotal = 0;
  let stepMsMin = Infinity;
  let stepMsMax = 0;
  while (true) {
    const t0 = performance.now();
    const chunk = model.generation_step();
    const dt = performance.now() - t0;
    if (!chunk) break;
    stepMsTotal += dt;
    stepMsMin = Math.min(stepMsMin, dt);
    stepMsMax = Math.max(stepMsMax, dt);
    post('chunk', { data: chunk, step }, [chunk.buffer]);
    step++;
  }

  post('done', {
    promptMs,
    numSteps: step,
    stepMsAvg: step > 0 ? stepMsTotal / step : 0,
    stepMsMin: step > 0 ? stepMsMin : 0,
    stepMsMax,
  });
}

self.onmessage = async (e) => {
  const { type, ...data } = e.data;
  try {
    if (type === 'load') {
      await handleLoad(data.language, data.voiceName);
    } else if (type === 'clear_cache') {
      await clearAssetCache();
    } else if (type === 'generate') {
      await handleGenerate(data.text, data.voiceName, data.temperature);
    }
  } catch (err) {
    post('error', { message: err.message || String(err) });
    console.error(err);
  }
};
