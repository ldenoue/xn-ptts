import init, { Model } from './idle_tts_wasm.js';

const HF_BASE = 'https://huggingface.co/idle-intelligence/pocket-tts-gguf/resolve/main';
const VOICE_BASE = 'https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/main';
const MODEL_URL = `${HF_BASE}/pocket-tts-q8_0.gguf`;
const TOKENIZER_URL = `${HF_BASE}/tokenizer.model`;
const ASSET_CACHE = 'pocket-tts-idle-assets-v1';
const CACHE_KEY_PREFIX = '/__pocket_tts_idle_asset__?url=';
const VOICE_SET = 'embeddings_v2';
const VOICE_NAMES = ['alba', 'marius', 'javert', 'fantine', 'cosette', 'eponine', 'azelma'];

let model = null;
let tokenizer = null;
let voiceIndexMap = {};

function voiceUrl(name) {
  return `${VOICE_BASE}/${VOICE_SET}/${name}.safetensors`;
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

  const buf = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.length;
  }
  return { bytes: buf, contentType };
}

async function fetchCachedBytes(url, label) {
  if (!('caches' in self)) {
    const { bytes } = await fetchWithProgress(url, label);
    return bytes;
  }

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

  const deleted = await caches.delete(ASSET_CACHE);
  const message = deleted
    ? 'Cached Idle Intelligence Pocket TTS assets cleared.'
    : 'No cached Idle Intelligence Pocket TTS assets were found.';
  post('cache_cleared', { message });
}

function decodeSentencepieceModel(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let pos = 0;

  function readVarint() {
    let result = 0, shift = 0;
    while (pos < buffer.length) {
      const b = buffer[pos++];
      result |= (b & 0x7f) << shift;
      shift += 7;
      if ((b & 0x80) === 0) return result;
    }
    return result;
  }

  function readBytes(n) {
    const data = buffer.slice(pos, pos + n);
    pos += n;
    return data;
  }

  function readVarIntFrom(buf, p) {
    let result = 0, shift = 0;
    while (p < buf.length) {
      const b = buf[p++];
      result |= (b & 0x7f) << shift;
      shift += 7;
      if ((b & 0x80) === 0) return { val: result, pos: p };
    }
    return { val: result, pos: p };
  }

  function decodePiece(data) {
    let pPos = 0, piece = '', score = 0, type = 1;
    const pView = new DataView(data.buffer, data.byteOffset, data.byteLength);
    while (pPos < data.length) {
      const key = readVarIntFrom(data, pPos);
      pPos = key.pos;
      const fieldNum = key.val >>> 3;
      const wireType = key.val & 0x7;
      if (fieldNum === 1 && wireType === 2) {
        const len = readVarIntFrom(data, pPos);
        pPos = len.pos;
        piece = new TextDecoder().decode(data.slice(pPos, pPos + len.val));
        pPos += len.val;
      } else if (fieldNum === 2 && wireType === 5) {
        score = pView.getFloat32(pPos, true);
        pPos += 4;
      } else if (fieldNum === 3 && wireType === 0) {
        const v = readVarIntFrom(data, pPos);
        type = v.val;
        pPos = v.pos;
      } else {
        if (wireType === 0) { const v = readVarIntFrom(data, pPos); pPos = v.pos; }
        else if (wireType === 1) { pPos += 8; }
        else if (wireType === 2) { const len = readVarIntFrom(data, pPos); pPos = len.pos + len.val; }
        else if (wireType === 5) { pPos += 4; }
        else break;
      }
    }
    return { piece, score, type };
  }

  const pieces = [];
  while (pos < buffer.length) {
    const key = readVarint();
    const fieldNum = key >>> 3;
    const wireType = key & 0x7;
    if (fieldNum === 1 && wireType === 2) {
      const len = readVarint();
      pieces.push(decodePiece(readBytes(len)));
    } else {
      if (wireType === 0) { readVarint(); }
      else if (wireType === 1) { pos += 8; }
      else if (wireType === 2) { const len = readVarint(); pos += len; }
      else if (wireType === 5) { pos += 4; }
      else break;
    }
  }
  return pieces;
}

class UnigramTokenizer {
  constructor(pieces) {
    this.vocab = new Map();
    this.unkId = 0;
    for (let i = 0; i < pieces.length; i++) {
      const p = pieces[i];
      if (p.type === 2) this.unkId = i;
      if (p.type === 1 || p.type === 4 || p.type === 6) {
        this.vocab.set(p.piece, { id: i, score: p.score });
      }
    }
  }

  encode(text) {
    return this._viterbi('\u2581' + text.replace(/ /g, '\u2581'));
  }

  _viterbi(text) {
    const n = text.length;
    const best = new Array(n + 1);
    best[0] = { score: 0, len: 0, id: -1 };
    for (let i = 1; i <= n; i++) best[i] = { score: -Infinity, len: 0, id: -1 };

    for (let i = 0; i < n; i++) {
      if (best[i].score === -Infinity) continue;
      for (let len = 1; len <= n - i && len <= 64; len++) {
        const entry = this.vocab.get(text.substring(i, i + len));
        if (entry) {
          const newScore = best[i].score + entry.score;
          if (newScore > best[i + len].score) best[i + len] = { score: newScore, len, id: entry.id };
        }
      }
      if (best[i + 1].score === -Infinity) {
        const ch = text.charCodeAt(i);
        const byteEntry = this.vocab.get(`<0x${ch.toString(16).toUpperCase().padStart(2, '0')}>`);
        best[i + 1] = {
          score: best[i].score + (byteEntry ? byteEntry.score : -100),
          len: 1,
          id: byteEntry ? byteEntry.id : this.unkId,
        };
      }
    }

    const ids = [];
    let p = n;
    while (p > 0) {
      ids.push(best[p].id);
      p -= best[p].len;
    }
    ids.reverse();
    return new Uint32Array(ids);
  }
}

async function handleLoad() {
  voiceIndexMap = {};

  await init('./idle_tts_wasm_bg.wasm');
  post('status', { message: 'Candle WASM initialized. Loading tokenizer and model...' });

  const tokData = await fetchCachedBytes(TOKENIZER_URL, 'Idle tokenizer');
  tokenizer = new UnigramTokenizer(decodeSentencepieceModel(tokData));
  post('status', { message: 'Idle tokenizer loaded' });

  const modelWeights = await fetchCachedBytes(MODEL_URL, 'Idle Q8_0 model weights');
  post('status', { message: 'Initializing Idle Intelligence model...' });
  model = new Model(modelWeights);

  for (const name of VOICE_NAMES) {
    post('status', { message: `Loading voice: ${name}...` });
    const voiceData = await fetchCachedBytes(voiceUrl(name), `Idle voice ${VOICE_SET}: ${name}`);
    voiceIndexMap[name] = model.add_voice(voiceData);
  }

  post('loaded', {
    sampleRate: model.sample_rate(),
    voiceSet: VOICE_SET,
    engineLabel: 'Idle Intelligence Candle Q8_0',
  });
}

async function handleGenerate(text, voiceName, temperature) {
  const voiceIndex = voiceIndexMap[voiceName];
  if (voiceIndex == null) throw new Error(`Voice is not loaded: ${voiceName}`);

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
    if (dt < stepMsMin) stepMsMin = dt;
    if (dt > stepMsMax) stepMsMax = dt;
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
      await handleLoad();
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
