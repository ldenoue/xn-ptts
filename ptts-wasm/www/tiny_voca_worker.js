// Browser adapter for tiny-voca's official Pyodide + ONNX Runtime Web demo.
// Large model and pronunciation assets stay on the pinned Hugging Face revision.

const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/';
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const REVISION = 'e5f9b11651a4db525fb207a56a91e4df2d1dd387';
const ASSET_ROOT = `https://huggingface.co/n4ze3m/tiny-voca/resolve/${REVISION}`;
const CACHE_NAME = `tiny-voca-${REVISION}`;
const DATA = ['lexicon.tsv', 'align.json', 'g2p.json', 'g2p.npz', 'homographs.json'];
const VOICES = { F2: 0, M2: 1 };

importScripts(`${PYODIDE}pyodide.js`, `${ORT}ort.wasm.min.js`);

let pyodide;
let phonemize;
let session;
let config;
let loadPromise;

function post(type, data = {}, transfer = []) {
  self.postMessage({ type, ...data }, transfer);
}

function progress(label, received, total) {
  const pct = total ? Math.round(received / total * 100) : -1;
  const detail = total
    ? `${(received / 1_000_000).toFixed(1)} / ${(total / 1_000_000).toFixed(1)} MB`
    : `${(received / 1_000_000).toFixed(1)} MB`;
  post('progress', { label, pct, detail });
}

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    progress(label, received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

async function download(url, label) {
  let cache;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (_) {
    cache = null;
  }
  const bytes = await fetchBytes(url, label);
  if (cache) cache.put(url, new Response(bytes)).catch(() => {});
  return bytes;
}

async function loadPhonemizer() {
  post('progress', { label: 'Starting Pyodide phonemizer', pct: -1, detail: 'Loading Python runtime' });
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  await pyodide.loadPackage('numpy');
  const [source, ...files] = await Promise.all([
    download('./tiny-voca-runtime/dizione.zip', 'Loading phonemizer'),
    ...DATA.map((name) => download(`${ASSET_ROOT}/phonemizer/${name}`, `Loading ${name}`)),
  ]);
  pyodide.FS.mkdirTree('/dizione-data');
  DATA.forEach((name, index) => pyodide.FS.writeFile(`/dizione-data/${name}`, files[index]));
  pyodide.unpackArchive(source.buffer, 'zip', { extractDir: '/dizione-src' });
  pyodide.runPython(`
import json, os, sys
os.environ['DIZIONE_DATA'] = '/dizione-data'
sys.path.insert(0, '/dizione-src')
import dizione

def phonemize_json(text):
    try:
        result = dizione.phonemize(text)
    except ValueError:
        if any(word for mark, words in dizione.pieces(text) if not mark for word in words):
            raise
        return 'null'
    return json.dumps({'ids': list(result.ids), 'stress': list(result.stress), 'text': str(result)})

phonemize_json('Warm up.')
`);
  phonemize = pyodide.globals.get('phonemize_json');
}

async function loadModel() {
  const [configBytes, modelBytes] = await Promise.all([
    download(`${ASSET_ROOT}/config.json`, 'Loading configuration'),
    download(`${ASSET_ROOT}/onnx/model.onnx`, 'Loading tiny-voca model'),
  ]);
  config = JSON.parse(new TextDecoder().decode(configBytes));
  post('progress', { label: 'Starting tiny-voca', pct: -1, detail: 'Creating ONNX session' });
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'] });
}

async function load() {
  if (!loadPromise) {
    loadPromise = Promise.all([loadPhonemizer(), loadModel()]).catch((error) => {
      loadPromise = null;
      throw error;
    });
  }
  await loadPromise;
}

const ABBREVIATIONS = /(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$/i;

function sentences(text) {
  const parts = [];
  let current = '';
  for (const piece of text.replace(/\s+/g, ' ').trim().split(/(?<=[.!?…]["')\]]?)\s+(?=["'(\[]?[A-Z0-9])/)) {
    current = current ? `${current} ${piece}` : piece;
    if (!ABBREVIATIONS.test(current)) {
      parts.push(current);
      current = '';
    }
  }
  if (current) parts.push(current);
  return parts.filter((part) => /[\p{L}\p{N}]/u.test(part));
}

function finished(text) {
  const body = text.replace(/\s+$/, '');
  if (!body) return text;
  const last = body.split(/\s+/).pop();
  const marked = /[\p{L}\p{N}]/u.test(last)
    ? /[.!?…,;:]["'”’)\]»]*$/.test(last)
    : /[.!?…]/.test(last) || /^[,;:]$/.test(last);
  return marked ? text : `${body}.`;
}

function fit(sentence) {
  const result = JSON.parse(phonemize(sentence));
  if (!result) return [];
  if (result.ids.length <= config.model.max_tokens - 12) return [result];
  const pieces = sentence.split(/(?<=[,;:—])\s+/);
  if (pieces.length > 1) return pieces.flatMap(fit);
  const words = sentence.split(' ');
  if (words.length < 2) return [];
  const half = Math.ceil(words.length / 2);
  return [...fit(words.slice(0, half).join(' ')), ...fit(words.slice(half).join(' '))];
}

async function synthesize(phones, voice, speed) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor('int64', BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor('float32', Float32Array.from([value]), [1]);
  const output = await session.run({
    tokens: integers(phones.ids),
    stress: integers(phones.stress),
    voice: new ort.Tensor('int64', BigInt64Array.from([BigInt(voice)]), [1]),
    speed: scalar(speed),
    pitch_shift: scalar(0),
    pitch_range: scalar(1),
  });
  return output.audio.data;
}

const PEAK = 0.99;
function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) {
    for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
  }
}

async function generate(text, voiceName, speed) {
  await load();
  const voice = VOICES[voiceName];
  if (voice == null) throw new Error(`Unknown tiny-voca voice: ${voiceName}`);
  if (!(speed >= 0.5 && speed <= 2)) throw new Error('Speed must be between 0.5 and 2.0.');

  const started = performance.now();
  const list = sentences(finished(text));
  const chunks = [];
  const stepTimes = [];
  let phonemeMs = 0;
  let steps = 0;
  post('gen_start', { numTokens: null });

  for (const sentence of list) {
    const mark = performance.now();
    const utterances = fit(sentence);
    phonemeMs += performance.now() - mark;
    for (const phones of utterances) {
      const stepStarted = performance.now();
      const audio = await synthesize(phones, voice, speed);
      stepTimes.push(performance.now() - stepStarted);
      chunks.push(audio);
      steps += 1;
      const heard = audio.slice();
      limitPeak(heard);
      post('chunk', { data: heard.buffer }, [heard.buffer]);
    }
  }
  if (!chunks.length) throw new Error('tiny-voca only reads English for now.');
  post('done', {
    promptMs: phonemeMs,
    numSteps: steps,
    stepMsAvg: stepTimes.reduce((sum, value) => sum + value, 0) / stepTimes.length,
    stepMsMin: Math.min(...stepTimes),
    stepMsMax: Math.max(...stepTimes),
    totalMs: performance.now() - started,
  });
}

self.addEventListener('message', async ({ data }) => {
  try {
    if (data.type === 'load') {
      await load();
      post('progress_done');
      post('loaded', {
        sampleRate: config.sample_rate,
        voiceSet: 'F2 and M2 (American English)',
        engineLabel: 'tiny-voca 4.5M ORT',
      });
    } else if (data.type === 'generate') {
      await generate(data.text?.trim(), data.voiceName || 'M2', Number(data.speed || 1));
    } else if (data.type === 'clear_cache') {
      if (session) await session.release();
      session = null;
      config = null;
      loadPromise = null;
      await caches.delete(CACHE_NAME);
      post('cache_cleared', { message: 'tiny-voca model and pronunciation cache cleared.' });
    }
  } catch (error) {
    console.error(error);
    post('error', { message: error instanceof Error ? error.message : String(error) });
  }
});

post('status', { message: 'tiny-voca worker ready.' });
