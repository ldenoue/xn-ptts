import {
  PhononTTS,
  clearCache,
} from 'https://cdn.jsdelivr.net/npm/phonon-tts@0.6.0/index.js';

let tts = null;
let loadedKey = null;
let loadedLabel = 'Phonon 31m';
let registeredVoices = new Set();

function post(type, extra = {}, transfer = []) {
  self.postMessage({ type, ...extra }, transfer);
}

function progress({ file, loaded = 0, total = 0, cached = false }) {
  const label = file?.split('/').pop() || 'Phonon asset';
  const pct = total > 0 ? Math.round((loaded / total) * 100) : -1;
  const detail = cached
    ? 'browser cache'
    : total > 0
      ? `${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`
      : `${(loaded / 1e6).toFixed(1)} MB`;
  post('progress', { label: `Phonon: ${label}`, pct, detail });
}

async function load(data) {
  const quant = data.quant || 'q8';
  const key = `${data.model || 'default'}|${data.language || 'en'}|${quant}`;
  if (tts && loadedKey === key) return;

  tts?.dispose();
  tts = null;
  loadedKey = null;
  registeredVoices = new Set();

  tts = await PhononTTS.load({
    lang: data.language || 'en',
    quant,
    model: data.model,
    voices: ['default'],
    workerUrl: new URL('./phonon_runtime_worker.js', self.location.href),
    onProgress: progress,
  });
  loadedKey = key;
  loadedLabel = data.engineLabel || 'Phonon';
  registeredVoices = new Set(tts.voices);
  await ensureVoice(data.voiceName, data.voiceUrl);
}

async function ensureVoice(name = 'default', url) {
  if (registeredVoices.has(name)) return;
  if (!url) throw new Error(`Phonon voice ${name} has no catalog URL.`);
  post('status', { message: `Loading Phonon voice ${name}...` });
  await tts.addVoice(name, url);
  registeredVoices.add(name);
}

async function generate(data) {
  if (!tts) throw new Error('Load the Phonon model before generating speech.');
  await ensureVoice(data.voiceName, data.voiceUrl);

  post('gen_start');
  const speech = tts.stream(data.text, {
    voice: data.voiceName || 'default',
    temperature: data.temperature,
  });

  for await (const pcm of speech) {
    post('chunk', { data: pcm.buffer }, [pcm.buffer]);
  }

  const stats = await speech.done;
  post('done', {
    promptMs: stats.promptMs,
    numSteps: stats.frames,
    stepMsAvg: stats.stepMs?.avg,
    stepMsMin: stats.stepMs?.min,
    stepMsMax: stats.stepMs?.max,
  });
}

self.addEventListener('message', async ({ data }) => {
  try {
    if (data.type === 'load') {
      await load(data);
      post('progress_done');
      post('loaded', {
        sampleRate: tts.sampleRate,
        features: tts.features,
        voiceSet: tts.voices.join(', '),
        engineLabel: `${loadedLabel} (${tts.device}, ${tts.threads} thread${tts.threads === 1 ? '' : 's'})`,
      });
    } else if (data.type === 'generate') {
      await generate(data);
    } else if (data.type === 'clear_cache') {
      tts?.dispose();
      tts = null;
      loadedKey = null;
      registeredVoices = new Set();
      await clearCache();
      post('cache_cleared', { message: 'Phonon browser cache cleared.' });
    }
  } catch (error) {
    post('progress_done');
    post('error', { message: error?.message || String(error) });
  }
});

post('status', { message: 'Phonon worker ready.' });
