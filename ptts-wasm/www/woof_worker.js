// Browser adapter for Underdog Woof's WebGPU runtime.
// The upstream bundle expects a Window-like global for two feature checks.
globalThis.window = globalThis;
globalThis.matchMedia ??= () => ({ matches: false });

const MODEL_ROOT =
  'https://huggingface.co/ldenoue/Underdog-Woof-WebGPU/resolve/6455ef13506b6eaf4634ef77be95ec5f34f5ce3b';

let woofPromise = null;
let activeAbort = null;

function post(type, data = {}, transfer = []) {
  self.postMessage({ type, ...data }, transfer);
}

function progress(label, value) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  post('progress', { label, pct, detail: `${pct}%` });
}

async function loadModel() {
  if (!navigator.gpu) {
    throw new Error('Underdog Woof requires WebGPU. Try a current Chrome or Edge browser.');
  }
  if (!woofPromise) {
    woofPromise = (async () => {
      const { loadWoof } = await import('./woof-runtime/woof.js');
      const instance = await loadWoof(MODEL_ROOT, {
        download: (value) => progress('Downloading Woof', value),
        gpu: (value) => progress('Preparing WebGPU', value),
      });
      if (!instance) throw new Error('No compatible WebGPU adapter was found.');
      return instance;
    })().catch((error) => {
      woofPromise = null;
      throw error;
    });
  }
  return woofPromise;
}

async function generate(text) {
  const value = text?.trim();
  if (!value) throw new Error('Enter some text to speak.');

  const woof = await loadModel();
  activeAbort?.abort();
  activeAbort = new AbortController();
  post('gen_start', { numTokens: null });

  const started = performance.now();
  let chunks = 0;
  await woof.speak(
    value,
    (pcm) => {
      const copy = pcm.slice();
      chunks += 1;
      post('chunk', { data: copy.buffer }, [copy.buffer]);
    },
    activeAbort.signal,
    async () => 0,
    'narrator',
  );

  post('done', {
    promptMs: performance.now() - started,
    numSteps: chunks,
  });
}

self.addEventListener('message', async ({ data }) => {
  try {
    if (data.type === 'load') {
      const woof = await loadModel();
      post('progress_done');
      post('loaded', {
        sampleRate: woof.rate || 24000,
        voiceSet: 'Isla — Voice by Underdog Woof',
        engineLabel: 'Underdog Woof WebGPU',
      });
      return;
    }
    if (data.type === 'generate') {
      await generate(data.text);
      return;
    }
    if (data.type === 'cancel') {
      activeAbort?.abort();
      return;
    }
    if (data.type === 'clear_cache') {
      activeAbort?.abort();
      woofPromise = null;
      post('cache_cleared', {
        message: 'Woof released. Its Hugging Face files remain in the browser HTTP cache.',
      });
    }
  } catch (error) {
    console.error(error);
    post('error', { message: error instanceof Error ? error.message : String(error) });
  }
});

post('status', { message: 'Underdog Woof worker ready.' });
