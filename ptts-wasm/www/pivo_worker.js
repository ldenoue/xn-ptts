import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.wasm.min.mjs';
import { phonemize, SPECIAL } from './pivo_frontend.js?v=pivo-hf-1';

const ORT_ROOT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist';
const ortRuntimeUrl = `${ORT_ROOT}/ort-wasm-simd-threaded.mjs`;
const ortWasmUrl = `${ORT_ROOT}/ort-wasm-simd-threaded.wasm`;

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = { mjs: ortRuntimeUrl, wasm: ortWasmUrl };

const PIVO_REVISION = '515d84fc3420da6ae138f1b81017fcbc24d622a8';
const MODEL_ROOT = `https://huggingface.co/empero-ai/PiVo-1.4M/resolve/${PIVO_REVISION}/onnx`;
const VOICES = { Pi: 0, Vo: 1 };
const CONFIG = { width: 96, heads: 4, kvHeads: 2, maxTokenFrames: 200, maxUtteranceFrames: 3000 };
const CACHE_NAMES = ['key0', 'value0', 'key1', 'value1'];
const STATE_NAMES = ['memory', 'gru1', 'gru2', 'gru3', 'frame_state'];
const STATE_WIDTHS = [256, 160, 128, 128, 164];
let sessionsPromise;

const product = (values) => values.reduce((total, value) => total * value, 1);
const tensor = (type, data, dims) => new ort.Tensor(type, data, dims);
const int64 = (values, dims) => tensor('int64', BigInt64Array.from(values, BigInt), dims);

function loadSessions(progress) {
  if (!sessionsPromise) {
    sessionsPromise = (async () => {
      const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      const sessions = {};
      for (const [index, name] of ['encoder', 'decoder', 'conditioning', 'vocoder_step'].entries()) {
        progress(`Loading model graph ${index + 1} of 4…`);
        sessions[name] = await ort.InferenceSession.create(`${MODEL_ROOT}/${name}.onnx`, options);
      }
      return sessions;
    })();
  }
  return sessionsPromise;
}

async function releaseSessions() {
  if (!sessionsPromise) return;
  const sessions = await sessionsPromise.catch(() => null);
  sessionsPromise = null;
  if (sessions) await Promise.all(Object.values(sessions).map((session) => session.release()));
}

function phaseFeatures(durations) {
  const frames = durations.reduce((sum, value) => sum + value, 0);
  const result = new Float32Array(frames * 4);
  let frame = 0;
  for (const duration of durations) {
    for (let index = 0; index < duration; index++, frame++) {
      const fraction = (index + 0.5) / duration;
      result[frame * 4] = 2 * fraction - 1;
      result[frame * 4 + 1] = Math.sin(Math.PI * fraction);
      result[frame * 4 + 2] = Math.cos(Math.PI * fraction);
      result[frame * 4 + 3] = Math.log1p(duration) / Math.log1p(CONFIG.maxTokenFrames);
    }
  }
  return result;
}

function makeWav(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, value) => [...value].forEach((character, i) => view.setUint8(offset + i, character.charCodeAt(0)));
  write(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); write(8, 'WAVE'); write(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index++) view.setInt16(44 + index * 2, Math.round(Math.max(-1, Math.min(1, samples[index])) * 32767), true);
  return buffer;
}

async function synthesize(text, voice, speed, progress) {
  if (!(voice in VOICES) || speed < 0.5 || speed > 2) throw new Error('Invalid voice or speed.');
  const front = await phonemize(text);
  if (front.tokenIds.length > 512) throw new Error('This text exceeds 512 pronunciation tokens. Split it into shorter clauses.');
  if (!front.symbols.some((symbol) => !SPECIAL.includes(symbol))) throw new Error('Enter some spoken text.');
  const sessions = await loadSessions(progress);
  progress('Predicting pronunciation and timing…');
  const tokens = int64(front.tokenIds, [1, front.tokenIds.length]);
  const encodedResult = await sessions.encoder.run({ tokens, voice: int64([VOICES[voice]], [1]), lengths: int64([front.tokenIds.length], [1]) });
  const outputs = sessions.encoder.outputNames.map((name) => encodedResult[name]);
  const encoded = outputs[0]; const logDurations = outputs[1];
  const durations = Array.from(logDurations.data, (value, index) => {
    let duration = Math.round(Math.expm1(Math.max(0, Math.min(Math.log1p(CONFIG.maxTokenFrames), value))) / speed);
    duration = Math.max(0, Math.min(CONFIG.maxTokenFrames, duration));
    if (front.tokenIds[index] >= SPECIAL.length) duration = Math.max(1, duration);
    if (front.tokenIds[index] <= 2) duration = 0;
    return duration;
  });
  const frameCount = durations.reduce((sum, value) => sum + value, 0);
  if (frameCount > CONFIG.maxUtteranceFrames) throw new Error('The predicted speech exceeds 30 seconds. Split the text into shorter clauses.');
  const encodedWidth = encoded.dims.at(-1);
  const phase = phaseFeatures(durations);
  const expandedWidth = encodedWidth + 4;
  const expandedData = new Float32Array(frameCount * expandedWidth);
  let targetFrame = 0;
  for (let tokenIndex = 0; tokenIndex < durations.length; tokenIndex++) {
    for (let repeat = 0; repeat < durations[tokenIndex]; repeat++, targetFrame++) {
      expandedData.set(encoded.data.subarray(tokenIndex * encodedWidth, (tokenIndex + 1) * encodedWidth), targetFrame * expandedWidth);
      expandedData.set(phase.subarray(targetFrame * 4, targetFrame * 4 + 4), targetFrame * expandedWidth + encodedWidth);
    }
  }

  let cache = Object.fromEntries(CACHE_NAMES.map((name) => [name, tensor('float32', new Float32Array(0), [1, CONFIG.kvHeads, 0, CONFIG.width / CONFIG.heads])]));
  let state = Object.fromEntries(STATE_NAMES.map((name, index) => [name, tensor('float32', new Float32Array(STATE_WIDTHS[index]), [1, STATE_WIDTHS[index]])]));
  const queue = []; const pcm = new Float32Array(frameCount * 160);
  let outputOffset = 0; let previous = null; let deemphasisState = 0;
  const chunkFrames = 16;
  for (let position = 0; position < frameCount; position += chunkFrames) {
    if (position % 160 === 0) progress(`Synthesizing audio… ${Math.round(position / Math.max(frameCount, 1) * 100)}%`);
    const count = Math.min(chunkFrames, frameCount - position);
    const chunk = expandedData.slice(position * expandedWidth, (position + count) * expandedWidth);
    const decoded = await sessions.decoder.run({ expanded: tensor('float32', chunk, [1, count, expandedWidth]), position: int64([position], []), ...cache });
    const decoderOutputs = sessions.decoder.outputNames.map((name) => decoded[name]);
    const features = decoderOutputs[0];
    cache = Object.fromEntries(CACHE_NAMES.map((name, index) => [name, decoderOutputs[index + 1]]));
    for (let index = 0; index < features.dims[1]; index++) queue.push(features.data.slice(index * 20, index * 20 + 20));
    const final = position + count >= frameCount;
    while (queue.length >= 2 || (final && queue.length)) {
      const current = queue.shift(); const following = queue[0] ?? current; previous ??= current;
      const contextData = new Float32Array(60); contextData.set(previous); contextData.set(current, 20); contextData.set(following, 40);
      const periodValues = [previous, current, following].map((featuresRow) => Math.max(32, Math.min(255, Math.round(256 / (2 ** (Math.max(-10, Math.min(10, featuresRow[18])) + 1.5))))));
      const conditioned = await sessions.conditioning.run({ features: tensor('float32', contextData, [1, 3, 20]), periods: int64(periodValues, [1, 3]) });
      const conditioning = conditioned[sessions.conditioning.outputNames[0]];
      for (let subframe = 0; subframe < 4; subframe++) {
        const slice = conditioning.data.slice(subframe * 80, (subframe + 1) * 80);
        const generated = await sessions.vocoder_step.run({ conditioning: tensor('float32', slice, [1, 80]), period: int64([periodValues[1]], [1]), ...state });
        const vocoderOutputs = sessions.vocoder_step.outputNames.map((name) => generated[name]);
        state = Object.fromEntries(STATE_NAMES.map((name, index) => [name, vocoderOutputs[index + 1]]));
        for (const sample of vocoderOutputs[0].data) {
          deemphasisState = sample + 0.85 * deemphasisState;
          pcm[outputOffset++] = deemphasisState;
        }
      }
      previous = current;
    }
  }
  return { wav: makeWav(pcm.subarray(0, outputOffset)), seconds: outputOffset / 16000, unknownWords: front.unknownWords };
}

self.addEventListener('message', async ({ data }) => {
  const post = (status, extra = {}) => self.postMessage({ status, ...extra });
  const progress = (message) => post('progress', { message, pct: -1 });
  try {
    if (data.command === 'clear-cache') {
      await releaseSessions();
      post('ready');
      return;
    }
    if (data.command === 'load') {
      await loadSessions(progress);
      post('loaded', { backend: 'wasm', sampleRate: 16000 });
      return;
    }
    if (data.command !== 'tts') return;

    const result = await synthesize(data.text, data.voice || 'Pi', data.speed || 1, progress);
    post('complete', {
      audio: new Blob([result.wav], { type: 'audio/wav' }),
      durationSec: result.seconds,
      sampleRate: 16000,
      unknownWords: result.unknownWords,
    });
  } catch (error) {
    console.error(error);
    post('error', { error: error instanceof Error ? error.message : String(error) });
  }
});

self.postMessage({ status: 'ready' });
