// Workers must be same-origin. This local entry point lets phonon-tts load its
// runtime worker from the pinned CDN package while keeping the Worker URL local.
import 'https://cdn.jsdelivr.net/npm/phonon-tts@0.6.0/worker.js';
