// ינשוף — offline translation worker (Transformers.js, runs fully on-device once cached)
import { pipeline, env } from './lib/transformers.min.js';
env.allowLocalModels = false;
env.useBrowserCache = true;
// onnxruntime wasm is shipped with the app (lib/) so the engine itself works offline
env.backends.onnx.wasm.wasmPaths = new URL('./lib/', self.location.href).href;

let pipe = null, current = null, loading = null;

function load(model) {
  if (pipe && current === model) return Promise.resolve(pipe);
  if (loading) return loading;
  loading = pipeline('translation', model, {
    dtype: 'q8',
    device: 'wasm',
    progress_callback: p => postMessage({ type: 'progress', p })
  }).then(p => { pipe = p; current = model; loading = null; return p; })
    .catch(e => { loading = null; throw e; });
  return loading;
}

self.onmessage = async (e) => {
  const { id, type, model, text, src, tgt } = e.data;
  try {
    if (type === 'load') {
      await load(model);
      postMessage({ id, type: 'ready' });
    } else if (type === 'translate') {
      const p = await load(model);
      const out = await p(text, { src_lang: src, tgt_lang: tgt, max_new_tokens: 512 });
      postMessage({ id, type: 'result', text: (out[0] && out[0].translation_text) || '' });
    }
  } catch (err) {
    postMessage({ id, type: 'error', error: String((err && err.message) || err) });
  }
};
