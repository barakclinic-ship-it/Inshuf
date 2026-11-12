// ינשוף — offline translation worker (Transformers.js, runs fully on-device once cached)
import { pipeline, env } from './lib/transformers.min.js';
env.allowLocalModels = false;
env.useBrowserCache = true;
// onnxruntime wasm is shipped with the app (lib/) so the engine itself works offline
env.backends.onnx.wasm.wasmPaths = new URL('./lib/', self.location.href).href;
env.backends.onnx.wasm.numThreads = 1;

const pipes = {}, loading = {};

function load(model) {
  if (pipes[model]) return Promise.resolve(pipes[model]);
  if (loading[model]) return loading[model];
  loading[model] = pipeline('translation', model, {
    dtype: 'q8',
    device: 'wasm',
    progress_callback: p => postMessage({ type: 'progress', model, p })
  }).then(p => { pipes[model] = p; delete loading[model]; return p; })
    .catch(e => { delete loading[model]; throw e; });
  return loading[model];
}

self.onmessage = async (e) => {
  const { id, type, model, text, src, tgt } = e.data;
  try {
    if (type === 'load') {
      await load(model);
      postMessage({ id, type: 'ready' });
    } else if (type === 'unload') {
      const p = pipes[model]; delete pipes[model];
      if (p && p.dispose) await p.dispose();
      postMessage({ id, type: 'ready' });
    } else if (type === 'translate') {
      const p = await load(model);
      const opts = { max_new_tokens: 512 };
      if (src) { opts.src_lang = src; opts.tgt_lang = tgt; }
      const out = await p(text, opts);
      postMessage({ id, type: 'result', text: (out[0] && out[0].translation_text) || '' });
    }
  } catch (err) {
    postMessage({ id, type: 'error', error: String((err && err.message) || err) });
  }
};
