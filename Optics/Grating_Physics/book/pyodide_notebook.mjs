// Execute one generated notebook in a fresh, local Pyodide engine.
// This validates WebAssembly/Python compatibility, not browser widget rendering.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const [htmlRoot, notebookPath, origin, resultPath] = process.argv.slice(2);
const network = [];
const originalFetch = globalThis.fetch;
globalThis.fetch = async (input, options) => {
  const url = new URL(typeof input === 'string' ? input : input.url, origin);
  if (url.protocol !== 'data:' && url.origin !== origin) {
    network.push({ url: url.href, blocked: true });
    throw new Error(`External request blocked by validation: ${url.href}`);
  }
  const response = await originalFetch(url, options);
  if (url.protocol !== 'data:') network.push({ path: url.pathname, status: response.status });
  return response;
};

const started = Date.now();
let report;
try {
  const require = createRequire(import.meta.url);
  const { loadPyodide } = require(path.resolve(htmlRoot, 'runtime/pyodide/pyodide.js'));
  const pyodide = await loadPyodide({
    indexURL: path.resolve(htmlRoot, 'runtime/pyodide') + path.sep,
    stdout: () => {}, stderr: () => {},
    env: { MPLBACKEND: 'Agg' },
  });
  await pyodide.loadPackage(['micropip', 'ipython']);
  pyodide.globals.set('_validation_origin', origin);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(_validation_origin + '/pypi/piplite-0.4.7-py3-none-any.whl')
import piplite.piplite
piplite.piplite._PIPLITE_URLS = [_validation_origin + '/pypi/all.json']
piplite.piplite._PIPLITE_DISABLE_PYPI = True
`);
  const notebook = JSON.parse(fs.readFileSync(notebookPath, 'utf8'));
  const cells = notebook.cells.filter(c => c.cell_type === 'code')
    .map(c => Array.isArray(c.source) ? c.source.join('') : c.source);
  pyodide.globals.set('_validation_cells', JSON.stringify(cells));
  const output = await pyodide.runPythonAsync(`
import json, time
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
class _ValidationShell(InteractiveShell):
    # Match pyodide_kernel.interpreter.Interpreter's no-op GUI hook.
    # Plain InteractiveShell raises here for a valid %matplotlib inline cell.
    def enable_gui(self, gui=None):
        pass
_shell = _ValidationShell.instance()
_results = []
for _index, _source in enumerate(json.loads(_validation_cells)):
    _start = time.perf_counter()
    with capture_output(display=True) as _captured:
        _result = await _shell.run_cell_async(_source)
    _error = _result.error_before_exec or _result.error_in_exec
    _results.append(dict(cell=_index, seconds=round(time.perf_counter()-_start, 3),
        stdout=_captured.stdout[-3000:], stderr=_captured.stderr[-1500:],
        display_outputs=len(_captured.outputs), error=str(_error) if _error else None))
    if _error:
        break
json.dumps(_results)
`);
  const results = JSON.parse(output);
  report = { notebook: path.basename(notebookPath), passed: results.length === cells.length &&
      results.every(r => !r.error), cells: results, network,
    seconds: (Date.now() - started) / 1000, evidence: 'Pyodide in Node; browser UI not tested' };
} catch (error) {
  report = { notebook: path.basename(notebookPath), passed: false,
    error: String(error.stack || error), network, seconds: (Date.now() - started) / 1000 };
}
fs.writeFileSync(resultPath, JSON.stringify(report, null, 2) + '\n');
process.exit(report.passed ? 0 : 1);
