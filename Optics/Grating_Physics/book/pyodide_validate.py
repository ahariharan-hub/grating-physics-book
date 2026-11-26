"""Execute staged lessons in fresh local WebAssembly engines, blocking remote fetches.

Requires the Node runtime already managed by Jupyter Book. No browser profile,
interactive desktop, external computation server, or downloaded test runner is used.
"""
import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from urllib.parse import unquote, urlsplit

from jupyter_book.nodeenv import find_nodeenv_path

ROOT = Path(__file__).resolve().parents[1]


def input_key(stage, name):
    """Ignore saved outputs; identify the code and files the engine executes."""
    stage = Path(stage)
    notebook = json.loads((stage / name).read_text(encoding='utf-8'))
    code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    assets = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (stage / '_public').iterdir() if p.is_file()}
    manifest = json.loads((ROOT / '_build/pyodide-runtime/site/runtime-manifest.json').read_text(encoding='utf-8'))
    # Node executes Python/runtime files, not the browser's widget JS, fonts or
    # math renderer. Those assets require the separate browser checks.
    runtime = {p: entry['sha256'] for p, entry in manifest['files'].items()
               if p.startswith(('runtime/pyodide/', 'pypi/'))}
    return hashlib.sha256(json.dumps(dict(code=code, assets=assets, runtime=runtime),
                                    sort_keys=True).encode()).hexdigest()


def summarize(stage):
    """Collect matching individual receipts, including targeted reruns."""
    stage = Path(stage)
    result_dir = stage.parent / 'wasm-validation'
    names = json.loads((stage.parent / 'profile.json').read_text(encoding='utf-8'))['notebooks']
    records = []
    for name in names:
        path = result_dir / (Path(name).stem + '.json')
        record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        if record.get('input_key') != input_key(stage, name):
            record = dict(notebook=name, passed=False, error='No matching execution receipt')
        records.append(record)
    report = dict(passed=all(r.get('passed') for r in records), notebooks=len(names),
                  code_cells=sum(len(r.get('cells', [])) for r in records),
                  evidence='Local Pyodide/Node execution; browser rendering and controls require separate checks',
                  results=records)
    (result_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def validate(stage, names=None):
    stage = Path(stage).resolve()
    profile = json.loads((stage.parent / 'profile.json').read_text(encoding='utf-8'))
    base_url = profile.get('base_url', '').rstrip('/')
    # Compute validation uses the same verified runtime files that are later
    # copied into HTML. It can run while the independent HTML renderer builds.
    html = ROOT / '_build/pyodide-runtime/site'
    result_dir = stage.parent / 'wasm-validation'
    result_dir.mkdir(parents=True, exist_ok=True)
    if not names:
        names = profile['notebooks']
    # The renderer may regenerate its working sources during a rebuild. Give
    # this validation run immutable notebook and helper inputs of its own.
    temporary = tempfile.TemporaryDirectory(prefix='wasm-inputs-', dir=result_dir.parent)
    frozen = Path(temporary.name)
    shutil.copytree(stage / '_public', frozen / '_public')
    for name in names:
        notebook = stage / name
        if notebook.parent != stage or not notebook.is_file():
            raise RuntimeError('Unexpected staged notebook: ' + name)
        shutil.copyfile(notebook, frozen / name)
    node = find_nodeenv_path('22.17.0') / ('Scripts/node.exe' if sys.platform == 'win32' else 'bin/node')
    if not node.is_file():
        raise RuntimeError('The Jupyter Book managed Node runtime could not be found.')
    class QuietHandler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            relative = unquote(urlsplit(path).path).lstrip('/')
            prefix = base_url.lstrip('/') + '/'
            if base_url and relative.startswith(prefix):
                relative = relative[len(prefix):]
                path = '/' + relative
            if relative.startswith('_public/'):
                candidate = (frozen / relative).resolve()
                if frozen in candidate.parents:
                    return str(candidate)
            return super().translate_path(path)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(html)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{server.server_port}'
    records = []
    try:
        for name in names:
            notebook = frozen / name
            source = json.loads(notebook.read_text(encoding='utf-8'))
            destination = result_dir / (notebook.stem + '.json')
            if not any(c['cell_type'] == 'code' for c in source['cells']):
                record = dict(notebook=name, passed=True, cells=[], evidence='No code cells')
                destination.write_text(json.dumps(record), encoding='utf-8')
            else:
                log = destination.with_suffix('.log')
                execution_result = frozen / (notebook.stem + '.result.json')
                with log.open('w', encoding='utf-8') as stream:
                    try:
                        result = subprocess.run([str(node), str(ROOT / 'book/pyodide_notebook.mjs'),
                            str(html), str(notebook), origin, str(execution_result)],
                            stdout=stream, stderr=subprocess.STDOUT, timeout=900)
                        record = json.loads(execution_result.read_text(encoding='utf-8')) if execution_result.exists() else dict(
                            notebook=name, passed=False, error=f'Runner exited {result.returncode}; see {log.name}')
                        record['passed'] = record['passed'] and result.returncode == 0
                    except subprocess.TimeoutExpired:
                        record = dict(notebook=name, passed=False, error='WebAssembly execution exceeded 900 seconds')
            record['input_key'] = input_key(frozen, name)
            destination.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            records.append(record)
            print(f"WASM [{'PASS' if record['passed'] else 'FAIL'}] {len(records)}/{len(names)} {name}", flush=True)
            if not record['passed']:
                print(str(record.get('error') or next((c['error'] for c in record.get('cells', []) if c.get('error')), 'unknown error'))[:1000], flush=True)
    finally:
        server.shutdown()
        server.server_close()
        temporary.cleanup()
    summarize(stage)
    return dict(passed=all(r['passed'] for r in records), results=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=ROOT / '_build/pyodide-book/source')
    parser.add_argument('--notebook', action='append')
    args = parser.parse_args()
    outcome = validate(args.stage, args.notebook)
    raise SystemExit(0 if outcome['passed'] else 1)
