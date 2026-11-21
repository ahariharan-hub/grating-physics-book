"""Execute and preview the configured notebooks without public local credentials.

Run with the grating-book Python: execute, build, or preview. All executed
notebooks, logs, tokens and local site configuration live under ignored _build.
"""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import nbformat
import yaml
from public_content import PUBLIC_VERIFICATION_FILES, check_readers, source_files

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / '_build'
def toc_files(entries):
    """Walk native MyST chapter groups in reading order, excluding URL entries."""
    files = []
    for entry in entries:
        if 'file' in entry:
            files.append(entry['file'])
        files.extend(toc_files(entry.get('children', [])))
    return tuple(files)


PAGES = toc_files(yaml.safe_load((ROOT / 'myst.yml').read_text(encoding='utf-8'))['project']['toc'])
PILOT = tuple(name for name in PAGES if Path(name).suffix == '.ipynb')
PACKAGES = ('jupyter-book', 'numpy', 'scipy', 'matplotlib', 'plotly', 'anywidget', 'ipywidgets',
            'ipykernel', 'jupyter-server', 'jupyter-client', 'nbclient', 'nbformat', 'PyYAML')


def kernel_directory():
    directory = BUILD / 'kernels' / 'python3'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Grating Book pilot', 'language': 'python'}), encoding='utf-8')
    return directory.parent


def execute():
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    destination = BUILD / 'executed'
    destination.mkdir(parents=True, exist_ok=True)
    versions = {name: metadata.version(name) for name in PACKAGES}
    results = []
    for name in PILOT:
        notebook = nbformat.read(ROOT / name, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
        notebook.metadata.pop('widgets', None)
        notebook.metadata.kernelspec = dict(name='python3', display_name='Python 3', language='python')
        manager = KernelManager(kernel_name='python3', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_directory())]))
        start = time.monotonic()
        client = NotebookClient(notebook, km=manager, timeout=180, resources={'metadata': {'path': str(ROOT)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(notebook, destination / name)
        errors = [o for c in notebook.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
        if errors:
            raise RuntimeError(f'{name}: execution returned error outputs')
        code_count = sum(c.cell_type == 'code' for c in notebook.cells)
        results.append(dict(notebook=name, code_cells=code_count, elapsed_seconds=round(time.monotonic()-start, 2), errors=0))
        print(f'Executed {name}: {code_count} code cells; no errors.', flush=True)
    record = dict(python=sys.version.split()[0], platform=sys.platform, dependencies=versions, notebooks=results)
    (ROOT / 'book/tested-environment.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'book/requirements-tested.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n', encoding='utf-8')


def stage(local=False):
    selected = source_files(ROOT)
    check_readers(ROOT, [path.relative_to(ROOT).as_posix() for path in selected])
    folder = BUILD / ('local-preview' if local else 'public-pilot')
    folder.mkdir(parents=True, exist_ok=True)
    # Replace staged sources while preserving the renderer/template cache.
    for path in folder.iterdir():
        if path.name not in ('_build', 'myst.yml'):
            remove_generated(path, folder)
    for path in selected:
        destination = folder / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for name in PILOT:
        executed = BUILD / 'executed' / name
        if not executed.exists():
            raise RuntimeError('Run python book/pilot.py execute first.')
        shutil.copy2(executed, folder / name)
    config = yaml.safe_load((ROOT / 'myst.yml').read_text(encoding='utf-8'))
    if local:
        state_path = BUILD / 'local-runtime.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'token': secrets.token_urlsafe(32)}
        state_path.write_text(json.dumps(state), encoding='utf-8')
        config['project']['jupyter'] = {'kernelName': 'python3',
                                      'server': {'url': 'http://127.0.0.1:8897/', 'token': state['token']}}
    else:
        if 'server' in config['project'].get('jupyter', {}):
            raise RuntimeError('Public configuration must not use a local Jupyter server.')
    text = yaml.safe_dump(config, sort_keys=False)
    config_path = folder / 'myst.yml'
    if not config_path.exists() or config_path.read_text(encoding='utf-8') != text:
        config_path.write_text(text, encoding='utf-8')
    return folder


def remove_generated(path, parent):
    """Remove one derived child only after checking its resolved boundary."""
    resolved, boundary = path.resolve(), parent.resolve()
    if boundary not in resolved.parents or path.is_symlink():
        raise RuntimeError('Refusing an unexpected generated-output path.')
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def executable():
    candidate = Path(sys.executable).parent / ('Scripts/jupyter-book.exe' if os.name == 'nt' else 'bin/jupyter-book')
    return str(candidate) if candidate.exists() else shutil.which('jupyter-book')


def scan_public(folder):
    state_path = BUILD / 'local-runtime.json'
    forbidden = [b'http://127.0.0.1:8897', b'http://localhost:8897']
    if state_path.exists():
        forbidden.append(json.loads(state_path.read_text())['token'].encode())
    for path in folder.rglob('*'):
        if path.is_file() and any(value in path.read_bytes() for value in forbidden):
            raise RuntimeError(f'Local execution credentials/configuration in public output: {path.relative_to(folder)}')
    print('Public output scan passed: no local server address or local token.', flush=True)


def build():
    folder = stage()
    # Hashed assets and removed pages must not survive a prior build.
    for name in ('html', 'site'):
        path = folder / '_build' / name
        if path.exists():
            remove_generated(path, BUILD / 'public-pilot')
    subprocess.run([executable(), 'build', '--html'], cwd=folder, check=True)
    # Provider verification files must retain their exact names and bytes at
    # the deployed artifact root, without MyST rendering or theme wrappers.
    for name in PUBLIC_VERIFICATION_FILES:
        shutil.copyfile(folder / name, folder / '_build/html' / name)
    scan_public(folder / '_build/html')
    print('Public artifact: _build/public-pilot/_build/html')


def preview():
    folder = stage(local=True)
    state_path = BUILD / 'local-runtime.json'
    state = json.loads(state_path.read_text())
    def available(url, headers=None, expected=None):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=3) as response:
                return response.status == 200 and (expected is None or expected in response.read())
        except (OSError, urllib.error.URLError):
            return False
    server_ready = available('http://127.0.0.1:8897/api/status',
                             {'Authorization': 'token '+state['token']})
    preview_ready = available('http://localhost:3010/', expected=b'Scalar theory of grating diffraction')
    if server_ready and preview_ready:
        print('Refreshed the existing local preview: http://localhost:3010/')
        return
    def listening(port):
        try:
            with socket.create_connection(('localhost', port), timeout=1):
                return True
        except OSError:
            return False
    server_listening, preview_listening = listening(8897), listening(3010)
    config = BUILD / 'jupyter-local.py'
    config.write_text('\n'.join([
        "c = get_config()", "c.ServerApp.ip = '127.0.0.1'", "c.ServerApp.port = 8897",
        'c.ServerApp.port_retries = 0', 'c.ServerApp.open_browser = False',
        f'c.ServerApp.root_dir = {str(folder)!r}', "c.ServerApp.allow_origin = 'http://localhost:3010'",
        'c.ServerApp.allow_credentials = True', f'c.IdentityProvider.token = {state["token"]!r}',
    ])+'\n', encoding='utf-8')
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == 'nt' else 0
    def spawn(command, log_name):
        # Import caches inside the watched source tree trigger a full site reload
        # and disconnect live widgets. Preview kernels inherit this setting.
        environment = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg',
                           PYTHONDONTWRITEBYTECODE='1')
        environment['JUPYTER_PATH'] = str(kernel_directory().parent) + os.pathsep + environment.get('JUPYTER_PATH', '')
        with (BUILD / log_name).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=flags, start_new_session=os.name != 'nt', env=environment)
        return process.pid
    if not server_ready and not server_listening:
        state['jupyter_pid'] = spawn([sys.executable, '-m', 'jupyter_server', '--config', str(config)], 'jupyter-local.log')
    if not preview_ready and not preview_listening:
        state['preview_pid'] = spawn([executable(), 'start', '--port', '3010'], 'preview-local.log')
    state_path.write_text(json.dumps(state), encoding='utf-8')
    print('Local preview: http://localhost:3010 (waiting for new or existing services).')
    for _ in range(45):
        if (available('http://127.0.0.1:8897/api/status', {'Authorization': 'token '+state['token']})
                and available('http://localhost:3010/', expected=b'Scalar theory of grating diffraction')):
            return
        time.sleep(1)
    raise RuntimeError('Preview startup failed; inspect the ignored local logs under _build.')


def reset_preview_kernels():
    """Reset only kernels belonging to this tool's dedicated preview server."""
    state = json.loads((BUILD / 'local-runtime.json').read_text())
    headers = {'Authorization': 'token '+state['token']}
    base = 'http://127.0.0.1:8897/api/kernels'
    with urllib.request.urlopen(urllib.request.Request(base, headers=headers), timeout=10) as response:
        kernels = json.load(response)
    for kernel in kernels:
        request = urllib.request.Request(base+'/'+kernel['id']+'/restart',
                                         headers=headers, method='POST', data=b'')
        with urllib.request.urlopen(request, timeout=30):
            pass
    print(f'Reset {len(kernels)} preview kernels. Refresh the browser, then activate the examples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('execute', 'build', 'preview', 'scan'))
    args = parser.parse_args()
    {'execute': execute, 'build': build, 'preview': preview,
     'scan': lambda: scan_public(BUILD / 'public-pilot/_build/html')}[args.command]()
