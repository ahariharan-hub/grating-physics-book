"""Execute and preview the two-notebook pilot without public local credentials.

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
import subprocess
import sys
import time

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / '_build'
PILOT = ('0_0_Foundations_of_Scalar_Optics.ipynb', '0_1_Scalar_Fields_and_Helmholtz.ipynb')
PACKAGES = ('jupyter-book', 'numpy', 'scipy', 'matplotlib', 'plotly', 'ipywidgets',
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
    folder = BUILD / ('local-preview' if local else 'public-pilot')
    folder.mkdir(parents=True, exist_ok=True)
    for path in list(ROOT.glob('*.ipynb')) + list(ROOT.glob('*.md')):
        shutil.copy2(path, folder / path.name)
    for directory in ('wave_optics', 'Wave_solver_py_scripts', 'downloads', 'supplement'):
        shutil.copytree(ROOT / directory, folder / directory, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.pyc'))
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
    (folder / 'myst.yml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return folder


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
    subprocess.run([executable(), 'build', '--html'], cwd=folder, check=True)
    scan_public(folder / '_build/html')
    print('Public artifact: _build/public-pilot/_build/html')


def preview():
    folder = stage(local=True)
    state_path = BUILD / 'local-runtime.json'
    state = json.loads(state_path.read_text())
    config = BUILD / 'jupyter-local.py'
    config.write_text('\n'.join([
        "c = get_config()", "c.ServerApp.ip = '127.0.0.1'", "c.ServerApp.port = 8897",
        'c.ServerApp.port_retries = 0', 'c.ServerApp.open_browser = False',
        f'c.ServerApp.root_dir = {str(folder)!r}', "c.ServerApp.allow_origin = 'http://localhost:3010'",
        'c.ServerApp.allow_credentials = True', f'c.IdentityProvider.token = {state["token"]!r}',
    ])+'\n', encoding='utf-8')
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == 'nt' else 0
    def spawn(command, log_name):
        environment = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
        environment['JUPYTER_PATH'] = str(kernel_directory().parent) + os.pathsep + environment.get('JUPYTER_PATH', '')
        with (BUILD / log_name).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=flags, start_new_session=os.name != 'nt', env=environment)
        return process.pid
    state['jupyter_pid'] = spawn([sys.executable, '-m', 'jupyter_server', '--config', str(config)], 'jupyter-local.log')
    state['preview_pid'] = spawn([executable(), 'start', '--port', '3010'], 'preview-local.log')
    state_path.write_text(json.dumps(state), encoding='utf-8')
    print('Private local preview starting at http://localhost:3010 (credential stored only under ignored _build).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('execute', 'build', 'preview', 'scan'))
    args = parser.parse_args()
    {'execute': execute, 'build': build, 'preview': preview,
     'scan': lambda: scan_public(BUILD / 'public-pilot/_build/html')}[args.command]()
