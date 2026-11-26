"""Build and assemble Binder and browser-Python editions for GitHub Pages.

This command writes only ignored build artifacts. Source sync, Git commits,
pushes and Pages deployment are separate explicit publication operations.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import execution_cache
import pilot
import pyodide_assets
import pyodide_book
from pyodide_validate import summarize, validate
from public_content import PUBLIC_NOTEBOOKS, PUBLIC_VERIFICATION_FILES

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '_build/release'
SITE = WORK / 'site'


def run(script, *args):
    subprocess.run([sys.executable, '-B', str(ROOT / 'book' / script), *args],
                   cwd=ROOT, check=True,
                   env=dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg'))


def assemble(binder, browser, destination=SITE):
    binder, browser, destination = map(lambda p: Path(p).resolve(),
                                       (binder, browser, destination))
    if destination != SITE.resolve() or ROOT.resolve() not in destination.parents:
        raise RuntimeError('Refusing an unexpected release output directory.')
    for source in (binder, browser):
        if not (source / 'index.html').is_file():
            raise RuntimeError('Missing completed edition: ' + str(source))
        if source == destination or destination in source.parents or source in destination.parents:
            raise RuntimeError('Release sources and output must be separate.')
        if any(p.is_symlink() for p in source.rglob('*')):
            raise RuntimeError('Pages artifacts must not contain symlinks.')
    if (binder / 'pyodide').exists():
        raise RuntimeError('The Binder artifact already contains a browser edition.')
    if destination.is_dir():
        shutil.rmtree(destination)
    shutil.copytree(binder, destination)
    shutil.copytree(browser, destination / 'pyodide')
    (destination / '.nojekyll').write_text('', encoding='utf-8')
    for name in PUBLIC_VERIFICATION_FILES:
        if (destination / name).read_bytes() != (ROOT / name).read_bytes():
            raise RuntimeError('Provider verification bytes changed: ' + name)
    pilot.scan_public(destination)
    report = dict(notebooks=len(PUBLIC_NOTEBOOKS), binder='/', browser='/pyodide/',
                  bytes=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),
                  symlinks=False, credential_scan='passed', deployed=False)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'assembly.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Both editions assembled: ' + str(destination), flush=True)
    return report


def build(base_url, wasm='all'):
    base_url = base_url.rstrip('/')
    if base_url and (not base_url.startswith('/') or any(s in base_url for s in ('..', '?', '#', '://'))):
        raise ValueError('base-url must be an absolute URL path.')
    os.environ['BASE_URL'] = base_url
    # Keep the author's running preview separate from the deployment profile.
    authoring_cache = pyodide_book.WORK / 'native'
    pyodide_book.WORK = WORK / 'browser'
    pyodide_book.STAGE = pyodide_book.WORK / 'source'
    pyodide_book.HTML = pyodide_book.STAGE / '_build/html'
    if authoring_cache.is_dir() and not (pyodide_book.WORK / 'native').exists():
        shutil.copytree(authoring_cache, pyodide_book.WORK / 'native')
    if not execution_cache.valid(ROOT):
        run('pilot.py', 'execute')
        for check in ('solutions', 'boundary', 'propagation', 'scattering', 'orders', 'masks'):
            run('check_' + check + '.py')
        execution_cache.save(ROOT)
    run('pilot.py', 'build')
    run('check_build.py')
    if not (pyodide_assets.SITE / 'runtime-manifest.json').is_file():
        pyodide_assets.Builder().prepare()
    pyodide_book.build(base_url + '/pyodide')
    if wasm == 'all':
        # Reuse only receipts with identical code, helper and runtime inputs.
        existing = summarize(pyodide_book.STAGE)
        missing = [r['notebook'] for r in existing['results'] if not r.get('passed')]
        if missing and not validate(pyodide_book.STAGE, missing)['passed']:
            raise RuntimeError('Browser computation validation failed.')
        if not summarize(pyodide_book.STAGE)['passed']:
            raise RuntimeError('The full browser computation check is incomplete.')
    elif wasm == 'smoke':
        names = [n for n in PUBLIC_NOTEBOOKS if n.startswith(('0_2_', '5_1_', '7_1_', '7_2_', '7_3_', '8_1_', '9_3_'))]
        if not validate(pyodide_book.STAGE, names)['passed']:
            raise RuntimeError('Browser computation smoke check failed.')
    pyodide_book.check()
    return assemble(ROOT / '_build/public-pilot/_build/html', pyodide_book.HTML)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'assemble'))
    parser.add_argument('--base-url', default=os.environ.get('BASE_URL', ''))
    parser.add_argument('--wasm', choices=('all', 'smoke', 'skip'), default='all',
                        help='Use all for publication; narrower checks are for development only.')
    args = parser.parse_args()
    if args.command == 'build':
        build(args.base_url, args.wasm)
    else:
        assemble(ROOT / '_build/public-pilot/_build/html', WORK / 'browser/source/_build/html')
