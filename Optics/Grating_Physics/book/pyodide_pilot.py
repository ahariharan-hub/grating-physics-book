"""Isolated browser-execution pilot; never sync, commit, push, or deploy a book.

Run with the existing grating-book Python environment:
    python book/pyodide_pilot.py stage
    python book/pyodide_pilot.py build
    python book/pyodide_pilot.py check
    python book/pyodide_pilot.py serve
    python book/pyodide_pilot.py vendor
    python book/pyodide_pilot.py build --same-site
    python book/pyodide_pilot.py serve --same-site

Generated lesson copies and reports live under ignored _build/pyodide-pilot.
The default pilot uses the standard CDN runtime. --same-site builds a separate
candidate in _build/pyodide-packaged with local runtime assets and a policy
blocking remote requests. Verify its live execution before organizational use.
The source notebooks, Binder configuration and publishing workflow are not edited.
"""

import argparse
import ast
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import nbformat
import yaml


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / '_build' / 'pyodide-pilot'
HTML = PILOT / '_build' / 'html'
SAME_SITE = False
BASE_URL = ''
LESSONS = (
    ('0_2_Solutions_of_the_Helmholtz_Equation.ipynb', 'Matplotlib and sliders'),
    ('0_5_From_Helmholtz_BVP_to_Diffracted_Field.ipynb', 'Numerical grating solver'),
    ('5_2_Direction_Cosines_and_Spatial_Frequency_Coordinates.ipynb', 'Plotly widgets'),
)


def widget_stylesheet():
    """Keep Plotly's SVG layers overlaid even when runtime CSS is not retained.

    The widget bundle normally injects these rules into the document head.
    Without them its SVG layers become consecutive blocks, leaving a large
    scrollable blank area. Ship the matching dependency's rules as a normal
    book stylesheet instead of imposing a height that could clip the figure.
    """
    import plotly.graph_objects as go
    widget = go.FigureWidget()
    try:
        bundle = str(widget._esm)
        anchor = bundle.index('"X .main-svg"')
        start = bundle.rfind('={', 0, anchor) + 1
        end = bundle.index('for(', anchor)
        expression = ast.parse(bundle[start:end].rstrip(';'), mode='eval').body
        if not isinstance(expression, ast.Dict):
            raise RuntimeError('The Plotly widget style format changed.')
        rules = {}
        for key, value in zip(expression.keys, expression.values, strict=True):
            selector = key.id if isinstance(key, ast.Name) else ast.literal_eval(key)
            declarations = ast.literal_eval(value)
            if not isinstance(selector, str) or not isinstance(declarations, str):
                raise RuntimeError('Unexpected nonliteral Plotly style rule.')
            rules[selector] = declarations
        assert 'position:absolute' in rules['X .main-svg']
        css = '\n'.join(
            selector.replace('X', '.js-plotly-plot .plotly')
                    .replace('Y', '.plotly-notifier') + ' {' + declarations + '}'
            for selector, declarations in rules.items())
        return (ROOT / 'assets/textbook.css').read_text(encoding='utf-8') + (
            '\n/* Plotly.js widget styles (MIT); see /_public/plotly-license.txt. */\n') + css + '\n'
    finally:
        widget.close()


def protected_files():
    """Fingerprint existing scientific sources and execution/publishing settings."""
    files = list(ROOT.glob('*.ipynb')) + sorted((ROOT / 'wave_optics').rglob('*.py'))
    files += [ROOT / 'myst.yml', ROOT / 'reader-guide.md', ROOT / 'book/workflow.py']
    files += sorted((ROOT / 'book/publishing').rglob('*'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files)) if p.is_file()}


def reference():
    """Small, reproducible solver comparison, not a convergence assessment."""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from wave_optics.absorbing_grating import (
        AbsorbingGratingConfiguration, solve_absorbing_grating,
    )
    solution = solve_absorbing_grating(AbsorbingGratingConfiguration(M=6))
    values = [solution.reflectance, solution.transmittance, solution.absorption]
    assert np.isfinite(values).all()
    assert abs(sum(values) - 1) < 1e-7, 'Native reference failed energy balance'
    return dict(M=6, numpy=np.__version__, flux=values,
                transmission_real=solution.transmission.real.tolist(),
                transmission_imag=solution.transmission.imag.tolist(),
                rtol=1e-7, atol=1e-9)


def bootstrap(*, same_site=None, base_url=None):
    # These are generated test cells, not modifications of the teaching sources.
    # Versioned compiled packages are selected by Pyodide's own package set.
    same_site = SAME_SITE if same_site is None else same_site
    base_url = BASE_URL if base_url is None else base_url
    if same_site:
        installer = '''import piplite
import piplite.piplite
piplite.piplite._PIPLITE_DISABLE_PYPI = True
await piplite.install(['numpy', 'scipy', 'matplotlib', 'pillow', 'ipywidgets',
                       'plotly==7.1.0', 'anywidget==0.11.0'])'''
    else:
        installer = '''# This pilot has no JupyterLite wheel index yet. Use Pyodide's installer
# directly so it does not probe the nonexistent /pypi/all.json on each lookup.
import micropip
await micropip.install(['numpy', 'scipy', 'matplotlib', 'pillow', 'ipywidgets',
                       'plotly==7.1.0', 'anywidget==0.11.0'])'''
    return f'''# Browser pilot initialization (generated copy only).
import sys
if sys.platform != 'emscripten':
    raise RuntimeError('Activate this pilot with the browser Python kernel.')
{installer}
from pyodide.http import pyfetch
from pathlib import Path
import io, json, zipfile
_response = await pyfetch('{base_url}/_public/wave_optics_browser.zip')
if not _response.ok:
    raise RuntimeError('Could not load the same-site wave_optics bundle.')
with zipfile.ZipFile(io.BytesIO(await _response.bytes())) as _archive:
    _archive.extractall(Path.cwd())
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
_response = await pyfetch('{base_url}/_public/pyodide-reference.json')
if not _response.ok:
    raise RuntimeError('Could not load the native numerical reference.')
_reference = await _response.json()
import numpy as np
from wave_optics.absorbing_grating import (
    AbsorbingGratingConfiguration, solve_absorbing_grating,
)
_solution = solve_absorbing_grating(AbsorbingGratingConfiguration(M=_reference['M']))
for _actual, _expected in (
    ([_solution.reflectance, _solution.transmittance, _solution.absorption], _reference['flux']),
    (_solution.transmission.real, _reference['transmission_real']),
    (_solution.transmission.imag, _reference['transmission_imag']),
):
    np.testing.assert_allclose(_actual, _expected,
                               rtol=_reference['rtol'], atol=_reference['atol'])
print('PILOT_NUMERICS_PASS: browser flux and complex coefficients match native Python.')
'''


def stage():
    before = protected_files()
    if SAME_SITE:
        from pyodide_assets import verify
        verify()
    PILOT.mkdir(parents=True, exist_ok=True)
    assets = PILOT / '_public'
    assets.mkdir(exist_ok=True)
    style = PILOT / 'assets/browser-widgets.css'
    style.parent.mkdir(exist_ok=True)
    style.write_text(widget_stylesheet(), encoding='utf-8')
    from importlib.metadata import distribution
    plotly = distribution('plotly')
    licenses = [p for p in plotly.files if p.name in ('LICENSE', 'LICENSE.txt')]
    if not licenses:
        raise RuntimeError('The installed Plotly license could not be located.')
    shutil.copyfile(plotly.locate_file(licenses[0]), assets / 'plotly-license.txt')
    # Only package Python runtime modules; no private tests, reports or notebooks.
    with zipfile.ZipFile(assets / 'wave_optics_browser.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted((ROOT / 'wave_optics').glob('*.py')):
            bundle.write(path, path.relative_to(ROOT).as_posix())
    (assets / 'pyodide-reference.json').write_text(
        json.dumps(reference(), indent=2) + '\n', encoding='utf-8')
    for name, _ in LESSONS:
        notebook = nbformat.read(ROOT / name, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
        notebook.metadata.pop('widgets', None)
        notebook.metadata.kernelspec = dict(
            name='python', display_name='Python (Pyodide)', language='python')
        notebook.cells.insert(0, nbformat.v4.new_code_cell(
            bootstrap(), id='browser-pilot-' + name.split('_', 2)[0] + '-' + name.split('_', 2)[1],
            metadata={'tags': ['hide-input']}))
        nbformat.write(notebook, PILOT / name)
    config = yaml.safe_load((ROOT / 'myst.yml').read_text(encoding='utf-8'))
    config['project']['id'] += '-pyodide-pilot'
    config['project']['title'] += ' — browser pilot'
    config['project']['jupyter'] = {'lite': True, 'binder': False, 'kernelName': 'python'}
    config['project']['toc'] = [{'file': 'index.md'}] + [
        {'file': name, 'title': title} for name, title in LESSONS]
    # Copy the cached theme so rendering cannot change the existing preview cache.
    theme = ROOT / '_build/templates/site/myst/book-theme'
    isolated_theme = PILOT / '_build/templates/site/myst/book-theme'
    if theme.is_dir() and not isolated_theme.is_dir():
        shutil.copytree(theme, isolated_theme, ignore=shutil.ignore_patterns('book-theme-main'))
    config['site']['template'] = str(isolated_theme) if isolated_theme.is_dir() else 'book-theme'
    for value in config['site'].get('options', {}).values():
        if isinstance(value, str):
            source = ROOT / value
            if source.is_file() and ROOT in source.resolve().parents:
                destination = PILOT / value
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    config['site'].setdefault('options', {})['style'] = 'assets/browser-widgets.css'
    (PILOT / 'myst.yml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    runtime_note = (
        '**Same-site candidate:** Python, required packages and widget scripts are '
        'served from this book. A content policy blocks external asset requests. '
        'Live validation is still required before organizational use. '
        if SAME_SITE else
        '**Development preview:** runtime and widget assets may still load from '
        'external CDNs. This build is not ready for the restricted organization. ')
    (PILOT / 'index.md').write_text(
        '# Browser execution pilot\n\n'
        'This private pilot tests existing lessons inside Jupyter Book. '
        'Open a lesson and select its power button. Python computation should '
        'run on this computer through Pyodide.\n\n'
        + runtime_note +
        'Original lesson text may still describe Binder.\n\n'
        'Check slider updates, Matplotlib output, Plotly rotation and reset controls. '
        'The generated initialization cell also compares a small grating calculation '
        'against native Python; this is a compatibility check, not convergence evidence.\n',
        encoding='utf-8')
    assert protected_files() == before, 'Authoring files changed during staging'
    (PILOT / 'source-hashes.json').write_text(json.dumps(before, indent=2) + '\n', encoding='utf-8')
    print('Staged three lessons without modifying authoring sources: ' + str(PILOT), flush=True)


def validation_identity():
    """Reuse live evidence only for the same sources, runtime and loader assets."""
    build = json.loads((PILOT / 'same-site-build.json').read_text(encoding='utf-8'))
    record = dict(
        sources=json.loads((PILOT / 'source-hashes.json').read_text(encoding='utf-8')),
        bootstrap=bootstrap(),
        runtime=hashlib.sha256((HTML / 'runtime-manifest.json').read_bytes()).hexdigest(),
        style=hashlib.sha256((PILOT / 'assets/browser-widgets.css').read_bytes()).hexdigest(),
        loaders={row['file']: hashlib.sha256((HTML / row['file']).read_bytes()).hexdigest()
                 for row in build['derived_javascript_changes']})
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def check():
    expected = json.loads((PILOT / 'source-hashes.json').read_text(encoding='utf-8'))
    assert protected_files() == expected, 'Authoring sources changed after staging; regenerate the pilot'
    config = yaml.safe_load((PILOT / 'myst.yml').read_text(encoding='utf-8'))
    assert config['project']['jupyter'] == {'lite': True, 'binder': False, 'kernelName': 'python'}
    counts = {}
    for name, _ in LESSONS:
        original = nbformat.read(ROOT / name, as_version=4)
        staged = nbformat.read(PILOT / name, as_version=4)
        nbformat.validate(staged)
        assert staged.cells[0].source == bootstrap()
        assert [(c.cell_type, c.source) for c in staged.cells[1:]] == [
            (c.cell_type, c.source) for c in original.cells], name
        counts[name] = sum(c.cell_type == 'code' for c in original.cells)
    with zipfile.ZipFile(PILOT / '_public/wave_optics_browser.zip') as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == {
            p.relative_to(ROOT).as_posix() for p in (ROOT / 'wave_optics').glob('*.py')}
    html_built = (HTML / 'index.html').is_file()
    if html_built:
        rendered = json.loads((PILOT / '_build/site/config.json').read_text(encoding='utf-8'))
        assert rendered['projects'][0]['thebe'] == {'lite': True, 'kernelName': 'python'}, (
            'Rendered book did not select browser execution')
        for name in ('wave_optics_browser.zip', 'pyodide-reference.json'):
            assert (HTML / '_public' / name).read_bytes() == (PILOT / '_public' / name).read_bytes(), (
                'Runtime bundle changed after the HTML build; rebuild the pilot')
        css = (PILOT / 'assets/browser-widgets.css').read_bytes().replace(b'\r\n', b'\n')
        assert any(p.read_bytes().replace(b'\r\n', b'\n') == css for p in HTML.rglob('*.css')), (
            'The static Plotly widget stylesheet is missing from the built book')
        if SAME_SITE:
            from pyodide_assets import verify
            manifest = verify()
            for relative, entry in manifest['files'].items():
                assert hashlib.sha256((HTML / relative).read_bytes()).hexdigest() == entry['sha256'], relative
            assert 'Content-Security-Policy' in (HTML / 'index.html').read_text(encoding='utf-8')
    live_passed = False
    validation = PILOT / 'browser-validation.json'
    if SAME_SITE and html_built and validation.is_file():
        evidence = json.loads(validation.read_text(encoding='utf-8'))
        live_passed = evidence.get('identity') == validation_identity() and all(
            evidence.get('lessons', {}).get(name, {}).get('execution') == 'passed'
            and evidence.get('lessons', {}).get(name, {}).get('controls') == 'passed'
            for name, _ in LESSONS)
    report = dict(authoring_sources='unchanged', backend='JupyterLite/Pyodide',
                  binder_enabled=False, original_code_cells=counts,
                  html_built=html_built,
                  plotly_styles='bundled with book stylesheet',
                  browser_execution='passed for three pilot lessons' if live_passed else 'not yet verified for this build',
                  widget_callbacks='passed; user-confirmed' if live_passed else 'not yet verified for this build',
                  same_site_only='enforced by CSP; see live validation' if SAME_SITE else 'not yet implemented',
                  full_series_validated=False,
                  organization_ready=False)
    (PILOT / 'checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def build():
    stage()
    command = Path(sys.executable).parent / 'Scripts' / 'jupyter-book.exe'
    executable = str(command) if command.is_file() else shutil.which('jupyter-book')
    if not executable:
        raise RuntimeError('Run with the existing grating-book environment.')
    log_path = PILOT / 'build.log'
    with log_path.open('w', encoding='utf-8') as log:
        result = subprocess.run([executable, 'build', '--html'], cwd=PILOT,
                                stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, BASE_URL=BASE_URL, PYTHONIOENCODING='utf-8'))
    if result.returncode:
        print(log_path.read_text(encoding='utf-8')[-1800:], flush=True)
        raise RuntimeError('Book build failed; see ' + str(log_path))
    shutil.copytree(PILOT / '_public', HTML / '_public', dirs_exist_ok=True)
    if SAME_SITE:
        from pyodide_assets import package_site
        package_site(HTML, BASE_URL)
    check()


def serve_static(folder, port, base_url="", same_site=True, report_folder=None):
    """Serve one book artifact; the server performs no Python computation."""
    get_folder = folder if callable(folder) else lambda: Path(folder)
    report_folder = Path(report_folder) if report_folder else Path(get_folder()).parent.parent
    from urllib.parse import urlsplit
    class PreviewHandler(SimpleHTTPRequestHandler):
        extensions_map = dict(SimpleHTTPRequestHandler.extensions_map,
                              **{'.wasm': 'application/wasm', '.whl': 'application/octet-stream'})

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(get_folder()), **kwargs)

        def translate_path(self, path):
            if base_url:
                if not urlsplit(path).path.startswith(base_url + '/'):
                    return str(report_folder / '__outside_base_url__')
                path = path[len(base_url):]
            return super().translate_path(path)

        def end_headers(self):
            if same_site:
                from pyodide_assets import content_policy
                self.send_header('Content-Security-Policy', content_policy() +
                                 '; report-uri ' + base_url + '/__pilot_csp__')
                self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def do_POST(self):
            if same_site and self.path == base_url + '/__pilot_csp__':
                size = int(self.headers.get('Content-Length', '0'))
                if 0 < size <= 16384:
                    try:
                        report = json.loads(self.rfile.read(size))
                    except ValueError:
                        self.send_error(400)
                        return
                    with (report_folder / 'csp-violations.jsonl').open('a', encoding='utf-8') as log:
                        log.write(json.dumps(report) + '\n')
                    self.send_response(204)
                    self.end_headers()
                    return
            self.send_error(404)

    print(f'Static preview only (no Python computation server): http://127.0.0.1:{port}{base_url}/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), PreviewHandler).serve_forever()


def main():
    global PILOT, HTML, SAME_SITE, BASE_URL
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('stage', 'build', 'check', 'serve', 'vendor'))
    parser.add_argument('--same-site', action='store_true')
    parser.add_argument('--base-url', default='')
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    BASE_URL = args.base_url.rstrip('/')
    if BASE_URL and (not BASE_URL.startswith('/') or any(p in BASE_URL for p in ('..', '?', '#', '://'))):
        parser.error('base-url must be an absolute URL path without traversal, query or fragment.')
    SAME_SITE = args.same_site
    if SAME_SITE:
        PILOT = ROOT / '_build/pyodide-packaged'
        HTML = PILOT / '_build/html'
    if args.command == 'vendor':
        from pyodide_assets import Builder
        Builder().prepare()
        return
    if args.command == 'serve':
        if not (HTML / 'index.html').is_file():
            parser.error('Run build before serve.')
        serve_static(HTML, args.port or (3022 if SAME_SITE else 3021), BASE_URL, SAME_SITE, PILOT)
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
