"""Build the complete Pyodide edition from the shared teaching sources.

Use the grating-book environment:
    python book/pyodide_book.py build
    python book/pyodide_book.py serve
    python book/pyodide_book.py check

The Binder configuration and publishing workflow stay unchanged. Generated
sources, native execution copies and runtime assets stay under _build. A full
build is separate from the three-lesson pilot and never commits or publishes.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

import nbformat
import yaml

from public_content import PUBLIC_NOTEBOOKS, check_readers, source_files
from pyodide_assets import package_site, verify
from pyodide_pilot import bootstrap, reference, serve_static, widget_stylesheet
from pyodide_static import static_outputs, check_exported_figures

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '_build/pyodide-book'
STAGE = WORK / 'source'
HTML = STAGE / '_build/html'
PLOTLY_LESSONS = {
    '5_1_Three_Dimensional_Orders_and_Conical_Diffraction.ipynb',
    '5_2_Direction_Cosines_and_Spatial_Frequency_Coordinates.ipynb',
    '7_1_Interference_and_Mutual_Coherence_of_Two_Plane_Waves.ipynb',
    '7_2_Interference_of_Diffraction_Orders_from_a_Grating.ipynb',
    '7_3_Finite_Diffraction_Beams_and_Experimental_Recombination.ipynb',
}
COMPACT_LESSON = '7_2_Interference_of_Diffraction_Orders_from_a_Grating.ipynb'


def presentation_cell(name):
    if name != COMPACT_LESSON:
        return None
    return nbformat.v4.new_code_cell(
        (ROOT / 'book/compact-order-interference.py').read_text(encoding='utf-8'),
        id='browser-compact-interference', metadata={'tags': ['hide-input']})


def add_presentation(notebook, name):
    cell = presentation_cell(name)
    if cell is not None:
        position = next(i for i, c in enumerate(notebook.cells) if c.id == 'n72-explorer')
        notebook.cells.insert(position, cell)

GUIDANCE = {
    'Binder supplies Python in your browser; no local installation is needed.':
        'Python runs on your computer through Pyodide; no installation is needed.',
    'Select the power button to connect to a Binder-backed Python session. No local Python installation is required.':
        'Select the power button to start Python in your browser. Calculations run on your computer; no Python installation is required.',
    'Binder startup can take time. A new or restarted session may require running the cells again.':
        'The first activation loads Python and its libraries from this website and can take time. A restarted session may require running the cells again.',
}


def plotly_bootstrap(name):
    """Use the same strict-JSON widget transport in native and browser kernels."""
    code = ''
    if name in PLOTLY_LESSONS:
        adapter = ((ROOT / 'book/plotly-widget-layout.js').read_text(encoding='utf-8')
                   if name in (COMPACT_LESSON,
                       '5_1_Three_Dimensional_Orders_and_Conical_Diffraction.ipynb') else '')
        code += f'''
import plotly.graph_objects as _book_go
if not getattr(_book_go.FigureWidget, '_book_buffer_adapter', False):
    import copy as _book_copy
    import math as _book_math
    import numpy as _book_np
    from plotly.serializers import _py_to_js as _book_original_serializer
    _book_plotly_source = str(_book_go.FigureWidget._esm)
    for _needle in ('MD=class', 'Md.default.newPlot(ne.el', 'MD as FigureView'):
        if _needle not in _book_plotly_source:
            raise RuntimeError('Review the changed Plotly widget renderer before using its book layout adapter.')
    # The anywidget model applies raw serialized batch messages to its cache.
    # Skip the wire-format sentinel as well as JavaScript undefined; otherwise
    # an update for another trace overwrites retained coordinates before render.
    _book_skip = 'Tr!==void 0&&qs.set(Ar,je,Tr)'
    if _book_plotly_source.count(_book_skip) != 1:
        raise RuntimeError('Review the changed Plotly sparse-update model cache.')
    _book_plotly_source = _book_plotly_source.replace(
        _book_skip, 'Tr!==void 0&&Tr!=="_undefined_"&&qs.set(Ar,je,Tr)')
    def _book_array_serializer(value, widget):
        if isinstance(value, _book_np.ndarray):
            return _book_array_serializer(value.tolist(), widget)
        if isinstance(value, _book_np.generic):
            return _book_array_serializer(value.item(), widget)
        if isinstance(value, float) and not _book_math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {{key: _book_array_serializer(item, widget) for key, item in value.items()}}
        if isinstance(value, (list, tuple)):
            return [_book_array_serializer(item, widget) for item in value]
        return _book_original_serializer(value, widget)
    _book_traits = {{}}
    for _name, _trait in _book_go.FigureWidget.class_traits().items():
        if _trait.metadata.get('to_json') is _book_original_serializer:
            _copy = _book_copy.copy(_trait)
            _copy.metadata = dict(_trait.metadata, to_json=_book_array_serializer)
            _book_traits[_name] = _copy
    if not {{'_widget_data', '_py2js_update', '_py2js_addTraces'}} <= _book_traits.keys():
        raise RuntimeError('Review the changed Plotly widget serialization traits.')
    _book_go.FigureWidget = type('_BookFigureWidget', (_book_go.FigureWidget,),
        dict(_book_traits, _book_buffer_adapter=True, _esm=_book_plotly_source + {adapter!r}))
'''
    return code


def markdown_bootstrap(name):
    if name == '7_1_Interference_and_Mutual_Coherence_of_Two_Plane_Waves.ipynb':
        return (ROOT / 'book/phase-geometry-markdown.py').read_text(encoding='utf-8')
    return ''


def browser_bootstrap(name, base_url):
    code = (bootstrap(same_site=True, base_url=base_url) + plotly_bootstrap(name)
            + markdown_bootstrap(name))
    if name == '7_3_Finite_Diffraction_Beams_and_Experimental_Recombination.ipynb':
        # This lesson adds already-3D collections with autoscaling disabled.
        # Matplotlib 3.8 lacks Axes3D's newer autolim keyword; its base Axes
        # implementation supports the required operation. Scope the adapter
        # to this lesson's browser kernel and leave scientific sources intact.
        code += '''
import inspect as _inspect
from matplotlib.axes import Axes as _BaseAxes
from mpl_toolkits.mplot3d import Axes3D as _BrowserAxes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection as _Line3D, Poly3DCollection as _Poly3D
if 'autolim' not in _inspect.signature(_BrowserAxes3D.add_collection3d).parameters:
    _original_add_collection3d = _BrowserAxes3D.add_collection3d
    def _browser_add_collection3d(self, collection, zs=0, zdir='z', *, autolim=True):
        if autolim:
            return _original_add_collection3d(self, collection, zs=zs, zdir=zdir)
        if not isinstance(collection, (_Line3D, _Poly3D)):
            raise TypeError('The browser adapter requires an already-3D collection.')
        return _BaseAxes.add_collection(self, collection, autolim=False)
    _BrowserAxes3D.add_collection3d = _browser_add_collection3d
'''
    return code


def reader_text(text):
    for source, replacement in GUIDANCE.items():
        text = text.replace(source, replacement)
    return text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject_json_constant(value):
    raise ValueError('Generated notebook contains a nonstandard JSON number: ' + value)


def selection():
    return source_files(ROOT)


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in selection()}


def browser_stylesheet():
    return widget_stylesheet() + '\n' + (ROOT / 'book/browser-layout.css').read_text(encoding='utf-8')


def toc_files(entries):
    for entry in entries:
        if 'file' in entry:
            yield entry['file']
        yield from toc_files(entry.get('children', []))


def initialize(base_url):
    verify()
    STAGE.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    config = yaml.safe_load((ROOT / 'myst.yml').read_text(encoding='utf-8'))
    pages = list(toc_files(config['project']['toc']))
    notebooks = [p for p in pages if p.endswith('.ipynb')]
    if set(notebooks) != set(PUBLIC_NOTEBOOKS) or len(notebooks) != len(set(notebooks)):
        raise RuntimeError('Review the difference between the public notebook selection and the book TOC.')
    for source in selection():
        relative = source.relative_to(ROOT)
        destination = STAGE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == '.md':
            destination.write_text(reader_text(source.read_text(encoding='utf-8')), encoding='utf-8')
        else:
            shutil.copy2(source, destination)
    config['project']['id'] += '-pyodide-book'
    config['project']['jupyter'] = {'lite': True, 'binder': False, 'kernelName': 'python'}
    config['site'].setdefault('options', {})['style'] = 'assets/browser-widgets.css'
    (STAGE / 'assets/browser-widgets.css').write_text(browser_stylesheet(), encoding='utf-8')
    theme = ROOT / '_build/templates/site/myst/book-theme'
    isolated = STAGE / '_build/templates/site/myst/book-theme'
    if theme.is_dir() and not isolated.is_dir():
        shutil.copytree(theme, isolated, ignore=shutil.ignore_patterns('book-theme-main'))
    config['site']['template'] = str(isolated) if isolated.is_dir() else 'book-theme'
    (STAGE / 'myst.yml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (WORK / 'source-hashes.json').write_text(json.dumps(before, indent=2) + '\n', encoding='utf-8')
    (WORK / 'profile.json').write_text(json.dumps(dict(base_url=base_url, pages=pages,
        notebooks=notebooks), indent=2) + '\n', encoding='utf-8')
    return notebooks


def native_outputs(notebooks):
    """Fresh native kernels produce static book outputs, never modify authoring notebooks."""
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    directory = WORK / 'kernels/python3'
    directory.mkdir(parents=True, exist_ok=True)
    kernel_env = dict(IPYTHONDIR=str(WORK / 'ipython'), MPLCONFIGDIR=str(WORK / 'matplotlib'),
                      PYTHONDONTWRITEBYTECODE='1')
    (directory / 'kernel.json').write_text(json.dumps(dict(argv=[sys.executable, '-B', '-m',
        'ipykernel_launcher', '-f', '{connection_file}'], display_name='Book build', language='python',
        env=kernel_env)), encoding='utf-8')
    versions = {n: metadata.version(n) for n in ('numpy', 'scipy', 'matplotlib', 'plotly',
                                               'ipywidgets', 'anywidget', 'nbclient')}
    dependencies = {p.relative_to(ROOT).as_posix(): sha(p) for p in selection()
                    if p.parts and p.relative_to(ROOT).parts[0] in ('wave_optics', 'assets')}
    cache = WORK / 'native'
    cache.mkdir(exist_ok=True)
    results = []
    for name in notebooks:
        source = nbformat.read(ROOT / name, as_version=4)
        key = hashlib.sha256(json.dumps(dict(notebook=sha(ROOT / name),
            dependencies=dependencies, versions=versions,
            **({'plotly_transport': plotly_bootstrap(name) + markdown_bootstrap(name)}
               if name in PLOTLY_LESSONS else {}),
            **({'presentation': presentation_cell(name).source} if name == COMPACT_LESSON else {})),
            sort_keys=True).encode()).hexdigest()
        receipt = cache / (Path(name).stem + '.json')
        executed = cache / name
        saved = json.loads(receipt.read_text()) if receipt.exists() else {}
        start = time.monotonic()
        cached = saved.get('key') == key and executed.exists() and saved.get('sha256') == sha(executed)
        if cached:
            previous = nbformat.read(executed, as_version=4)
            models = previous.metadata.get('widgets', {}).get('application/vnd.jupyter.widget-state+json', {}).get('state', {})
            if any(m.get('model_name') == 'AnyModel' for m in models.values()):
                cached = 'browser_plotly_figures' in previous.metadata
        if not cached:
            notebook = nbformat.from_dict(source)
            notebook.metadata.pop('widgets', None)
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    cell.outputs = []
                    cell.execution_count = None
            notebook.metadata.kernelspec = dict(name='python3', display_name='Python 3', language='python')
            if name in PLOTLY_LESSONS:
                notebook.cells.insert(0, nbformat.v4.new_code_cell(
                    plotly_bootstrap(name) + markdown_bootstrap(name),
                    id='book-native-plotly-transport'))
            add_presentation(notebook, name)
            if any(c.cell_type == 'code' for c in notebook.cells):
                manager = KernelManager(kernel_name='python3', kernel_spec_manager=KernelSpecManager(
                    kernel_dirs=[str(directory.parent)]))
                try:
                    client = NotebookClient(notebook, km=manager, timeout=600,
                        resources={'metadata': {'path': str(STAGE)}})
                    client.execute()
                    # FigureWidget transports incremental updates rather than
                    # replacing _widget_data. Saved widget metadata alone can
                    # therefore contain empty initial traces. Read the final
                    # figures from their live Python objects before shutdown.
                    capture = nbformat.v4.new_code_cell('''
import json as _snapshot_json
from ipywidgets.widgets.widget import _instances as _snapshot_widgets
from plotly.utils import PlotlyJSONEncoder as _SnapshotEncoder
print(_snapshot_json.dumps({key: widget.to_plotly_json()
    for key, widget in list(_snapshot_widgets.items())
    if callable(getattr(widget, 'to_plotly_json', None))}, cls=_SnapshotEncoder))
''')
                    notebook.cells.append(capture)
                    try:
                        client.execute_cell(capture, len(notebook.cells)-1, store_history=False)
                        payload = ''.join(o.text for o in capture.outputs
                                          if o.output_type == 'stream' and o.name == 'stdout')
                        notebook.metadata['browser_plotly_figures'] = json.loads(payload)
                    finally:
                        notebook.cells.pop()
                finally:
                    if manager.has_kernel:
                        manager.shutdown_kernel(now=True)
            notebook.cells = [c for c in notebook.cells if c.id not in (
                'browser-compact-interference', 'book-native-plotly-transport')]
            nbformat.write(notebook, executed)
            receipt.write_text(json.dumps(dict(key=key, sha256=sha(executed))), encoding='utf-8')
        notebook = nbformat.read(executed, as_version=4)
        if any('Message serialization failed' in output.get('text', '')
               for cell in notebook.cells for output in cell.get('outputs', [])):
            raise RuntimeError('Invalid widget JSON was saved in native output: ' + name)
        # Only execution metadata and outputs can differ from the authoring source.
        assert [(c.cell_type, c.source) for c in notebook.cells] == [(c.cell_type, c.source) for c in source.cells], name
        results.append(dict(notebook=name, code_cells=sum(c.cell_type == 'code' for c in source.cells),
                            elapsed_seconds=round(time.monotonic() - start, 2)))
        print(f'Native outputs ready [{len(results)}/{len(notebooks)}]: {name}', flush=True)
    (WORK / 'native-validation.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


def browser_sources(notebooks, base_url):
    assets = STAGE / '_public'
    assets.mkdir(exist_ok=True)
    bundle_path = assets / 'wave_optics_browser.zip'
    temporary = bundle_path.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in selection():
            relative = path.relative_to(ROOT)
            if relative.parts[0] in ('wave_optics', 'assets'):
                bundle.write(path, relative.as_posix())
    temporary.replace(bundle_path)
    (assets / 'pyodide-reference.json').write_text(json.dumps(reference(), indent=2) + '\n', encoding='utf-8')
    plotly = metadata.distribution('plotly')
    license_file = next(p for p in plotly.files if p.name in ('LICENSE', 'LICENSE.txt'))
    shutil.copyfile(plotly.locate_file(license_file), assets / 'plotly-license.txt')
    snapshots = {}
    for name in notebooks:
        notebook = nbformat.read(WORK / 'native' / name, as_version=4)
        snapshots[name] = static_outputs(notebook)
        add_presentation(notebook, name)
        for cell in notebook.cells:
            if cell.cell_type == 'markdown':
                cell.source = reader_text(cell.source)
        if any(c.cell_type == 'code' for c in notebook.cells):
            cell_id = 'browser-init-' + hashlib.sha256(name.encode()).hexdigest()[:12]
            notebook.cells.insert(0, nbformat.v4.new_code_cell(browser_bootstrap(name, base_url),
                                  id=cell_id, metadata={'tags': ['hide-input']}))
        notebook.metadata.kernelspec = dict(name='python', display_name='Python (Pyodide)', language='python')
        nbformat.write(notebook, STAGE / name)
    (WORK / 'static-figures.json').write_text(json.dumps(snapshots, indent=2) + '\n', encoding='utf-8')


def check():
    profile = json.loads((WORK / 'profile.json').read_text(encoding='utf-8'))
    expected = json.loads((WORK / 'source-hashes.json').read_text(encoding='utf-8'))
    assert source_hashes() == expected, 'Authoring sources changed after the build'
    code_cells = 0
    for name in profile['notebooks']:
        original = nbformat.read(ROOT / name, as_version=4)
        json.loads((STAGE / name).read_text(encoding='utf-8'), parse_constant=reject_json_constant)
        staged = nbformat.read(STAGE / name, as_version=4)
        nbformat.validate(staged)
        assert not any('Message serialization failed' in output.get('text', '')
                       for cell in staged.cells for output in cell.get('outputs', [])), name
        has_code = any(c.cell_type == 'code' for c in original.cells)
        cells = staged.cells[1:] if has_code else staged.cells
        if has_code:
            assert staged.cells[0].source == browser_bootstrap(name, profile['base_url'])
        presentation = presentation_cell(name)
        if presentation is not None:
            generated = [c for c in cells if c.id == presentation.id]
            assert len(generated) == 1 and generated[0].source == presentation.source
            position = next(i for i, c in enumerate(cells) if c.id == presentation.id)
            assert cells[position + 1].id == 'n72-explorer'
            cells = [c for c in cells if c.id != presentation.id]
        assert [(c.cell_type, c.source) for c in cells] == [
            (c.cell_type, reader_text(c.source) if c.cell_type == 'markdown' else c.source)
            for c in original.cells], name
        code_cells += sum(c.cell_type == 'code' for c in original.cells)
    built = json.loads((STAGE / '_build/site/config.json').read_text(encoding='utf-8'))
    assert built['projects'][0]['thebe'] == {'lite': True, 'kernelName': 'python'}
    assert sum('slug' in p for p in built['projects'][0]['pages']) + 1 == len(profile['pages'])
    manifest = verify()
    for path, entry in manifest['files'].items():
        assert sha(HTML / path) == entry['sha256'], path
    import re
    for page in HTML.rglob('*.html'):
        text = page.read_text(encoding='utf-8')
        assert not re.search(r'<(?:link|script)[^>]+(?:href|src)=[\"\']https?://', text), page
        assert 'Content-Security-Policy' in text, page
    check_readers(STAGE)
    exported_style = (HTML / 'myst-theme.css').read_text(encoding='utf-8')
    staged_style = (STAGE / 'assets/browser-widgets.css').read_text(encoding='utf-8')
    assert staged_style.endswith((ROOT / 'book/browser-layout.css').read_text(encoding='utf-8'))
    assert exported_style.startswith('/* User Provided Stylesheet */\n\n')
    assert exported_style.removeprefix('/* User Provided Stylesheet */\n\n') == staged_style
    plotly_figures = check_exported_figures(STAGE, HTML, profile['notebooks'], profile['base_url'])
    from pyodide_validate import summarize
    wasm = summarize(STAGE)
    initialization_cells = sum(any(c.cell_type == 'code' for c in nbformat.read(ROOT / name, as_version=4).cells)
                               for name in profile['notebooks'])
    initialization_cells += sum(presentation_cell(name) is not None for name in profile['notebooks'])
    if wasm['passed']:
        assert wasm['code_cells'] == code_cells + initialization_cells
    report = dict(notebooks=len(profile['notebooks']), pages=len(profile['pages']), original_code_cells=code_cells,
                  native_execution='passed', original_sources='unchanged', backend='JupyterLite/Pyodide',
                  runtime='served with book; remote requests blocked by CSP',
                  wasm_execution='passed' if wasm['passed'] else 'not yet verified for this full build',
                  static_plotly_figures_verified=plotly_figures,
                  browser_controls='not yet verified across the full series', published=False)
    (WORK / 'checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


def build(base_url):
    notebooks = initialize(base_url)
    native_outputs(notebooks)
    browser_sources(notebooks, base_url)
    render(base_url)


def render(base_url):
    # Cached exports can retain a cell's previous MIME data after only its
    # saved output changes. Preserve native execution and theme caches, but
    # regenerate the derived pages and output assets from the staged notebooks.
    build_root = (STAGE / '_build').resolve()
    for name in ('site', 'html'):
        target = (build_root / name).resolve()
        if target.parent != build_root or STAGE.resolve() not in target.parents:
            raise RuntimeError('Unexpected generated render directory: ' + str(target))
        if target.is_dir():
            shutil.rmtree(target)
    executable = Path(sys.executable).parent / (
        'Scripts/jupyter-book.exe' if sys.platform == 'win32' else 'jupyter-book')
    log = WORK / 'build.log'
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([str(executable), 'build', '--html'], cwd=STAGE,
            stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ, BASE_URL=base_url, PYTHONIOENCODING='utf-8'))
    if result.returncode:
        raise RuntimeError('Book build failed; see ' + str(log))
    shutil.copytree(STAGE / '_public', HTML / '_public', dirs_exist_ok=True)
    package_site(HTML, base_url)
    publish_snapshot(base_url)


def publish_snapshot(base_url):
    check()
    # Preview only completed, verified snapshots. Rebuilding must not serve the
    # renderer's intermediate HTML while it still contains CDN references.
    snapshot = WORK / 'served' / uuid.uuid4().hex
    shutil.copytree(HTML, snapshot)
    pointer = WORK / 'preview.json'
    temporary = pointer.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(directory=str(snapshot), base_url=base_url)), encoding='utf-8')
    temporary.replace(pointer)
    print('Completed book snapshot: ' + str(snapshot), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build', 'check', 'serve'))
    parser.add_argument('--base-url', default='')
    parser.add_argument('--port', type=int, default=3023)
    args = parser.parse_args()
    base_url = args.base_url.rstrip('/')
    if base_url and (not base_url.startswith('/') or any(p in base_url for p in ('..', '?', '#', '://'))):
        parser.error('base-url must be an absolute URL path without traversal, query or fragment')
    if args.command == 'build':
        build(base_url)
    elif args.command == 'check':
        check()
    else:
        pointer = WORK / 'preview.json'
        if not pointer.exists():
            parser.error('Build the complete edition before serving it')
        selected = json.loads(pointer.read_text(encoding='utf-8'))
        def completed_snapshot():
            current = json.loads(pointer.read_text(encoding='utf-8'))
            directory = Path(current['directory']).resolve()
            if (WORK / 'served').resolve() not in directory.parents:
                raise RuntimeError('Unexpected preview snapshot directory')
            if current['base_url'] != selected['base_url']:
                raise RuntimeError('Restart preview after changing the deployment base URL')
            return directory
        serve_static(completed_snapshot, args.port, selected['base_url'], True, WORK)


if __name__ == '__main__':
    main()
