"""Prepare and verify the pinned browser runtime, using network only on request.

Generated downloads and the static tree remain under ignored _build. This is
for the isolated Pyodide pilot, not the private-to-public publishing workflow.
"""

from collections import deque
import base64
from email.parser import BytesParser
import hashlib
from html import escape
import json
from pathlib import Path
import re
import shutil
import tarfile
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '_build/pyodide-runtime'
CACHE = WORK / 'cache'
SITE = WORK / 'site'
PYODIDE = '0.27.0'
KERNEL = '0.4.7'
PYODIDE_URL = f'https://cdn.jsdelivr.net/pyodide/v{PYODIDE}/full/'
KERNEL_URL = f'https://unpkg.com/@jupyterlite/pyodide-kernel@{KERNEL}/pypi/'
ROOT_PACKAGES = ('micropip', 'ssl', 'sqlite3', 'ipykernel', 'comm', 'pyodide-kernel',
                 'ipython', 'numpy', 'scipy', 'matplotlib', 'pillow', 'ipywidgets',
                 'plotly==7.1.0', 'anywidget==0.11.0', f'piplite=={KERNEL}')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Builder:
    def __init__(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        SITE.mkdir(parents=True, exist_ok=True)
        self.files = {}
        self.selected = {}
        self.constraints = {}
        self.index = {}
        self.queue = deque(Requirement(p) for p in ROOT_PACKAGES)
        self.environment = dict(default_environment(), python_version='3.12',
                                python_full_version='3.12.7', sys_platform='emscripten',
                                platform_system='Emscripten', platform_machine='wasm32', extra='')

    def fetch(self, url, name, sha256=None):
        path = CACHE / name
        if CACHE.resolve() not in path.resolve().parents:
            raise ValueError('Unexpected cache path')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            request = Request(url, headers={'User-Agent': 'grating-book-browser-build/1'})
            with urlopen(request, timeout=90) as response:
                data = response.read()
            temporary = path.with_suffix(path.suffix + '.part')
            temporary.write_bytes(data)
            temporary.replace(path)
            print(f'Fetched {name}: {path.stat().st_size / 1_000_000:.2f} MB', flush=True)
        if sha256 and digest(path) != sha256:
            raise RuntimeError('Download checksum mismatch: ' + name)
        return path

    def publish_file(self, source, relative, url):
        destination = SITE / relative
        if SITE.resolve() not in destination.resolve().parents:
            raise ValueError('Unexpected runtime output path')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        self.files[relative] = dict(sha256=digest(destination), bytes=destination.stat().st_size,
                                    source=url)

    def json(self, url, name):
        return json.loads(self.fetch(url, name).read_text(encoding='utf-8'))

    def stylesheet(self, relative, url):
        path = self.fetch(url, relative)
        self.publish_file(path, relative, url)
        for target in re.findall(r'url\(\s*[\'\"]?([^\s)\'\"]+)', path.read_text(encoding='utf-8')):
            if target.startswith(('data:', '#')):
                continue
            remote = urljoin(url, target)
            if urlsplit(remote).netloc != urlsplit(url).netloc:
                raise RuntimeError('Review external stylesheet dependency: ' + remote)
            # Keep the CSS's relative font paths working within the static tree.
            destination = (SITE / relative).parent / urlsplit(target).path
            local = destination.resolve().relative_to(SITE.resolve()).as_posix()
            self.publish_file(self.fetch(remote, local), local, remote)

    def mathjax(self):
        """Thebe's widget math renderer loads this exact MathJax distribution."""
        version = '2.7.5'
        metadata = self.json(f'https://registry.npmjs.org/mathjax/{version}', f'mathjax-{version}.json')
        distribution = metadata['dist']
        archive = self.fetch(distribution['tarball'], f'mathjax-{version}.tgz')
        algorithm, expected = distribution['integrity'].split('-', 1)
        actual = base64.b64encode(hashlib.new(algorithm, archive.read_bytes()).digest()).decode()
        if actual != expected:
            raise RuntimeError('MathJax archive integrity mismatch')
        with tarfile.open(archive, 'r:gz') as bundle:
            for member in bundle.getmembers():
                parts = Path(member.name).parts
                if not member.isfile() or not parts or parts[0] != 'package' or '..' in parts:
                    continue
                relative = '/'.join(parts[1:])
                if not relative.startswith(('config/', 'extensions/', 'jax/', 'fonts/', 'localization/')) and relative not in ('MathJax.js', 'LICENSE'):
                    continue
                destination = f'runtime/cdnjs/mathjax/{version}/' + relative
                cached = CACHE / destination
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(bundle.extractfile(member).read())
                self.publish_file(cached, destination, distribution['tarball'])

    def wheel(self, name, version, artifact, url):
        path = self.fetch(url, artifact['filename'], artifact['digests']['sha256'])
        self.publish_file(path, 'pypi/' + path.name, url)
        self.selected[name] = dict(version=version, source=url)
        self.index[name] = {'releases': {version: [{
            'filename': path.name, 'url': './' + path.name, 'packagetype': 'bdist_wheel',
            'digests': {'sha256': digest(path)}, 'yanked': False,
        }]}}
        with zipfile.ZipFile(path) as archive:
            metadata_name = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
        for text in metadata.get_all('Requires-Dist', []):
            requirement = Requirement(text)
            if not requirement.marker or requirement.marker.evaluate(self.environment):
                self.queue.append(requirement)

    def package(self, requirement):
        name = canonicalize_name(requirement.name)
        self.constraints.setdefault(name, set()).add(str(requirement.specifier))
        specifier = SpecifierSet(','.join(sorted(filter(None, self.constraints[name]))))
        if name in self.selected:
            if not specifier or Version(self.selected[name]['version']) in specifier:
                return
            # Pure Python packages can be upgraded to meet a later dependency.
            # Compiled packages must remain compatible with the pinned runtime.
            previous = self.builtins.get(name)
            if previous and not previous['file_name'].endswith('none-any.whl'):
                raise RuntimeError(f'Conflicting compiled browser package requirement: {name}{specifier}')
            self.selected.pop(name)
            self.builtins.pop(name, None)
        builtin = self.packages.get(name)
        if builtin and specifier and Version(builtin['version']) not in specifier:
            if not builtin['file_name'].endswith('none-any.whl'):
                raise RuntimeError(f'The pinned Pyodide package cannot satisfy {name}{specifier}')
            builtin = None
        if builtin:
            filename = builtin['file_name']
            url = PYODIDE_URL + filename
            path = self.fetch(url, filename, builtin['sha256'])
            self.publish_file(path, 'runtime/pyodide/' + filename, url)
            self.selected[name] = dict(version=builtin['version'], source=url, builtin=True)
            self.builtins[name] = builtin
            self.queue.extend(Requirement(p) for p in builtin['depends'])
            return
        if name in self.lite_index:
            releases = self.lite_index[name]['releases']
            version = next((v for v in sorted(releases, key=Version, reverse=True)
                            if Version(v) in specifier), None)
            if version is None:
                raise RuntimeError('No matching JupyterLite wheel: ' + str(requirement))
            artifact = releases[version][0]
            self.wheel(name, version, artifact, urljoin(KERNEL_URL, artifact['url']))
            return
        metadata = self.json(f'https://pypi.org/pypi/{name}/json', 'pypi-' + name + '.json')
        releases = metadata['releases']
        for version in sorted((v for v in releases if not Version(v).is_prerelease), key=Version, reverse=True):
            if Version(version) not in specifier:
                continue
            for artifact in releases[version]:
                filename = artifact['filename']
                python = artifact.get('requires_python')
                if artifact.get('yanked') or not filename.endswith(('-py3-none-any.whl', '-py2.py3-none-any.whl')):
                    continue
                if python and Version('3.12.7') not in Requirement('python' + python).specifier:
                    continue
                self.wheel(name, version, artifact, artifact['url'])
                return
        raise RuntimeError('No compatible pure Python wheel: ' + str(requirement))

    def prepare(self):
        lock = self.json(PYODIDE_URL + 'pyodide-lock.json', 'pyodide-lock.json')
        if lock['info']['version'] != PYODIDE:
            raise RuntimeError('Unexpected Pyodide distribution version')
        self.packages = {canonicalize_name(k): v for k, v in lock['packages'].items()}
        self.lite_index = self.json(KERNEL_URL + 'all.json', 'lite-all.json')
        self.builtins = {}
        for filename in ('pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip'):
            url = PYODIDE_URL + filename
            self.publish_file(self.fetch(url, filename), 'runtime/pyodide/' + filename, url)
        while self.queue:
            self.package(self.queue.popleft())
        # Retain only the built-in packages actually shipped with the book.
        lock['packages'] = self.builtins
        lock_path = CACHE / 'local-pyodide-lock.json'
        lock_path.write_text(json.dumps(lock), encoding='utf-8')
        self.publish_file(lock_path, 'runtime/pyodide/pyodide-lock.json', PYODIDE_URL + 'pyodide-lock.json')
        index_path = CACHE / 'local-all.json'
        index_path.write_text(json.dumps(self.index), encoding='utf-8')
        self.publish_file(index_path, 'pypi/all.json', 'generated from verified wheel metadata')
        javascript = {
            'runtime/require.min.js': 'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js',
            'runtime/npm/anywidget@0.11.0/dist/index.js': 'https://cdn.jsdelivr.net/npm/anywidget@0.11.0/dist/index.js',
            'runtime/licenses/pyodide.txt': 'https://raw.githubusercontent.com/pyodide/pyodide/0.27.0/LICENSE',
            'runtime/licenses/anywidget.txt': 'https://cdn.jsdelivr.net/npm/anywidget@0.11.0/LICENSE',
        }
        for relative, url in javascript.items():
            self.publish_file(self.fetch(url, relative), relative, url)
        stylesheets = {
            'runtime/npm/jupyter-matplotlib@0.11.3/css/mpl_widget.css':
                'https://cdn.jsdelivr.net/npm/jupyter-matplotlib@0.11.3/css/mpl_widget.css',
            'runtime/cdnjs/font-awesome/4.7.0/css/font-awesome.css':
                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css',
            'runtime/npm/katex@0.15.2/dist/katex.min.css':
                'https://cdn.jsdelivr.net/npm/katex@0.15.2/dist/katex.min.css',
        }
        for relative, url in stylesheets.items():
            self.stylesheet(relative, url)
        self.mathjax()
        manifest = dict(pyodide=PYODIDE, kernel=KERNEL, packages=self.selected, files=self.files,
                        total_bytes=sum(f['bytes'] for f in self.files.values()))
        (SITE / 'runtime-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        print(f"Prepared {len(self.selected)} packages; {manifest['total_bytes'] / 1_000_000:.1f} MB of static assets.", flush=True)
        verify()


def verify():
    path = SITE / 'runtime-manifest.json'
    if not path.is_file():
        raise RuntimeError('Prepare assets first: python book/pyodide_pilot.py vendor')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    for relative, record in manifest['files'].items():
        candidate = SITE / relative
        if SITE.resolve() not in candidate.resolve().parents or digest(candidate) != record['sha256']:
            raise RuntimeError('Runtime asset verification failed: ' + relative)
    return manifest


def content_policy():
    """Block remote runtime traffic without changing machine/browser settings."""
    return ("default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval' blob:; "
            "connect-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob: data:; font-src 'self' data:; "
            "worker-src 'self' blob:; frame-src 'self' blob:; object-src 'none'; base-uri 'self'")


def pin_widget_loader(text):
    """Static hosts cannot resolve the semver ranges accepted by npm CDNs.

    anywidget's Python model advertises ~0.11.*. Both the lazy Thebe bundle and
    the theme's bundled loader must request the exact frontend shipped here.
    Keep other widget modules on their original paths so missing assets fail.
    """
    pins = 0
    for base in ('r', 'cdn'):
        needle = '${' + base + '}${n}@${t}/dist/'
        replacement = '${' + base + '}${n}@${n==="anywidget"?"0.11.0":t}/dist/'
        pins += text.count(needle)
        text = text.replace(needle, replacement)
    return text, pins


def use_memory_filesystem(text):
    """MyST serves book pages, not a JupyterLite contents filesystem.

    Once a service worker takes control, the default kernel attempts to mount
    that contents API. It can fail before Python executes any lesson cell.
    This read-only book loads its helper bundle into Pyodide's ordinary FS.
    """
    needle = 'mountDrive:i,loadPyodideOptions:g,contentsManager:o'
    count = text.count(needle)
    return text.replace(needle, 'mountDrive:!1,loadPyodideOptions:g,contentsManager:o'), count


def guard_plotly_resize(text):
    """A static figure can disappear while its async redraw is finishing."""
    needle = 'n.Plotly.Plots.resize(this.node)'
    if text.count(needle) != 1:
        raise RuntimeError('Review changed Plotly MIME renderer resize lifecycle.')
    return text.replace(needle,
        'this.node.isConnected&&this.node.getClientRects().length&&n.Plotly.Plots.resize(this.node)')


def wait_for_cell_kernel(text):
    """Queue early Run/Run-all requests until their real session is attached."""
    replacement = (
        'var __bookKernelWait=Date.now();while(!this.session||!this.session.kernel){'
        'if(Date.now()-__bookKernelWait>120000){'
        'var __bookKernelError=new Error("Python is still starting. Restart the kernel and run again.");'
        'this.setAsIdle();this.clearOnError(__bookKernelError);'
        'this.events.triggerError(__bookKernelError.message);return null;}'
        'yield new Promise(__bookResolve=>setTimeout(__bookResolve,50));}'
    )
    count = 0
    for quote in ('"', '`'):
        needle = ('if(!this.session||!this.session.kernel)return console.warn(' + quote +
                  'Attempting to execute on a cell without an attached kernel' + quote + '),null;')
        count += text.count(needle)
        text = text.replace(needle, replacement)
    return text, count


def reading_outline(text):
    """Start each page's Contents closed, preserving the native reader toggle.

    The theme otherwise reopens Contents on mount and margin-occlusion changes.
    Reset on route changes, so a reader can open it without resize closing it.
    """
    needle = ('[u,d]=(0,z.useState)(!1),{occluded:f}=qt();return'
              '(0,z.useEffect)(()=>{d(!0)},[s]),'
              '(0,z.useEffect)(()=>{s&&d(!f)},[f,s]),')
    replacement = ('[u,d]=(0,z.useState)(!1),'
                   '__bookOutlinePage=typeof window<`u`?window.location.pathname:``;return'
                   '(0,z.useEffect)(()=>{d(!1)},[__bookOutlinePage]),')
    return text.replace(needle, replacement), text.count(needle)


def presentation_prerequisites():
    """Apply generated presentation cells that precede the selected lesson cell.

    These depend on normal lesson setup, so they cannot run during the initial
    browser bootstrap. Their own Python guards safely reapply after a lesson
    reload. Never auto-execute scientific lesson cells.
    """
    return (
        'const __bookPosition=z?.code?.indexOf(L)??-1;'
        'for(const __bookPresentation of (z?.code??[]).slice(1,Math.max(1,__bookPosition))){'
        'if(__bookPresentation.source.startsWith(\'"""Presentation for generated lesson 7.2;\')){'
        'const __bookPresentationResult=await __bookPresentation.execute();'
        'if(!__bookPresentationResult||__bookPresentationResult.error?.length){'
        'L.render(__bookPresentation.model.toJSON());return;}}}'
    )


def initialize_before_individual_cell(text):
    """Run the generated browser prerequisite before any individual code cell.

    The theme's Run-all path already executes notebook cells in order, but its
    Run-cell path bypasses the hidden prerequisite. Wrap that request in a
    kernel-side readiness check without changing the generated notebook code.
    Only successful setup sets the marker; restart clears it. Compile the
    original source intact with Python's top-level-await flag (8192), so string
    literals, imports and global definitions keep their original semantics.
    The client shares only an in-flight request, never a success cache.
    """
    needle = (
        'setTimeout(()=>{if(!L)throw Error(`no cell found on execute`);'
        'L.execute().then(()=>{if(!L)throw Error(`no cell found after execute`);'
        'p.clearCell(N,P,L?.id,`execute`)})},100)'
    )
    replacement = (
        'setTimeout(async()=>{if(!L)throw Error(`no cell found on execute`);'
        'try{const __bookInit=z?.code?.[0];'
        'if(__bookInit&&__bookInit!==L&&'
        '__bookInit.source.startsWith("# Browser pilot initialization")){'
        'if(!__bookInit.__bookPendingInit){'
        'const __bookKey=JSON.stringify("_book_runtime_ready_"+__bookInit.id);'
        'const __bookCode="if not globals().get("+__bookKey+", False):\\n"+'
        '"    await eval(compile("+JSON.stringify(__bookInit.source)+'
        '", \'<browser initialization>\', \'exec\', 8192), globals())\\n"+'
        '"    globals()["+__bookKey+"] = True\\n";'
        '__bookInit.__bookPendingInit=__bookInit.execute(__bookCode);}'
        'let __bookResult;try{__bookResult=await __bookInit.__bookPendingInit;}'
        'finally{delete __bookInit.__bookPendingInit;}'
        'if(!__bookResult||__bookResult.error?.length){'
        'L.render(__bookInit.model.toJSON());return;}}'
        + presentation_prerequisites()
        + 'await L.execute();}catch(__bookError){L.clearOnError(__bookError);}'
        'finally{p.clearCell(N,P,L.id,`execute`);}},100)'
    )
    return text.replace(needle, replacement), text.count(needle)


def preserve_restart_session(text):
    """Keep JupyterLite's session attached when its kernel is replaced.

    The bundled server treats disposal of the old kernel during restart as a
    permanent shutdown and removes its session. The client then cancels the
    new kernel's execution futures. Wait for replacement and rebind the
    session's shutdown listener; genuine shutdown still removes the session.
    """
    patches = {
        'kernel_restart': (
            'async restart(e){const t=this._kernels.get(e);'
            'if(!t)throw Error(`Kernel ${e} does not exist`);'
            'const{id:n,name:i,location:s}=t;'
            'return t.dispose(),this.startNew({id:n,name:i,location:s})}',
            'async restart(e){const t=this._kernels.get(e);'
            'if(!t)throw Error(`Kernel ${e} does not exist`);'
            'const{id:n,name:i,location:s}=t;let __bookRestartDone;'
            't.__bookRestartComplete=new Promise(__bookResolve=>{__bookRestartDone=__bookResolve});'
            'try{t.dispose();return await this.startNew({id:n,name:i,location:s});}'
            'finally{__bookRestartDone();}}',
        ),
        'session_rebind': (
            'async _handleKernelShutdown({kernelId:e,sessionId:t}){'
            'const n=await this._kernels.get(e);'
            'n&&n.disposed.connect((()=>{this.shutdown(t)}))}',
            'async _handleKernelShutdown({kernelId:e,sessionId:t}){'
            'const n=await this._kernels.get(e);'
            'n&&n.disposed.connect((async()=>{'
            'if(n.__bookRestartComplete){await n.__bookRestartComplete;'
            'const __bookNext=await this._kernels.get(e);'
            'if(__bookNext&&__bookNext!==n){'
            'await this._handleKernelShutdown({kernelId:e,sessionId:t});return;}}'
            'await this.shutdown(t);}))}',
        ),
    }
    counts = {}
    for name, (needle, replacement) in patches.items():
        counts[name] = text.count(needle)
        text = text.replace(needle, replacement)
    return text, counts


def disable_contents_serviceworker(text):
    """Use ordinary Pyodide memory files without the optional contents bridge.

    The bridge registers a root-scoped service worker and heartbeat, although
    this static book has no Jupyter contents server. Its optional consumers
    already support an unavailable service; kernel Web Workers are unaffected.
    """
    needle = ('id:"@jupyterlite/server-extension:service-worker",autoStart:!0,'
              'provides:P.u,activate:e=>new B.J')
    replacement = needle.replace('activate:e=>new B.J', 'activate:e=>null')
    return text.replace(needle, replacement), text.count(needle)


def stop_closed_widget_sync(text):
    """Stop deferred view updates from syncing after their widget is closed.

    The bundled widget model deletes its comm but leaves comm_live true. View
    removal then calls save_changes against the deleted channel. Mark the
    model offline before removing its views; missing comms on open models
    retain the upstream error behavior.
    """
    needle = 'if(this._closed=!0,this.comm&&!e&&this.comm.close(),'
    replacement = needle.replace('this._closed=!0,', 'this._closed=!0,this.comm_live=!1,')
    return text.replace(needle, replacement), text.count(needle)


def package_site(html, base_url=''):
    """Adapt derived theme assets only; retain guards against upstream changes."""
    html = Path(html)
    manifest = verify()
    for relative in (*manifest['files'], 'runtime-manifest.json'):
        destination = html / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SITE / relative, destination)
    replacements = {
        PYODIDE_URL: base_url + '/runtime/pyodide/',
        KERNEL_URL: base_url + '/pypi/',
        'https://cdn.jsdelivr.net/npm/': base_url + '/runtime/npm/',
        'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js': base_url + '/runtime/require.min.js',
        'https://cdnjs.cloudflare.com/ajax/libs/': base_url + '/runtime/cdnjs/',
    }
    counts = {url: 0 for url in replacements}
    widget_pins = 0
    memory_filesystems = 0
    plotly_resize_guards = 0
    kernel_attach_waits = 0
    reading_outlines = 0
    individual_cell_initializers = 0
    restart_session_guards = {'kernel_restart': 0, 'session_rebind': 0}
    disabled_contents_workers = 0
    closed_widget_guards = 0
    changes = []
    for path in [*html.glob('*.js'), *(html / '_assets').glob('*.js')]:
        original = path.read_text(encoding='utf-8')
        text = original
        for source, destination in replacements.items():
            counts[source] += text.count(source)
            text = text.replace(source, destination)
        text, pins = pin_widget_loader(text)
        widget_pins += pins
        text, mounts = use_memory_filesystem(text)
        memory_filesystems += mounts
        if path.name.startswith('plotly-renderer-'):
            text = guard_plotly_resize(text)
            plotly_resize_guards += 1
        text, waits = wait_for_cell_kernel(text)
        kernel_attach_waits += waits
        text, outlines = reading_outline(text)
        reading_outlines += outlines
        text, initializers = initialize_before_individual_cell(text)
        individual_cell_initializers += initializers
        text, restart_guards = preserve_restart_session(text)
        for name, count in restart_guards.items():
            restart_session_guards[name] += count
        text, disabled_workers = disable_contents_serviceworker(text)
        disabled_contents_workers += disabled_workers
        text, closed_guards = stop_closed_widget_sync(text)
        closed_widget_guards += closed_guards
        # Thebe's defaults otherwise permit PyPI fallback for the kernel itself.
        needle = f'pipliteUrls:["{base_url}/pypi/all.json"],pipliteWheelUrl:'
        text = text.replace(needle, needle.replace('],pipliteWheelUrl:', '],disablePyPIFallback:!0,pipliteWheelUrl:'))
        if text != original:
            changes.append(dict(file=path.relative_to(html).as_posix(),
                                before=hashlib.sha256(original.encode()).hexdigest(),
                                after=hashlib.sha256(text.encode()).hexdigest()))
            path.write_text(text, encoding='utf-8')
    if not all(counts[url] for url in replacements if 'https://cdnjs.cloudflare.com/ajax/libs/' != url):
        raise RuntimeError('The theme runtime loader changed; review URL adaptation: ' + str(counts))
    if widget_pins < 2:
        raise RuntimeError('Review changed Thebe widget loader before publishing static assets.')
    if memory_filesystems != 1:
        raise RuntimeError('Review changed JupyterLite filesystem configuration.')
    if kernel_attach_waits != 2:
        raise RuntimeError('Review changed Thebe cell execution readiness guards.')
    if reading_outlines != 1:
        raise RuntimeError('Review changed theme outline state before building the book reading layout.')
    if individual_cell_initializers != 1:
        raise RuntimeError('Review changed theme individual-cell initialization before building the book.')
    if any(count != 1 for count in restart_session_guards.values()):
        raise RuntimeError('Review changed JupyterLite kernel/session restart lifecycle.')
    if disabled_contents_workers != 1:
        raise RuntimeError('Review changed JupyterLite optional contents service-worker plugin.')
    if closed_widget_guards != 2:
        raise RuntimeError('Review changed widget model closure lifecycle.')
    meta = '<meta http-equiv="Content-Security-Policy" content="' + escape(content_policy(), quote=True) + '">'
    for path in html.rglob('*.html'):
        text = path.read_text(encoding='utf-8')
        for source, destination in replacements.items():
            text = text.replace(source, destination)
        if '<head>' in text:
            path.write_text(text.replace('<head>', '<head>' + meta, 1), encoding='utf-8')
    report = dict(runtime_bytes=manifest['total_bytes'], packages=len(manifest['packages']),
                  loader_url_replacements=counts, derived_javascript_changes=changes,
                  widget_loader_pins=widget_pins, widget_module_versions={'anywidget': '0.11.0'},
                  plotly_resize_guards=plotly_resize_guards,
                  kernel_attach_waits=kernel_attach_waits,
                  reading_outlines=reading_outlines,
                  individual_cell_initializers=individual_cell_initializers,
                  restart_session_guards=restart_session_guards,
                  disabled_contents_workers=disabled_contents_workers,
                  closed_widget_guards=closed_widget_guards,
                  filesystem='Pyodide memory filesystem; JupyterLite contents mount disabled',
                  remote_requests='blocked by Content-Security-Policy; live validation still required')
    (html.parent.parent / 'same-site-build.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    Builder().prepare()
