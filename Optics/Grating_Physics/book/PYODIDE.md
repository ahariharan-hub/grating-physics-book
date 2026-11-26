# Browser Python edition

The Binder edition continues to use the original `myst.yml` and Python backend.
`pyodide_book.py` builds a separate edition from the same selected
teaching sources and the same nested table of contents. It never commits,
pushes, syncs the public checkout, or deploys a website.

Use the existing `grating-book` Python environment from `Grating_Physics`:

```powershell
python book/pyodide_pilot.py vendor
python book/pyodide_book.py build
python book/pyodide_book.py serve
```

`vendor` prepares the versioned Python/WebAssembly runtime, required Python
packages, widget JavaScript, stylesheets and fonts in an ignored build cache.
It needs network access when an asset is not cached. The book reader loads
these assets from the book's own site. Reader calculations use their computer's
CPU and memory through Pyodide; the hosting service only serves static files.
Thebe's MathJax 2.7.5 renderer and its supporting files are also bundled locally.

The complete preview uses `http://127.0.0.1:3023/`. The three-lesson pilot remains
available through its original commands and uses separate output folders.

## Build and validation

The complete build copies only `public_content.py`'s selected files, executes
fresh native kernels to generate static outputs, and adds initialization cells
to generated notebook copies. Completed native results have content fingerprints
and can be reused. Teaching notebook sources, scientific helper modules, Binder
settings are preserved byte for byte. The public workflow builds and validates
both editions and deploys them together only after all build checks succeed.

The browser edition's generated activation instructions tell readers that
Python runs on their computer. Saved widget containers are converted to ordinary
image and Plotly outputs so the initial figures are visible before activation.
Plotly snapshots come from the final Python figure objects before their native
kernels shut down; widget transport metadata can retain empty initial traces.
Rendered pages and output assets are regenerated each build so stale MIME
exports cannot override a notebook's updated saved figures. Native execution
receipts and the downloaded theme/runtime caches are retained.
Running the unchanged code replaces those snapshots with live widgets.
Individual Run-cell actions first run the generated browser initialization
cell, so readers can start with the lesson's visible setup cell. Initialization
is idempotent within a kernel and repeats after a kernel restart. A failed
initialization stops the requested cell and remains retryable. Scientific
cells still follow the usual notebook order for variables and lesson setup.
The bundled JupyterLite server originally removed its session when disposing
the old kernel during restart. A guarded browser adapter waits for replacement
and reattaches the session's shutdown listener. Ordinary shutdown still removes
the session. Three consecutive restarts with live figure/slider checks passed
in Chrome; deployment-path and published-site checks are separate evidence.
Navigation, references and downloads come from the shared teaching sources.
The browser-only layout moves the collapsible Contents section above the lesson
and keeps the wider desktop article area available for interactive outputs.
Wide widget outputs scroll within their own output area on
smaller screens. These styles are kept in `book/browser-layout.css` and are not
applied to the Binder edition.

All 52 selected notebooks use the reviewed reading layout. Markdown cells have
a serif reading column up to 56rem (896px) wide, close to the navigation. The
theme's direct notebook-code spinner marker distinguishes executable cells
from Markdown, including Markdown containing fenced examples. Code cells and
live widget outputs keep the wider article area and their existing typography.
Text-only notebooks hide the unused compute toolbar. Generated initialization
source bars and repeated chapter titles are hidden while their outputs and
anchors remain available. Contents starts collapsed on each route and retains
its native toggle and the reader's choice during resizing.

The shared lesson prose uses connected introductions, transitions and figure
guidance. Technical derivations, exercises, equations, code, tables, original
link targets and references are retained. The accepted 0.0, 0.2, 5.1 and 7.2
samples form the baseline for the series-wide pass. The application-specific
lithography row was removed from 0.0's notation table during its earlier review.

Validate the full WebAssembly/Python execution separately:

```powershell
python book/pyodide_validate.py
python book/pyodide_book.py check
```

The validator uses the Node runtime already managed by Jupyter Book and a fresh
Pyodide engine per notebook. Fetches to other origins are rejected. This checks
Python/package compatibility and cell execution; it does not simulate the
browser DOM, WebGL, pointer input, or every possible control setting. Browser
rendering and control checks remain a separate validation activity. A single
lesson can be selected using `--notebook <filename.ipynb>`.
Each run freezes its notebook and helper inputs. Individual receipts identify
the code, helper bundle and runtime versions, so targeted reruns update the
full-series summary without accepting results from different inputs.

Generated sources, execution receipts, detailed logs and reports are under
`_build/pyodide-book/`. Native validation is recorded separately from WebAssembly
validation. A build alone does not mean all browser behavior has been tested.

The preview serves only completed, checked snapshots. While a new build is
running it continues serving the previous completed snapshot, rather than the
renderer's intermediate HTML. This avoids transient CDN references during a
rebuild. External resource requests are also blocked by the book's content
security policy.

## Deployment path

For an edition hosted under a subdirectory, pass its path when building:

```powershell
python book/pyodide_book.py build --base-url /grating-physics-book/pyodide
```

Restart the preview server if this deployment path changes. No deployment is
performed by this command.

The publication builder keeps its own profile under `_build/release/browser`,
so it does not change the author's running root-path preview:

```powershell
python book/release.py build --base-url /grating-physics-book --wasm all
```

It validates the Binder edition, builds browser Python under `/pyodide/`,
executes all browser notebooks against the deployed URL prefix, and assembles
`_build/release/site`. Runtime downloads are reused only after checksum
verification; computation receipts require exact matching inputs. Development
options `--wasm smoke` and `--wasm skip` are not the publication checks.
The Pages workflow builds pull requests without deploying. A successful main
branch build deploys the same two-edition artifact. Binder remains at the
existing book URL and browser Python is at the corresponding `/pyodide/` URL.

The versioned runtime is currently Pyodide 0.27.0 with JupyterLite's 0.4.7 kernel.
Its NumPy/SciPy/Matplotlib versions differ from the native build environment.
Execution checks and numerical references are therefore evidence of compatibility,
not a claim that the dependency versions or every floating-point result are identical.
The generated runtime manifest records the actual packages and file checksums.

The derived theme adapters are deliberately guarded against upstream changes.
They pin the local `anywidget` renderer, restore Plotly's positioning styles,
skip Plotly resizes after a static preview is removed, and disable JupyterLite
contents-drive synchronization. The book loads its
Python helpers and runtime assets into Pyodide's ordinary memory filesystem.
Early cell execution requests wait for an attached Python session instead of
returning while the notebook remains busy. A failed attachment has a bounded
two-minute wait and reports an error.

Lesson 7.3's generated initialization also adapts Matplotlib 3.8's older 3D
collection API. It adds already-3D collections through the base Axes API when
autoscaling is disabled. This keeps the lesson's fixed geometry limits and
avoids changing its teaching code or scientific helper module.

Lessons 5.1 and 7.2 use a guarded Plotly frontend adapter from
`book/plotly-widget-layout.js`. It measures the widget after attachment and when
its container changes width, then fits the layout after fonts are ready. It
expands the `conical-spatial` orders and `conical-incidence-wavefronts` figures
to their containing widgets' widths. The incident view reserves space for its
colour-bar title and uses fixed legend-entry widths to prevent label overlap.
The compact `two-order-detector` figure retains its 1000-pixel container limit.
Other conical views keep their existing layouts. Figure data
and Python control callbacks remain in the shared scientific implementation.
All five Plotly lessons serialize NumPy coordinates as ordinary JSON arrays in
their generated browser initialization. This avoids losing coordinate arrays
through binary widget transport. The subclass copies the affected trait
metadata and leaves the original Plotly class and scientific arrays intact.
Non-finite display values are encoded as JSON `null`, including intentional
`NaN` gaps between grating openings. The same serializer runs in temporary
native snapshot kernels, preventing Jupyter message-serialization warnings
from becoming saved book output. Scientific arrays retain their original gaps.
The frontend model cache also skips the serialized `_undefined_` sentinel in
sparse batch updates. Without that guard, changes to other traces can overwrite
retained coordinates before a view renders.

Lesson 7.2 also receives a generated presentation cell after its module reloads.
It reuses the original controls and callbacks, reduces the figure height from
950 to 700 pixels, keeps the detector square, and collapses detailed wavevector
calculations and reading notes. Fringe spacing, visibility and detector motion
remain visible. Native snapshots use the same presentation cell; the authoring
notebook and shared helper modules are unchanged.
The compact result row also carries the selected orders' colour key instead of
the long Plotly legend. Explicit detector-axis and intensity labels avoid
automatic title measurements in the widget loader's hidden frame.

Lesson 7.1's generated initialization also adapts the optional phase-geometry
explanation. Its Markdown-only Output widgets are replaced with HTMLMath
widgets, so paragraphs, emphasis and TeX use the supported renderer and the
bundled MathJax assets. The diagram, lazy drawing callbacks and explanation
source remain in the shared helper module; replaced widgets are closed.
