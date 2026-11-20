# Pilot validation

The book navigation contains only notebooks 0.0 and 0.1. Notebook 0.0 has no
code cells; all four code cells in 0.1 have passed fresh local execution.
`requirements-tested.txt` and `tested-environment.json` record the tested
Python environment. These are top-level pins, not a complete Conda lock.

Five native MyST dropdowns contain parsed mathematical expressions. The
build validation checks these and verifies the two-page scope. A separate
local kernel protocol check has verified authenticated execution and widget
trait updates; rendered browser interaction still needs an end-to-end check.

The deployment workflow repeats notebook execution, HTML generation and
validation on Ubuntu. It builds with `BASE_URL=/grating-physics-book` and
publishes `_build/public-pilot/_build/html/`. Local tokens and server
configuration are generated only under ignored `_build/`.

Public in-page execution points to this repository's `main` branch on Binder.
The root `.binder/` configuration supplies the scientific/widget environment
and makes the book subdirectory the Jupyter server root. A public Binder
launch and real browser slider updates must be tested independently of the
static build. MyST currently labels in-page execution as beta.

The upstream book theme currently emits a nonfatal schema warning for a
`multiple` option. It does not prevent the tested build.
