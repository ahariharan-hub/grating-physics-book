# Pilot validation

The book navigation contains only notebooks 0.0, 0.1 and 0.2. Notebook 0.0 has no
code cells; all eight code cells across 0.1 and 0.2 have passed fresh local execution.
`requirements-tested.txt` and `tested-environment.json` record the tested
Python environment. These are top-level pins, not a complete Conda lock.

Eight native MyST dropdowns contain parsed mathematical expressions. The
build validation checks these and verifies the three-page scope. A separate
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

## Public execution check: 2026-11-20

GitHub Pages deployment and the Ubuntu execution/build workflow passed. A real
Binder session launched in 120.9 seconds on its first uncached build, exposed the
correct book subdirectory, executed all four cells in notebook 0.1, and sent 29
widget comm frames after changing `Trial q/k` to 1.25. The tested source commit
was `abe918f377d21f41ba2bc7edecc9ad7d5b0d2db2`; the subsequent download/report
edits preserve the physics code and Binder environment. See
`public-validation.json`. These are backend/protocol checks; rendered browser
slider interaction remains unverified.

Both downloadable ZIP bundles are directly linked from the pilot and checked
for archive integrity during the build validation.

## Notebook 0.2 preparation

Notebook 0.2 is included after 0.1. Its four code cells execute without errors,
and its three Answer dropdowns use native MyST. The radial Helmholtz residual,
Bessel-cone residual and propagation amplitude change are below 2e-16 in the
recorded calculation. All eight solution families produce PNG figures; the
plane/Bessel/Gaussian beam views, Gaussian waist/axial controls and reset pass
backend callback checks. See `solutions-validation.json`. These checks do not
verify rendered browser slider interaction.
