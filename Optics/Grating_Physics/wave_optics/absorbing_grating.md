# Binary amplitude mask versus physical scalar absorber

Notebook 0_5 determines the physical absorbing field; 0_6 decomposes and
propagates it; 0_7 compares it with the ideal amplitude mask and explores
thickness. These preserve former sections 1–3, 4–7 and 8–9 respectively.
Historical validation entries below refer to the original combined lesson.
The hierarchy in 0_7 is A (ideal amplitude),
B (physical absorber in air), and C (future chrome-on-glass requiring substrate
modeling). No substrate or vector solver is implemented.

The validated lossless backend, phase-mask functions and tests remain available.
The former phase-grating teaching material is preserved in
[supplement/Binary_Amplitude_and_Dielectric_Phase_Masks.ipynb](../supplement/Binary_Amplitude_and_Dielectric_Phase_Masks.ipynb),
outside the main narrative. The current correction was made from clean HEAD
`9237c9f43bc5e8aa481b53d0e31e40e54741585d`; it does not modify either validated
scattering solver or the ideal-amplitude coefficient formula.

## Geometry and phase reference

`AbsorbingGratingConfiguration` defines a free-standing layer in air, with period
`period`, centred **clear** opening `clear_width`, thickness `thickness`, and
absorber index `n_absorber + 1j*kappa`. The layer occupies `-h < z < 0`.
`opening_shift` translates the clear opening. This is intentionally different
from the old configuration, whose centred region is the dielectric ridge.

Backend defaults (micrometres): wavelength 0.633, period 1.5, clear width 0.75, thickness
0.3; absorber index 1.5 + 0.8i; M=24 (49 harmonics). These are illustrative
optical-scale parameters, not measured or fitted chrome data. Both exterior
half-spaces remain lossless air. There is no substrate or absorbing incoming
half-space. The scalar model matches U and its normal derivative, not general
polarization-dependent Maxwell boundary data.

Notebooks 0_5–0_7 compute with `M=24` and pass `display_order_limit=7` to the
explorer interfaces. Coefficient plots and order selectors show -7 through +7.
The shared solution, both field reconstructions, FFT agreement checks and power
totals still use all 49 harmonics, including every thickness case. The display
range is independent of numerical truncation; no solver or physics is changed.

The time convention is exp(-i omega t). Unit illumination at z=0 is
U_i=exp(-i k0 z), directed toward -z. The ideal aperture is placed at the exit:

```text
D = clear_width / period
t_m = D sinc(m D) exp(-i 2 pi m opening_shift / period)
C_ideal,m(-h) = exp(+i k0 h) t_m
C_m(-h-distance) = C_m(-h) exp(+i beta_m distance)
beta_m = sqrt(k0² - (2 pi m / period)²), outgoing/decaying branch
```

The exp(+i k0 h) factor is required by the shared entrance-plane illumination.
There is no fitted phase, coefficient rescaling, or power normalization.
`ideal_amplitude_exit_field` returns the exact binary field times this phase;
`IdealAmplitudeGrating.reconstruct` returns its retained Fourier sum. The latter
rings at discontinuities and is not the exact local transmission. Outside the
exit plane the ideal field diffracts. Phase of an exact zero is undefined;
small Fourier-reconstruction ripples can still have phase.

## Complex-layer solver and independent absorption

`solve_absorbing_grating` reuses the original bounded modal basis, equilibrated
boundary solve, reconstruction, and exterior flux factors. The internal shared
engine uses `scipy.linalg.eig` for a complex non-Hermitian layer, checking both
eigenvector conditioning and residual. It retains `eigh` for the original real
lossless path. Passive eigenvalues are checked before selecting decaying roots;
small negative imaginary roundoff is clamped at the real axis.

Downward modal waves are anchored at the entrance and upward waves at the
exit, avoiding exponentially growing transfer factors. Near zero modal
wavenumber the constant/linear limiting basis is retained. The solver checks
boundary residuals and rejects a numerically unresolved eigenbasis or singular
matching problem. As before, exact exterior grazing thresholds can be singular
in special uniform problems; change parameters slightly and assess convergence.

For unit incident amplitude, scalar efficiency uses Re(beta_m)/k0 times the
squared coefficient. Grazing and evanescent orders have zero outgoing normal
flux. `power_balance` retains the original meaning R+T; `energy_balance` is
R+T+A. No negative absorption or flux defect is silently clipped.

Absorption is calculated independently from

```text
A = (k0 / period) integral_-h^0 integral_period Im(epsilon) |U|² dx dz.
```

The transverse integral is exact for the retained field, using the material
convolution's anti-Hermitian part `(E-E†)/(2i)`; elementwise imaginary parts
would be wrong for shifted openings. Adaptive depth quadrature resolves the
surface-loss regions. R and T come from exterior fields, not this integral.
The backend checks the resulting balance to max(1e-7, 10 times quadrature error).
Tests use tighter tolerances for the validated cases. Conservation alone does
not prove convergence to the continuous, infinite-order solution.

## Acceptance results

The absorbing backend and original lossless suite passed **before** comparison
UI implementation (29 tests at that gate). Further tests subsequently cover
the full thickness range, ideal coefficients, common-plane propagation and UI.

At the backend default configuration (M=24):

| Quantity | Result |
| --- | ---: |
| R | 0.0640879242725684 |
| T | 0.4251501703262573 |
| Integrated A | 0.5107619054011746 |
| R+T+A | 1.0000000000000002 |
| Absorption quadrature error estimate | 7.82e-13 |
| Equilibrated boundary condition number | 5.48 |
| Eigenvector condition number | 5.14 |
| Relative boundary residual | 1.16e-17 |
| Relative eigensystem residual | 5.41e-16 |

Checks include:

- Uniform absorbing-medium forward-wave phase/decay and local flux loss. This
  is a homogeneous modal benchmark, not scattering from an absorbing exterior.
- An independent complex-index Fabry–Perot slab calculation: complex r, t and
  internal fields agree within 3e-14 for kappa 0.001, 0.8, 4 and thickness
  0.02, 0.3, 3. Integrated absorption agrees within 1e-10.
- Zero absorption matches the old lossless solver with the same geometry
  (air as its centred ridge and dielectric as its groove); kappa=1e-9 approaches
  that result. All-air illumination has the correct absolute propagation phase.
- Horizontal interface continuity, the coupled Helmholtz equation, shift
  covariance, depth-dependent flux loss, and R+T+A conservation.
- Transmitted complex coefficients for |m|<=4, compared with M=72:

| Retained half-order M | Maximum complex difference, default h=0.3 |
| --- | ---: |
| 8 | 7.36e-4 |
| 16 | 8.83e-5 |
| 24 | 2.55e-5 |
| 36 | 6.88e-6 |
| 48 | 2.34e-6 |

Across all seven interactive thicknesses, the M=24 versus M=72 difference for
|m|<=4 stays below 4.02e-5 (test tolerance 5e-5). These are convergence estimates,
not rigorous error bounds for every retained high order or near-edge field.
For stronger contrast kappa=4, the M=24 and M=48 differences against M=96 are
1.12e-4 and 1.35e-5. Raising contrast therefore requires another order check.

A stress case kappa=8, h=30 remains finite even when a naive growing transfer
factor would overflow (exponent >709). Its independent energy balance agrees
within 1e-10. This is a stability test, not an accuracy claim at that truncation.
No convergence or conditioning failure occurred in the validated ranges.

## Analytical coefficients and ideal normalization

Tests independently evaluate the rectangular-aperture integral,
D sinc(mD) exp(-i 2 pi m x0/p), then compare a 65,536-point period projection
against the analytic coefficients
for orders -12 through 12, centred/shifted apertures and open-fraction limits.
The tolerance 3/N accommodates pixel/edge sampling. The Python analytical
agreement is checked to 2e-15. Parseval approaches sum|t_m|²=D;
this is mean squared boundary amplitude, not automatically transmitted normal
flux. The ideal model predicts no reflection or absorbed-power allocation.

## Interactive experiment and thickness

`AbsorbingComparisonData` supports seven bounded thickness cases:
0.03, 0.08, 0.15, 0.30, 0.50, 0.75 and 1.00 micrometres.
Notebook 0_7 supplies `solution=grating_solution`, reusing its setup's initial
solution and solving only the six other thicknesses. The parameters match 0_5 and 0_6. The
existing `ScalarGratingExplorer` displays the complex material through separate
n/kappa selectors and reports physical R/T/A in its efficiency view. Section 3 of 0_5
names the Scalar Fourier Modal Method (FMM) and explains its relationship to
RCWA while retaining scalar interface assumptions.

`AbsorbingGratingExplorer` provides two principal tabs:

- A: material geometry, calculated physical intensity, exact ideal transmission,
  a selectable real/imaginary/amplitude/phase/intensity profile, wrapped phase,
  and a local-field Argand diagram coloured by transverse position. One x slider
  highlights the same complex sample on the spatial and Argand plots. The field
  plots show HBVP alone by default; **Compare with ideal mask** overlays the
  ideal field on the profile, phase and Argand plots. Reset returns to HBVP
  alone. The exact ideal-transmission panel remains visible. R/T/A
  accounting remains visible in the shared readout.
- B: magnitudes, phases, selected coefficient Argand vectors and difference
  arrow, complex differences, flux-weighted efficiencies, relative phases, and
  a compact table. Green open markers overlay coefficients extracted by FFT
  from the same sampled HBVP complex field; the readout gives the discrepancy
  against direct modal coefficients.

`field_fourier_coefficients` uses N uniform ascending samples over exactly one
period, excluding the repeated endpoint. With x_j=x_start+j*p/N, it computes
FFT(U)[m % N]/N times exp(-i*2*pi*m*x_start/p), preserving negative-order bins,
normalization and absolute phase. The view uses N=max(512, 4*(2*M+1)), above
the retained-order Nyquist requirement. The displayed absolute coefficient
tolerance is 1e-12 for unit incident amplitude; negligible phases remain masked
at amplitude 1e-6. This is a representation-consistency check of the truncated
solution, not independent scattering validation. It transforms complex U,
never |U| or |U|². The field samples and FFT are cached until the reference
plane or thickness changes; representation, x and order controls reuse them.

Its expandable thickness figure plots selected coefficient magnitudes/phases,
physical R/T/A, and |Delta C|. Each case uses z_ref=-h-distance, at fixed distance
below its own exit; the comparison is not at one fixed laboratory z across h.
All absolute phases retain the entrance reference. Phase points are not joined
across wraps. Observation controls do not re-solve the BVP. Figures, artists,
sampled material/field maps and Fourier bases are reused; hidden figures are
not rendered. `close()` releases callbacks, widgets, figures and cached arrays.

For m=1 at the exit, |Delta C| decreases from 0.22735 at h=0.03 to 0.08099 at
h=0.30, then increases to 0.19968 at h=1.00. Greater opacity alone does not
remove reflection, edge coupling or phase changes. General material-parameter
sliders, an absorbing substrate, and full Maxwell TE/TM scattering are outside
this milestone. Pistor's technical reference and the existing vector bridge
remain in the notebook.

## Reproduce checks

```text
python -m unittest supplement.tests.test_absorbing_grating supplement.tests.test_scalar_grating
python -m unittest supplement.tests.test_amplitude_grating supplement.tests.test_absorbing_grating_view
python -m unittest supplement.tests.test_absorbing_lesson supplement.tests.test_scalar_grating_view
```

The broader scalar propagation, dielectric mask, binary mask and existing
visualization tests should also be run after changes. Execute Notebook 0_5 in a
fresh kernel; saved notebook outputs and widget image state remain empty.

### Corrective milestone validation (2026-10-18)

All **9 code cells** of the revised main notebook executed in a fresh kernel.
For that run, the lossless `solve_scalar_grating` entry point was replaced by
a function that raises immediately: no active cell or explorer fell back to
the former model. Additional checks verified that all five early explorers
share the exact absorbing solution, that section 8 reuses that same solution
at its initial thickness, and that both views report matching coefficients
at a common plane. Material n/kappa, field/harmonic data, order/plane/thickness
controls, undefined-phase handling, reset and disposal were exercised.

The preserved supplementary lesson executed all **3 code cells** in its own
fresh kernel from the `supplement` directory. Both notebooks validate with
nbformat, contain no saved outputs/widget state, and pass KaTeX 0.16.22 parsing:
**143 expressions** in the main notebook and **60** in the supplement.
The main notebook's three Markdown tables render with consistent columns.
Material n/kappa maps, solved fields, harmonics, reconstruction, reference-plane
plots, the central field/coefficient comparison, and the thickness sweep were
rendered to PNG and visually inspected. These are kernel, renderer and widget
data checks; a live VS Code/Jupyter frontend was not exercised.

The focused absorbing-lesson, existing scalar-view and amplitude-model run
passed **22 tests**, including six new regressions for the corrected narrative
and shared absorbing-solution workflow. Neither scattering solver was changed.

Full repository discovery ran **301 tests in 587.56 seconds: 298 passed, one
failed and two errored**. These are the same three pre-existing Notebook 1
expectation failures described below: missing builder cells and setup cell ID
`288c5c98`. The Notebook 1 file and its tests are unchanged from HEAD.
All absorbing-solver, ideal-amplitude, lossless dielectric, phase-mask,
propagation and related explorer regressions passed. No new numerical or
physical inconsistency was found in the tested ranges. `git diff --check`
passed, temporary validation artifacts were removed, and no commit or push
was performed.

### Follow-up: recovery from an older cached solver

A running notebook kernel could retain `scalar_grating` from before
`_solve_layer` was introduced. Importing `absorbing_grating` before refreshing
that dependency then raised `ImportError`, even though the helper existed on
disk. The setup cell now imports and reloads modules sequentially in dependency
order, beginning with the scalar solver and refreshing dependent class
references before constructing new solutions and explorers.

A subprocess regression reproduced the reported failure before the fix and
passed afterward, including a second setup run with cached dependents and live
explorers. All **7 focused lesson tests** passed, and all **9 notebook code
cells** were executed again in a fresh kernel, with shared-class, field and
coefficient-widget checks. Outputs were not saved. The numerical solvers were
not changed; the full-suite result above predates this focused reload fix.

### Complex-field visualization and Fourier-decomposition amendment

Section 8 now links selectable Re U, Im U, amplitude and wrapped phase to a
local-field Argand diagram, with one transverse-position selector. The existing
reference-plane and thickness controls also drive the coefficient view, where
FFT-extracted HBVP coefficients overlay direct modal coefficients. The ideal
amplitude-mask comparison retains its analytical coefficients and absolute
entrance/output-plane phase convention. Sections 1–7 and both scattering
backends are unchanged by this amendment.

Modified files:

- `0_5_From_Helmholtz_BVP_to_Diffracted_Field.ipynb`: exploration prompts and
  concise complex-field/Fourier explanation, including FFT normalization.
- `wave_optics/amplitude_grating.py`: sampled-complex-field Fourier extraction.
- `wave_optics/absorbing_grating_view.py`: shared controls, field Argand plot,
  linked sample markers, FFT coefficient overlays and discrepancy readout.
- `supplement/tests/test_amplitude_grating.py` and
  `supplement/tests/test_absorbing_grating_view.py`: numerical and UI checks.
- `wave_optics/README.md` and this report: behavior and validation documentation.

Validation:

- All **9 notebook code cells** executed in a fresh kernel. Further checks
  exercised representation/x/plane/thickness/order controls, reset, disposal,
  shared solution identity and unchanged R+T+A accounting.
- Across seven thicknesses and three output distances (0, 0.23 and 1 µm), the
  largest complex coefficient difference was **1.388e-16**, below the stated
  **1e-12 absolute tolerance** for unit incident field amplitude.
- **12 amplitude-model/comparison-view tests** passed. These cover odd/even
  sample counts, grid-origin phase, signed and unordered diffraction indices,
  FFT normalization, invalid sample ordering/repeated endpoints, reference-plane
  phase, linked spatial/Argand markers and reuse without another solve or FFT
  when changing representation or selected position/order.
- **49 existing lesson, scalar-view, absorbing-solver and lossless-solver
  regression tests** passed, including cached-kernel setup recovery.
- All **165 math expressions** parsed with KaTeX; three Markdown tables and
  notebook format/empty saved outputs were checked. Amplitude, imaginary and
  phase views plus the coefficient comparison were rendered and inspected.

No scattering model, solver or material physics was added. This agreement
checks the representation of the existing truncated field, not independent
scattering accuracy. No commit or push was performed.

### Original additive implementation at commit 9237c9f

The then-current notebook was executed in a fresh Python kernel: all 11 code cells
and additional thickness/plane/order/profile/reset/cleanup checks passed.
All 218 notebook math expressions passed KaTeX 0.16.22 parsing; all four Markdown
tables passed structural rendering checks. The Matplotlib field, coefficient
and thickness figures were visually inspected. This does not substitute for
checking a particular VS Code/Jupyter frontend; no live frontend was available.

That baseline full repository discovery ran 295 tests: **292 passed; one failure and two errors
are pre-existing Notebook 1 test expectations** in `test_wave_optics.py`.
The tests seek the removed `wave_1d`, `wave_2d`, `wave_3d` builder cells and setup
cell ID `288c5c98` in `1_Plane_Waves_and_Wavevectors.ipynb`. Both the notebook and
that test file are unchanged from HEAD, where those cells are already absent.
The same three failures reproduce in isolation. They require a separate update
to the older Notebook 1 tests; this milestone does not alter that lesson.
All 98 tests covering the new absorber, ideal/binary/dielectric models, scalar
grating, propagation, and their associated lesson views passed in the full run.
