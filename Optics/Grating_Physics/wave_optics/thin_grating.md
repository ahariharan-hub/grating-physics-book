# Thin phase grating: numerical gate for 0_5C.1

This report documents the retained **lossless dielectric/phase-mask** backend
and the earlier Notebook 0_5 comparison. That teaching material now lives in
[the standalone supplement](../supplement/Binary_Amplitude_and_Dielectric_Phase_Masks.ipynb).
Main Notebooks 0_5–0_7 instead study physical absorbers and binary amplitude masks;
see [its acceptance report](absorbing_grating.md). Historical section numbers
and test counts below describe the earlier implementation.

Implemented on baseline `f1b17ad9846ce0d2c53476d0a1df681f21b0f81c`.
The milestone attachment did not separately define C.1 and C.2. C.1 is the
analytical backend and numerical validation; C.2 is the notebook extension
and comparison views. The C.1 checks passed before starting C.2.

## Conventions and API

`thin_grating.py` reuses `ScalarGratingConfiguration`. It requires equal
exterior indices, real lossless materials and unit normal illumination at
z=0. It leaves `scalar_grating.py`, `scalar_grating_view.py` and the ASM
backend unchanged.

```python
from wave_optics.thin_grating import ThinPhaseGrating, compare_grating_models

# solution is an existing solve_scalar_grating(...) result.
thin = ThinPhaseGrating(solution.configuration)
plane = compare_grating_models(solution, -.6, thin_model=thin)
u_thin = thin.reconstruct(x_samples, plane.z_ref)[0]
u_hbvp = solution.reconstruct(x_samples, plane.z_ref)[0]
```

`thin_mask_coefficients(config, orders=None)` uses the analytical centred
rectangle coefficients; `thin_mask_exit_field(config, x)` evaluates the
exact discontinuous mask for independent comparisons. Neither uses an FFT.
The absolute exit factors are `exp(i*k0*n_ridge*h)` and
`exp(i*k0*n_groove*h)`. Dropping the common exterior propagation phase would
misreference the comparison. Negative-z outgoing propagation is
`C(-h-D) = C(-h)*exp(i*beta*D)` for D >= 0. This also gives evanescent decay.

`ThinPhaseGrating` exposes `harmonics`, `at_reference_plane`, `reconstruct`,
order classifications and flux-weighted `transmission_efficiency`.
`GratingComparisonPlane` contains both coefficient arrays, their complex
difference (HBVP minus thin), masked phases and phase differences,
efficiencies, and all-order power totals. Returned stored arrays are
read-only. Phases are NaN below 1e-6 of unit incident amplitude; a phase
difference requires both amplitudes to exceed that threshold. Comparison
does not solve a BVP, align phases or normalize either model's power.

## Numerical acceptance results

All 15 C.1 tests passed on 2026-10-18. The 46 existing scalar HBVP, grating
view and ASM backend/view/lesson checks also passed before the extension.
The attachment supplies acceptance topics but no numerical tolerances;
the following tolerances are explicit in the tests.

| Independent check | Acceptance | Observed |
| --- | --- | --- |
| Analytical coefficients vs piecewise quadrature, three duties and signed orders | maximum absolute error < 2e-14 | 3.26e-16 |
| Fully uniform medium: thin and HBVP vs exact plane wave | maximum absolute error < 3e-14 | thin/HBVP difference 0 at tested samples |
| Retained field propagated by existing FFT ASM | maximum absolute error < 8e-15 | 1.16e-15 |
| Common-plane complex difference propagation | maximum absolute error < 2e-15 | passed |
| Evanescent amplitude after one decay length | absolute error < 3e-16 | passed |
| Reconstructed discontinuous mask, M=512, >0.08d from jumps | maximum absolute error < 0.005 | 0.001582 |
| Lossless HBVP total scalar flux | unity to 12 decimal places | 1.0000000000000004 |

Additional tests cover centred-origin symmetry and translation phase,
absolute reference phase, empty/full ridge duty, a mismatched uniform slab,
propagating magnitude invariance, grazing constant fields and zero flux,
integrated field flux vs modal efficiencies, phase masking, unit rescaling,
input validation and reuse without a scattering solve.

The high-M mask test assesses Fourier convergence without increasing the
HBVP truncation or introducing a discontinuity FFT. At f=0.37, the same
away-from-edge error is 0.03168, 0.007808 and 0.001582 for M=32,128,512.
At a jump a Fourier series tends to the two-sided average. The notebook's
M=20 reconstruction is a retained-harmonic field, not an exact step mask.

## Physical discrepancies to retain in the lesson

For the notebook geometry (M=20, wavelength 0.633, d=1.5, h=0.3, f=0.5,
ridge index 1.5, other indices 1), the HBVP predicts
R=0.0694872031 and T=0.9305127969. The thin phase mask predicts outgoing
propagating flux T=0.8782735043 with reflection omitted by assumption.
Its deficit is **not absorption or calculated reflection**. Unit local
transmission amplitude does not impose the correct normal-flux balance
when appreciable diffraction angles and evanescent components are present.
Increasing M beyond all propagating orders does not repair this deficit.

For f=0.5 the thin mask's nonzero even orders vanish, whereas the coupled
HBVP gives C_2(-h)=0.08340184+0.15582039i. Its phase difference relative to
the negligible thin coefficient must therefore be omitted. A uniform
layer with an index different from its surroundings also need not match
the thin model: the HBVP retains Fresnel reflection and internal round trips.
Only a fully matched uniform medium is their exact common plane-wave limit.

These are expected differences between physical models, not detected
defects in the existing solver. No solver changes were required.

Run the numerical gate with:

```text
python -m unittest supplement.tests.test_thin_grating supplement.tests.test_scalar_grating supplement.tests.test_scalar_grating_view supplement.tests.test_scalar_propagation supplement.tests.test_scalar_propagation_view supplement.tests.test_scalar_propagation_lesson
```

## C.2 integration verification

After passing C.1, section 8 was added with `GratingComparisonExplorer`.
All **66 focused tests** passed together: 15 new numerical tests, 5 new
comparison-view tests and 46 existing scalar HBVP/visualization/ASM checks.
The nine notebook code cells executed in a fresh kernel; additional checks
switched both tabs, moved the comparison plane and closed the widgets.
No executed outputs or widget state were saved.

Sections 1–7 were compared against HEAD and are unchanged. The existing
scalar solver, scalar grating visualization module and ASM backend are
unchanged. Field and coefficient plots were visually inspected, all field
quantities rendered without missing-glyph warnings, all three Markdown
tables exported with consistent column counts, and all 134 inline/display
math expressions passed a MathJax 3.2.2 render check. These checks do not
constitute inspection of a live Jupyter frontend.

The vector bridge follows the user's requested scope: no Maxwell solver,
angle controls or polarization simulation. It explicitly distinguishes
general vector matching from the special nonmagnetic, y-invariant TE case
represented by the current scalar equation. The cited Pistor sections were
checked against the official Berkeley thesis PDF.

## Binary amplitude/phase amendment

The ideal-mask introduction now precedes the dielectric/HBVP comparison.
`binary_transmission.py` provides one analytical centred-rectangle coefficient
formula and one exact local-field evaluator for either pair of complex
transmission levels. The original thin phase-mask API delegates to these
helpers, retaining the absolute exit phase and its propagation conventions.
The amplitude model uses transparent/opaque levels (1,0); the dielectric
phase model uses exp(i*k0*n_r*h) and exp(i*k0*n_g*h).

`BinaryTransmissionExplorer` shows exact transmission magnitude and phase,
coefficient magnitude and phase, and a selected coefficient in the complex
plane. Readers can select either model or both, retain -M through M, and
optionally display the finite Fourier sum. Exact local curves do not change
with truncation. Zero and negligible amplitudes have no displayed phase.
Only the dielectric phase prescription enters the existing HBVP comparison;
no opaque, absorbing or conducting material was added to the solver.

Nine added analytical tests cover both models' independent period-average
integrals, centred edges, signed orders, phase-only diffraction, uniform and
endpoint limits, absolute dielectric phase, common-phase rotation and
Parseval normalization. The infinite-series mean square is f for the
amplitude mask and one for the phase mask; finite-sum convergence is checked
against an analytic tail bound. An independent sampled finite-field check
verifies the same Fourier normalization. No mean square is identified with
longitudinal-flux efficiency.

Five added view tests cover undefined phases, truncation without changes to
the exact mask or BVP, linked Argand selection, visibility, cache reuse,
reset independence and complete widget cleanup. All **80 focused tests**
passed together after the amendment, including the previous 66 checks.
The **ten notebook code cells** execute in a fresh kernel, and the phase
mask coefficients in the new explorer match the later thin/HBVP comparison
exactly. All **154 mathematical expressions** pass KaTeX 0.16.22 rendering;
Markdown table structure is valid. Plot variants were visually inspected,
with no missing-glyph warnings and no clipped header after repeated updates.
Notebook outputs remain unsaved. These checks do not claim live Jupyter
frontend inspection.

The existing scalar solver, ASM backend, HBVP visualization modules,
sections 1–7, conceptual vector section and Pistor reference are preserved.
Run the added checks with:

```text
python -m unittest supplement.tests.test_binary_transmission supplement.tests.test_binary_transmission_view
```
