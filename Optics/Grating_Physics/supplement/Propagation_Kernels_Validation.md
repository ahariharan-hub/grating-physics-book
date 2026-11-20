# 8.3: kernel theory and interactive comparison

## Scope and conventions

The notebook now introduces the propagation weight before its live comparison:
kernel meaning, full RS I and equivalent ASM multiplier, Fresnel approximation,
then the additional usual paraxial Fraunhofer approximation. It does not
prescribe a thin-mask field, solve material scattering, calculate an exit field
or propagate an order sum. The previous field-comparison helper and its
numerical tests remain available for later use.

Section 7 reserves grating-field applications for the next notebook:
connect U_exit, boundary contributions and C_m, then propagate the same
field. Thin-mask or FMM/RCWA exit-field formation remains separate from
propagation. The introduction now points to this closing handoff.

Section 1 now introduces the integral kernel as a continuous matrix, then
connects spatial convolution to a Fourier multiplier. A compact table states
what each representation weights and its units in this plane-to-plane problem.
The diagonal Fourier representation is explicitly restricted to transverse
translation invariance; general optical systems can couple spatial frequencies.
That preliminary-theory edit left the existing code cells and numerical
and interactive backends unchanged. Notebook schema, the primer's table and
four displayed-math blocks, local links and code compilation were checked.

Coordinates are (x,y,s), with exp(-i omega t) and positive s towards -z from
the reference plane. The medium is homogeneous and lossless; lambda is the
medium wavelength. Q=(xi,eta,0) is a boundary sample and P=(x,y,s) an observation
point. The plotted y=eta=0 cut uses the full 3D kernels and inverse-area units.

## Mathematics and display

`wave_optics/propagation_kernels.py` evaluates:

- Full K = -2 partial_s g, with g=exp(ikR)/(4*pi*R), retaining 1-ikR.
- Fresnel: exp(iks)/(i*lambda*s) times exp[ik*rho^2/(2s)].
- Fraunhofer: the Fresnel kernel divided by exp[ik*(xi^2+eta^2)/(2s)].

The Markdown identifies both the path and amplitude approximations needed
for Fresnel, then the extra omitted boundary phase needed for Fraunhofer.
Sections 3 and 4 now show the intervening algebra: exact factorization of
the full kernel, separate path/prefactor approximations, the longitudinal
wavevector expansion yielding H_F, and separation of the three Fresnel
phase factors. Inserting the Fraunhofer kernel into the generic boundary
integral explicitly identifies the scaled input Fourier transform. No
boundary field is selected or evaluated in that derivation. The edit changes
only those two Markdown cells. The later geometric presentation described
below adds linked live views without altering the numerical kernels.
The displayed full-kernel factorization, separated Fresnel phases, omitted
Fraunhofer phase and H_F expression were checked numerically against the
existing backends at three parameter sets. Notebook schema, Markdown/math
blocks, local links and code compilation also passed.
Full RS I / ASM equivalence uses the transverse Fourier transform, inverse
measure dkx*dky/(2*pi)^2 and the outgoing/decaying branch of kappa. The usual
Fraunhofer expression is identified as an integral kernel and scaled Fourier
mapping, rather than a third same-coordinate convolution multiplier.

## Geometry beside the derivations

`propagation_geometry.py` now supplies compact live views in Sections 1–4:

Sections 2–4 follow the same reading order: derivation, live panel, then an
explicit spatial-kernel magnitude. The Fraunhofer magnitude equals the
Fresnel magnitude, 1/(lambda*s), since the additional discarded factor has
unit magnitude. This layout edit preserves all code cells, saved outputs and
metadata; notebook schema and code compilation were checked.

- Section 1: projected reference and observation planes, both transverse axes,
  selected Q and P, and the right triangle R^2=s^2+rho^2.
- Section 2: the same planes with full-kernel phase fronts and contribution
  strength. A single Markdown equation immediately below the live panel gives
  the exact magnitude and its spreading/directional/near-field factors.
  The selected numerical values remain in one compact widget readout.
  ASM is identified as the equivalent plane-wave representation.
- Section 3: unchanged physical geometry beside two linked comparisons:
  exact extra path R-s versus quadratic rho^2/(2s) above, and full/Fresnel
  kernel magnitudes below. The magnitude axis stays fixed at [0,1.15] in
  lambda^2 units, so both transverse weighting and distance decay remain
  visible. Markers track the selected Q in both plots. Short readout lines
  give selected magnitudes, extra paths and the complex prefactor.
  A matching magnitude line below the panel shows 1/(lambda*R) becoming
  1/(lambda*s), cos(theta) becoming one, and the near-field factor becoming
  one; the text explicitly retains quadratic phase.
- Section 4: unchanged planes, observation direction from the reference
  origin, and relative Fresnel/Fraunhofer phases. The shaded difference is
  exactly k*xi^2/(2s) in the displayed eta=0 section.
  The derivation now connects local wavefront flattening to these terms:
  the cross term carries linear tilt, while the boundary quadratic term
  carries curvature. Exploration links moving P to changing the tilt and
  identifies the green phase line as tangent to the Fresnel parabola at
  xi=0. This is a kernel phase across Q for fixed observation point P.
  The panel note uses the same interpretation. Numerical formulas and
  controls are unchanged; notebook schema and code compilation passed.

The main explorer reuses the same plane renderer. Window outlines are crops
of infinite planes, not aperture edges; their extent follows the camera.
Selected points and phase-front arcs lie in the y=eta=0 section. The plotted
right triangle labels lengths, not a bent propagation trajectory. Choosing
an approximation never moves the physical planes or changes their separation.

One non-widget GeometryState links s, x and xi across all five panels; local
controls remain beside each explanation. Every panel owns its own widgets
and removes only its own synchronization callbacks when closed. Rerunning a
cell adopts the existing geometry and leaves sibling panels active. The
notebook now has 26 cells, including eight code cells. The magnitude annotations
preserve existing code cells, execution outputs and notebook metadata.

The explorer varies s/lambda=1..100 and P and Q transverse coordinates within
[-3,3] wavelengths. The profile interval is a comparison window, not an
aperture. Every control changes geometry, selected weights or displayed
kernel values. Distance and P change the profiles; Q moves the selected
sample, profile guides and complex arrows.

The default Patch response view interprets the same kernels as the contribution
of one differential boundary patch, with unit amplitude and zero phase at Q.
Section 5 now connects K=|K| exp(i*phi) directly to the real contribution
|K|*U(Q)*dA*cos(phi-omega*t): magnitude sets oscillation height and phase
sets its time shift. The repeated magnitude factorization was removed;
complex-arrow views are introduced as optional exploration after the panel.
The phase is explicitly that of the complete kernel, including its prefactor.
The cosine form was checked against all three plotted responses at four
geometries. The decay experiment uses the normal connection x=xi=0, avoiding
an unconditional decay claim for changes in s at fixed oblique offsets.
This revision changes only two Markdown cells and preserves saved outputs,
code and metadata; notebook schema, math delimiters and compilation passed.
It plots Re[lambda^2*K*exp(-i*omega*t)] over two periods, i.e. the real field
contribution per U(Q)*dA/lambda^2. No finite patch area is assigned or integrated.
The waveform and magnitude axes are fixed, preserving absolute phase and
distance-dependent decay. The geometry marks Q as a boundary patch, P as the
observation point, and theta from the positive-s normal. Blue fronts use the
full response phase kR-atan(kR), not the point-source Green phase kR. Radii are
solved at integer phase cycles; any thinning is labeled. Fronts belong to this
local response, not a total aperture field. The exact magnitude is interpreted
as cos(theta)/(lambda*R) times sqrt(1+1/(kR)^2).

Compare shapes applies the same i*lambda*s*exp(-iks) factor to all three
kernels. Absolute weights retains carrier and spreading, displayed as
lambda^2*K. Complex axes stay at [-1.15,1.15] in both directions, with equal
aspect; magnitude axes stay at [0,1.15]. Absolute magnitudes are always listed.
There is no per-kernel normalization. Geometry uses equal spatial scales
and a camera that contains both P and Q. Analytical unwrapped phase
differences include full turns; they are never formed by subtracting two
wrapped angles. A fixed central phase band prevents arbitrarily enlarging
small residuals. A selected-point readout also gives the omitted boundary
phase directly, linking motion of Q to the Fraunhofer approximation.
The phase-error axis now explicitly defines approximation minus full RS I.
The caption connects Section 4's positive omitted curvature to the additional
negative error: Delta_phi_FF = Delta_phi_F - k*xi^2/(2*s). The green curve
resembles that negative curvature when Fresnel's own error is small; at other
settings it includes both errors. The live readout reports the signed extra
Fraunhofer error, FF minus F. This identity and readout updates were checked,
all 11 kernel tests passed (6.91 seconds), and default/long-distance figures
were inspected for readable labels. Numerical kernels are unchanged.
Symbolic equations live in notebook Markdown; widget
readouts use plain HTML and Unicode.

## Section 6: Fourier-space comparison

`propagation_fourier.py` reuses `scalar_propagation` for full/Fresnel
multipliers. Angular frequencies kx,ky are converted to fx,fy by dividing
by 2*pi. Its signed error is Fresnel minus ASM, reversing the existing
backend's ASM-minus-Fresnel convention to agree with Section 5. Errors are
analytical and unwrapped; the propagating-wave comparison is undefined
outside the propagating band and plotted as NaN there.

The first explorer has local separation and kx/k controls. Its ky=0 cut
shows the equal-scale wavevector circle, the imaginary evanescent branch,
Fresnel parabola, carrier-removed phasors, magnitudes and phase errors.
Phasor axes stay at +/-1.15 and magnitude axes at [0,1.12]. The same carrier
factor is removed from both multipliers; no individual normalization hides
decay. Grey regions mark the evanescent band, where the formal Fresnel
extension is explicitly outside validity. Section 1's linked-control note
is restricted to Sections 1–5; the spectral controls are independent.

The companion fixes s=30 wavelengths and ky=0. Moving P changes x/s and
selects the paraxial input-spectrum coordinate (kx/k,ky/k)=(x/s,0). The
marker is a coordinate, not a spectrum amplitude. The general 2D map was
checked against the existing Fraunhofer kernel's Fourier weight. It is
not represented as a third same-frequency transfer function. No exit
field, FFT grid, finite aperture or integrated field is introduced.

Eight focused tests check angular-frequency conversion and outgoing
branches, zero-distance identity, composition and unit scaling, phase
error sign/whole turns/paraxial limit, the 2D Fraunhofer map, changing
mathematical objects under controls, fixed display scales, actual
notebook-cell reruns, invalid inputs and failed-render cleanup.
All 25 kernel/geometry/Fourier tests passed (25.31 seconds).

A fresh project-Python kernel executed all 25 notebook cells, exercised
each new control including both cutoff points and evanescent components,
and reran the two new cells independently. All seven panels stayed active
until explicitly closed; 141 widget models remained stable on rerun and
all were released at the end. Spectral updates took 0.346–0.481 seconds.
The saved notebook was unchanged by execution. Default, long-distance,
evanescent and both mapping-direction figures were rendered and inspected;
the intended VS Code frontend was not directly inspected.

## Section 7: calculation-chain summary

The Markdown-only closing section collects the real-space patch weighting
and coherent integration chain, the ASM/Fresnel Fourier-multiplier and
inverse-transform chain, and the Fraunhofer spectrum-to-position chain.
The latter explicitly includes the coordinate mapping and full observation
prefactor. It is not labeled as a third same-frequency multiplier. The
input field remains unspecified, with a specific grating field deferred
to the next notebook. This edit preserves all existing code cells, outputs
and metadata. Schema, local links, math delimiters and code compilation
passed; no numerical backend or widget behavior changed.

## Verification

Eleven focused kernel tests cover:

- Full kernel normalization/sign against the established 3D Green gradient.
- Direct integration of this kernel against the independently implemented
  full 2D angular spectrum (2e-11 absolute tolerance).
- The exact omitted boundary phase, including the central Q case where
  Fresnel and Fraunhofer coincide while neither equals the full kernel.
- Translation invariance of full/Fresnel kernels, and the separate-coordinate
  dependence of the usual Fraunhofer kernel.
- Inverse-area units, common-factor scaling and analytic phase continuity.
- Fixed-transverse-position limits; whole phase turns remain in the errors.
- Full-kernel phase fronts, their angular magnitude variation, and the real
  response at quarter-period intervals, including absolute phase and decay.
- Every control, all three displays, fixed response/arrow/magnitude limits, and visible
  P/Q coordinates after layout.
- Initial-render failure cleanup, idempotent closure, and invalid inputs,
  including booleans after cached numerical calls.

Six additional geometry checks cover identical plane locations across stages,
visible endpoint geometry, the path/phase identities, the plotted full and
Fresnel magnitudes with their fixed scale, linked updates, local
widget ownership, actual notebook-cell reruns, render-failure cleanup and
state validation. Near-zero comparisons use absolute roundoff tolerances.

After adding the Section 3 magnitude comparison, all 17 geometry/kernel tests
passed (20.76 seconds). The Fresnel readout was exercised after changes to
distance, P and Q. Notebook-cell rerun checks select the explorer calls by
source, allowing frontends to regenerate cell IDs. Saved notebook outputs
and metadata were preserved.

The 13 existing field-comparison tests remain part of the regression run.
All 30 focused/regression checks passed (56.29 seconds), including the six
checks for linked geometry and panel ownership.
A fresh project-Python kernel runs the complete notebook and every main
control, all three displays, slider extremes and the Q=0 equality. Reexecuting
the actual explorer cell closes its old panel and preserves the widget count.
The earlier single-panel run took 0.66–0.89 seconds and retained 24 widget
instances. The current fresh-kernel run exercises eight geometry changes
through different panels, all three summary views, and independent reruns
of all five panel cells. It retains 108 widget instances, with synchronized
updates taking 1.24–3.10 seconds while other validation ran concurrently.
Sliders update on release. Summary-view changes do not rerender the four
geometry-only panels.
At that verification step the primary notebook had empty outputs and no
widget-state metadata; subsequent text-only edits preserve saved execution state.

## Presentation verification

Default, short-distance and long-distance scientific figures were rendered
and inspected. The notebook now presents the live explorer and one validation
link in its references; the duplicate static-view block and unused SVG export
have been removed. Numerical tests and their validation records remain available.

The four section-local plane sketches and the reused summary geometry were
also rendered and inspected at default, short/long separation, zero-offset
and Q=O states. Reference/observation plane labels, both transverse coordinate
systems, projected endpoints and the approximation-specific plots remain
visible. Review renders and execution helpers are temporary and were removed.
The updated two-comparison Fresnel figure was also inspected at default,
short and long separations, with the magnitude scale unchanged.

This verification distinguishes Python execution/callback checks and figure
inspection from inspection in a live notebook frontend. The intended VS Code
notebook frontend has not been inspected in this revision.
