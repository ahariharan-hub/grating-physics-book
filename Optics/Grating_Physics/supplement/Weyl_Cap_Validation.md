# Retained Weyl grating/cap application

This application is retained for later use. It is no longer Section 5 of
notebook 8.2, which now teaches only
[spherical-wave reconstruction](Weyl_Reconstruction_Validation.md).
The records below describe earlier versions and their numerical validation.
Its [static cap illustration](figures/8_2_weyl_cap.svg) remains supplementary.

References retained for that later application:

- [MIT Optics, lecture 16](https://ocw.mit.edu/courses/2-71-optics-spring-2009/c753a13064038b59a5596af428a8bc41_MIT2_71S09_lec16.pdf): thin-grating spectral shifts.
- [Foreman and Torok, collection/imaging treatment](https://mrforeman.com/downloads/pubs/pub011.pdf): optical-system context; the retained explorer does not implement a lens mapping.

## Model and conventions

One live explorer now follows four guided steps: spherical illumination,
plane-wave components, thin-grating diffraction, and reconstruction on a fixed
entrance cap. A static final-step export is supplementary. It starts with a known spherical Green response,
reveals its continuous plane-wave spectrum, applies a thin scalar grating, and
evaluates the resulting field on an object-side spherical reference cap.

- exp(-i omega t), wavelength=1, k=2 pi; propagation coordinates (x,y,s), s=-z.
- Unit point source P0=(x0,0,-s0): x0=-1..1, s0=1..2.5 wavelengths;
  initial source (0,0,-1.5). Reference/grating plane s=0.
- FIXED cap centre O=(0,0,0), the optical-axis intersection with the grating
  plane; radius 6, meridional angles -40 to +40 degrees. O is a reference
  object point, not a restriction of the illumination to one boundary patch.
  The field normalization is the original unobstructed spherical response on
  that cap. Moving P0 changes both the physical field and its plotted ratio;
  it never moves the cap or resets the amplitude/phase reference.
- Grating t(x)=0.5+0.5*cos(2 pi x/d), infinite and invariant in y, period 1.2..4.
  It shifts the continuous incident spectrum with coefficients 1/4, 1/2, 1/4.
  Output longitudinal wave numbers are recomputed after each shift.
- Cap values are a direct angular-spectrum sum at curved 3D coordinates.
  The existing `field_propagation.component_field` supplies the outgoing phase
  convention (negative global kz, positive distance s). Both transverse
  frequencies are integrated; the display is a spatial cross-section.
- The finite cap is the surface being sampled, not a prior NA filter on the
  spectrum. No material-grating solver, polarization, efficiency calculation,
  lens mapping, far-field reduction or Debye approximation is implemented.

Section 5 now explicitly calls the cap a curved observation surface in the
same homogeneous medium. U_ref is identified with the prescribed U_exit,
and U_cap is the pointwise complex field evaluated on that surface. Its
curvature neither focuses the field nor imposes a spherical wavefront.
A short note explains that sine-condition sqrt(cos(theta)) or reciprocal
apodization belongs to a specified optical mapping, while factors may also
arise from a separately defined power/area amplitude normalization. Neither
operation is part of this explorer. The fixed spherical-reference ratio
is distinguished as display normalization only.

The live final-step note and geometry caption repeat this scope next to the
plot; the supplementary SVG was regenerated. The stage/control and
notebook-cell rerun tests passed (2 checks, 11.41 seconds), including cleanup.
The updated final-step figure was rendered and inspected. Notebook schema,
math delimiters and code compilation passed; code cells, saved outputs and
metadata were preserved. That scope-only edit left propagation and quadrature
unchanged; the subsequent recentering and accuracy update are recorded below.

## Numerical integration

The Weyl integrand contains 1/kappa at incoming grazing incidence. For the
propagating part use mu=kappa/k; for the evanescent part use kappa=i beta.
These changes of variable remove the incoming singularity analytically.
For shifted spectra the output grazing boundaries are split in polar and
azimuthal coordinates, with quadratic substitutions at transition endpoints.
The full 3D sum is evaluated in small observation-point batches to bound memory.

The cap uses quadrature order 64, with complete azimuth integration. General
`reconstructed_field` calls retain their order-48 default. Evanescent
beta is truncated at 20/s0. Its incident-plane omitted absolute tail is bounded
by exp(-20)/(4 pi*s0): at most 1.65e-10 over the source-distance range,
and much smaller at cap points. A grazing transition outside the truncation
range is not used to split an interval; all integration intervals stay positive.
Numerical weights and curves are independent of the spectrum-image grid.
Only the picture floors |kappa| at 0.02 k and uses a fixed log-amplitude scale
to keep the integrable grazing ring readable.

Earlier source-centred-cap baseline, before the geometry revision below:

- No grating: maximum complex relative error against the analytic spherical
  response, 2.1e-13.
- Grating periods 1.2, 2, 4: maximum complex differences between order 48 and 64,
  divided by the ideal sphere amplitude, 1.4e-10, 2.1e-11, 1.4e-11.
- Incident-plane grating boundary checks at three non-collinear 3D points:
  maximum absolute error approximately 1.1e-10, consistent with the beta tail.

These are representative checks over the declared model, not universal error
bounds for arbitrary apertures, periods, cap radii or source positions.

## Interactions and interpretation

The cap was recentered from the initial illumination source to reference
object point O=(0,0,0) in the grating plane. The receiving coordinates now
have s=6*cos(theta), with on-axis vertex (0,0,6); source motion leaves them
unchanged. Geometry plots identify O separately from P0 and include the
whole downstream cap. Notebook text distinguishes this reference point
from the full exit field contributing to the sum.

The longer paths produced order-48 versus order-64 relative discrepancies
up to 6.65e-8 at two moved-source edge cases. Cap quadrature was increased
to 64 and checked against order 96, retaining the existing 3e-8 relative
and 3e-10 absolute convergence tolerances. All 16 Weyl/cap tests passed
(45.38 seconds), including analytic no-grating recovery, shifted-mask
boundary values, dispersion, Helmholtz, fixed cap geometry under source
motion, live controls and rerun cleanup. The notebook's existing code
cells, outputs and metadata were preserved; schema and compilation passed.

A fresh project kernel ran all 13 cells of 8.2, exercised all four cap
steps, both cap edges, moved-source/period controls and isolated cap-cell
rerun. The three panels retained 105 widget models on rerun and released
all models when closed. Two five-control source sweeps took 7.086 and
6.278 seconds in total. The saved notebook was unchanged by this check.
Default, initial-illumination and moved-source figures were inspected;
the latest supplementary SVG, notebook links/anchors and math delimiters
passed validation. This was kernel/callback and rendered-figure inspection,
not direct inspection of the live VS Code notebook frontend.

The step control keeps the same laboratory geometry and source state throughout:

1. Spherical illumination: move P0; physical phase fronts and the complex plane
   field change together. Source x and positive source-plane distance s0 are the
   only numerical controls shown.
2. Components: component angle selects a term, with matching flat fronts,
   spectral marker and complex phasor on a fixed amplitude scale. The analytic
   spectrum includes exp(i*kappa*s0-i*kx*x0). Moving the source sideways
   changes phase without changing the ungrated spectrum magnitude. The phasor
   makes this visible; the displayed spectrum is log magnitude, not intensity.
3. Diffraction: enable the mask and change period. The plane field becomes
   t*Ui and the COMPLEX spectrum is the weighted sum of shifted spectra. The
   right plot shows incident and prescribed fields at the same plane. Component
   markers/arrows show a selected input component's orders, not the full field.
4. Cap: sum all 3D components at the actual curved coordinates. Cap point moves
   P and the selected samples. Source movement changes the total cap field;
   component angle only changes highlighting. Both cap geometry and the
   original spherical normalization stay fixed. The raw phase generally varies
   even for the initial source, whose centre P0 differs from O. A unity ratio
   in the initial no-grating case confirms recovery of the analytic field;
   it does not imply constant raw phase on the cap.

Only controls relevant to a step are displayed. Source position persists
between steps. Numerical-model details are optional; symbolic equations stay
in notebook Markdown and short widget readouts use plain HTML and Unicode.
The helper retains the established owned-panel lifecycle, including controls
that are temporarily omitted from the visible tree.

Notebook notation now reuses U_ref, Ui, Q=(xi,eta,0), P=(x,y,s), P0, r0, k and
s=-z. Section 4 defines the Fourier transform and the longitudinal wave number
kappa once; source distance is s0 throughout, with no competing z0. Equations
use transverse wavevectors kx,ky in radians per unit length, with propagation
factor exp(i*(kx*x+ky*y+kappa*s)) and kappa=sqrt(k^2-kx^2-ky^2). The Fourier
transform has no forward prefactor; its inverse measure is dkx*dky/(2*pi)^2.
Weyl's amplitude remains i/(2*kappa) times the source phase. Grating shifts are
2*pi*m/d in kx. Spectrum axes use kx/k and ky/k, keeping the unit circle.
One explicit link back to 5.2 gives kx=2*pi*fx and ky=2*pi*fy (Goodman).
Section 4 introduces Weyl's expansion of the existing spherical Green response
before the Section 5 application. Applying K_RS1=-2*d_s*g gives the spectral
kernel exp(i*kappa*s), explicitly connecting full RS I to angular spectrum.
The real-space/Fourier-space Green pair is labelled explicitly, and the
derivative chain includes the intermediate transformed response g-tilde_s.
Only transverse separation is transformed; s stays a parameter.
The revised Markdown passed schema/link checks, and a finite-difference check
confirmed that derivative for both propagating and evanescent components.

## Verification outcome after source/step revision

- 78 Weyl, shared field-propagation, Green/Kirchhoff interaction and propagation
  picture regression tests passed. One added grazing-transition/truncation
  regression also passed (79 distinct checks total).
- Moved sources (-1,1), (0.6,1.2), (1,2.5), represented as (x0,s0), recover the
  analytic spherical field at non-collinear 3D points, with observed absolute
  errors below 2e-11. Thin-mask boundary data and fixed-cap invariance pass.
- At these moved sources, grating periods 1.2, 2 and 4 were compared at orders
  48 and 64. Largest observed cap difference relative to the original sphere
  amplitude was 3.9e-9. Representative real/imaginary curves stayed within the
  fixed [-1.65,1.65] axes. This is a sampled check, not a universal bound.
- Tests distinguish spectral phase from magnitude, verify coherent shifted
  spectra, exercise every guided step and relevant control, decode PNG output,
  and check cleanup after cell reruns and failed initial rendering.
- A fresh project-Python kernel executed the combined 8.2 notebook and exercised
  source movement in every step, component angle, grating period, cap point,
  analytic moved-source recovery and isolated explorer-cell rerun. All passed;
  other panels stayed open and the widget count was unchanged after the rerun.
  Callbacks took 0.34-0.95 seconds locally; sliders update on release.
- Default and moved-source scientific figures were rendered and visually
  inspected, including a 60-degree component and a point near the cap edge.
  Source labels, plane labels, math-text notation and fixed phasor axes were
  checked. Source images are cross-sections of the 3D field, not a 2D solver.
- Notebook schema, local links, reference anchors, clean saved outputs and SVG
  XML validation passed. The static fallback represents the final step.
- These are Python/callback checks and scientific-figure inspection. The revised
  controls have not been inspected in a live notebook frontend.

## Wavevector notation revision

Sections 4-5 and the spectrum helper now use kx,ky in radians per unit length.
The forward transform, inverse measure, Weyl source phase and grating shifts
are written consistently; plotting remains in normalized kx/k,ky/k coordinates.
This is a coordinate-convention change, preserving the physical field and the
full outgoing/evanescent quadrature. The revised analytic-spectrum test checks
absolute complex amplitude as well as translation phases and grating shifts.

The 16 Weyl/cap tests and 6 shared propagation tests passed (22 total). The
updated source/component figure was rendered and inspected; notebook schema,
local reference links, SVG XML and clean diff checks passed. Live notebook
frontend inspection remains separate and was not performed for this revision.
A fresh kernel also ran the combined notebook, exercised all four steps and
their controls, recovered the analytic moved-source field and reran the explorer
cell with unchanged widget count. Callback times were 0.36-0.97 seconds locally.

## Merge into 8.2

The Weyl/grating/cap material now follows the angular-spectrum motivation in
Section 5 of `8_2_Rayleigh_Sommerfeld_Propagation.ipynb`. It shares the notebook
setup, uses a single final reading list, and keeps its own independently
rerunnable explorer cell. The former standalone 8.3 is removed after transfer;
its numerical helper and validation remain. The later source/step revision
extends the quadrature with the source translation phase and source-distance
evanescent cutoff; it retains the same full 3D propagation method.

The merged notebook executed in a fresh project-Python kernel. All three
explorers produced PNGs; RS response switching, patch selection, Weyl reference
recovery and grating controls were exercised. Rerunning all three explorer
cells replaced only their owned panels and preserved the total widget count.
Schema, internal/external file links, reference anchors and SVG XML passed.

The merged-notebook Weyl and RS regression suites passed: **55 tests**.
