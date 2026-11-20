# wave_optics

Shared calculations and interactive views for the optics notebooks. This package
and the numbered notebooks live in the project root; unnumbered notebooks,
scripts, tests, figures, and other supporting material live in `supplement/`. Notebook
setup cells locate this helper package from either directory. Run the notebooks
with a kernel that has NumPy, SciPy, Matplotlib, Plotly, and ipywidgets installed.

The numbered teaching route connects theory to measurement: identify the
apparatus and recorded signal, state the model relation, then explain what can
be inferred. Complex U and instantaneous signed fields are model quantities;
ordinary camera data constrain intensity, while phase reconstruction needs
additional information. Experimental connections cover amplitudes from signal
ratios, directions from calibrated order positions, decay lengths with a stated
probe response, and relative phase from interference. Keep these connections
brief and retain the algebra linking each observable to the inferred quantity.

The two-dimensional scalar FMM backend is documented in
[scalar_grating_2d.md](scalar_grating_2d.md), including Bloch illumination,
field reconstruction, validation, and truncation limits.

| Module | Responsibility |
| --- | --- |
| `waves.py` | Wavevectors, direction cosines, scalar fields, and plane-wave phasors |
| `coordinates.py` | Cartesian, angular, spatial-frequency, and normalized pupil descriptions |
| `exit_field_view.py` | Exact thin-mask field, periodicity, and order-reconstruction figures for Notebook 9.1 |
| `field_sampling.py` | Prescribed complex periodic fields, physical-coordinate FFT coefficients, and independent sampling-error comparisons for Notebook 9.2 |
| `field_sampling_view.py` | Paired grid-comparison figures and an owned sampling explorer with fixed physical coefficients and adjustable grid spacing, window size, and origin |
| `diffraction.py` | Allowed transmission orders and Harvey/Ewald coordinates |
| `plotting.py` | Matplotlib renderers, Plotly figures, tables, and HTML export |
| `widgets.py` | Jupyter controls, animation, and callbacks |
| `propagation_pictures.py` | Original static SVG teaching diagrams for notebook 8.1 |
| `propagation_interactive.py` | Validated Green responses, full RS I square-aperture quadrature, and owned interactive panels for notebook 8.1 |
| `phasor_view.py` | Notebook one's independent spatial-point and complex-plane explorers, plus an optional local-phasor grid |
| `grating_intro.py` | Introductory grating diagrams, ideal amplitude mask, and incident surface-phase explorer |
| `order_formation.py` | Arrival phases, finite-array interference, and optional cell phase matching |
| `order_lesson.py` | Earlier surface-phase and phase-matching views, plus introductory path diagrams |
| `order_waves.py` | Incident and selected outgoing plane waves, real-order directions, and selection controls |
| `order_paths.py` | Signed incoming/outgoing path geometry and its connection to the phase-repeat condition |
| `order_array.py` | Finite sums of spreading narrow-opening sources, normalized far-field pattern, and standalone controls |
| `slit_kinematics.py` | Continuous single-slit candidate directions and wavevector components, without strengths |
| `evanescent.py` | Complex normal components and decaying fields for individual orders in the outgoing half-space |
| `evanescent_view.py` | Notebook four's selectable orders, time-dependent fields, amplitude envelopes, and grazing comparison |
| `conical.py` | Three-dimensional order components, detector-plane intersections, and negative-z fields, including evanescent candidates |
| `conical_plotting.py` | Incident wavefronts, spatial rays and detector plane, selected Cartesian wavevector projections, and Harvey/Ewald constructions |
| `conical_view.py` | Linked Plotly views for notebook five and the standalone Ewald notebook |
| `amplitude_view.py` | Notebook six's cycle-average, intensity, and field-versus-detector views; reusable amplitude, phase, and phasor views |
| `superposition.py` | Two-wave field sums, interference terms, wavevector differences, and an ideal calibrated camera response |
| `interference_setup.py` | Notebook seven's mirror and detector controls, cylindrical overlap estimate and Matplotlib apparatus sketch |
| `point_interference_view.py` | Retained standalone one-point field addition view; no longer displayed in notebook seven |
| `detector_interference.py` | Phase traced through notebook seven's two arms, Gaussian envelopes, area-averaged detector images and a resolved fringe view |
| `interference_wavefronts.py` | Linked wavefronts, detector slice, intensity profile and wavevector-subtraction inset for notebook seven |
| `phase_exposure.py` | Steady EO phase control and optional exposure averaging of the same detector patch, with coherence and visibility readouts |
| `spatial_coherence.py` | Young experiment with a uniform incoherent line source, spatial correlation, and averaged detector fringes |
| `phase_geometry.py` | Mirror-to-detector phase projection and enlarged wavefront intersections along z, using notebook seven's apparatus direction |
| `longitudinal_interference.py` | Retained optional fields-along-z view; no longer displayed in notebook seven |
| `superposition_view.py` | Reusable two-wave explorers from the previous notebook-seven draft, plus a camera view |
| `scalar_theory.py` | Scalar Helmholtz residuals, prescribed wave families, boundary choices, and a Fraunhofer slit model |
| `helmholtz_families.py` | Cylindrical, Bessel, regular spherical-multipole, and evanescent Helmholtz fields using SciPy special functions |
| `scalar_theory_view.py` | Independent controls and illustrations for the scalar-optics foundations notebook |
| `scalar_propagation.py` | Outgoing ASM and Fresnel transfer functions, spectral geometry, phase-error estimates, and finite-grid FFT propagation |
| `scalar_grating.py` | Normal-incidence scalar Helmholtz scattering by one rectangular dielectric layer; complex orders, fields and scalar flux efficiencies |
| `scalar_grating_2d.py` | Two-periodic scalar FMM for a single lossless or absorbing layer, arbitrary propagating incident Bloch wavevectors, and volume-integrated absorption |
| `amplitude_grating_2d.py` | 2D absorbing-screen/ideal-opening comparison at a common exit reference, with complex-field FFT2 coefficient extraction |
| `absorbing_grating_2d_view.py` | Linked 2D unit-cell/exit-field, order-lattice and propagated-field notebook display |
| `absorbing_grating.py` | Passive complex-index grating with air openings, scalar scattering and independently integrated absorption |
| `amplitude_grating.py` | Ideal amplitude aperture at the common exit phase, with reusable absorbing thickness cases |
| `absorbing_grating_view.py` | Main Notebook 0_5 amplitude-mask/absorber field and coefficient comparison, including thickness dependence |
| `scalar_grating_view.py` | Notebook 0_5's shared solved field, harmonic and reference-plane views, linked order evolution and scalar-efficiency table |
| `thin_grating.py` | Analytical thin phase-mask coefficients, negative-z propagation and common-plane comparisons with an existing scalar BVP solution |
| `binary_transmission.py` | Shared analytical Fourier coefficients and exact local fields for ideal binary amplitude and dielectric phase masks |
| `binary_transmission_view.py` | Ideal transmission amplitudes/phases, finite Fourier reconstructions, order coefficients and a selected coefficient in the complex plane |
| `grating_comparison_view.py` | Two linked tabs comparing complex fields, order coefficients, Argand vectors, differences and flux-weighted efficiencies |
| `scalar_propagation_view.py` | Spectral geometry, transfer functions, and linked slit propagation explorers for Notebook 0_4 and the propagation reference |

The numerical modules use NumPy; `helmholtz_families.py` also uses SciPy
special functions, and `scalar_grating.py` uses SciPy linear algebra.
Importing `wave_optics` does not create figures or widgets.
Further propagation and imaging modules can be added as those topics develop.

The [scalar grating backend guide](scalar_grating.md) documents the 0_5A
scattering API, negative-z reference planes, boundary matching, and physical
validation. Unlike the kinematics helpers, it solves for complex reflected and
transmitted amplitudes from the dielectric structure. No new notebook is needed
to use or test it.

## Calculations

```python
from wave_optics import wavevector, scalar_plane_wave, diffraction_orders

k = wavevector(wavelength=2.0, theta_deg=50, phi_deg=30)
field = scalar_plane_wave([[0, 0, 0], [1, 0, 0]], k)

orders = diffraction_orders(
    wavelength=500, period=1000, theta_deg=30, phi_deg=30)
print(orders["allowed"])
print(orders["directions"])  # km/k for real orders
```

The retained phasor helper API uses the same coordinates in 1D, 2D, and 3D.
These names remain available for existing callers; numbered notebooks defer
phasor diagrams as a teaching method:

```python
from wave_optics import plane_wave_phasor, field_from_phasor, wavenumber

phasor = plane_wave_phasor([[0, 0, 0], [1, 0, 0]], k)
field = field_from_phasor(phasor, time=0.2, omega=wavenumber(2.0))  # phase speed = 1
```

The phasor excludes time. The real field is recovered using the convention
`Re{phasor * exp(-1j * omega * time)}`.

Wavevectors use radians per length unit: `k = 2*pi/wavelength`. Polar angles
are measured from +z, and azimuth from +x toward +y. The `diffraction_orders`
model has its period along x, grooves along y, and transmission toward +z, with the
same isotropic medium on both sides. Wavelength is the wavelength in that
medium and must use the same unit as the period. The model gives allowed
directions; it does not calculate order efficiencies.

## Notebook views

The former combined grating notebook is split into three independent lessons:

| Lesson | Former sections | Existing views |
| --- | --- | --- |
| [0_5 Physical gratings and the Helmholtz BVP](../0_5_From_Helmholtz_BVP_to_Diffracted_Field.ipynb) | 1–3 | Geometry, solved field, coupled harmonics |
| [0_6 From complex fields to diffraction orders](../0_6_From_Complex_Fields_to_Diffraction_Orders.ipynb) | 4–7 | Reconstruction, reference plane, order evolution, efficiency/R/T/A |
| [0_7 Ideal binary masks versus physical gratings](../0_7_Ideal_Binary_Masks_vs_Physical_Gratings.ipynb) | 8–9 and vector bridge | Complex field and coefficient comparison, FFT, Argand plots, thickness sweep |

Each setup computes one explicit `AbsorbingGratingConfiguration` and solved
`AbsorbingGratingSolution`, with M=24 (49 harmonics). The lessons pass
`display_order_limit=7` to both explorer interfaces: coefficient plots and
order selectors show -7 through +7, while fields, FFT checks and power totals
use all solved harmonics. Change this display setting independently of M.
Omitting it from an explorer preserves the full-order display for other callers.
Section 3 introduces Scalar
Fourier Modal Method (FMM) and its relationship to electromagnetic RCWA.

`ScalarGratingData(grating_solution)` caches that solution's complex material,
field and harmonic samples: 512 x samples over two periods, 601 z samples,
and 256 slice samples over one period. The no-argument `ScalarGratingData()`
retains the old lossless M=20 default for existing callers; the main notebook
never uses that default. Absorbing geometry/field explorers offer separate
real-index n and extinction-coefficient kappa maps, without casting away the
imaginary material component. The `reference` explorer's lazy `evolution_ui`
and `efficiency_ui` share one output-plane control in 0_6 sections 2–4. The
efficiency view plots coefficient magnitude, phase relative to order 0 and
transmitted flux per displayed order, with explicit markers for zero normal
power. A separate bar chart includes R, T and independently integrated A,
using all solved orders for R+T+A conservation. Moving the plane updates the
field coefficients while efficiencies, totals and plot scales stay fixed.

Notebook 0_7 centrally compares the ideal binary amplitude mask with the same
physical absorber. `AbsorbingComparisonData(solution=grating_solution)` reuses
the exact initial solution and solves only six additional thickness cases.
`AbsorbingGratingExplorer` supplies linked structure/field and coefficient tabs,
including common-plane amplitude, phase, Argand vectors, complex differences
and flux-weighted efficiencies. The field profile also selects real or imaginary
components. One x-position selector highlights the same local HBVP field value
in the spatial profile, phase plot and field Argand diagram; the Argand colours
encode x. These field plots show only HBVP by default; **Compare with ideal
mask** enables the ideal overlays on the spatial and Argand plots. Reset hides
them again. The separate ideal-transmission panel and coefficient comparison
remain available. The coefficient tab overlays FFT extraction from the sampled complex
field on direct modal coefficients, reporting maximum absolute error against
1e-12. It uses an endpoint-excluded period grid, FFT/N normalization and the
sampling-origin phase correction; it never transforms amplitude or intensity
to obtain field coefficients. This checks the representation of the existing
solution, not independent scattering physics. The expandable thickness plot supports
0_7 section 2. Changing the plane or selected order does not invoke the solver.
The separate 0_5 and 0_6 lessons retain the initial thickness.
Both models share U_i(0)=1 and the absolute exit phase exp(+i k0 h); there is no
fitted phase or power normalization. See [absorbing_grating.md](absorbing_grating.md)
for numerical acceptance results and analytical normalization checks.

Call `close()` before replacing an explorer. The notebook setup closes old views
and reloads dependencies one at a time in dependency order, starting with
`scalar_grating` before importing the absorber. This recovers a running kernel
that cached the older solver without `_solve_layer`, and refreshes dependent
class references before constructing new solutions. After rerunning the
reference-view cell in 0_6, redisplay
sections 3 and 4. Saved notebooks contain no executed outputs or widget state.

The former amplitude/phase-mask and lossless dielectric comparison is preserved
in the standalone [supplementary lesson](../supplement/Binary_Amplitude_and_Dielectric_Phase_Masks.ipynb).
It is outside the main 0_5–0_7 narrative. `scalar_grating.py`,
`thin_grating.py`, `binary_transmission.py`, their legacy explorers, and their
lossless/phase tests remain supported. `GratingComparisonExplorer` is the
legacy dielectric comparison; `AbsorbingGratingExplorer` is the main amplitude
comparison. The Pistor reference remains in 0_7, with a concise
future vector-scattering discussion.

All 0_1–0_7 introductions distinguish essential prerequisites, optional
mathematics and a first reading through the visualizations. The 0 series is
a conceptual overview; later numbered lessons provide systematic development.
The pre-edit [component inventory](../supplement/validation/notebook_0_split_inventory.json)
maps every original cell, figure group, control, derivation and reference.

The [split validation record](../supplement/validation/notebook_0_split_validation.json)
contains full counts and checks against baseline `af39df06e12f497f85dd9f5d9eee2bf12f4a4cd1`.
Fresh-kernel runs of 0_5, 0_6 and 0_7 matched sampled baseline fields,
coefficients and flux totals exactly. All nine interactive view groups were
retained. The 65 retained regression tests passed, as did a one-time audit
confirming every physics/view module was unchanged. KaTeX parsed 660
expressions; Markdown tables and rendered lesson/plot previews were checked.

| Lesson | Educational prose words | Cells (code) |
| --- | ---: | ---: |
| 0_5 | 1,430 | 13 (4) |
| 0_6 | 1,394 | 16 (5) |
| 0_7 | 1,817 | 11 (2) |

Counts exclude formulas and code. The shorter lessons follow the conceptual
boundaries without padding. Scientific citations remain present, including
Pistor (2001). Durnin's author-hosted PDF has a certificate error; 0_1 and 0_2
also provide its verified publisher DOI. Publisher access may require a browser
check or subscription.

Focused checks:

```text
python -m unittest supplement.tests.test_absorbing_grating supplement.tests.test_absorbing_lesson supplement.tests.test_amplitude_grating supplement.tests.test_absorbing_grating_view supplement.tests.test_scalar_grating supplement.tests.test_scalar_grating_view
python -m unittest supplement.tests.test_foundations_sequence
python -m unittest supplement.tests.test_binary_transmission supplement.tests.test_binary_transmission_view supplement.tests.test_thin_grating supplement.tests.test_grating_comparison_view
```

`Scalar_Wave_Propagation.ipynb` restructures `Scalar_wave_propagation.docx`
into a derivation from the scalar Helmholtz equation through the transverse
Fourier ODE, outgoing angular spectrum, Fresnel approximation and phase error,
then shift invariance, convolution, and the exact Rayleigh–Sommerfeld kernel.
It keeps the document's coordinate representations and all twelve source
figures (including both animated GIFs) in
`supplement/figures/scalar_wave_propagation_source/`, with provenance in `inventory.json`.
Two independent `ScalarPropagationExplorer` stages, `geometry` and `transfer`,
link spectral coordinates to a signed field slice and compare exact/Fresnel
propagation. An editable 2D Gaussian-boundary example illustrates FFT
propagation with separate transverse widths and a fixed intensity reference.
The numerical helper explicitly represents a finite periodic grid; it adds
no automatic padding, resampling, or band limiting. Optional sections retain
mode, angular-measure, index-scaled, global Fourier-shell, and statistical
representations. The source Word file is unchanged. Saved notebook outputs
and widget state are clear; rerunning setup or a view disposes earlier widgets.

`0_Foundations_of_Scalar_Optics.ipynb` previews scalar optics and diffraction
around the central complex scalar field U. Its core reading route introduces
what U represents, the Helmholtz constraint, and the choice of plane waves as
building blocks for the following notebooks. Other solution families,
linearity, boundary conditions, and the first diffraction example give context;
detailed derivations and checks are optional on a first reading. Plane waves
are a chosen representation, not the only diffraction solutions. The closing
question leads directly to notebook 1's wavevector geometry. Four independent
`ScalarTheoryExplorer` stages illustrate `helmholtz`, `solutions`, `boundary`,
and `diffraction`; six editable calculations and
thirteen checkpoint answers provide numerical and conceptual checks. No
earlier notebook must be run.

Visible Sections 5.4–5.8 introduce cylindrical waves, Bessel beams, spherical
multipoles, evanescent fields, and the paraxial Gaussian families. One
`solutions` view compares seven exact families, including plane, standing,
and spherical waves, alongside a fundamental Gaussian beam explicitly labelled
paraxial. Signed-field slices, amplitude profiles, and beam intensity share one set of controls. Its
shared controls retain their values when switching families, while specialized
controls appear only when needed. A cone-average calculation constructs a Bessel profile
from plane waves. Section 5.8 gives the Gaussian field, width, Rayleigh range,
curvature, and Gouy phase before the combined explorer, then derives Gaussian
and Bessel intensity profiles. For plane waves, Bessel beams, and Gaussian beams,
select `View: Beam intensity` to link a longitudinal xz intensity map, a transverse
xy observation plane, and its intensity profile along x at y=0. The observation
plane is marked in cyan on the longitudinal map. Gaussian `Axial plane z/z_R`
and the other families' `Observation z/λ` move this plane. All intensity maps use
the same fixed 0–1 unit-amplitude reference rather than renormalizing each plane;
phase and model time are hidden in this view. Distances are in micrometres, with
unequal xz scales explicitly labelled and equal xy scales. Dashed width guides
mark Gaussian w(z) (the local 1/e² intensity radius) or the first Bessel zero.
The plane-wave reference is uniform and has no finite beam edge.

The default `Field & amplitude` view retains signed fields and amplitude profiles.
Its Gaussian transverse xy slice uses waist units at the selected z/z_R. Changing
waist updates physical scales and divergence; at fixed z/z_R the amplitude
profile in waist units is unchanged. Gaussian, Hermite–Gaussian, and Laguerre–Gaussian
formulas are distinguished as paraxial solutions rather than exact Helmholtz
fields; the latter two families are discussed but not plotted.

This notebook uses exp(-i omega t), wavelength in the medium, and positive-z
propagation. The spherical illustration is a slice of a three-dimensional
field with its source region masked for display. The boundary view contrasts
fields sharing U(0) but having different directional content and derivatives.
The slit view assumes an ideal scalar aperture and the paraxial Fraunhofer
limit, with screen curves normalized to their own central intensity. It does
not implement general near-field, ASM, or Rayleigh-Sommerfeld propagation.
Saved outputs and widget state remain clear, and rerunning setup or a section
closes its previous explorers.

`7_1_Interference_and_Mutual_Coherence_of_Two_Plane_Waves.ipynb` connects notebook 5's diffraction orders
and notebook 6's intensity to interference between two beams. A beam splitter
and one mirror prepare the beams, inspired by Catunda, Sartori and Nunes
(1998), DOI:10.1119/1.18825. The original experiment's magnifying optics are
outside the model.

The whole lesson uses right-handed laboratory (x,y,z) coordinates with
propagation toward negative z. The splitter is S=(0,0,0), mirror M=(h,0,0),
and detector centre P=(0,0,z_D), where z_D<0. The apparatus is drawn in xz:
x right, z up, +y into the page. Beam 1 propagates along (0,0,-1); after
reflection beam 2 propagates along (-sin(theta),0,-cos(theta)).
`two_beam_geometry` returns actual 3D positions and direction vectors.
Its `detector_z` argument and the explorer's `detector_z` slider use signed
coordinates, ranging from -10 to -3.5 schematic units (default -6.9).
More negative z_D moves the detector farther from the splitter.
Its square stays parallel to xy and centred at x=y=0. The mirror reference
remains aligned at z_D=-6.9 regardless of the selected detector position.
Positive mirror tilt is counterclockwise in the xz side view and gives
theta=theta_0-2*delta. At delta=+2 degrees, the axes meet near z_D=-9.15.

The apparatus uses Matplotlib and shared ipywidgets controls. The source
example is 632.8 nm, 1 mW, and a 0.54 mm 1/e^2 intensity diameter, based on
Newport N-STP-912 specifications. Polarization is along y. The beam-width
contours describe geometry; their shading is not calculated intensity.
Dashed axes continue beyond the moving detector. Readouts distinguish
overlap, separate footprints, and a single intercepted beam.

The overlap calculation precedes the apparatus drawing. For equal circular
cylinders, `cylindrical_overlap_volume(D_c,theta)` gives
2*D_c^3/(3*abs(sin(theta))) in mm^3 (diameter in mm, angle in degrees).
This is the full cylinder intersection (Hubbell 1965,
DOI:10.6028/jres.069c.017), independent of detector position.
The assumed D_c=0.54 mm is separate from the source specification.
The default estimate is 0.361 mm^3. For Gaussian beams, these contours
are a chosen boundary; tails extend beyond them.

Section 2 displays `mirror_tilt_explorer.detector_ui` once: a rotatable
3D wavefront view, its matching xy detector slice, an intensity profile
through P, and a wavevector-subtraction inset. Mirror tilt and detector-position
controls sit above these measured quantities; `ui` contains only the linked
apparatus drawing and readout. `wavefront_ui` is an alias
for the same widget, with no duplicate owner or callbacks. The display
selector switches the upper panels between either signed field and the
combined time-averaged intensity; the lower profile always shows intensity.
All three spatial panels share a 10-wavelength window (6.328 um by default).
Local 3D coordinates are (x,y,z-z_D), with z-z_D>0 on the incoming side.
The common snapshot puts beam 1 at a crest at P, retaining relative phase.
The vector inset shows k_1/k and k_2/k with common tails and Delta k/k
from tip 1 to tip 2. Mirror tilt changes this triangle; detector motion
does not. Readouts give Delta k_x/k, Delta k_z/k and fringe spacing.

The full detector appears in a collapsed overview, calculated only when
opened and refreshed with apparatus changes while open. Its central patch
is marked to scale. Full-image bins use Gauss-Legendre averaging to resolve
the carrier before averaging. Image columns are x and rows are y.
The explicit scale is 1.8 mm per schematic unit, making the detector
2.34 mm square and h=3.78 mm. Ideal 50:50 splitting, a lossless mirror,
fixed Gaussian profiles, planar phases and optics_phase_deg=0 are assumed.
Intensity scales are fixed to the default single-arm on-axis intensity.
No camera response, propagation-induced spreading, curvature or grating
efficiencies are calculated.

Fields, profiles and crest planes use `detector_field(x_mm,y_mm,geometry)`;
an optional `z_mm` array allows axial samples of those same fields.
The camera survives apparatus and mode changes; colour scales stay fixed.
Section 1 establishes the apparatus; Section 2 places its controls beside the
detector view. Section 3 explains phase and intensity.

The algebra leads with DeltaPhi=(k_2-k_1).r+DeltaPhi_0, then uses
k_1=k*(0,0,-1) and k_2=k*(-sin(theta),0,-cos(theta)).
The components k_x, k_y and k_z are explicit, with both k_y=0.
The inset's difference is Delta k=k*(-sin(theta),0,1-cos(theta)).
At z=z_D this becomes k*[-x*sin(theta)+z_D*(1-cos(theta))]+DeltaPhi_0.
The apparatus contributes DeltaPhi_0=k*h*(1+sin(theta))+optics_phase,
which is spatially constant at a fixed mirror setting. It changes with
theta. Fringe spacing is 2*pi/abs(Delta k_x)=lambda/abs(sin(theta)), about 2.17 um
at the default angle. The intensity derivation averages the expanded field sum,
then identifies Gamma_12=<U_1* U_2> and normalizes it to gamma_12.
A short historical note connects the experimental motivation to van Cittert,
Zernike, Wolf, and Thompson and Wolf's 1957 interference experiment.
The arriving-field convention includes the apparatus phase in arg(gamma_12);
the displayed steady beams have gamma_12=exp(i*DeltaPhi), with unit magnitude.
Visibility separates the intensity-balance factor from |gamma_12|, so reduced
contrast from unequal local intensities is distinguished from loss of coherence.
An optional note introduces correlation at an additional time delay. The
lesson links Gbur and Visser (2010), Agarwal and Classen (2020), and Goodman,
and leads into Section 4, "Control the EO phase; observe the detector".
The notebook first displays `mirror_tilt_explorer.modulated_ui`: the apparatus
drawing and readout, with an ideal EO modulator between splitter and mirror.
It then displays `detector_modulated_ui`, matching Section 2's wavefronts,
detector slice, intensity profile and wavevector inset. Mirror tilt, detector
position and EO phase (0 to 180 degrees) sit above that detector figure.
Geometry controls are shared across the two detector views. EO changes affect
only the Section 4 views: orange crest planes, field and detector fringes move
together, while blue planes, individual arm intensities and Delta k stay fixed.
Both figures preserve camera/zoom, fixed intensity scales and lazy full-detector
overviews. The default holds phase steady, with unit coherence magnitude.

Geometry changes calculate one shared zero-EO field dataset; the modulated
view applies a complex phase factor while sharing the unchanged amplitudes,
coordinates and directions. Display-mode switches reuse those datasets.
Plotly updates modify existing traces without rebuilding subplot layouts;
detector-position changes also leave the wavevector inset alone.
The apparatus drawings retain their Matplotlib artists. Geometry moves the
existing artists, and EO-only changes redraw just the phase label over a cached
background before PNG encoding. The spatial sampling and image resolution are
unchanged. The phase-geometry accordion renders only when opened and stale,
as do the full-detector overviews and optional exposure example.


The collapsed "Optional: phase changes during exposure" opens a separate
two-panel demonstration with Play/Pause and a current-phase selector. Its left
panel switches the orange crest planes and current detector slice together
between zero and beta. Its right panel always shows the completed equal-time
intensity average on the same patch and scale. The four-panel view above stays
at the selected steady phase. Playback is slowed for illustration; the common
optical carrier is frozen rather than animated. The two snapshots are cached,
and playback updates only the current detector surface and orange planes.
Changing geometry or EO phase pauses and refreshes the demonstration; closing
the accordion pauses it. The optional view is allocated only on opening.
The collapsed explanation derives
gamma_12=exp(i*DeltaPhi)*(1+exp(i*beta))/2, whose magnitude is abs(cos(beta/2)).
At 180 degrees the average is I_1+I_2, even with unequal arms. Readouts distinguish
coherence from local visibility, which also includes intensity imbalance.
Normalized coherence is undefined if either local arm is dark. The sketch
labels the switching experiment while the optional example is open.
A constant modulator delay is included in optics_phase_deg; the ideal EO
element changes neither beam direction nor individual intensity.

`exposure_ui` retains the earlier standalone Matplotlib comparison, allocated
only if explicitly requested. It shares the EO controls with the new view.
`wavefront_data` keeps selected-phase field intensity separate from
recorded_intensity, which includes exposure averaging when enabled.
Apparatus changes preserve the EO settings. One explorer owns the shared
widgets and closes each once. The original `ui` and detector display retain
the unmodified apparatus and its steady beams.

Section 5 introduces an independent Young experiment to sample spatial coherence
between Q1=(-d/2,0,0) and Q2=(d/2,0,0). A uniform incoherent line source at
z=R illuminates both openings, and the detector lies at z=-L. Source width
and opening separation control a retained four-panel Matplotlib view: apparatus,
visibility versus separation, detector image, and sample source-point patterns
with their continuous-source average. The narrowband paraxial model uses
gamma=sinc(a*d/(lambda*R)), with NumPy's normalized sinc convention, and
I/(I1+I2)=1+gamma*cos(k*d*x/L). Visibility is abs(gamma); a negative gamma
shifts fringes by half a period. Source brightness is normalized, temporal
coherence is assumed sufficient, and the small-opening envelope is omitted
within the central detector patch. The zero-width limit preserves full contrast.
Controls update on release; figures and callbacks are released by close().
The new section distinguishes sampling spatial correlation from the preceding
EO exposure example and links TU Delft and Gbur/Visser for further reading.

`phase_geometry_ui` is a collapsed "Where does the phase offset come from?"
accordion. It traces the splitter-to-mirror segment kh and the reflected
phase k_2x*(x-h)+k_2z*z_D, then illustrates the projection to P.
`phase_projection` returns xyz vectors, with
MF=h*sin(theta)-z_D*cos(theta). At centred alignment F=P.
The optional longitudinal wavefront spacing is
s=lambda/abs(cos(theta)), distinct from detector fringe spacing across x.
The retained `longitudinal_ui` helper samples z through x=y=0 and is not
displayed in the notebook.

All views share one explorer and are allocated lazily. Rerunning setup
closes the previous explorer; rerun later display cells to attach the new
instance. `close()` releases callbacks, images and Plotly widgets, including
the exposure views' playback link, controls, cached frames and figures.
The saved notebook contains no outputs, attachments or widget state.
The previous lesson remains in `supplement/Notebook_7_Previous_Draft.ipynb`;
its independent `TwoWaveExplorer` helpers are preserved.

`6_Amplitude_and_Intensity_of_a_Scalar_Wave.ipynb` starts from notebook 5's
known order directions and asks what a detector measures from one selected
order. Three sections connect the local signed field, its mean square and
relative intensity, then compare a spatial field snapshot with the ideal
detector image. Amplitude is explicitly supplied, rather than calculated
from grating structure. Three independent `AmplitudeIntensityExplorer`
views (`average`, `intensity`, `order`) and two editable NumPy calculations
have predictions and collapsible answers. The closing question leads to
notebook 7: why can relative phase affect intensity when two fields overlap,
although a constant phase shift leaves either field's own intensity unchanged?

Field values use one fixed arbitrary unit; all plot scales remain fixed as
amplitude varies from 0 to 2. Relative intensity uses A_ref = 1 in the same
medium and direction. The lesson distinguishes mean(u^2) = A^2/2 from
the squared complex-field magnitude |U|^2 = A^2, with the physical conversion
constant absorbed into the intensity reference. Phase is undefined at zero
amplitude. The final view fixes lambda = 2 micrometres, d = 4 micrometres,
normal incidence and m = +1, with outgoing k_m/k = (0.5, 0, -sqrt(3)/2).
Its left panel shows a signed xz field snapshot and a cyan detector line
at z = -1.5 micrometres. The right panel shows the same detector face-on in
xy with uniform relative intensity and a fixed grayscale from 0 to 4.
Phase and model time move the field bands while the detector image stays
unchanged. Both windows sample a broad plane wave; no finite-beam edges or
order efficiencies are modelled. The reference has the same direction and
detector geometry, so the oblique-incidence projection cancels in the ratio.
At zero amplitude the propagation arrow becomes a dashed direction guide.

Setup closes previous views before reloading the shared module. Rerunning
a section replaces only that section's view. `close()` is idempotent and
releases callbacks, figures and widgets. The saved notebook has no outputs,
embedded figures or widget state. Goodman second edition, section 4.1.1,
and OpenStax section 16.3 support the stated intensity conventions.

`5_Grating_Diffraction_in_Three_Dimensions.ipynb` extends the series to
incidence with a nonzero component along the uniform grooves. It retains
incidence from z>0, transmission into z<0, and wavelength in the common
lossless medium. Six independent small calculations cover incident components,
surface translations, grazing, the cone angle, observation surfaces, and NA
collection. After Harvey's direction-cosine diagram, optional section 7 introduces NA
coordinates and the sine condition as a prelude to physical pupil mapping.
It distinguishes coordinate scaling from the optical mapping. A worked DUV
lithography example follows mask orders through 4X reduction, distinguishes
mask-side and wafer-side NA, and separates angular pupil coordinates from
physical internal-pupil positions. A projection-optics paper
and a microlithography objective design disclosure support that example; microscopy,
condenser and mirror-system examples remain as comparisons. The final calculation holds an outgoing direction
fixed while changing the collection NA; n and the illustrative pupil radius
are independent written parameters, not additional explorer controls.
Checked source conventions are retained in `Notebook_5_Reference_Map.md`.

`ConicalDiffractionExplorer` introduces three views as the lesson progresses:
`incidence_ui`, `orders_ui`, and `harvey_ui`. Each is created on
first display and reuses one shared set of wavelength, period and incident
component controls. Defaults are lambda=2, d=4 micrometres, kix/k=0.3,
kiy/k=0 and selected m=+1. Presets restore these parameters and set kiy/k
to 0, 0.4, 0.6 or 0.8; the last three take m=+1 through propagation, grazing
and evanescence. Increasing the across-groove component adjusts the other
slider's bounds to keep incidence physical, with a notice if it changes.

All real orders, their nearest evanescent neighbours and any retained selected
integer appear in the candidate list. An evanescent selection has a disk point,
but no real outgoing ray or hemisphere marker.
The spatial cone is optional; the finite groove patch depicts an infinite
grating, and ray endpoints use a fixed 4-micrometre reference radius. No
efficiencies are assigned. The Harvey renderer reuses the existing
construction and reflects its normal coordinates into the series' -z convention.
Rotating a view preserves its camera when parameters change. Plotly 5 supports
FigureWidget with ipywidgets; Plotly versions that require anywidget need that
package as well. Rerunning setup closes the previous instance and reloads the
three new modules; rerun later display cells to attach the replacement views.
The idempotent `close()` releases all figures, observers, widgets and styles.
No plots, widget state or executed outputs are saved in the notebook.

The outgoing spatial view uses the same 50%-open bar grating as the incidence
view. Its optional detector-like plane is at z=-L (default L=3 micrometres).
Round markers retain the fixed radius R=4 micrometres; open squares show
(x,y,z)=(-L*kx/kz, -L*ky/kz, -L). The drawn plane patch spans ±6 micrometres
in x and y. Finite intersections outside it are reported without moving them
onto its edge or expanding the scale near grazing. Grazing orders have no
finite intersection and evanescent orders have no real ray intersection.
These are reference positions, not simulated detector powers or spot sizes.

Below the order table, a second rotatable 3D view resolves the selected order
into kx, ky and kz arrows in rad/micrometre, with Cartesian projection guides.
The table lists all three components divided by k. Zero components are stated
numerically; an evanescent selection draws only Re(k), with its imaginary kz
reported separately as decay. All three component axes have equal spatial
scale; their window follows the selected vector. Changing L leaves these
wavevectors and the radius markers unchanged. Both cameras survive control
updates, and both new controls belong to the same disposable explorer.

Notebook 5's hemisphere-and-disk view has a **Coordinates** selector for
direction cosines, spatial frequency in Goodman's convention (cycles/µm),
or wavevector (rad/µm). The notebook cites Goodman's transverse-frequency
definitions while retaining order subscripts and outgoing propagation toward -z.
It scales all markers, rays, projection guides, axis ranges and hover values
by `1`, `1/lambda`, or `k`, respectively. The radius and neighbouring-order
step are shown with the selected components. Axes rescale to fit the same
construction; the 3D camera, selected integer and physical state persist.
Evanescent selections keep their transverse marker without a real sphere
point. The Ewald view retains its existing normalized coordinates.
The accompanying calculation connects `f_m = k_m/(2*pi)` to `k_m/k` and
checks the spatial-frequency step `1/d`, including grazing and decay.

`Ewald_Construction_for_Gratings.ipynb` holds the former Ewald section of
notebook five, its coordinate comparison, and its checkpoint questions. It
is unnumbered pending its later place in the series. Run it independently
from top to bottom: its setup imports NumPy, reloads the shared modules, and
creates its own `ewald_explorer`. It starts at the conical example (lambda=2,
d=4 micrometres, kix/k=0.3, kiy/k=0.4, selected m=+1). No executed cells or
variables from notebook five are needed.

The standalone lesson recalls the outgoing components before introducing
q=km-ki, the shifted sphere, reciprocal rods, and the selected vector triangle.
Its Ewald and later Harvey views share one parameter state within that
notebook. The small calculation has independent editable parameters. An
evanescent candidate retains its selected rod but has no real sphere point
or outgoing triangle. `Ewald_Construction_Reference_Map.md` retains the
checked source passages, the Pflüger paper's 2018 erratum, and the convention
notes. Rerunning setup closes the previous explorer; neither notebook stores
executed outputs, images, or widget state.

`4_Evanescent_Diffraction_Orders.ipynb` follows the candidates excluded from
the earlier real-direction plots. It retains incidence from z>0 and the
outgoing side z<0, with ky=0 and wavelength measured in the same lossless
medium on both sides. The decaying branch is kz=-i*kappa, so at distance
s=-z the envelope is exp(-kappa*s). Actual scattering amplitudes and power
are not calculated; every illustrated component has unit surface amplitude.

Section 4 uses `EvanescentOrderExplorer`. Its default is lambda=d=2 micrometres,
theta_i=30 degrees, and the evanescent order m=+1. The selector keeps all seven
sample order numbers (-3 through +3), including evanescent candidates, and
retains the selected integer across grazing. A component ruler, a signed-field
view, and a depth cut with the amplitude envelope share the controls. Time is
t/T, with one cycle split into forty playback steps. Depth changes only the
view; the field colour scale stays linear at [-1,1]. The three state presets
use lambda=2, normal incidence and m=+1 with d=3, 2, and 1.5 micrometres.
Section 6 displays these same states together. Four independent calculations
cover classification, time versus distance, decay length, and the cutoff.
Rerunning the explorer cell closes the previous instance and reloads both new
modules. All image buffers remain in memory; no outputs or widget state are
stored in the saved notebook.

`3_Why_Diffraction_Orders_Appear.ipynb` begins with the spatial phase change
of one incident plane wave across one grating period, delta_phi = kx*d.
Section 4 compares incident and outgoing phase changes. For an ideal linear,
infinitely periodic grating, their difference must be 2*pi*m; this introduces
the diffraction-order integer and kmx = kix + m*2*pi/d. A normal-incidence
example makes the whole-cycle condition explicit. Section 5 draws the real
directions satisfying abs(kmx)<=k, with grazing orders marked separately. The phase difference is B minus A at the same time, with
B displaced by +d along x at z=0. Point 3 reuses `GratingSurfaceExplorer`
from `grating_intro.py`, the same view and controls as notebook two. The left
plot shows the incident field above the grating; the right shows surface phase
with dots at opening centres. The notebook initializes lambda=d=2 micrometres
and theta_i=30 degrees, giving delta_phi=pi radians. Normal incidence sets the
angle to zero; rerunning the cell restores the starting values. The previous
explorer (including the older simplified view) is closed before replacement.
All images are generated in memory; no previews or widget state are saved.
The earlier simplified and path-diagram helpers remain available but are not
displayed in this notebook.

Section 5 uses `OrderWaveExplorer` from `order_waves.py`, extending section
3's incident-field picture across z=0. The left panel displays the incident
wave above the grating and one selected outgoing component below it. The right
shows all real order directions and a table of order numbers, signed angles,
and kx values. The selected order is orange. Wavelength, period, and incidence
controls update the available-order dropdown; an unavailable selection falls
back to m=0 with a notice. Grazing orders remain selectable and are marked.
The default is lambda=d=2 micrometres, theta_i=30 degrees, and m=-1; reset
restores it. These controls are independent of section 3. The readout shows
both phase steps and their difference 2*pi*m. The shared diffraction helper
supplies real directions, reflected into this notebook's negative-z geometry.
Each component has unit amplitude and zero phase at the origin; the drawing
is not a full boundary-field solution and does not assign efficiencies.
The cell closes both the earlier phase dial and the current explorer before
replacement. It saves no image/widget state. Earlier dial helpers remain
available but are not displayed in this notebook.

The optional **Compare non-integer values** panel retains the main order view
and adds a continuous m_test slider plus two field plots. The left is a free
continuous plane wave; the right copies one cell with the incident phase
offset, using the existing `wavefront_fields` reference helper. At integers
the fields agree; between integers the copied field has seam phase jumps.
Both comparison plots share the main outgoing view's fixed x range
(-4.5 to 4.5 micrometres) and z range (-4.5 to 0 micrometres). Cell boundaries
are recomputed at integer multiples of the current d, so their spacing moves
visibly when d changes. A labelled period ruler appears in the main field and
both comparison plots; the comparison readout explicitly repeats the shared
period. Both wave plots overlay a schematic k arrow and its horizontal kx
projection inside the reference cell, plus points A and B separated by d at
the same height. A phase graph beneath them measures phase change from A:
its orange endpoint is the outgoing kx*d, and its green circle is the next
copy's starting phase kix*d. Their difference explains the seam; whole 2*pi
cycles represent the same phase. Field colours show cos(phi), not phase.
The arrows show direction and relative components, scaled to fit; the
readout gives k and kx in rad/micrometre. The notebook's normal-incidence
example uses lambda=2 and d=4 micrometres: m_test=0.5 compares pi with zero,
while m_test=1 compares 2*pi with zero. This tests compatibility, not a
physical discontinuous field or scattering solution. The slider range is
limited to real directions; parameter changes clamp an out-of-range value
with a notice. Selecting an order or using **Use selected order** restores an
exact integer. The comparison stays hidden until enabled, renders in memory,
and releases its figure, images and observers together with the main explorer.

Section 6 displays `order_waves.paths_ui`, a later view of that same explorer.
Its controls are shared with section 5, including the integer selector and
optional continuous comparison. The left picture retains the outgoing plane
wave, k and kx, with A and B at the grating surface. The right draws paths
through those openings from a common incident wavefront to a common outgoing
reference plane, using parallel far-field directions. The orange and green
segments show the outgoing and incoming extra lengths; signed values are
given separately. With delta_L defined as total route A minus total route B,
delta_L = d*(sin(theta_out)-sin(theta_i)), so k*delta_L is precisely
(kx-kix)*d. Normal incidence isolates the outgoing path difference first:
lambda=2 and d=4 micrometres give delta_L=lambda/2 at m_test=0.5 and
delta_L=lambda at m_test=1. A later tilted example includes both contributions.
The path view is allocated on first display and reused on subsequent calls;
it follows the main selected order while comparison is off, and the comparison
direction while it is on. It adds no independent parameter state or observers.
The explorer closes both displayed widget trees and all three figures. After
rerunning section 5, rerun section 6 to display the new explorer's path view.
No images, widget state or executed outputs are saved to the notebook.

Notebook 3 also follows the earlier notebooks' **Small calculation** format:
each activity has a short explanation, editable NumPy code, a prediction to
try, and a collapsible answer. Four cells reproduce the incident phase step,
one selected order's wavevector/field sample, the cell seam, and the
path-to-phase conversion. Section 4 links back to notebook two's opening
single-slit view before deriving the periodicity constraint.
Their written parameters are independent of all sliders.

`Spreading_Waves_and_Finite_Arrays.ipynb` holds the former section 7 and its
two calculations. It is unnumbered pending its later place in the series.
Run it independently from top to bottom; it imports NumPy and creates its
own `OpeningArrayExplorer` from `order_array.py`, with no `order_waves`
instance or previously executed notebook required. Its wavelength, spacing,
incidence and opening-count controls are independent of notebook 3.
The defaults are lambda=2, d=4 micrometres, normal incidence and N=1.
Reset restores all four; rerunning the setup closes the old explorer.
Its observer callbacks, image buffer, widgets and figure are released by
an idempotent `close()` call.

Start at normal incidence and change N from 1 to 2, 5 and 15. Opening
positions remain on fixed grating sites including x=0, with new neighbours
added on either side. The spatial window is independent of N.
The left plot sums prescribed outgoing cylindrical-wave approximations
exp(i*k*r)/sqrt(r), each driven by exp(i*kix*x_j). Regions within one wavelength
of any source are omitted because this is an approximation away from sources.
Signed fields are normalized to each snapshot's peak complex magnitude;
a symmetric logarithmic colour mapping makes weaker ripples visible.
The right curve is the far-field array factor squared, divided by N².
Integer order directions are marked for N>1. Single sources have an isotropic
response in this model; no finite-width slit envelope, coupling, polarization
or efficiency is assigned. Finite N gives peaks of finite angular width
and side lobes; the field is not replaced by ideal plane waves.
The notebook explains the distance, drive phase, propagation phase and
amplitude of each source contribution, image normalization, and contrast.
Its first small calculation adds two contributions at one point; its second
reproduces the directional curve. The former shared `OrderWaveExplorer.array_ui`
API remains available for older sessions, but notebook 3 no longer uses it.
All plots remain in memory and both notebooks are saved without outputs,
attachments or widget state.


`n_Diffraction_Order_Geometry.ipynb` is retained as a broader reference.
It uses `OrderFormationExplorer` from
`order_formation.py`. The default arrival view starts with two openings at
normal incidence, observed along the normal. It shows schematic paths to the
same distant point P, the real time traces of the individual contributions at
P, and their sum divided by N. The path sketch compresses the observation
distance; the phase calculation uses far-field path differences.
The lower plot shows the normalized coherent array intensity for 2–32
identical illuminated openings, excluding the single-opening envelope.
After the Fourier section, a separate optional instance with
`view='wavefronts'` compares a continuous plane wave with copies of one cell
carrying the incident Floquet phase offset. Its five-cell tiled field is a
compatibility illustration, not a scattering solution. The two explorers
have independent controls.
Lengths are in micrometres and wavelength is in the common exterior medium.
The transmitted coordinate +s corresponds to -z in notebook two.
Choose an order to follow its exact angle as parameters change, or move the
observation-angle slider to compare arrivals. Grazing directions are included;
if a followed order becomes evanescent, the angle is retained in free-angle
mode with a notice. Rerunning the setup cell closes both previous explorers before
reloading the module. Its figure and image buffers remain in memory.

`2_A_Plane_Wave_Meets_a_Grating.ipynb` starts with one isolated slit.
Section 1 uses `SlitKinematicsExplorer` to show the continuous outgoing
wavevector semicircle. Its independent wavelength and candidate-angle
controls select kx=k*sin(theta), kz=-k*cos(theta), with ky=0. Grey arrows
sample the continuum; the arc and fixed length k represent candidate
directions without assigning strengths. A small calculation checks the
components and length. The section introduces the negative-z angle convention
before the repeating pattern in section 2. Rerunning the cell closes the
previous explorer. Sections and seven small calculations are numbered in
reading order; notebook 3 links back to this introductory view.

The incident-wave view in section 4 uses `GratingSurfaceExplorer` from
`grating_intro.py`. Its three controls set period, wavelength in the medium
(both in micrometres), and signed incidence angle from -z toward +x. Incidence
is from z > 0 and transmission is toward z < 0, so normal incidence has
`k = (0, 0, -2*pi/wavelength)`. This angle is distinct from the shared helpers'
polar angle measured from +z. The explorer shows the incident field and
surface phase at `t = 0`, with openings of width `d/2`. The order preview
mirrors the shared +z directions across z = 0, preserving their tangential
components. Neither view propagates the masked field or calculates efficiencies.
The notebook closes the old explorer before replacing it; `close()` is
idempotent and releases its figure, callbacks, widgets, layouts and styles.
Its SVG diagrams and widget image buffers are generated in memory only.

The same notebook also uses `OrderSurfaceExplorer`, a 2D x-z view of
orders on a sphere and a plane at `z = -L`, with a horizontal projection
ruler below it. The ruler shares the spatial view's x scale and shows the
sphere markers at `x_sphere = R*kx/k`, including grazing orders. It is a
coordinate projection, not another observation surface. Wavelength, grating period and
signed incidence angle determine the allowed directions. Sphere radius `R`
and plane distance `L` set the reference surfaces. A reset button restores
the five-order example. The view keeps `ky = 0`, uses micrometres, and holds
axis limits fixed so that moving either surface is easy to compare.
Grazing orders appear only on the sphere; finite plane intersections outside
the displayed x range are identified in the readout and by continuing rays.
Markers label geometrical intersections, not spot sizes or powers.
The readout counts all real directions, then separates transmission into
`z < 0` from grazing directions. Written example counts and numerical answers
refer to their stated parameters; the controls supply the current counts.
`order_surface_intersections` supplies the geometry for real outgoing
wavevectors. Close the old explorer before replacing it.

`OrderSurfaceExplorer.components_ui` displays linked spatial, wavevector,
and spatial-frequency triangles in a later notebook cell, using the same
wavelength, period, incidence angle and radius controls. Its order selector retains the selected
order while it is allowed, includes grazing directions, and falls back to
`m=0` with a notice if the order becomes unavailable. Reset restores `m=+1`.
Each panel has a fixed scale in its own units; changing `R` changes only the
spatial triangle and changing `L` leaves all three triangles unchanged.
The frequency semicircle has radius `1/lambda`, with components `fx=kx/(2*pi)`
and `fz=kz/(2*pi)` in cycles per micrometre. Its axis limits remain fixed as
wavelength changes. Zero and grazing directions produce degenerate triangles
with explicit readouts.
The explorer owns and closes both displays, figures and the order observer;
redisplaying `components_ui` does not add callbacks or allocate a new explorer.
`order_component_svg()` remains available as a standalone default illustration.
The notebook connects the three triangles through the horizontal
direction cosine `alpha = kx/k = x_sphere/R = lambda*fx`, checks neighbouring-order
spacing with a small NumPy calculation, then compares with the plane's
tangent mapping. At fixed period, changing wavelength changes the direction-
cosine spacing `lambda/d` but leaves the wavevector step `2*pi/d` and
spatial-frequency step `1/d` fixed.
An optional pupil preview introduces an ideal sine mapping; no collecting
optical system or pupil propagation is simulated.

```python
from IPython.display import display
from wave_optics.widgets import PlaneWaveExplorer, ScalarPlaneWave3D
from wave_optics.widgets import DiffractionGeometryExplorer
from wave_optics.widgets import WavevectorCoordinateExplorer

wave = PlaneWaveExplorer(dimensions=2)
display(wave.ui)

wave_3d = ScalarPlaneWave3D()
display(wave_3d.ui)

geometry = DiffractionGeometryExplorer(500, 1000, 30, 30)  # lengths in nm
display(geometry.ui)

coordinates = WavevectorCoordinateExplorer()
display(coordinates.ui)
```

`n_Coordinate_Systems.ipynb` uses the coordinate explorer. Its wavelength
control is **vacuum wavelength in micrometres**, with refractive index supplied
separately. The same-side NA defines an ideal circular collection cone about
+z; backward directions are never counted as collected. The projection menu
switches among transverse wavevector, direction cosines, spatial frequencies,
and normalized pupil coordinates. Dragging preserves the camera across updates.
The numerical function `wavevector_coordinates` returns the same representations
without importing any plotting tools. Pupil position uses the ideal sine mapping.

Notebook one uses `include_phasors=False` for its 1D, 2D, and 3D physical-wave
explorers. It follows notebook 0's choice of plane waves and develops their
phase and wavevector geometry before handing over to a periodic surface in
notebook 2. Its separate phasor lessons have been removed. The reusable
`LocalPhasorExplorer` and legacy phasor stages remain available to existing
callers, but are not part of the numbered teaching route.

The wave explorers use unit amplitude and phase speed `c = 1` length unit per
time unit, with wavelength measured in the medium. Playback advances in `t/T`,
so every wavelength takes the same on-screen time per cycle. The physical
period is `T = wavelength/c`; shortening the wavelength still increases the
physical frequency. Changing wavelength holds `t/T` fixed and updates the
displayed physical time `t`.

Call `wave.close()` or `wave_3d.close()` before replacing an explorer. This
stops playback, disconnects callbacks and the clock link, and releases its
widgets, layouts, styles, and plots. Calling `close()` again is harmless.
The closed explorer is disposed; create a new instance to display it again.
The plane-wave notebook refreshes its local modules in dependency order in
its first code cell: numerical modules, plotting, widgets, then package
exports. This also recovers a kernel with older plotting or numerical modules
cached; refreshing widgets alone can fail when it imports newer functions.
The notebook uses `close_wave_explorer(previous)` when replacing an explorer. This
helper also releases older instances whose class did not yet have `close()`.
After a backend edit, rerun the notebook from its first code cell. Merely
repeating a `from ... import ...` statement does not refresh a cached module
or upgrade an existing instance.

Plots are rendered in memory. Keep saved notebooks free of image previews:
do not create PNG snapshots or image attachments, and remove saved widget
state containing image buffers before writing an executed notebook to disk.
Run the cells to display the interactive plots.

Wave renderers are also accessible through `wave.plot`, or directly as
`PlaneWavePlot` and `ScalarPlaneWave3DPlot` in `plotting.py`, without widgets.
The standalone HTML export retains rotation, zoom, and hover. Changing physical
parameters through the notebook controls requires a running Python kernel.
Exports default to `supplement/figures/` from either notebook directory; an
explicit relative export target is resolved from the current working directory.

After editing a backend module, restart the notebook kernel and run all cells
to use the updated definitions.

## Reusable two-wave observation-plane view

The previous draft's `TwoWaveExplorer('directions')` links four panels: a standard xz
side view (x horizontal, z vertical, equal scales), a face-on detector at z=-1 um,
the normalized wavevectors, and the detector's y=0 intensity profile. Crest planes
satisfy k.r + phi - 2*pi*t/T = 2*pi*n and appear as straight lines perpendicular
to the direction arrows in the side view. Finite patch edges are drawing
limits only; the calculation remains two broad constant-amplitude plane waves.
One arrow per plane wave represents its uniform propagation direction; the
arrows meeting at the detector do not imply focusing. Model time advances
the fronts while the detector intensity remains fixed. Amplitude, phase, and
half-angle controls update all related panels, with fixed intensity scale 0 to 4.
The wavevector panel displays Lambda=lambda/(2*sin(theta)) and its live value.
A Lambda ruler marks adjacent bright fringes when two fit inside the detector
view and both amplitudes are nonzero. The detector appears edge-on at z=-1 um
in the xz plot.
The four-panel layout uses the same phase-2-minus-phase-1 convention as the
archived draft. These views remain available for later teaching sections.

## Interactive performance

Explorers retain the latest fields and geometry for each set of physical
parameters. Time and observation controls reuse those arrays when appropriate;
changing wavelength, direction, amplitude or relative phase invalidates the
affected results. These caches belong to each explorer and are released by
`close()`, rather than collecting every slider position.

Linked views update only the panels affected by a control. In notebook 5,
detector distance changes the spatial detector view, order selection moves
highlights, and coordinate units change the Harvey view. Playback reuses
artists where possible. Uniform sampled fields are rendered as images with
their original sample spacing and fixed physical colour scales.

After a backend update, restart the kernel and run the notebook from the top
to replace existing explorer instances. Python callback benchmarks exclude
browser painting and communication time; their timings are not playback
frame-rate guarantees.

## Checks

Notebook 0_5 uses an ideal amplitude mask versus a physical absorber throughout:
`absorbing_grating.py` adds a passive complex-layer solver using the shared
scalar matching engine; `amplitude_grating.py` provides the consistently phased
ideal exit aperture and bounded thickness cases; `absorbing_grating_view.py`
provides the linked physical-field and coefficient views. Both exterior media
are air. The original dielectric solver and comparisons remain available.

```python
from wave_optics.absorbing_grating import AbsorbingGratingConfiguration, solve_absorbing_grating
from wave_optics.amplitude_grating import AbsorbingComparisonData
from wave_optics.absorbing_grating_view import AbsorbingGratingExplorer

solution = solve_absorbing_grating(AbsorbingGratingConfiguration())
print(solution.reflectance, solution.transmittance, solution.absorption)
print(solution.energy_balance)  # R + T + independently integrated A
cases = AbsorbingComparisonData(solution=solution)  # Reuse this solve; calculate six other thicknesses.
absorber = AbsorbingGratingExplorer(cases)
display(absorber.ui)
# absorber.close() before replacement
```

The centred feature in this geometry is the **clear opening**, with
`open_fraction = clear_width / period`. The ideal field includes exp(+i k0 h)
at the exit, preserving the BVP's incident phase at z=0. `power_balance` retains
its earlier meaning R+T; use `energy_balance` for R+T+A. See
[absorbing_grating.md](absorbing_grating.md) for conventions, analytical and
numerical acceptance results, analytical checks and limitations.

Run the shared test suite from the project root:

```text
python -m unittest discover -s supplement/tests
```

Run the standalone polarization checks from the `supplement/` directory:

```text
python -m unittest discover -s . -p test_polarization.py
```

`lattice_grating.py` supports Lessons 5.3, 5.4 and 7.4: reciprocal bases, unequal-medium negative-z branches, circular collection, shifted rectangular coefficients, and origin-corrected sampled 2D Fourier coefficients. It uses vacuum wavelength; sampled arrays use [y, x]. No material scattering is solved.

Chapter 5 now separates conical geometry (5.1), direction/frequency coordinates (5.2), two-period lattices (5.3), and angular sigma coordinates and the declared physical sine-pupil mapping (5.4). `lattice_orders` needs no objective and returns k, f, normalized components and propagation status. `collect_orders(data, NA_obj)` subsequently adds q and collection flags without dropping labels or source metadata. Explicit legacy NA_obj calls delegate to that step. `lattice_plotting.py` draws geometry on the three coordinate scales; detailed unit-cell excitation checks are in 7.4.

`field_propagation.py` and `field_propagation_view.py` retain the worked Chapter 8 experiment for future lessons in `supplement/Field_Propagation_Worked_Material.ipynb`: prescribed coherent order sums with positive distance s toward -z, reusing the conical outgoing branch. The independent explorer compares complex fields and intensity at two planes, and the phase-information comparison uses equal-intensity inputs. No scattering, FFT or lens solve is added.

Notebook 8.0 is a Markdown-only chapter introduction. 8.1 connects Helmholtz, Green and Kirchhoff; 8.2 develops RS propagation and the Weyl bridge; 8.3 compares ASM, Fresnel and Fraunhofer. The retained experiment helpers and physical tests remain available.

`weyl_reconstruction.py` supplies the focused Section 5 of 8.2. One "Build the sum" control moves from a single flat wave through increasingly dense weighted 3D Weyl sums, beside a fixed spherical target. Matching phase colours make the change from straight to curved fronts visible; intensity is not encoded. Quadrature counts and complex-field errors are optional details. Cached maps and sums keep repeated updates quick. Grating/cap helpers remain in `weyl_cap.py` for later lessons. See `supplement/Weyl_Reconstruction_Validation.md`.

`propagation_kernels.py` supports the current 8.3 with the full RS I spatial kernel (equivalent to full ASM), Fresnel and usual paraxial Fraunhofer kernels. The default Patch response view links a tiny boundary patch, outgoing full-kernel phase fronts and obliquity to the real oscillation contributed at P per unit patch strength. The two complex-weight views, magnitude profiles and unwrapped phase differences remain available on fixed scales. The plot is a section of 3D kernels; no finite aperture or exit field is calculated. See `supplement/Propagation_Kernels_Validation.md`.

`propagation_geometry.py` places compact live reference/observation-plane sketches in 8.3 Sections 1–4 and shares the same drawing with the summary explorer. They show the selected right triangle, full response weighting, exact versus quadratic extra path alongside full/Fresnel magnitudes, and the boundary phase omitted by Fraunhofer. `GeometryState` links the physical controls while each panel owns its widgets and detaches its observers on closure. Recreating one panel preserves the geometry and leaves the other panels active.

`propagation_fourier.py` adds 8.3 Section 6: an independently controlled plane-wave multiplier explorer and a Fraunhofer observation-to-frequency map. It reuses `scalar_propagation.py` for the full/Fresnel transfer functions, converting angular spatial frequencies to the backend's cycles-per-length convention. The new readout uses Fresnel minus ASM phase error, preserving whole turns. Fixed phasor scales retain evanescent decay; the Fraunhofer view assigns a frequency coordinate without prescribing a boundary field or spectrum amplitude. Both cells replace only their own panel and retain their local controls on rerun.

`propagation_comparison.py` retains the earlier field-comparison backend for a later lesson: a prescribed Gaussian-illuminated sinusoidal grating, full 2D angular-spectrum quadrature and analytic Fresnel/Fraunhofer fields at matching observation points. Its numerical validation remains in `supplement/Propagation_Comparison_Validation.md`; it is no longer the main 8.3 walkthrough.

`propagation_pictures.py` retains figure constructors and shared drawing styles. `write_illustrations` regenerates the three current shared SVGs: propagation relationships, boundary data and boundary consistency. Historical context is in Appendix 3; the 8.1 fallback figures are exported separately by `green_kirchhoff.write_static`. Superseded standalone SVG exports are excluded from the shared export list.

Course prose follows the [shared exit-field notation](../0_0_Foundations_of_Scalar_Optics.ipynb#exit-field-notation): `t` and its Fourier coefficients describe the prescribed mask; `U_exit` and `C_m` describe the output field. `U_ref` identifies a general reference plane. Backend identifiers remain compatible.

`exit_field_view.py` supports 9.1. `ThinMaskExitFieldExplorer` adapts the 2.2 `GratingSurfaceExplorer` to a fixed 10 µm field with period d (nm) and duty D (open fraction) controls. The readout derives open/closed widths, grating spatial frequency and reciprocal spacing G. Defaults d=2000 nm, D=0.5 and incidence 30 degrees give 1 µm open and 1 µm closed. Solid curves show the actual field; dashed curves repeat the central cell without its inter-cell Bloch factor. The same amplitude panel applies to both. `data['U_ordinary_pbc']` retains the comparison field. Both owned views share controls, mask and incident phase. The current complex field is available as `data['U_exit']`; phase is undefined (NaN) behind opaque strips. `close()` releases both views. Static periodicity and exact three-order checks are retained in the [Fourier companion](../supplement/Exit_Field_Fourier_Representations_and_Checks.ipynb); 9.1 follows only the interactive binary grating. No propagation or scattering solve is performed. Default 2.2 controls retain their original behaviour.

Notebook 5.4 now follows one mask-side lithography example through wavevectors, normalized angular sigma coordinates and an explicitly assumed physical sine mapping. Its extended LSI/reference-cap material is preserved in `supplement/Pupil_Mapping_Aplanatism_and_Imaging_Notes.ipynb`; shared numerical helpers are unchanged.

`pupil_view.py` owns the interactive 5.4 coordinate-chain plot. `SigmaPupilExplorer` links source sigma coordinates, mask pitches and physical pupil radius while keeping wavelength, index and mask-side projection NA fixed. It uses `lattice_grating` for order geometry and explicitly assumes the declared sine mapping for physical positions. `data` exposes current coordinates/collection flags; `close()` disconnects only its own callbacks and widgets. The main notebook contains the equations and a short display call.

## Coordinate notation

Teaching formulas use angular sigma = k_parallel/(k0 NA), Delta k = k_out - k_in for scattering-vector changes, and bar k_x = k_x/k0 for vacuum-normalized components. Lattice collection data expose angular coordinates as `sigma`; the existing `q` key remains a compatibility alias. Local FFT bin indices, layer-mode roots and beam-frame coordinates retain their explicitly defined symbols.
