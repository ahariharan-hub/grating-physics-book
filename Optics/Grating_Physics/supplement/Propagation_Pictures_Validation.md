# 8.1: Chapter 18 integration and verification

## Workspace cleanup (2026-11-01)

Removed obsolete standalone figure exports and temporary review/execution
artifacts after checking notebook links and figure metadata. The shared SVG
exporter now writes only propagation relationships, boundary data and boundary
consistency. Figure constructors, numerical helpers and tests remain available.
Earlier revision logs below retain the historical verification results; their
temporary screenshots and executed notebook copies are no longer retained.

Post-cleanup checks: all 46 main notebooks are byte-for-byte unchanged; all
64 main/supplementary notebooks pass schema validation, all 13 notebook image
links resolve, and the edited documentation has no unresolved local links.
The shared exporter produces exactly three XML-valid, byte-stable SVGs in a
temporary verification directory. The existing environment still imports
the numerical/widget packages and renders the 8.3 kernel panel. Verification
artifacts were removed; `.gitignore` excludes future caches and temporary files.
The retained propagation/interaction regression suite passed all 45 tests
(89.12 seconds), with bytecode writing and pytest's cache disabled.

## 8.2 follow-on: Rayleigh-Sommerfeld to angular spectrum (2026-10-31)

### Fixed-stage navigation and aperture clarity

Removed the unused stage selector from fixed-stage displayed widget trees instead of depending on display:none (the user screenshot showed the disabled selector). Fixed-stage panels now render only their relevant control rows. Cleanup explicitly releases owned controls outside the rendered tree as well, preserving failure/rerun cleanup.

In the patch side view, dense phase markers at s=0 made the opening look closed. Phase data remain in the front view; the side view now marks only selected Q, with screen end caps and a clear A opening bracket. All source patches are still integrated; numerical helpers and profile/phasor linkage are unchanged. Rendered centre, edge-row and narrow-opening states and visually inspected the gap. Existing 54 propagation/interaction tests passed, and five focused fixed-panel/failed-render checks passed after strengthening the regression tests. Actual 8.2 code cells executed and released every owned widget. No frontend UI inspection was attempted.



Rebuilt `8_2_Rayleigh_Sommerfeld_Propagation.ipynb` around the question of propagating known field values without independently guessing their normal slope. This work preserves the pre-existing 8.1 changes. RS I is the main route: mirror-probe difference, zero G on the complete plane, surviving derivative, full kernel, patch quadrature, and intensity. RS II is a brief additive-probe counterpart in the same existing boundary-choice explorer.

Retained both live 8.2 panels and their numeric backends. Default response is RS I; all three response choices remain available for comparison. Shortened the HTML captions so the plots are prominent. Replaced the copied long history/bibliography and broad approximation diagram with focused RS references and links to the 8.1 companion. Setup no longer regenerates unrelated static drawings.

The closing section distinguishes the burden of general analytic RS calculations/direct oscillatory quadrature from the theory itself. Weyl's expansion and the convolution theorem lead to the angular-spectrum multiplier exp(i kappa s), with the outgoing/evanescent branch retained. Full RS I and full angular spectrum have the same plane data and represent the same solution. The text explicitly recognizes FFT acceleration of RS as well, rather than claiming that RS is intrinsically inefficient or AS is a new approximation.

Source checks: Handbook Vol. 2, printed pp. 48-50 (RS mirror construction), 54-56 (convolution, Weyl, propagation factors and evanescence); Chew Lecture 35, section 35.1, especially Eqs. (35.1.9)-(35.1.11), confirms the exp(-i omega t) outgoing identity. Kept the source normalization 1/(4 pi) and inward normal; derivative at Q gives K_RS1=-2 partial_s g. Shen-Wang remains the full-kernel/FFT reference.

Validation: **42 existing propagation tests passed**. A fresh project-Python kernel executed the complete notebook, exercised all three Green-response choices, decoded the PNGs, checked default RS I, row-highlight invariance, amplitude-halving/intensity-quartering and isolated panel reruns without duplicates. Notebook schema, local links, reference anchors, equation delimiters and whitespace checks passed. Source notebook outputs remain empty. This is kernel/callback verification; no live frontend inspection was attempted.

## Current revision: a shorter main route (2026-10-31)

### Conceptual coherence pass

Read the short notebook as one continuous argument and checked its source/probe roles, geometry, normalization, normal direction, derivative location and transition from exact theorem to aperture prescription. Corrected the restored aperture schematic to use curved illumination from the same external P0 as the live examples; O is the coordinate origin, Q is green, P orange, and the plane is labeled A+B rather than reusing the full closed-surface label S.

The main text now states that the displayed illumination amplitude and unit-source Green normalization differ; free-space G(Q,P)=G(P,Q) connects the P-centred probe view to the Q-to-P aperture view; derivatives act at Q; actual field AND normal slope are required before approximation; and plotted weight magnitudes/scaling do not replace complex weights in the integral. The ending points forward to numerical aperture propagation in 8.2. No new section or live panel was added.

Verification: explicit scalar reciprocity and plane-normal derivative comparisons passed against the earlier planar helpers; all three current plot stages rendered; notebook schema, local links, equation delimiters, SVG XML and git whitespace checks passed. The revised aperture schematic was rendered and visually inspected. No frontend UI inspection was attempted.



Built on the existing working revision, preserving its numerical backend, HTML readouts, reference material, and earlier rendering fixes. The main notebook now contains three physical questions and three live panels (Helmholtz, Green response, Kirchhoff boundary weights), rather than seven live panels plus the full derivation/history. Main prose is approximately 610 words before the short reading list; the notebook has 10 cells instead of 24.

Moved the boundary-transformation explorer, shared-face cancellation, small-sphere limits, radiation-condition detail, reconstruction check, compatibility proof, full historical narrative/date index, static fallbacks, and complete bibliography to `supplement/8_1_Boundary_Mathematics_and_History.ipynb`. All four deeper interactive panels remain available there. Relative image/notebook links were adjusted for the companion's location. The main notebook retains three key references and links directly to the companion.

The historical narrative, date index and historical bibliography were subsequently transferred to `Appendix_3_Historical_Context_and_Original_Sources.ipynb`. That appendix is prose-only, without equations or executable cells. The companion retains boundary mathematics, all five code cells and the reading needed by its derivations; historical links from 8.0-8.2 now point to Appendix 3. Reference anchors and local notebook links were checked after the transfer.

Validation: both notebooks passed schema/local-link/internal-anchor checks and executed in fresh project-Python kernels from their own directories. The updated lifecycle test exercised and reran all seven panels across both notebooks without duplicate widgets. No numerical helper changes were made for this shortening, so previously recorded numerical checks remain applicable. The Helmholtz and Green plots now explicitly distinguish physical illumination U from P0 and the mathematical probe G centred at observation point P. Both updated plot exports were visually inspected; the existing geometry/scale and seven-panel rerun checks passed. No frontend UI inspection was attempted in this pass.

## Earlier revision: reliable readouts and physical questions (2026-10-31)

Baseline: `62957490881b96a63287e0edabeb99f77942b683`; earlier interaction reference: `f82d8b866a44045520beefb84be64acf91958fe4`. Inspected both implementations before editing. The starting working tree was clean.

### Rendering repair and frontend evidence

The old `_equation` assembled long dollar-delimited TeX strings inside `HTMLMath`; the user supplied a confirmed literal-TeX rendering failure. An isolated probe was prepared with explicit `\(...\)` delimiters and HTML line breaks outside the math. The intended VS Code notebook window was located and inspected through Computer Use. The currently open user notebook had unsaved changes, so the plan was to open the separate probe rather than overwrite that tab. The user stopped Computer Use with physical Escape before the probe could be run and its slider changed. No further UI actions were performed.

Consequently, neither a successful explicit-delimiter rendering fix nor initial/post-update frontend validation is claimed. The authorized fallback is implemented: fixed symbolic equations are in notebook Markdown, and `_owned_panel` uses ordinary `HTML` for short labelled numerical readouts. No readout contains dollar delimiters or TeX commands; HTML lines wrap naturally without an overflow/scroll container. This changes widget readouts specifically, not Matplotlib's math-text delimiters. The numbers now sit below the plot. Setup no longer regenerates unchanged illustration files on every run; checked-in fallback exports can be regenerated explicitly after figure edits. Detailed numerical balances are collapsed or in optional sections.

### Physical interaction and scope

- Main story: wavelength -> point separation -> Green identity/Kirchhoff boundary bridge -> closed boundary becoming A+B+C -> Kirchhoff's aperture prescription. Detailed face cancellation, small-sphere limits, closure-limit algebra, and analytic reconstruction remain live in the optional section. References and scientific distinctions are retained; RS remains in 8.2.
- Helmholtz emphasizes visible wavefront spacing and real/imaginary field oscillations when wavelength changes. The full three-dimensional curvature and selected-point balance remain available as a secondary inset/readout; no ray-only second derivative is substituted.
- Green's explorer restores a simple Distance action in addition to selecting P/Q, phase-front geometry, and one complex response arrow. Its amplitude axes stay fixed at +/-0.20 throughout distance=0.6..3.0, so magnitude changes are visible without clipping the allowed responses.
- Kirchhoff restores the earlier boundary-weight pattern: |G| and |normal derivative G|/k across the plane. Distance changes their magnitude/shape, opening width shades A, and eta selects Q. The curves agree with the earlier planar helpers at distance=0.6, 1.3, 3.0 within floating-point roundoff (maximum observed difference below 6e-17). The selected assumed-data arithmetic is a secondary inset after reveal; it is not an exact transmitted-field solution.
- Geometry and the source/probe distinction stay prominent. Closure expansion has a fixed overview plus a large near-P view; the detector remains fixed. Curved-surface derivatives, sphere quadrature, inward normals and finite-radius inner-sphere truth are unchanged.

### Verification

A fresh project-Python kernel ran the notebook and exercised all main and optional controls. Every relevant plot changed and decoded as PNG. Initial and updated readouts were checked to be ordinary HTML with no raw widget TeX. All seven actual cell bodies reran with no duplicate widget count. These are Python execution/callback tests, **not intended-frontend rendering verification**.

**Result: 68 tests passed.** Notebook schema, local links, SVG XML, deterministic static exports and git whitespace checks also passed.

Focused checks cover added distance controls, earlier-backend weight equivalence, wavelength-dependent spatial oscillations, and the renderer-independent HTML contract before and after updates. Existing full-gradient, complex reconstruction, sphere deformation, inner-limit, cleanup and 8.2 tests remain. Scientific PNG previews were inspected locally for prominent geometry, stable scales, labels, and clipping; serialization now includes tight padding. Actual final notebook width and slider behavior in VS Code could not be re-inspected after Computer Use was stopped.

## Earlier revision: live conceptual mathematics (2026-10-31)

Baseline inspected: `4a6815d6a92fb4690cf42e1689fdef1e200ff255`; interaction reference: `f82d8b866a44045520beefb84be64acf91958fe4`. The working tree was clean. This revision replaces the main static walkthrough with six fixed-concept panels placed immediately after their explanations. A seventh, independent panel owns the closed-sphere check. Static exports appear only in one collapsed supplementary fallback.

### What each control changes

- Helmholtz: P transverse and wavelength change the selected complex field, full spherical Laplacian, plotted balance, and numeric equation. Coordinates stay in fixed reference-wavelength units while k changes.
- Green response: P and Q move the connecting geometry and complex response together. Phasor axes stay at +/-0.12 so arrow length reflects magnitude rather than being normalized away.
- Green identity: four accounting steps highlight one cell, add its neighbour, cancel the sampled face terms, and leave S plus S'. A midpoint face term uses the actual 3D quantity (U grad G - G grad U).n times a representative face area; the opposite normal contributes its negative. The grid/face area are schematic mathematical accounting, not physical interference or an exact 2D discretization.
- Small sphere: epsilon changes the actual sphere and a fixed-scale local zoom, normal position, area, exact complex radial Green derivative, selected curve markers, both complex integrals, and numeric equation. The complete integral remains -U(P) at finite radius.
- Surface transformation: a parameterized closed contour deforms from the original sphere cross-section into the disk plus hemisphere. P and P0 stay fixed. Intermediate contours are labeled geometric schematics with no assigned quadrature value. At the endpoint R_C becomes available; fixed global/zoom viewports show real expansion, and the calculated closure integral and selected curve point update.
- Kirchhoff: opening width and selected eta change the aperture and selected region. Before reveal, transmitted field/slope are unknown while the incident data are shown. Revealing retains Ui and hi in A, sets both to zero in B, and updates both complex boundary densities on fixed axes. These are prescribed densities per unit area, not a computed transmitted diffraction pattern.

All panels share the established 3D response and full-gradient helpers and the owned PNG/ipywidgets lifecycle. Unrelated controls are not created in a panel. Inner-radius and closure reference curves are cached independently of point highlighting. The planar RS kernels, numerical aperture sum and 8.2 notebook are unchanged.

### Verification and limits

- **65 tests passed:** preserved numerical physics/8.2 regression checks plus focused tests for actual mathematical changes, stable scales, relevant controls, fixed detector position, region crossing, finite-radius inner invariance, PNG rendering, failure cleanup and seven-panel reruns. The focused nine interaction tests were rechecked after equation-display changes.
- A fresh project-Python kernel ran the complete notebook. Its acceptance cell exercised every required control, verified the plotted PNG changed each time, checked A/B transitions, and decoded the images. All seven actual panel cells then reran with the same widget count. Callback timing in that run: median 0.310 s, maximum 0.398 s (local measurements, not a frontend performance guarantee).
- Rendered and inspected 14 representative interaction states, including both face-accounting endpoints, both epsilon extremes, intermediate/expanded closures, and A/B prescriptions. Figures use fixed amplitude scales and fixed closure viewports; labels, legends and selected points were inspected. Notebook execution and callbacks were exercised programmatically; no live browser session is claimed.
- Existing analytic reconstruction accuracy, curved-surface normals, full 3D Helmholtz residuals, and quadrature convergence checks remain in the suite. Static SVG/XML, notebook schema/local links, deterministic export and whitespace checks are also verified.

## Earlier revision: linked Green / Kirchhoff geometry (2026-10-31)

Baseline: `f82d8b866a44045520beefb84be64acf91958fe4`; the working tree was clean. Inspected 8.1, the existing Green/planar helpers, pictures, 8.2, and notebook lifecycle tests. This entry supersedes the older feature counts below.

### Scope and teaching sequence

- 8.1 now follows an external physical source P0, observation point P, and boundary point Q through one six-stage explorer with a synchronized equation and explanation area. Its lower panel is one explicitly labeled closed-sphere reconstruction check, separate from the physical-screen model.
- Stages: full spherical Helmholtz balance; free-space Green response; Green identity with shared-face cancellation; exclusion-sphere signs and separate limits; hemispherical closing surface at fixed detector distance; Kirchhoff's assumed aperture data.
- Seven original SVGs preserve the six default states and reconstruction check without a kernel. The existing aperture/vector schematic is retained as the destination. Internal grid cells and the enlarged hole in C are schematic cross-sections of 3D surfaces. Closure and inner-sphere curves are genuine quadrature results, not drawn trend lines. Phasor curves are actual complex ring contributions, not invented arrows.
- Fourier/imaging history retains its names/references but its equations have been removed from the historical introduction. RS and its aperture patch sum remain in 8.2; their numerical backend has not been changed.

### Conventions and numerical model

Coordinates are (x,y,s), with s=-z; exp(-i omega t); k=2 pi in wavelength units. Physical source P0=(0,-0.4,-3), observation point P=(0,y,1.3), sphere centre=(0,0,1.5). The bounded controls keep P inside the sphere, the physical source outside, and the exclusion sphere inside the outer surface. U=exp(ik r0)/r0 is a source of strength 4 pi; G includes 1/(4 pi).

All curved-surface derivatives use the full Cartesian Green gradient dotted with the appropriate unit normal. Outer-sphere normals point inward; exclusion-sphere normals point away from P. The existing planar normal helper is not reused for arbitrary normals. Sphere quadrature uses 48 Gauss-Legendre polar nodes by 96 uniform azimuth samples. Inner-radius curves use 24 by 48, with the selected point independently evaluated at 48 by 96. Closure curves use 80 by 160, selected-radius values 96 by 192. Reconstruction reference is evaluated independently at P.

At y=0.3 the analytic complex reference is -0.14250549901680948 + 0.1799422733500358j. Outer radii 1.5, 2.1, 2.8 give relative complex errors 5.34e-14, 8.63e-14, 1.69e-13. Complete inner-sphere errors relative to -U(P), at epsilon=0.04, 0.2, 0.6, are below 1.1e-14. The total is correct at finite radius: the separate terms, not the total, approach their limiting values. These are representative numerical errors, not universal precision guarantees.

At closure radii 4, 6, 14, the combined C contribution has relative magnitudes 0.212675, 0.107233, 0.021271 for the unobstructed reference field. It tends toward zero with expanding mathematical closure; the detector remains at s=1.3. This check does not substitute an unobstructed wave for the unknown field transmitted by a real screen. Stage F makes that model change explicit.

### Verification

Numerical tests cover the full gradient, Cartesian Helmholtz residual, complex reconstruction and sphere deformation, off-axis source/observation geometry, finite-radius inner signs and limiting terms, quadrature refinement, closure convergence, control synchronization, PNG decoding, extrema, failed-render cleanup and widget lifecycle. Existing RS/8.2 tests are retained, with notebook lifecycle targets updated to 8.2 following the already-authorized relocation. Figure previews were inspected locally; no claim is made that a live notebook browser frontend was operated.

Results: 16 new tests passed (15-test full run plus the added real notebook-cell rerun check; all six extreme-stage renders rechecked after layout edits); 42 existing propagation/8.2 tests passed. Both notebooks executed in fresh project-Python kernels; every 8.1 stage produced a decodable PNG. Notebook schema, local asset links, SVG XML, equation delimiters, deterministic SVG regeneration, and git whitespace checks passed. No saved notebook outputs were added.

## Earlier revision: history, physics, mathematics, experiment (2026-10-31)

Baseline inspected: `f9f2d9c30a9985b940028c5de5d10a9f28338966` (HEAD at the start of this revision; working tree was clean). Read 8.0, the current 8.1, both supporting helpers and the earlier validation log before editing. Notebook 8.0 and unrelated repository files were left unchanged. The entries under "Earlier revision log" below are historical records, not the current feature count.

### Teaching structure and scope

The main route now follows history -> common aperture/screen geometry -> point response and boundary mathematics -> linked numerical experiment -> brief representation/approximation outlook. The main narrative is approximately 1,120 words including the optional Lucke reading note, compared with 1,479 words in the baseline (equations, image alternatives and bibliography excluded). Detailed compatibility reasoning, the sine-curve illustration, and the Buchwald/Yeang notation crosswalk are retained in a linked optional extension.

- The historical figure links dates/authors, geometries, questions and mathematical ideas. It remains separate from Figure 3's modern equivalence/approximation diagram.
- The main geometry now includes an observation screen. The time convention remains exp(-i omega t); rightward transmission has positive distance s=-z, and partial_n=partial_s=-partial_z.
- A compact table distinguishes Kirchhoff's exact theorem/aperture prescription from independent RS I field or RS II derivative data. Compatible field and derivative data may coexist; independently guessed data need not be compatible. The full complex RS I kernel is retained.
- The optional Lucke note is next to this discussion. Checked the local paper's printed pp. 194-197 and 199-204 (PDF pp. 3-6 and 8-13, including download cover); visually inspected Fig. 3. Section 3's calculated disk limits are RS intensity -> 0 and Babinet-regularized FK -> 1/4 as distance/radius -> 0. A Fresnel extrapolation of unity is outside its validity there. "Exact" retains scalar/short-wavelength assumptions; section 4.2 removes evanescent components and uses stationary phase, as does the 3D stationary-phase derivation in Appendix B. None of these reduced-kernel qualifications replaces our full kernel or the full outgoing/decaying spectral equivalence. No main Poisson-Arago section or disk visualization was restored.

### Linked aperture calculation

`patches` and `screen_profile` share one full RS I midpoint-quadrature implementation, including both source-plane coordinates. The patch exercise links aperture phase, side geometry, row phasors, and a calculated screen cut at x=0. The marked intensity is the square modulus of the selected phasor sum; it agrees with the cut at that y. Moving P or selecting a row reuses the cached profile; selecting a row cannot change the numerical field or curve.

The default source has 41 x 41 midpoint patches; the displayed cut covers -8 <= y/lambda <= 8 at spacing 0.05 lambda. Intensity is always |U|^2 relative to a fixed unit incident-field reference, without per-curve peak normalization. The amplitude control changes this field relative to the same reference, so halving amplitude quarters intensity. The phase-ramp explanation avoids assuming a rigid shift for the full, potentially nonparaxial model. Three prediction/observation/explanation exercises cover symmetry, distance and phase. Curves are identified as numerical predictions, separate from proposed optical measurements and cited measurements.

### Verification

- Dedicated backend suite: **42 passed**. Includes Green normal derivative and Helmholtz residuals, independent Gauss-Legendre checks, symmetry, phase-ramp reflection, amplitude scaling and zero field, invalid inputs, actual curve/marker/phasor agreement, no row/point reintegration, all 16 corners of visible distance/position/opening/phase controls, real PNG decoding, failed-render cleanup, and isolated/full cell rerun lifecycle checks.
- After correcting the layout of nearly vertical phasor chains and moving the cut legend outside the data, **4 relevant rendering/linkage tests passed again**. Square, equally scaled real/imaginary axes avoid unreadable narrow boxes.
- Executed the complete notebook from fresh project-Python kernels using nbclient from both the project root and supplement. Exercised all three experiments, choice changes, selected-point/marker agreement, amplitude scaling, PNG decoding, and a full notebook rerun with stable widget-model counts and released old panels. No error outputs. This checks kernel callbacks and image payloads; a live browser frontend was not operated.
- Notebook schema, code compilation, Markdown/table/math parsing, all local image paths, reference anchors and diff whitespace checked. Source code outputs remain empty. SVG regeneration was checked for XML validity and byte stability; changed SVGs are canonicalized without rewriting untouched diagrams, and unmanaged static figures are cleared after serialization.
- Rendered and visually inspected the revised timeline and geometry, default and phase-ramp linked plots, a range-edge/high-amplitude case and a zero-field case. SVGs remain native vector diagrams; dynamic views use portable PNG widget images.

At unit prescribed amplitude, the following refinement results use the fixed incident-intensity units. The aperture comparison uses the whole displayed cut; the screen-spacing comparison checks intervening samples against a finer 0.025-lambda cut at 81 x 81 aperture samples.

| s/lambda, width/lambda, phase slope (rad/lambda) | max intensity error: 41^2 vs 161^2 | 81^2 vs 161^2 | 0.05 vs 0.025 screen-grid interpolation error |
| --- | ---: | ---: | ---: |
| 3, 3, 0 (default) | 0.004959 | 0.001014 | 0.003691 |
| 0.8, 6, 0 | 0.009104 | 0.001838 | 0.002124 |
| 0.8, 6, 4 | 0.010636 | 0.002122 | 0.003746 |
| 8, 6, 4 | 0.004839 | 0.000989 | 0.000661 |
| 8, 0.8, 0 | 0.000000616 | 0.000000126 | 0.000000255 |

The refined aperture errors decrease in all five cases. These are representative/extreme checks, not a uniform error bound over every continuous slider setting. Errors scale with amplitude squared. The finite displayed cut does not collect all transmitted power, and this activity makes no claim to be an energy-flux or polarization calculation. It is an educational finite quadrature (not adaptive, and not a numerical s->0 boundary-limit test); the camera comparison assumes the boundary-field model is physically adequate. No experimental dataset was fitted or presented as the calculated curve.

### Changed files and reproduction

- `8_1_Physical_Pictures_Behind_Scalar_Field_Propagation.ipynb`
- `wave_optics/propagation_interactive.py`
- `wave_optics/propagation_pictures.py`
- Historical timeline export, now superseded by the tables in Appendix 3.
- `supplement/figures/8_1_boundary_data.svg`
- `supplement/tests/test_propagation_pictures.py`
- This validation note.

Run the regression suite from the project root:

```powershell
.venv\Scripts\python.exe -m pytest supplement/tests/test_propagation_pictures.py -q
```

For the lesson, select the project Python kernel, restart it and run all cells. Each embedded experiment also supports independent reruns and closes only its own previous widgets. Fresh-kernel automation uses nbclient and ipykernel; the notebook itself uses the existing NumPy/Matplotlib/ipywidgets/IPython environment. Diagnostic files were temporary and are not deliverables.

---

## Earlier revision log (retained)

Updated 2026-10-30. The notebook now follows four physical questions: specified reference data and observation field; weighted elementary responses; the equivalent plane-wave/Fourier representation; and approximation/validity. Historical attribution is retained rather than replaced by the textbook narrative.

## Main reference consulted

Wolfgang Singer, Michael Totzeck and Herbert Gross, *Handbook of Optical Systems*, Volume 2, *Physical Image Formation*, edited by Herbert Gross (Wiley-VCH, 2005), Chapter 18, "Scalar Diffraction." The supplied PDF's title/copyright pages identify the authors, year and ISBN 3-527-40378-7. The book is now reference [1]; the eleven existing references remain, renumbered [2]-[12].

Consulted printed pages (the PDF page number is the printed page plus 26):

- Section 18.1, pp. 42-43: motivation, available descriptions, prescribed boundary values versus solving a physical material problem.
- Sections 18.2-18.3, pp. 44-50: boundary reconstruction, radiation condition, Kirchhoff prescription and the distinct RS I/II boundary data.
- Section 18.5, pp. 52-54: reinterpretation of a mathematical probe as weighted secondary contributions, full directional kernels and convolution; Figure 18-7.
- Section 18.6, pp. 54-57: convolution-to-multiplication, Weyl representation, plane-wave components, phase/decay; Figure 18-9.
- Section 18.8, pp. 60-63, and section 18.10, pp. 69-70: quadratic phase/amplitude approximations, additional far-field approximation and dependence of validity on geometry and permitted phase error.

The figure pages and relevant approximation pages were rendered and visually inspected. Physical concepts were paraphrased; no book figure was copied. Intermediate book operator labels/prefactors were not transcribed into the notebook. The established negative-z convention and the exact prescribed-field RS I/angular-spectrum equivalence remain explicit.

## Files changed in this revision

- [Notebook 8.1](../8_1_Physical_Pictures_Behind_Scalar_Field_Propagation.ipynb): about 806 explanatory/caption words excluding references, link destinations and mathematical expressions; one displayed equation. Four question-led sections integrate the main book reference with original historical references. Debye remains a short near-focus outlook; the diffraction lecture remains supplementary.
- [Plotting helper](../wave_optics/propagation_pictures.py): paired reference-surface S / observation-point P sketches show the probe and secondary-response viewpoints. The spectral sketch shows two plane-wavefront families between the same reference/observation geometry. The relationship diagram explicitly names real-space convolution and Fourier-space multiplication.
- The historical timeline graphic (now superseded by the tables in [Appendix 3](../Appendix_3_Historical_Context_and_Original_Sources.ipynb)) and [relationship SVG](figures/8_1_propagation_relationships.svg) were regenerated as original vector assets. Historical chronology remains separate from mathematical equivalence and approximation arrows.
- This validation note. Notebook 8.0, numerical propagation helpers and historical source content were not changed by this integration.

## Verification

The single figure-generation cell executed in a fresh Python namespace. The notebook passed Jupyter schema validation. Markdown rendering confirmed two image references and one displayed equation. All twelve numbered reference anchors and local links resolve. The explanatory text remains within 650-900 words.

Both final SVGs were rendered at normal notebook width (900 pixels) and inspected for legibility, overlap, clipping, reference/observation geometry and propagation-arrow versus wavefront distinctions. Wavefront segments in the spectral sketch were explicitly bounded to keep them clear of labels. SVGs contain no embedded raster images; font outlines avoid font substitution. Saved code outputs remain empty, so the Markdown-linked static assets provide readability without a kernel.

The physical review retained the distinction between producing boundary data and propagating it, Kirchhoff's theorem and approximate aperture prescription, point-source Green response and planar kernel, full versus reduced propagation, and paraxial versus near-focus assumptions. No FFT, field propagation simulation or sampling study was added. Whitespace checks passed; temporary source-page renders and authoring files were removed.

Run 8.1's one code cell to regenerate the two tracked figures. Original source-check details from the first version remain represented in its retained bibliography: Huygens's preface/Chapter I, Fresnel's Academy record, Fraunhofer's telescope discussion, Buchwald-Yeang's historical notes, Debye/Weyl publisher records, Booker-Clemmow's summary/section 3, Chew's Weyl identity lecture, the supplementary diffraction slides, Shen-Wang's full kernel/multiplier relation and Sheppard's Debye validity discussion.

## Green-function intuition illustration

At the user's request, a third compact SVG was added beside the intuitive Green-function explanation: one unit oscillating source, spherical wavefronts and two complex response phasors. The illustrative phasors agree with the drawn free-space geometry (nearer point at two wavefront spacings, farther point at three-and-a-half): the farther response is weaker and opposite in phase. No propagation simulation is added. Figure numbering and the regeneration cell were updated. The notebook remains within the requested word range; its schema, rendered Markdown image links and local paths passed checks, and the new SVG was visually inspected.


### Boundary Green-function comparison

Added an original three-panel SVG comparing the free-space Kirchhoff probe, the opposite-sign image construction for RS I, and the equal-sign image construction for RS II. The graphic connects cancellation on the plane to the required boundary datum, while distinguishing the valid Kirchhoff theorem from its approximate aperture prescription. Mathematical probes and the outgoing normal are explicitly identified. All four SVGs regenerate successfully; the new figure was rendered and visually inspected. Notebook schema and image paths pass; explanatory text remains within the 650-900-word target (883 words).


Moved the boundary Green-function comparison into Section 2, after the unit-source response and before the surface-patch construction. Added an explanation of equal path lengths, image signs, and which unknown boundary-data term disappears. Renumbered the figures in reading order. Notebook schema, image paths, and section placement checked; narrative remains below 900 words.


### Interactive Section 2

Added a three-stage ipywidgets explorer: point response, boundary choice and square-aperture patch sum. Full RS I integrates both transverse dimensions using 41 x 41 midpoint patches. The selected phasor segment represents a complete row. Verified the Green normal derivative by finite differences, aperture symmetry and quadrature convergence (default field differs by 0.00141 from 161 x 161 patches). Exercised widget callbacks and visually reviewed rendered plots. Static figures retained; schema and code compilation pass. Narrative remains within 900 words.


Removed the static boundary-construction Figure 3 from the notebook, retaining its interactive replacement. Renumbered the relationship diagram and its reference to Figure 3. Expanded the three experiment instructions to explain controls, expected observations and links to boundary terms and complex superposition. Schema, remaining image paths and figure references checked.


Connected the Helmholtz PDE to the point-source Green equation, boundary identity, and full RS I integral in Section 2. Defined both transverse integration coordinates and the exact kernel, identified the prescribed square-aperture phase ramp and 41 x 41 midpoint quadrature, and distinguished boundary integration from a volume PDE solve. Boundary recovery is qualified at continuous points. Schema, Markdown/math parsing, image paths and code compilation pass. Narrative: 900 words.


Added explicit xi/eta source-plane labels and a front view of the square aperture to the interactive patch experiment. Selected rows link to the side view and phasor chain; the side view identifies x=xi=0, Q=(0,eta,0), P=(0,y,s), distance s, and separation y-eta. Other modes identify observation coordinates too. Added coordinate definitions before the RS I integral. Visually reviewed the three-panel plot, exercised stage/row/distance/response callbacks and rendered parameter extremes. Notebook schema passes.


Restyled Figure 3 with a centred title, equal mirrored full-representation boxes and connectors, a centred Fresnel/Fraunhofer chain, and a separate full-width Debye outlook. Regenerated the SVG and visually inspected the rendering for alignment and label clearance.


Expanded the Boundary choice explanation beside the interactive plot: curve definitions, normal-derivative scaling, boundary-integral weights versus full products, Kirchhoff/RS cancellations and the geometric peak. Renamed plot titles, axis and legend to identify Green-function weight magnitudes. Exercised all response callbacks and rendered the revised plot.


### Backend robustness audit (2026-10-31)

- Added explicit finite-real, positive-distance/wavenumber/aperture, quadrature-count, stage, response and row-index validation. Invalid inputs raise ValueError rather than silently selecting an experiment or row.
- Replaced output-context rendering with a single SVG Image widget. Each update replaces its bytes and clears the unmanaged Matplotlib figure after serialization.
- Added an owned PropagationPanel.close() that detaches callbacks and releases child widgets, layouts and styles. Closing is idempotent; initial-render failures also release their widget tree. The notebook closes its previous panel before recreating it.
- Dedicated tests: 25 passed, including finite-difference Green normal derivative and Helmholtz residuals; integrated-field Helmholtz residual; mirror symmetry; image cancellations; midpoint convergence; independent 100-point-per-axis Gauss-Legendre reference integrals for four representative/extreme configurations (absolute complex-field error below 0.004 for the default 41 x 41 quadrature); all 16 corners of distance/position/opening/phase controls rendered to SVG; invalid inputs; three notebook-cell reruns without growing widget counts; unrelated-widget survival; failed-render cleanup.
- Existing adjacent propagation tests: 13 passed (13 subtests).
- Executed the complete notebook through nbclient in actual fresh project-Python kernels from both the project root and supplement. Exercised every experiment and reran the interactive cell in each kernel; no error outputs. The temporary executed copies were removed after verification; the source notebook remains unexecuted.
- This certifies Python/kernel execution and SVG generation. Live frontend behavior still requires an ipywidgets-capable Jupyter frontend; no browser frontend was tested in this audit.

Reproduce backend regression checks from the project root:

```powershell
.venv\Scripts\python.exe -m pytest supplement/tests/test_propagation_pictures.py supplement/tests/test_field_propagation.py supplement/tests/test_scalar_propagation.py -q
```

The notebook runtime uses NumPy, Matplotlib, ipywidgets and IPython. Tests additionally use pytest and nbformat; fresh-kernel verification uses nbclient and ipykernel. These execution tools were installed in the project virtual environment for this audit. The 41 x 41 midpoint sum remains an educational finite quadrature, not an adaptive solver or an exact screen-scattering solution.


### Broken-image regression fix

The previous audit verified SVG bytes but missed the Image widget frontend format/MIME mismatch. Replaced Image(format=svg) with portable PNG serialization and added PNG-format/signature plus image-decoder verification. Checked actual widget image payloads in all modes, visually inspected the point-response payload, and executed all modes plus the reload/rerun path in a fresh kernel. The notebook interactive cell now reloads the helper after closing its previous panel, so rerunning that cell loads the corrected renderer without a kernel restart. Dedicated backend suite: 25 passed.


### Boundary-data comparison and Poisson-Arago connection

Added a domain map, an inward-normal Green boundary identity, and an equation table distinguishing the exact Kirchhoff representation from its overprescribed aperture model and RS I/II independent boundary data. Explained zero Cauchy data on an open screen patch and its incompatibility with nonzero aperture illumination. The interactive boundary choice now reports its assumed opening/screen values. Added a separate disk illustration and a computed Fresnel/Babinet intensity profile, explicitly distinct from the full RS I square experiment. Supplementary Cooper, Feynman and experimental demonstration references are linked beside claims; the handbook remains the main physics reference.

The axisymmetric Fresnel calculation integrates angle analytically (J0) and radius by Gauss-Legendre quadrature. Tests verify its analytic on-axis field, unit normalized axial intensity, transverse symmetry, convergence and vanishing-disk limit. Dedicated suite: 26 passed. Fresh-kernel notebook execution, all boundary-choice PNG updates, reference anchors, local image paths and Markdown/table/math parsing pass. Both new illustrations were rendered and visually inspected. Narrative word count, excluding equations and references as before: 896.


Removed the Poisson-Arago section, its notebook figure and its two dedicated references at user request. Removed the disk illustration from notebook figure regeneration. Boundary-data comparison and the Kirchhoff/RS explanations remain. Notebook schema validates.


Added a mode-to-equation table and a grouped patch-sum equation matching blue row segments, orange selection and green U(P). Explicitly identified x=0 observation geometry and integration over both source coordinates. Interactive captions and plot titles distinguish point response g, boundary weights and propagated field U. Both tables and math parse correctly; notebook schema passes. Targeted regression checks confirm plotted phasor increments/resultant equal actual RS I row contributions and notebook reruns still clean up their widgets: 2 passed.


### Stepwise lesson and vector geometry

Split Section 2 into six named stages: coherent addition, one point response, Kirchhoff boundary identity, aperture-data conflict, RS boundary-data choice, and full RS I reconstruction. Each experiment is embedded immediately after its explanation, with its stage selector hidden and fixed. Independent controls retain explicit owned cleanup; full setup closes all previous notebook-owned panels. Added tests for three independent experiment reruns and isolation: dedicated suite 28 passed.

Replaced the domain schematic with an original aperture/vector construction showing incident wave, opening width 2b, O, Q, P, r-prime, r, R=r-r-prime, inward normal and angle theta in notebook coordinates. Rendered and visually inspected it. The new Totzeck (1991) reference is explicitly distinguished: its two-dimensional cylindrical/slit setting is not substituted for this notebook's three-dimensional Green function. Schema, local image paths and Markdown/math parsing pass.


Rewrote sections 2.3-2.4 to introduce boundary value versus normal spatial slope before the integral. Added a two-question symbol table, a small-normal-step relation, underbraces identifying both weighted integral terms, explicit derivative location and inward-normal sign, and an opening/screen assumption table. The example U=C sin(ks) demonstrates zero boundary value with nonzero slope and is explicitly distinguished from the outgoing aperture solution. Replaced unexplained Cauchy/continuation jargon with a direct overconstraint explanation and a transition to RS I/II. Schema and both table/math renderings pass; computational backend unchanged.


Checked the full local 49-page Buchwald/Yeang PDF: sections 5, 6 and 8, with visual inspection of the historical surfaces and proof pages. Added a notation crosswalk distinguishing the paper's illuminated B / shadowed C faces from the notebook's opaque planar B region, and explicitly distinguished the modern sine-curve illustration, Poincare's original Green-integral argument, and the later boundary-recovery test. Added precise PDF-page locators to reference 5. Notebook schema and Markdown parsing pass.


### Compact historical table

Replaced the historical timeline illustration in the lesson with a six-row Markdown table of dates/developments, physical questions and main contributions. Removed its notebook image reference and default figure-regeneration entry, retaining the legacy drawing function/asset. The aperture geometry, source-response illustration, modern relationship diagram and interactive views remain. Renumbered the two remaining numbered figures and the supplementary-reference cross-reference. Notebook schema, history table parsing and helper compilation pass.


### Abbe and Hopkins in Section 1

Added Abbe (1873) chronologically to the history table, with pupil selection/recombination as the bridge from grating components to image formation. Added a short Hopkins (1953) outlook toward partially coherent image intensity, explicitly outside this lesson's single prescribed-field propagation calculation. Added references 16-17 to the original publications and a modern ZEISS explanation. Notebook schema, seven-row table parsing and reference-anchor checks pass; numerical helpers unchanged.


### Four intersecting historical threads

Replaced the small Section 1 timeline/table and the isolated Hopkins outlook with four compact date/contributor/contribution tables: physical interference; boundary representations; plane-wave/Fourier representations; and imaging/focusing systems. Included all names in the consolidated discussion, with selected contribution/publication dates and repeated Born/Wolf/Sommerfeld entries where their roles differ. Clarified that interference underlies all groups, Sommerfeld's 1896 half-plane and later lecture formulations differ, Collins is paraxial, and vector/partial-coherence developments are outlooks for this scalar lesson. Added references 18-26 to the previously checked original papers/books and publication records. Four-table rendering, reference anchors and notebook schema pass; numerical helpers unchanged.


### Foundational learning priorities

Added subsection 1.5 at the end of Section 1: core names, the concepts to understand and their motivation; a teaching sequence from superposition through boundary mathematics and RS prescriptions to the plane-wave bridge; and the role of later imaging/focusing/coherence developments. Explicitly distinguished the teaching route from historical succession and connected it to the next physical section. Notebook schema, all five Section 1 tables and reference anchors checked; backend unchanged.


### Motivational transition into the physical picture

Replaced the formal subsection 1.5 priority table with the user-approved motivational prose and light humor. Kept the four historical thread tables and used the closing passage to lead into the one-patch/one-observation-point geometry. Notebook schema and Markdown rendering checked; numerical content unchanged.


### Motivational Section 1 and compact history reference

Rewrote Section 1 around four reader-facing questions, retaining the approved motivational tone and restrained humor. Replaced the four large tables with one four-row learning map. Preserved all previously listed names, selected dates, contributions and citations in a linked optional reference near the end, grouping its original 30 rows into 16. The narrative leads from one patch and phase to the later boundary calculation, rather than asking the learner to memorize a chronology. Schema, table rendering and all reference/extension anchors checked; numerical helpers unchanged.


### Representative equations for Section 1

Added one equation block to each of subsections 1.1-1.4: coherent intensity/cross term; Helmholtz and the inward-normal boundary representation; full angular-spectrum transfer with outgoing/decaying branch; and a scalar optical-system kernel plus averaged image intensity. Each block defines its new symbols and supplies a plain-language reading. Identified the equations as modern signposts, kept vector/high-aperture and paraxial Collins roles distinct, and explained intensity averaging for partial coherence. Notebook schema, four display-math blocks, the compact table and reference anchors pass; numerical helpers unchanged.
