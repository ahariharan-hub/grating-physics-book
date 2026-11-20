# Notebook 5: Grating diffraction in three dimensions

Reference review: 20 September 2026. Public source links checked on 22 September 2026; sigma-space sources checked on 23 September 2026. This source and teaching map accompanies
[Chapter 5](../5_0_Grating_Diffraction_in_Three_Dimensions.ipynb). The source review began with the original unsplit lesson on 22 September 2026; the revised four-lesson route is recorded below.

## Purpose and priority

Central question: **What changes when the incident wave has a component along the grooves?**

The essential learning outcome is to extend the order condition from notebooks 3 and 4 to three dimensions, explain conical diffraction, and distinguish a physical propagation direction from its coordinates in a diagram. Harvey's representation makes those directions easier to read. Lesson 5.3 adds a second periodic direction. Lesson 5.4 then distinguishes angular acceptance from the optical condition needed for physical pupil mapping; sigma terminology is confined to its optional lithography application.

The first pass should retain the series' assumptions: an indefinitely periodic linear grating, uniform grooves, one incident plane wave, and the same homogeneous, isotropic, lossless exterior medium on both sides. Direction constraints follow from symmetry and the exterior dispersion relation; calculating scattering amplitudes needs additional physics.

## Verified reference material

All external sources below have public web links. PDF page numbers are one-based positions in the copies reviewed; printed page labels, sections and equation numbers remain the more reliable locators when cover pages differ. Relevant text and the principal diagrams were inspected during the reference review. The 1998 paper's publisher page is public, although full-text access may require a subscription.

### 1. Harvey and Vernold (1998): the original direction-cosine treatment

**Description of Diffraction Grating Behavior in Direction Cosine Space**, Applied Optics 37, DOI [10.1364/AO.37.008158](https://doi.org/10.1364/AO.37.008158).

[Publisher article page](https://opg.optica.org/ao/abstract.cfm?uri=ao-37-34-8158).

- Inspected: local PDF pages 2-4, carrying printed labels 8158-8160; article begins partway down the first of those pages.
- Figures 1-3 connect planar and conical diffraction, the observation hemisphere, and the direction-cosine disk. Equations (4)-(5) and Figure 3 give the row of order candidates and its propagating boundary. Figure 5 and Eq. (6) address groove rotation.
- Use for: the geometric motivation for Harvey's diagram, after the learner has constructed the orders.
- Convention note: the incident marker on a reflection observation hemisphere is an outward position vector, opposite to the incoming propagation vector. Translate this convention explicitly before comparing signs.
- Do not transfer a particular screen-pattern shape to a different screen geometry without deriving that projection.

### 2. Harvey and Pfisterer (2019): main explanatory reference

**Understanding diffraction grating behavior: including conical diffraction and Rayleigh anomalies from transmission gratings**, Optical Engineering 58(8), 087105, DOI [10.1117/1.OE.58.8.087105](https://doi.org/10.1117/1.OE.58.8.087105).

- Inspected closely: Sections 2 and 3, local PDF pages 5-7, printed 087105-4 through 087105-6; Figures 5-9.
- Use for: sign conventions, the conical order locus, the sphere-to-disk projection, and the distinction between reflection and transmission diagrams.
- Figures 6-8 are the main geometric references. Figure 9 helps distinguish direction-cosine spacing from angular spacing and distances on a plane.
- Convention note: with matching signed planar transmission angles, their Eq. (3) uses the opposite order label to this notebook series: m_H = -m.
- Precision note: Eqs. (8) and (11) obtain adjacent-order spacing from derivatives. Treat these as local differential estimates; calculate finite spacings directly when comparing widely separated orders near grazing.
- Keep the later scalar efficiency and anomaly discussions outside the core of Notebook 5. Geometric cutoff alone does not predict a measured intensity anomaly.

### 3. Maystre and Popov: rigorous foundation and conical mounting

E. Popov, editor, **Gratings: Theory and Numeric Applications**, second edition (2014), [author-hosted book](https://www.fresnel.fr/files/gratings/Second-Edition/GratingsTheoryandNumericApplicationsSecondEdition.pdf).

- D. Maystre, Chapter 2, Sections 2.3-2.4: local PDF pages 43-46, printed global pages 40-43 (chapter labels 2.8-2.11). Eqs. (2.46)-(2.58) establish pseudo-periodicity, discrete tangential components, the Rayleigh expansion, and outgoing-wave selection.
- D. Maystre and E. Popov, Chapter 4, Section 4.5: local PDF pages 113-115, printed global pages 110-112 (chapter labels 4.14-4.16). Figure 4.4 defines a conical mounting.
- Use for: the theoretical basis behind the phase-repeat argument and a check that the introductory geometric account agrees with grating theory.
- Convention notes: the book's alpha_n is a dimensional wavevector component, not our normalized direction cosine alpha. Its conical angle phi in Figure 4.4 is measured out of the xz-plane, not the spherical azimuth used by the existing explorer.
- Scope note: the efficiency invariance theorem in Section 4.5 has special perfectly conducting assumptions; it is not a general statement about dielectric or metallic grating efficiencies.

### 4. MKS Diffraction Grating Handbook: independent optical check

**Diffraction Grating Handbook**, eighth edition.

[Official MKS/Newport PDF](https://www.newport.com/mam/celum/celum_assets/np/resources/MKS_Diffraction_Grating_Handbook.pdf).

- Inspected: Section 2.1, printed page 22 / local PDF page 23, Eq. (2-3); Section 2.2.1, printed page 24 / local PDF page 25.
- Use for: an independent check of the conical angular equation and its reduction to in-plane incidence.
- Convention notes: the handbook uses reflection-angle conventions and G = 1/d for groove density. The series uses G = 2*pi/d for the wavevector increment. These differ by 2*pi and must not share a numerical value.
- For a fixed incidence, count orders using their full tangential components. The handbook's broad order bound is not the incidence-specific counting rule.

### 5. TU Delft: the three-dimensional propagating boundary

[Interactive Optics, Angular Spectrum Method](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap6_Diffraction/DiffractiveOptics_2022_01Clean.html#angular-spectrum-method), especially Eq. (375), its branch explanation, and Figure 87.

Use for the dependence of kz on both kx and ky, and the continuation to evanescent components. The source propagates toward +z; this series uses -z. Keep wavelength in the medium throughout. Eq. (375) and its explicit wavelength definition are the reference for normalization; avoid mixing vacuum and medium wavelength when rewriting the decay formula.

### 6. EM Possible lecture: supplementary teaching reference

[Diffraction Gratings and the Plane Wave Spectrum](https://empossible.net/wp-content/uploads/2019/08/Lecture-3b-Diffraction-Gratings-and-the-Plane-Wave-Spectrum.pdf), slides 9-14, PDF pages 5-7 of the official online lecture. Slide 12's configuration comparison was visually inspected.

Use for a visual distinction between planar incidence, conical incidence on a linear grating, and crossed gratings. Three-dimensional diffraction from a linear grating still uses one order integer. Coupled electromagnetic polarization and diffraction efficiencies belong to a later scattering treatment.

### 7. Tran and Oldenbourg: angular coordinates and pupil calibration

[Public article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8005462/), [DOI 10.1002/jemt.23625](https://doi.org/10.1002/jemt.23625), 2021 (online 2020).

Relevant passages: Eqs. (1)-(2), Figure 2, Section 2.1 and Table 2. Use the normalized relation r/R = n*sin(theta)/NA and the documented immersion-objective examples. Their wavelength in the grating equation is vacuum wavelength; the notebook retains wavelength in the medium.

### 8. Farre, Marsa and Montes-Usategui: collecting-condenser example

[Optics Express article, DOI 10.1364/OE.20.012270](https://doi.org/10.1364/OE.20.012270), 2012, especially Figure 11. The Nikon T-CHA oil-immersion condenser (NA 1.4) and the complete detection optics are checked against the sine condition using ray tracing and measurements. This supports a concrete design-and-calibration example.

### 9. Sironi: aplanatic mirror-system examples

[INAF public repository](https://openaccess.inaf.it/entities/publication/0c76045c-44a4-4277-8fc0-85e1c57d0f97), [DOI 10.1117/12.2275422](https://doi.org/10.1117/12.2275422), 2017. The Wolter-Schwarzschild discussion explicitly connects mirror design to the sine condition. Use as an example of designed aplanatism, without assuming that every two-mirror system is aplanatic or has a filled circular pupil.

### 10. Narasimhan and colleagues: definition of aplanatism

[Optics Express article, DOI 10.1364/OE.25.010710](https://doi.org/10.1364/OE.25.010710), 2017, Sections 1-2. Use only the definition combining stigmatic on-axis imaging with the sine condition, and its relation to spherical aberration and coma. The freeform design derivation is beyond this prelude.

### 11. Peng and colleagues: the lithography sine mapping

[DOI 10.1117/12.848252](https://doi.org/10.1117/12.848252), *Proceedings of SPIE* 7640, 76402Y (2010). [Author-uploaded manuscript](https://www.researchgate.net/publication/229050169_Toward_a_Consistent_and_Accurate_Approach_to_Modeling_Projection_Optics). Inspected the manuscript's Eqs. (27)-(30), which explicitly give M*n*alpha = -n'*alpha' and M*n*beta = -n'*beta', and the corresponding spatial-frequency scaling.

Convention checks: their positive demagnification factor M is the notebook's D = 4. The notebook uses M for signed transverse magnification: M = -1/4 for the inverted example, and D = 1/abs(M). Their wavelength is vacuum wavelength. Their longitudinal coordinate convention differs from the notebook; only the transverse mapping is used. The linked manuscript is an author version with some unresolved cross-references, so equation locators are identified as manuscript locators. The DOI verifies the published record; its full text was not fetched.

### 12. Carl Zeiss SMT: explicit sine-condition correction in a projection objective

[Shafer, Epple, Ulrich and Beierl, WO2005015316A2, Projection objective for microlithography](https://patents.google.com/patent/WO2005015316A2/en), published 17 February 2005. Inspected the Summary of the Invention and description of the radiance-transformation group. This primary design disclosure explains correcting the sine condition in wafer space so aperture zones share the intended image magnification. It also explicitly distinguishes the internal pupil mapping from the image-space sine correction.

Use to establish that the condition is an explicit lithography design target. The numerical example is a generic ideal model rather than a commercial lens prescription. Do not infer exact linear physical coordinates at every internal stop from an object-to-image sine relation.

### 13. Goodman: spatial-frequency convention

Joseph W. Goodman, *Introduction to Fourier Optics*, third edition (2005),
Section 3.10.1, Eqs. (3-58)-(3-62), printed pages 56-57.
[Book text](https://djvu.online/file/ej4aFH0hNKFsF).
The transverse coordinates are fx = alpha/lambda and fy = beta/lambda,
measuring phase cycles per length. Section 6.1 and the coordinate selector
identify this as Goodman's convention. The order index m and the three-component
vector f_m = k_m/(2*pi) are the notebook's extension; retain its outgoing -z
branch when comparing with Goodman's +z construction.

### 14. Mack: sigma space and diffraction-order collection

[CHE323/384 Homework 11 Solutions](https://lithoguru.com/scientist/CHE323/Homework11_solutions.pdf), University of Texas at Austin, problem 3, PDF page 2. The source explicitly calls the objective unit-circle construction sigma space and normalizes spatial frequencies by lambda/NA; a first-order offset is lambda/(pitch*NA). Inspected the source text on 23 September 2026. Keep vacuum wavelength lambda_0 explicit in this notebook and refer pitch, frequencies and collection NA to the same side.

### 15. Kirchauer: the scalar illumination sigma

[Partial Coherence and Advanced Apertures, Section 2.3.3](https://www.iue.tuwien.ac.at/phd/kirchauer/node21.html), TU Wien dissertation (1998), Eq. (2.4). Inspected on 23 September 2026. Defines the condenser-to-projection NA ratio sigma, conventionally called the partial-coherence factor. Use only the source-size definition here; do not infer arbitrary source shape, weighting or image contrast from a scalar radius.

## Common convention and derivation for Notebook 5

Retain incidence from z > 0 and transmission into z < 0. The grating repeats along x with period d and is uniform along y. Use exp(-i*omega*t), k = 2*pi/lambda, G = 2*pi/d, and actual propagation vectors throughout.

For each integer m:

\[
k_{mx}=k_{ix}+mG,\qquad k_{my}=k_{iy},\qquad
D_m=k^2-k_{mx}^2-k_{iy}^2.
\]

\[
k_{mz}=\begin{cases}
-\sqrt{D_m},&D_m>0,\\
0,&D_m=0,\\
-i\sqrt{-D_m},&D_m<0.
\end{cases}
\]

The last branch decays as exp(-sqrt(-D_m)*s) for distance s = -z. Grazing is a separate state. For complex components use algebraic squares in the dispersion relation, not a Hermitian norm.

For a real outgoing wavevector, the fixed ratio beta = kiy/k fixes its angle to the groove axis. The endpoints lie on the circle ky = kiy cut through the wavevector sphere; the rays lie on a cone. Only the outgoing half of that circle is used for transmission. At kiy = 0, the ray locus reduces to the familiar xz-plane.

Harvey coordinates are alpha_m = kmx/k and beta_m = kiy/k. Order candidates lie on an equally spaced row, delta alpha = lambda/d. The unit disk determines which candidates have real directions. Points outside it are retained as tangential-component coordinates, not assigned real rays.

For section 7, keep the collection axis along -z. Multiplying the two Harvey coordinates by the surrounding-medium index n gives dimensionless NA coordinates. A centered circular objective accepts a propagating order when (n*alpha)^2+(n*beta)^2 <= NA_obj^2. Under the ideal sine mapping, (xp/R,yp/R) = (n/NA_obj)*(alpha,beta), up to pupil-axis orientation. This geometry does not specify pupil amplitudes. The worked example fixes the outgoing direction while varying collection NA, so changing the aperture does not change the diffraction order's propagation state.


Section 7.3 defines the two-dimensional angular sigma coordinates as (n/NA_obj)*(alpha_m,beta_m). Their radius rho_m = n*sin(theta_m)/NA_obj reaches one at the collection edge; the full propagation disk instead has radius n/NA_obj. An order outside the collection disk can remain propagating. Under the specified sine mapping, collected sigma coordinates equal (xp/R,yp/R). The grating row advances by lambda_0/(d*NA_obj), with fixed sigma_y. Distinguish this point coordinate from the scalar illumination setting sigma, which is the incident source radius for a centered circular source. Both NAs in that ratio are mask-side quantities. Small calculation 7 prints sigma coordinates and rho for both collected and excluded real orders; reducing NA changes the normalized coordinates without changing the direction.

The lithography application is an ideal finite-conjugate projection model with equal signed magnification M in x and y: wafer-side NA coordinates = (1/M) times mask-side NA coordinates, and NA_wafer = NA_mask/abs(M). For the inverted example M = -1/4 and the positive reduction ratio D = 1/abs(M) = 4, so the coordinate multiplier is -D while the aperture-radius multiplier is D. Periods remain positive: d_wafer = abs(M)*d_mask. It introduces separately labelled mask and wafer media without changing the earlier grating boundary problem. Physical internal-pupil positions need their own mapping. The explicit bridge uses one chosen sine-mapped pupil with axes aligned to mask-side directions: (xp,yp)/R = q_mask/NA_mask = -q_wafer/NA_wafer for the inverted example. Its radius R is an independent physical design/relay scale, not determined by NA or image magnification alone. For the +1 order, an illustrative R = 3 mm gives xp = 2.144 mm. Sections 7.2-7.4 introduce the physical sine mapping once, name and normalize the resulting angular coordinates, then apply them in one mask-to-wafer example. The physical-pupil assumption is stated at its first use. The worked 193 nm example uses an illustrative 800 nm mask pitch, D = 4 and NA_wafer = 1.35: mask collection NA 0.3375; first orders at +/-0.24125 are collected and correspond to wafer coordinates -/+0.965; second orders at +/-0.4825 propagate on the mask side but are excluded. The ideal image pitch is 200 nm. No scattering amplitudes, resist result or scanner prescription are inferred.

## Teaching order and concrete checks

1. Start from the existing in-plane geometry and introduce phase variation along y.
2. Explain unchanged ky using continuous translation symmetry along the grooves.
3. Construct one outgoing vector, then classify all order candidates with D_m.
4. Draw the cone and its discrete order rays.
5. Introduce the sphere and Harvey disk as representations of those rays.
6. Rescale to NA coordinates, then introduce the sine condition as the additional optical requirement for physical pupil mapping. Normalize by the collection NA to introduce sigma space, its unit collection disk, and the separate scalar illumination setting.
7. Apply the ideal sine relation to lithography: distinguish mask-side and wafer-side NA, follow accepted grating orders through 4X reduction, and separate angular pupil coordinates from actual internal-pupil positions.

A useful example holds alpha_i = 0.3 and lambda/d = 0.5 while changing beta_i. At beta_i = 0.4, orders -2,-1,0,+1 propagate. At beta_i = 0.6, +1 is grazing. At beta_i = 0.8, only -1 and 0 propagate. This holds kix fixed to isolate the added along-groove component; simply changing azimuth at fixed polar angle changes both kix and kiy.

Implementation checks should cover the planar limit, the example above, zero-order transmission (k_0 = k_i in the common medium), grazing, decaying evanescent branches, and agreement of the same selected order in every representation. The existing unnumbered notebook and explorer use +z transmission, so their conventions require adaptation before reuse in this series.

The core source coverage is sufficient for this introductory notebook. Efficiency, polarization coupling, different exterior media, finite gratings, lens-design derivations and pupil-field transformations remain later topics. Section 7 is a geometrical prelude to that mapping treatment.

## Revised Chapter 5 route

1. [5.1: Conical diffraction](../5_1_Three_Dimensional_Orders_and_Conical_Diffraction.ipynb) retains the wavefront, ray and component explorers and the translation-symmetry derivation.
2. [5.2: Direction cosines and spatial frequencies](../5_2_Direction_Cosines_and_Spatial_Frequency_Coordinates.ipynb) retains Harvey diagrams and the established Fourier convention; these are coordinate representations, without collection optics.
3. [5.3: Two-dimensional gratings](../5_3_Two_Dimensional_Gratings_and_Diffraction_Order_Lattices.ipynb) extends the same translation argument to two integer labels and compares all three coordinate scales.
4. [5.4: From coordinates to pupils](../5_4_From_Diffraction_Coordinates_to_Pupils.ipynb) contains angular acceptance, the declared sine mapping, source supports and a brief coherent LSI model. The original pupil references remain there; lithography references apply only to the optional application.

Detailed rectangular and nonseparable thin-mask Fourier coefficients and translation checks moved to [7.4](../7_4_Diffraction_Kinematics_and_Complex_Order_Coefficients.ipynb#two-dimensional-thin-mask-coefficients). See [restructuring and validation notes](Chapter_5_Restructure_Validation.md). The [earlier validation](Lattice_Grating_Validation.md) records the original combined lesson and is historical.
