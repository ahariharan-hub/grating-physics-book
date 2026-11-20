# Ewald construction: reference and teaching map

This map accompanies [Ewald construction for diffraction gratings](Ewald_Construction_for_Gratings.ipynb). The source review and convention checks were moved with the former Ewald section of Notebook 5; the standalone notebook runs independently and has no assigned number in the series.

## Verified reference material

External sources are linked to their publisher, author, or institution.

### Pfluger et al. (2017), with the 2018 erratum: Ewald rods for a grating

M. Pfluger et al., **Grazing-incidence small-angle X-ray scattering (GISAXS) on small periodic targets using large beams**, IUCrJ 4, 431-438, [article and Section 2](https://journals.iucr.org/m/issues/2017/04/00/hf5342/).

- Inspected: Section 2, Figure 3 and its caption, and supplementary Section 2, especially Eqs. (5)-(13), PDF pages 3-4.
- [Supplementary derivation](https://journals.iucr.org/m/issues/2017/04/00/hf5342/hf5342sup1.pdf) explicitly intersects the sphere with grating rods. Exchange its groove and periodicity axes when translating to our notation.
- [Published 2018 erratum](https://journals.iucr.org/m/issues/2018/04/00/hf5934/index.html): Eq. (5) of the main article must use **arctan**, not arcsin. The supplementary derivation already uses arctan.
- Use the reciprocal geometry as a supporting connection; the experiment's reflection geometry and finite-target effects are not the transmission model of this notebook.

The [IUCr Ewald-sphere definition](https://dictionary.iucr.org/Ewald_sphere) supplies background. It uses reciprocal lengths 1/lambda; our wavevectors use 2*pi/lambda. A surface grating imposes tangential conditions and therefore rods, rather than the full reciprocal-lattice-point constraint of a bulk crystal.

The comparison with Harvey's direction-cosine diagram uses Harvey and Vernold (1998), DOI [10.1364/AO.37.008158](https://doi.org/10.1364/AO.37.008158), and Harvey and Pfisterer (2019), DOI [10.1117/1.OE.58.8.087105](https://doi.org/10.1117/1.OE.58.8.087105). Their public source links, inspected page locations and reflection/transmission convention notes are retained in the [direction-cosine reference map](Notebook_5_Reference_Map.md).

The three-dimensional normal component and its evanescent continuation are supported by [TU Delft's angular-spectrum treatment](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap6_Diffraction/DiffractiveOptics_2022_01Clean.html#angular-spectrum-method), Eq. (375); its outgoing side is +z, so the standalone notebook reverses that branch sign.

## Conventions and derivation

Incidence is from z > 0 and transmission into z < 0. The grating repeats along x with period d and is uniform along y. Use exp(-i*omega*t), k = 2*pi/lambda with wavelength in the common medium, G = 2*pi/d, and actual propagation vectors. The outgoing normal branch is negative real for propagating orders and negative imaginary for evanescent components.

For Ewald use q_m = k_m - k_i. For real vectors:

\[
|\mathbf q_m+\mathbf k_i|^2=k^2,\qquad q_{mx}=mG,\qquad q_{my}=0.
\]

The sphere is centered at -ki and has radius k. Each order gives a rod parallel to qz; qz is not fixed by surface periodicity. Select transmission by qz + kiz < 0, and mark equality as grazing. An evanescent candidate has no real rod-sphere intersection.

The coordinate connection is khat_m = (q_m + ki)/k. A translation, scaling, and hemisphere projection connect the Ewald construction to the direction-cosine disk. These operations should follow the same selected order in the explorer.

## Teaching order and checks

1. Recall the incident and outgoing components before introducing a new coordinate.
2. Define q as a difference of wavevectors, then place its endpoint on the shifted sphere.
3. Use tangential periodicity to obtain rods; select the outgoing intersection with qz+kiz<0.
4. Construct the editable vector triangle and track the same order through propagation, grazing and evanescence.
5. Translate and normalize the coordinates to compare the Ewald view with Harvey's disk.

At lambda=2, d=4, ai=0.3 and bi=0.4, m=+1 has km/k=(0.8,0.4,-0.447214) and q/k=(0.5,0,0.418812). Positive qz does not imply positive outgoing kz. At m=0 the triangle collapses; at bi=0.6 the selected m=+1 grazes, and at bi=0.8 its rod misses the sphere.

The standalone setup creates its own explorer. Rerunning it closes that explorer first; no previously executed notebook, cached variables or saved widget state are required.
