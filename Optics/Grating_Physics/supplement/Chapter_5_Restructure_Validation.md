# Chapter 5 restructure and verification

Completed 2026-10-28. Reference baseline: a06b5753b83d0e9d125c7be7fda284ac50147f97 (available locally).

## Content moves

| Previous content | Revised location |
| --- | --- |
| 5.1: three-dimensional orders and conical geometry | [5.1: Three-Dimensional Orders and Conical Diffraction](../5_1_Three_Dimensional_Orders_and_Conical_Diffraction.ipynb). Existing wavefront, ray, component and cone explorers retained; premature pupil discussion removed. |
| 5.2: Harvey and spatial-frequency coordinates | [5.2: Direction Cosines and Spatial-Frequency Coordinates](../5_2_Direction_Cosines_and_Spatial_Frequency_Coordinates.ipynb). Existing explorer and calculations retained, with explicit k, f and direction-cosine conversions. |
| 5.4: two-dimensional periodicity and order geometry | [5.3: Two-Dimensional Gratings and Diffraction-Order Lattices](../5_3_Two_Dimensional_Gratings_and_Diffraction_Order_Lattices.ipynb). Generic grating parameters replace the lithography example. No objective or coefficient calculation is needed. |
| 5.3: collection and sine mapping; 5.4: sampled illumination support | [5.4: From Diffraction Coordinates to Pupils](../5_4_From_Diffraction_Coordinates_to_Pupils.ipynb). Collection follows geometry. Generic q coordinates, reference caps and pupil scales are distinguished. Source identities remain intact. A brief coherent LSI model introduces H and the impulse response. Lithography and sigma terminology are confined to a short optional application. |
| 5.4: unit-cell Fourier integrals, rectangles and nonseparable cells, phases, FFT comparisons and excitation checks | [7.4: Diffraction Kinematics and Complex Order Coefficients](../7_4_Diffraction_Kinematics_and_Complex_Order_Coefficients.ipynb#two-dimensional-thin-mask-coefficients). Original two-wave experiment, absorber comparison and flux discussion remain. |

The overview, previous/next links, series README, reference map, helper notes and closing synthesis now follow the revised subjects. The closing synthesis uses a generic grating/collection example; the optional lithography application remains in 5.4. Superseded filenames were replaced; repository links were updated.

## Helper changes

`lattice_orders` computes geometry without objective NA: integer labels, k, f, normalized components and propagation states. `collect_orders(data, NA_obj)` adds q and collection flags in a subsequent step, retaining all labels, wavevectors and source metadata. Explicit older NA_obj calls delegate to collection for compatibility. `lattice_plotting` draws geometry on the three coordinate scales without collection optics. Existing conical explorers and their numerical controls are unchanged.

## Verification

Each notebook was executed in a separate fresh Python process, running its code cells in order with the project virtual environment and Matplotlib Agg. This was direct notebook-code execution, not execution through a Jupyter kernel. Notebook outputs and widget state remain cleared under the series convention.

| Notebook | Code cells | Result |
| --- | ---: | --- |
| 5.1 | 7 | Passed |
| 5.2 | 4 | Passed |
| 5.3 | 5 | Passed |
| 5.4 | 6 | Passed |
| 7.4 | 7 | Passed, including the preserved scalar absorber solve |
| Closing synthesis | 3 | Passed |

Twelve Matplotlib/widget figures and three Plotly views (incident wavefronts, outgoing rays/components and Harvey hemisphere/disk) were rendered and visually inspected. Plotly images used Kaleido with local Chrome; there was no browser UI interaction test. SciPy and Kaleido were installed in the local virtual environment for execution and figure QA. Temporary images and HTML were removed; the reproducible [execution helper](validate_chapter5_notebook.py) remains. Its optional `--render-plotly` flag requires Kaleido and local Chrome.

Focused checks passed for reciprocal duality (rectangular, rotated and skew bases), incidence translation, coordinate conversions, the conserved-component 1D row, algebraic dispersion and outgoing propagating/grazing/evanescent branches, evanescent decay, unequal outgoing indices, collection boundaries, and preservation of source identities. The brief declared complex-transfer example also passed without assigning it to the grating scattering model.

The relocated coefficient checks passed: Dy=1 eliminates n!=0 excitation, lateral sample rolls reproduce the negative Fourier translation phase, and analytic versus normalized origin-corrected FFT maximum errors are 6.7965e-5, 1.6959e-5 and 4.2377e-6 for rectangular cells sampled at 128, 256 and 512 points per axis. The 512-point nonseparable-cell error is 6.1775e-6. These validate prescribed transmission and Fourier representation, not material scattering.

Regression suites passed: 15 existing conical tests plus 5 new geometry/collection API tests. Local notebook link paths, unique cell IDs, empty saved outputs and whitespace checks passed.

To repeat the checks from Grating_Physics, use the project Python interpreter:

```text
python supplement/validate_chapter5_notebook.py <notebook filename>
python -m unittest discover -s supplement/tests -p test_conical.py
python -m unittest discover -s supplement/tests -p test_lattice_grating.py
```

Full convolution derivations, PSF calculations, lens field transformations and partially coherent imaging remain subsequent-series material.

## Teaching-cell cleanup

The developer assertion blocks in 5.3 and 5.4 were subsequently moved into `supplement/tests/test_lattice_grating.py`, extending that suite from five to nine tests. Existing tests already covered geometry/coordinate separation, reciprocal duality and collection preservation; the added tests retain the conical-row comparison, dispersion/decay, unequal-index comparison, source identity and declared complex-pupil checks.

The notebook replacements are short explained calculations: 5.3 prints one order's relative amplitude at increasing depths, and 5.4 compares the acceptance, output magnitude and phase of three specified orders under an explicitly illustrative pupil response. Both notebooks were executed independently after the change, and all nine supporting lattice tests passed. No physics helper behavior or figures changed.
