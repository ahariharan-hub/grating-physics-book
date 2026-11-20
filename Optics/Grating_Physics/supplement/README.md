# Supplementary material

The main teaching sequence, notebooks `0_` through `9_`, and the shared
`wave_optics/` package live one directory above this folder. Supplementary
notebooks, scripts, tests, figures, and other supporting material are collected here.

| Location | Contents |
| --- | --- |
| Unnumbered `.ipynb` files | Supplementary lessons and reference notebooks, including `Scalar_Wave_Propagation.ipynb` |
| [Appendix 4: Fourier Series, Transforms, and FFTs](../Appendix_4_How_to_Use_FFTs_in_Python.ipynb) | General transform-family reference, analytical FT/FFT comparison, and practical NumPy FFT examples |
| [Appendix 5: Formulae Sheet](../Appendix_5_Formulae_Sheet.ipynb) | Annotated formula chains for every current lesson and active companion, with links to source notebooks and archived versions |
| [Binary_Amplitude_and_Dielectric_Phase_Masks.ipynb](Binary_Amplitude_and_Dielectric_Phase_Masks.ipynb) | Preserved ideal amplitude/phase-mask lesson and lossless dielectric comparison, outside the main 0_5–0_7 amplitude-grating route |
| [Notebook_7_Previous_Draft.ipynb](Notebook_7_Previous_Draft.ipynb) | Preserved two-wave lesson before the incremental rebuild of Notebook 7 |
| [Pupil Mapping, Aplanatism and Imaging Notes](Pupil_Mapping_Aplanatism_and_Imaging_Notes.ipynb) | Extended 5.4 reference-cap, sine-mapping, LSI, illumination and lithography calculations |
| [Exit-Field Fourier Representations and Checks](Exit_Field_Fourier_Representations_and_Checks.ipynb) | Extended 9.1 Fourier formulas, Bloch-periodicity comparison and exact three-order verification |
| [Chapter 9: Complex Coefficients from One Unit Cell](Chapter_9_Unit_Cell_Coefficient_Calculations.ipynb) | Analytical and sampled rectangular/nonseparable thin-mask coefficients, one-dimensional limit, and translation checks |
| Standalone `.py` files | Earlier scalar/vector optics calculations and plotting scripts |
| `../wave_optics/` | Shared calculations and interactive views used by both notebook groups |
| `figures/` | Notebook figures and original source-document images and animations |
| `output/` | Standalone HTML illustrations |
| `tests/` | Checks for the shared calculations and notebook views |
| [validation/notebook_0_split_inventory.json](validation/notebook_0_split_inventory.json) | Pre-edit mapping of the combined 0_5 lesson into 0_5, 0_6 and 0_7 |
| [validation/notebook_0_split_validation.json](validation/notebook_0_split_validation.json) | Final word/cell counts, fresh-kernel results, numerical preservation and reference checks |
| `*_Reference_Map.md` | Source and convention notes for the relevant lessons |

Run each notebook's first code cell before its examples. It locates
the project-root `wave_optics/` package automatically from either notebook location.

Run the shared test suite **from the project root**:

```powershell
python -m unittest discover -s supplement/tests
```

Run the standalone polarization checks **from this folder**:

```powershell
python -m unittest discover -s . -p test_polarization.py
```

See [the helper-package guide](../wave_optics/README.md) for the individual
modules, notebook behavior, and conventions. Relative image links in the
supplementary notebooks remain beside their supporting assets.

## Freely accessible reading

Every current course notebook and active supplement has a selected-reading section with access status and a reading suggestion. The [source catalogue](reading_sources.json) records citations, historical public-domain evidence, coverage and retrieval limitations. Archived originals and the previous draft retain their historical references. Modern free manuscripts, open-access papers, reviews and book chapters are labeled separately from public-domain sources.

## Notation compatibility

Current lessons use `m,p` for diffraction orders and `n` for refractive index. Existing backend APIs and internal plot role keys may retain `n` for a second order; these identifiers are preserved for compatibility. They do not change the displayed mathematical convention. Source quotations retain their original symbols with translation notes.

The [shared exit-field convention](../0_0_Foundations_of_Scalar_Optics.ipynb#exit-field-notation) uses $U_{\mathrm{exit}}$ at the exit, $U_{\mathrm{ref}}$ at a general reference plane, $t_m$ for mask coefficients, and $C_m$ for field coefficients.
