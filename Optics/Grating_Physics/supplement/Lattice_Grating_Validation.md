# Historical validation: original combined lattice lesson

Validated on 2026-10-28 against the existing Chapter 5 and 7.4 conventions; reference baseline a86bb6a8acddc370cbcb437de58f4633de296878 was available locally. No existing numerical helper behavior was changed.

All nine code cells were executed in order in a fresh Python namespace using the project virtual environment (NumPy 2.5.4), with Matplotlib's noninteractive Agg backend. This executes the notebook code directly rather than through a Jupyter kernel. All six generated figures were visually inspected; verbose NumPy integer labels were corrected and the notebook was rerun. Notebook outputs and widget state remain empty, following the series convention; temporary figure files were removed.

- Reciprocal duality passed for rectangular, rotated rectangular, and skew direct bases.
- Rectangular analytic versus normalized, origin-corrected 2D FFT maximum absolute errors: 6.7965e-5, 1.6959e-5, 4.2377e-6 for 128, 256, 512 samples per axis.
- The nonseparable two-rectangle cell agreed with its analytic complex coefficients to 6.1775e-6 at 512 samples per axis; its coefficient matrix has a nonzero second singular value.
- Dy=1 suppresses every n!=0 coefficient. The populated row matches the existing conical helper, including complex longitudinal components.
- Rolling the sampled cell by known sample counts reproduced the negative-sign lateral-translation phase to absolute tolerance 1e-14.
- Propagating, grazing and evanescent states and collection were checked separately. The algebraic dispersion relation passed to 1e-12; an evanescent amplitude decreased from 1 to 0.336795 to 0.113431 at depths 0, 0.1, 0.2 um.
- Increasing output index preserved transverse components while increasing the number of propagating orders.
- At normal incidence (1,1) is collected; (2,0) propagates outside the pupil and has zero rectangular-mask coefficient. The sampled annular source retains 184 source identities and 31,096 source-order records.
- All relative links in the new notebook resolve. The existing conical regression suite passed: 15 tests.

The checks validate geometry, prescribed transmission, and Fourier representation. They do not validate material scattering or a full projection-lens field transformation. Candidate labels are finite for display and handoff; the notebook explicitly distinguishes them from the infinite evanescent spectrum.

The subsequent Chapter 5 restructure split geometry into 5.3, collection into 5.4, and coefficients into 7.4. See [current validation](Chapter_5_Restructure_Validation.md).
