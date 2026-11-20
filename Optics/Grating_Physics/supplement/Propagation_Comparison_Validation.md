# Retained field-comparison backend: ASM, Fresnel and Fraunhofer

The current 8.3 is restricted to kernel theory and kernel interactions; see
`Propagation_Kernels_Validation.md`. This document records the earlier
exit-field comparison retained in `wave_optics/propagation_comparison.py`
for later use. Its numerical backend and focused tests remain available.

## Model and conventions

The notebook compares one prescribed scalar boundary field: a circular Gaussian
envelope exp(-(x²+y²)/w²) times the thin amplitude mask
0.5+0.5*cos(2*pi*x/d), d=8 micrometres. Both transverse dimensions are finite;
the plot is a y=0 cut of the 3D field, not a one-dimensional slit propagator.
Controls span medium wavelength 0.5..1.5 um, amplitude radius w=2..12 um,
and distance s=20..800 um. Time convention is exp(-i omega t); s=-z>0.
The mask supplies boundary data; material scattering and lenses are not solved.

The Gaussian boundary has an analytic 2D Fourier transform: three overlapping
Gaussian spectra with amplitudes 1/4, 1/2, 1/4. The convention uses kx,ky in
radians per length, no forward prefactor and inverse measure dkx*dky/(2*pi)².
Fresnel is evaluated analytically as three tilted, propagated Gaussian terms.
Fraunhofer uses the analytic input spectrum at kx=k*x/s, ky=k*y/s and retains
the carrier, observation quadratic phase and 1/(i*lambda*s) amplitude factor.

## Numerical full-ASM reference

The existing scalar_propagation.transfer_function supplies the full outgoing
multiplier, including evanescent decay. Positive s is passed to its positive-
distance convention; no negative evanescent propagation is requested.

The Fourier quadrature uses window=2048 um and spacing=0.5 um: 4096 nodes per
transverse axis, delta_k=2*pi/window and cutoff pi/spacing. These are numerical
parameters; the Gaussian input has no imposed hard aperture. Gaussian spectral
amplitudes below 1e-14 are omitted. Integration over ky is chunked and collapsed
at y=0, then a dense inverse Fourier series and cubic interpolation evaluate
the same physical x coordinates used by the analytic approximations. Its
64-fold output oversampling is checked against direct complex Fourier sums.
The finite spectral sum is periodic; window refinement tests its influence.

Observation sampling includes a common broad cut and finer nodes around the
Fraunhofer peaks. Narrow, incorrect near-field Fraunhofer predictions are not
silently missed by coarse sampling. Complex-field differences use trapezoidal
L2 integration on these shared nonuniform coordinates, without phase alignment.

## Verification

All 13 focused tests passed:

- Independent integration of the full 3D RS I kernel over both source
  coordinates agrees with ASM at nontrivial observation points (2e-11 absolute
  test tolerance).
- Analytic Fresnel agrees in complex amplitude with the existing 512x512 FFT
  implementation (2e-14 absolute tolerance).
- Independent spatial integration confirms the analytic input spectrum and
  the Fraunhofer absolute complex prefactor.
- Four representative control cases compare window 2048 with 4096 and raise
  the spectral cutoff independently. At the most demanding checked case,
  lambda=1.5, w=2, s=800, the selected-point difference was 1.37e-7 absolute
  and 1.5e-5 relative. Other checked cases were near machine precision.
- Dense output interpolation agrees with direct angular sums (8e-11 absolute
  tolerance). Phase comparisons omit dark points and preserve constant offsets.
- All controls change their mathematical quantities and PNG plots. View changes
  reuse the same computed fields. Rerunning the notebook cell replaces its own
  panel with an unchanged widget count; failed initial rendering also cleans up.
- A fresh project-Python kernel ran the complete notebook, exercised every
  control and all three output views, decoded the PNGs and reran the explorer
  cell without duplicate widgets. Selected callbacks took 0.72-1.45 seconds;
  a separate extreme-width/distance calculation took about 4 seconds. Sliders
  update on release. No live notebook-frontend inspection is implied.

Representative relative complex-field differences over the displayed cut:

| Medium wavelength, width, distance (um) | Fresnel vs ASM | Fraunhofer vs ASM |
| --- | --- | --- |
| 1, 6, 120 | 0.0226 | 0.642 |
| 0.5, 3, 800 | 0.0364 | 0.0524 |
| 1.5, 2, 800 | 0.643 | 0.641 |
| 0.5, 12, 20 | 0.000334 | 6.60 |

Thus far-field agreement with Fresnel does not establish agreement with full
ASM. The diagnostics distinguish approximation error from numerical error.
The Gaussian width indicator w²/(lambda*s) is not treated as a universal cutoff.

## Presentation

The retained field explorer shows the input and spectrum plus a shared output plot for
intensity, amplitude or phase difference. There is no per-method peak
normalization. Input/spectrum axes are fixed; output axes follow the shared
physical observation range and all predicted values. Phase relative to ASM
has fixed [-pi,pi] limits; points below 1% of either field peak are omitted.
Detailed diagnostics and the static fallback are optional. Symbolic equations
stay in Markdown; widget readouts use plain HTML and Unicode.

Default and parameter-extreme scientific figures were rendered and visually
inspected. This is separate from inspection in a live notebook frontend;
the latter has not been performed for this notebook.
