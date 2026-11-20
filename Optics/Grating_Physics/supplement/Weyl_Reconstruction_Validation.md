# Weyl reconstruction in notebook 8.2

Section 5 asks one question: can flat plane-wave components reconstruct a
spherical wave? The source and target stay fixed. One **Build the sum** control
changes the actual complex field being added, rather than selecting a component
while displaying an unchanged finished sum.

## Model and interaction

- exp(-i omega t), downstream coordinate s=-z, wavelength=1.
- Physical source P0=(0,0,-1.5), reference plane s=0.
- Both pictures show the same y=0 cross-section of a 3D field.
- Left: start with one propagating plane-wave component at 30 degrees, then
  evaluate weighted Weyl sums at successively finer quadrature resolutions.
- Right: the fixed analytic field exp(ikR)/(4 pi R).
- Both use the same cyclic phase scale from -pi to pi. Colours encode phase,
  not amplitude or intensity. No intensity normalization is implied.
- The coarse sums are deliberately inaccurate numerical approximations;
  each new resolution resamples the continuous spectrum and recomputes its
  quadrature weights. These are not prefixes of one fixed list, and convergence
  is not guaranteed to be monotone from the single-component starting view.
- No source motion, probe motion, grating or optical-system mapping is added.

The intended observation is that individual components retain flat fronts,
while their interference produces the curved fronts of the summed field.
Weyl's specific relative amplitudes and phases are essential.

## Numerical method

The displayed sums use the existing full 3D `weyl_modes` quadrature, with
orders 8, 16 and 24: respectively 512, 2048 and 4608 weighted components.
The mode set includes all azimuths and both propagating and evanescent branches.
Separating the x and s factors allows a matrix product on the displayed y=0
plane without removing the ky modes from the integration.

The substitutions mu=kappa/k and kappa=i beta remove the grazing singularity
before Gauss-Legendre quadrature. The evanescent integral stops at beta=20/1.5.
Its omitted absolute tail at s=0 is bounded by exp(-20)/(4 pi*1.5), about
1.1e-10, and decreases further downstream. The first single-component view
isolates a propagating spectral term and is not a spherical-source solution.

An independent implementation integrates azimuth analytically to Bessel J0
before radial quadrature. The dense displayed sum agrees with that map to
3e-14; its maximum complex error against the analytic sphere over the entire
97 by 97 displayed grid is below 1.2e-10. The intermediate order-16 maximum
error is below 3e-5. Readouts put quadrature counts and errors in optional
HTML details; symbolic equations remain in notebook Markdown.

Cached states cover the four build positions. Closing a panel detaches its
observers and releases widget models. Rerunning its cell preserves the selected
build position and replaces only that panel. The cell also accepts migration
from the earlier three-control panel by starting at the first build position.

## Checks

Run from the project root:

```powershell
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider supplement/tests/test_weyl_reconstruction.py
```

Checks cover the full 3D map, off-meridian points, single-component flatness,
mode-sum convergence, an unchanged target, actual field/image changes at each
build position, reruns, invalid inputs and failed-render cleanup.

The earlier grating/cap application remains separately documented in
[Weyl_Cap_Validation.md](Weyl_Cap_Validation.md).

### Verification on 2026-11-01

All ten focused checks passed. A fresh project kernel executed the entire
notebook, exercised all four build positions, and verified that rerunning the
cell preserved its position without disturbing the other two panels. Closing
the three panels left zero widget models. The source notebook was not
overwritten by execution. Initial and built-sum plots were rendered and
visually inspected. These are Python/kernel and PNG checks; the intended live
VS Code frontend was not inspected.
