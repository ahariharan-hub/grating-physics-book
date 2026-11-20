# Scalar FMM for a two-dimensional periodic layer

`scalar_grating_2d.py` extends the scalar Helmholtz model to centered rectangular
inclusions in a rectangular x-y lattice. The layer is uniform through its
thickness. Layer indices may be real or passive complex numbers (positive real
part and nonnegative imaginary part); exterior indices stay real and lossless.
This is a scalar model,
matching U and dU/dz, with no polarization or vector Maxwell interface rules.

## Example

```python
import numpy as np
from wave_optics.scalar_grating_2d import (
    ScalarGrating2DConfiguration, solve_scalar_grating_2d,
)
c = ScalarGrating2DConfiguration(
    wavelength=.633, period_x=1.5, period_y=1.2,
    thickness=.3, fill_x=.5, fill_y=.4,
    n_inclusion=1.5, n_background=1., Mx=3, My=3,
    kx_incident=.4, ky_incident=.2,
)
s = solve_scalar_grating_2d(c)
x = np.linspace(-c.period_x/2, c.period_x/2, 80, endpoint=False)
y = np.linspace(-c.period_y/2, c.period_y/2, 80, endpoint=False)
exit_field = s.reconstruct(x, y, [-c.thickness])[0]  # shape (ny,nx)
C_mn = s.at_reference_plane(-c.thickness)
print(s.orders, s.transmission_efficiency, s.power_balance)
```

All lengths share one unit. Incident kx and ky use its inverse unit, include
refractive index, and must define a strictly propagating incident wave.
The default incident amplitude is one at z=0. With exp(-i omega t), illumination
travels toward -z; the layer occupies -h<z<0. Reflection is referenced to z=0,
transmission to z=-h. `beta_*` are positive-root longitudinal magnitudes;
outgoing transmitted kz is negative beta_output.

## Equations and ordering

Orders are pairs (m,n), with m outermost and n innermost, both ascending.
There are K=(2Mx+1)(2My+1) retained harmonics. Arrays of coefficients and
wavevectors have length K; `orders` has shape (K,2). Reshape coefficient arrays
to (2Mx+1,2My+1) for an order-lattice table (not a spatial image).

kx_m=kx_incident+2*pi*m/period_x, and similarly for ky_n.
For a centered rectangular inclusion, the material coefficients are

    epsilon_pq = n_background^2 delta_p0 delta_q0
                 + (n_inclusion^2-n_background^2)
                   fill_x fill_y sinc(p fill_x) sinc(q fill_y).

The dimensionless coupling matrix is epsilon_(m-m',n-n') minus the diagonal
(kx_m^2+ky_n^2)/k0^2. A Hermitian eigensolver is used for lossless layers;
absorbing layers use a general complex eigensolver with eigenbasis condition
and residual checks, and decaying modal roots. These modes decouple the layer
ODEs. Bounded modal functions and equilibrated boundary matching are shared
with the original 1D solver. `coupling_matrix` is reported in physical units
(inverse length squared). Both exterior evanescent fields decay away from
the layer. Flux efficiencies use Re(beta_order)/beta_incident_00 times the
squared coefficient magnitude; evanescent and grazing orders carry zero
normal scalar flux.

`harmonics(z, derivative=1)` returns dA/dz. `reconstruct(x,y,z)` returns
(nz,ny,nx), including the incident Bloch phase. `refractive_index` on the
configuration returns the exact material grid with the same axis convention.
Returned solution arrays are read-only.

## Numerical checks and scope

Run `python -m unittest discover -s supplement -p test_scalar_grating_2d.py -v`.
The checks cover an analytic oblique uniform slab (including unequal exterior
indices), the original 1D limit, both interface conditions, Bloch phase,
x-y interchange, scalar flux, evanescent decay, and grazing singularity handling.

Increase Mx and My independently and check complex coefficients as well as
power. Power conservation alone does not establish truncation convergence.
Dense factorization costs O(K^3), with O(K^2) matrix storage. Exact grazing or
trapped modes can make matching singular; the solver reports this rather than
returning an arbitrary solution. Defaults are starting cutoffs, not accuracy
promises. Spatial reconstruction sampling is separate from FMM truncation.

This version supports a single passive layer and rectangular inclusions.
It does not yet provide multilayers, arbitrary sampled geometries,
or a Gaussian illumination wrapper. Gaussian illumination can subsequently
use coherent quadrature over propagating incident (kx,ky) components; each
component requires its own Bloch solve and a consistent incident reference.
The existing 1D and absorbing solvers are unchanged.

## Absorbing screen versus ideal amplitude mask

`n_inclusion=1` and `n_background=1.5+.8j` describe centered rectangular air
openings surrounded by an illustrative absorber. `fill_x` and `fill_y` are
the **clear opening fractions** in this configuration. For the convention
exp(-i omega t), positive Im(n) attenuates outgoing waves inside the material.

The solution reports `reflectance`, `transmittance`, and `absorption`.
Absorption is computed by integrating k0^2 Im(epsilon)|U|^2 over the layer,
divided by incident normal flux and unit-cell area. The anti-Hermitian part of
the material convolution performs the transverse integral; depth is integrated
by quadrature. `absorption_quadrature_error` is the estimated depth-integration
error, not a bound on FMM truncation error. `power_balance` is R+T;
`energy_balance` is R+T+A. Absorption is not inferred by subtracting R and T.

```python
from wave_optics.amplitude_grating_2d import (
    absorbing_screen_configuration, AbsorbingComparison2DData,
)
from wave_optics.absorbing_grating_2d_view import AbsorbingGrating2DExplorer

configuration = absorbing_screen_configuration(Mx=3, My=3)
data = AbsorbingComparison2DData(configuration)
absorber_comparison = AbsorbingGrating2DExplorer(data)
display(absorber_comparison.ui)
# Call absorber_comparison.close() before replacing the explorer.
```

The three tabs show structure/exit field, the complex order lattice, and full
ASM propagation. FMM runs once at construction. Changing distance or selected
order only reconstructs the solved fields. Geometry, material and cutoff
changes are made in the configuration before constructing a new explorer.
An existing solution can be reused with `AbsorbingComparison2DData(solution=s)`.

The ideal model prescribes the rectangular opening at z=-h with the incident
plane wave continued to that plane. Its coefficients are

    C_mn = exp(i beta_incident_00 h) fill_x fill_y
           sinc(m fill_x) sinc(n fill_y).

The transverse incident Bloch phase is included in reconstructed spatial
fields. Both models share U_i(0,0,0)=1, with no phase fitting or power/peak
normalization. Equal real opening and exterior indices are required for this
comparison. The ideal prescription predicts no reflected field or physical
absorption budget. The exact binary boundary prescription and its finite
Fourier reconstruction are distinguished in the display (edge ringing is
expected in the latter). Propagation in this explorer uses full ASM; Fresnel
and Fraunhofer comparison controls are not included yet.

`field_fourier_coefficients_2d` extracts C_mn from complex spatial samples on
one endpoint-excluded cell. It removes the incident carrier and corrects for
the coordinate origin before extracting the FFT2 bins. The displayed FFT
agreement checks representation consistency, not the scattering model.

Additional checks for absorbing slabs, the existing absorbing 1D limit,
interface/flux balance, full/zero openings, and oblique FFT2 recovery run with
`python -m unittest discover -s supplement/tests -p test_absorbing_grating_2d.py -v`.
Independent external benchmarking remains deferred.

For the default absorbing screen, an initial cutoff sweep gave:

| Mx=My | C00 | R | T | A |
|---|---|---|---|---|
| 3 | -0.218778 - 0.011029i | 0.096691 | 0.183697 | 0.719612 |
| 4 | -0.217047 - 0.011896i | 0.097182 | 0.181627 | 0.721190 |
| 5 | -0.216257 - 0.009236i | 0.097364 | 0.182956 | 0.719680 |

These results conserve R+T+A but still change with cutoff. The display defaults
are suitable for an initial demonstration, not a claim of converged coefficients.

### Initial convergence measurement

For the default geometry with Mx=My=M, a local sweep gave:

| M | C00 | Total transmitted flux |
|---|---|---|
| 2 | -0.810349 - 0.140499i | 0.96403970 |
| 4 | -0.790181 - 0.161846i | 0.96383637 |
| 5 | -0.788647 - 0.164069i | 0.96379100 |
| 6 | -0.788578 - 0.164058i | 0.96378334 |

Flux balance stayed within 7e-16 of unity and matching residuals below 8e-17.
The change in C00 between M=5 and M=6 is about 7e-5; this is a measured change,
not an error bound. This illustrates why low-cutoff flux balance alone cannot
validate coefficient accuracy.
