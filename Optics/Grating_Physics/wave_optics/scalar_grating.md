# Scalar grating scattering backend — 0_5A

**Course notation:** the legacy transmitted amplitude `t_m` in these implementation notes is the exit-field coefficient $C_m(z_e)$, with $z_e=-h$, stored by `solution.transmission`. Course notebooks reserve $t_m$ for a thin mask's Fourier coefficients. [Shared convention](../0_0_Foundations_of_Scalar_Optics.ipynb#exit-field-notation).

`scalar_grating.py` solves for the complex diffracted amplitudes of a prescribed
dielectric structure. It supplies numerical data for a future Notebook 0_5B;
it has no plotting, widgets or notebook dependencies.

## Model and reference planes

The scalar field obeys

\[
(\partial_x^2+\partial_z^2+k_0^2 n^2)U=0,\qquad k_0=2\pi/\lambda_0,
\]

with time dependence \(e^{-i\omega t}\). A unit plane wave arrives normally
from \(z>0\), travelling toward negative \(z\). A rectangular periodic layer
occupies \(-h<z<0\), with homogeneous incident and output half-spaces.
Lengths must share one unit; **wavelength is in vacuum**. Defaults use µm:
\(\lambda_0=0.633\), \(d=1.5\), \(h=0.3\), duty cycle \(f=0.5\),
\(n_i=n_o=n_g=1\), \(n_r=1.5\).

Within each period the ridge is centred at \(x=0\) and occupies
\([-fd/2,fd/2)\); the remainder is groove material. This origin convention
matters for the phases of the diffraction coefficients. The material profile
method returns the prescribed piecewise index, not its Fourier approximation.

The model enforces continuity of **\(U\) and \(\partial U/\partial z\)** at
both horizontal boundaries. It is a scalar Helmholtz model, not a general
vector electromagnetic solver or a model of both TE and TM responses.
Only positive real indices, positive thickness, and normal incidence are
supported. Absorption, multilayers, oblique/conical incidence and arbitrary
profiles are outside this API.

## Where the coefficients come from

Write

\[
U(x,z)=\sum_{m=-M}^{M}A_m(z)e^{imGx},\qquad G=2\pi/d.
\]

The exact rectangular material coefficients are

\[
\varepsilon_p=n_g^2\delta_{p0}
 +(n_r^2-n_g^2)f\,\operatorname{sinc}(pf),
\qquad \operatorname{sinc}(v)=\frac{\sin(\pi v)}{\pi v}.
\]

Inserting the expansions into Helmholtz and collecting equal harmonics gives

\[
A_m''-k_{x,m}^2A_m+k_0^2\sum_{n=-M}^{M}\varepsilon_{m-n}A_n=0,
\qquad k_{x,m}=mG,
\]

or \(\mathbf A''+\mathbf Q\mathbf A=0\), with
\(Q_{mn}=k_0^2\varepsilon_{m-n}-k_{x,m}^2\delta_{mn}\).
The convolution uses material coefficients through \(|p|=2M\).
The Hermitian matrix \(\mathbf Q/k_0^2\) is diagonalized with SciPy `eigh`.
This is a Fourier–Galerkin truncation: the residual vanishes after projection
onto the retained harmonics. The finite field sum approximates the full
piecewise-material problem; pointwise accuracy still requires convergence.

Let \(\beta_{j,m}=\sqrt{(k_0n_j)^2-k_{x,m}^2}\), taking the positive real
root for propagating orders and positive imaginary root for evanescent ones.
The exterior fields are

\[
U_i=e^{-i\beta_{i,0}z}
 +\sum_m r_m e^{ik_{x,m}x+i\beta_{i,m}z},\qquad z\geq0,
\]

\[
U_o=\sum_m t_m e^{ik_{x,m}x-i\beta_{o,m}(z+h)},\qquad z\leq-h.
\]

Thus **\(r_m\) is referenced to \(z=0\)** and **\(t_m\) to \(z=-h\)**.
Reflected evanescent orders decay toward positive \(z\), transmitted ones
toward negative \(z\). No incoming output wave is allowed.

For numerical conditioning, use dimensionless depth \(s=-k_0z\), thickness
\(L=k_0h\), and \(B_j=\operatorname{diag}(\beta_{j,m}/k_0)\).
Eliminating reflection and transmission via field continuity leaves

\[
\mathbf A_s(0)+iB_i\mathbf A(0)=2iB_i\mathbf e_0,
\qquad
\mathbf A_s(L)-iB_o\mathbf A(L)=0.
\]

These conditions determine both layer solutions. The solver never prescribes
an arbitrary vector of diffracted amplitudes.

For each layer eigenvalue \(q_j^2\), the basis functions are
\(e^{iq_js}\) and \(e^{iq_j(L-s)}\). They are anchored at opposite
interfaces, so evanescent factors have magnitude at most one. For
\(|q_jL|<0.1\), the solver instead uses
\(\cos(q_js)\) and \(\sin(q_js)/(q_jL)\), evaluated with `sinc`.
Their cutoff limits are \(1\) and \(s/L\); the two independent solutions
are preserved even when \(q_j=0\).

One dense, row/column equilibrated boundary solve determines the coefficients.
There is no transfer matrix containing exponentially growing evanescent
factors. The broader stability issue is discussed by
[Moharam et al., JOSA A 12, 1077–1086 (1995)](https://doi.org/10.1364/JOSAA.12.001077);
the implementation here is a single-layer scalar boundary solve, not their
general multilayer algorithm.

## Numerical API

```python
import numpy as np
from wave_optics.scalar_grating import (
    ScalarGratingConfiguration, solve_scalar_grating,
)

config = ScalarGratingConfiguration(M=20)  # other defaults above, lengths in µm
solution = solve_scalar_grating(config)

x = np.linspace(-config.period/2, config.period/2, 256, endpoint=False)
z = np.linspace(.2, -.6, 161)
index = config.refractive_index(x, z)       # (161, 256)
harmonics = solution.harmonics(z)          # (161, 41), complex A_m(z)
normal_derivative = solution.harmonics(z, derivative=1)  # dA_m/dz
field = solution.reconstruct(x, z)         # (161, 256), complex U(x,z)
coefficients = solution.at_reference_plane(-.4)  # (41,), complex C_m
output_field = solution.reconstruct(x, -.4)[0]
print(solution.power_balance)              # sum of outgoing scalar efficiencies
```

Scalars and nonempty one-dimensional sample arrays are accepted. Sampling
methods always retain the sample dimension, even for a scalar. A reference
plane accepts exactly one scalar and returns just the order dimension.

| Attribute | Meaning |
| --- | --- |
| `configuration` | Geometry, indices, vacuum wavelength and truncation |
| `orders`, `kx` | Ascending integers `-M,...,M` and transverse wavevectors |
| `beta_incident`, `beta_output` | Positive-root longitudinal magnitudes; transmitted signed \(k_z=-\beta_o\) |
| `incident_kinds`, `output_kinds` | Per-order strings: `propagating`, `grazing`, `evanescent` |
| `reflection`, `transmission` | Complex \(r_m,t_m\), including evanescent orders |
| `reflection_efficiency`, `transmission_efficiency` | Nonnegative scalar flux fractions per order |
| `coupling_matrix` | Physical \(\mathbf Q\), in reciprocal length squared |
| `condition_number` | 2-norm condition estimate of the equilibrated boundary matrix |
| `boundary_residual` | Relative backward error of boundary matching |
| `power_balance` | Sum of reflected and transmitted efficiencies |

Solution arrays are read-only. `harmonics` and `at_reference_plane` return the
**total** field: incident plus reflected above the layer, the solved layer
field within it, and outgoing transmission below it. Consequently the total
incident-side coefficients can change magnitude as a reference plane moves.
The outgoing reflected coefficients alone obey the propagation law above.
Use \(z_{\rm ref}\leq-h\) to extract an outgoing output-plane field.

The sampled reconstruction is a direct Fourier sum, not an FFT propagation.
Choose enough samples to resolve all retained harmonics (over one period,
at least \(2M+1\) distinct points for Fourier recovery). All evanescent
terms remain available in the solution and in the reconstruction.

## Amplitude, intensity and scalar power

The scalar conserved normal current is proportional to
\(J_z=\operatorname{Im}(U^*\partial_zU)\). After averaging over one period,
normalize outgoing flux to the incident magnitude \(\beta_{i,0}\):

\[
\eta_m^{(R)}=\frac{\operatorname{Re}\beta_{i,m}}{\beta_{i,0}}|r_m|^2,
\qquad
\eta_m^{(T)}=\frac{\operatorname{Re}\beta_{o,m}}{\beta_{i,0}}|t_m|^2.
\]

These are **scalar power efficiencies**. A complex coefficient, its squared
magnitude, and its efficiency are different quantities. Evanescent and
exactly grazing orders have zero outgoing normal power but can have nonzero
field amplitudes. The code sums only strictly propagating orders for power.

Moving an output plane by a downward distance \(\ell>0\) gives
\(C_m(-h-\ell)=t_m e^{i\beta_{o,m}\ell}\). Propagating magnitudes are
fixed and phases advance; evanescent magnitudes decay.

## Validation report and numerical limits

Run from the project root:

```text
python -m unittest supplement.tests.test_scalar_grating -v
```

The 20 physics/API tests cover:

- Uniform media: zero reflection, no higher orders and \(t_0=e^{ik_0nh}\).
- Uniform slabs from both duty endpoints: complex Fabry–Perot reflection,
  transmission, interior fields and flux with unequal exterior indices.
- Analytical material coefficients against numerical integrals of the exact profile.
- Complex-coefficient convergence, including amplitude and phase separately.
- Both horizontal interface conditions, the coupled Helmholtz equation inside
  the layer, and conserved scalar flux inside and outside it.
- Reference-plane phase/decay, and independent FFT recovery of reconstructed
  field coefficients in all three regions.
- Grazing classification, excited internal cutoff modes, the thin-layer limit,
  a deep layer, length-unit rescaling, invalid inputs and conditioning warnings.

All 20 solver tests pass, and the API example above executes successfully.
The full `supplement/tests` run completes 231 tests: 228 pass, with one failure
and two errors in existing `test_wave_optics.InteractivePhasorTests` cases.
Those tests expect removed Notebook 1 cell ID `288c5c98` and old explorer-cell
source patterns. The same three failures reproduce in an isolated copy of
committed baseline `a0567bf`, without this solver. They are unrelated to the
new backend; the existing notebook and tests remain unchanged.

For the default physical configuration, comparing the same reflected and
transmitted orders \(m=-4,\ldots,4\) against \(M=60\) gives:

| M | Maximum complex coefficient difference | Maximum magnitude difference | Maximum phase difference (rad) |
| ---: | ---: | ---: | ---: |
| 5 | 2.10e-3 | 1.20e-3 | 3.96e-2 |
| 10 (default) | 4.28e-4 | 2.66e-4 | 7.36e-3 |
| 20 | 5.41e-5 | 3.37e-5 | 8.72e-4 |
| 30 | 1.48e-5 | 9.21e-6 | 2.36e-4 |
| 40 | 5.05e-6 | 3.15e-6 | 8.02e-5 |

Phase differences use `angle(c_M * conj(c_60))`; all compared coefficients
are nonzero. These are differences against a finer truncation, not certified
errors against an exact grating solution. Orders ±3 and ±4 are evanescent here.
At \(M=60\), the outgoing fractions are approximately
\(R=0.0694894830\), \(T=0.9305105170\), and
\(t_0=-0.5845420882-0.4358007284i\).
The sampled truncations conserve power within \(10^{-14}\), even when their
complex coefficients still differ noticeably. **Power balance does not
establish truncation convergence.** Slab coefficient errors are below
\(3\times10^{-14}\) in the tested cases.

The 30 µm thick, \(M=30\) stress case remains finite and conserves power
within \(10^{-12}\); a naive growing evanescent factor would overflow
float64. This tests numerical stability, not convergence for a thick grating.

Further limits:

- Dense eigendecomposition and matching cost roughly \(O((2M+1)^3)\).
  Modest truncations are intended. Sharp material edges can require more
  harmonics, especially for local near fields; test the quantity of interest.
- External orders are classified as grazing when the dimensionless squared
  longitudinal wavevector is within `64*eps*max(n**2,(kx/k0)**2)` of zero.
  These roots are set to zero. Resolved deviations retain their actual branch.
- Conditioning above \(10^{12}\) raises a warning; singular or numerically
  unresolved systems raise `numpy.linalg.LinAlgError`. Exact grazing in a
  transparent structure, for example, admits an arbitrary unforced constant
  harmonic. This backend reports that nonuniqueness instead of assigning
  arbitrary coefficients. Slight wavelength/period detuning can establish
  a limiting scattering solution; it is not applied silently.
- Nonfinite results and failed boundary residual checks raise errors; a scalar
  flux discrepancy above \(10^{-8}\) raises a warning. These diagnostics do
  not replace convergence checks or extend the model to vector optics.

Existing grating kinematics and positive-z ASM helpers are unchanged. This
module explicitly owns its negative-z scattering geometry and reference planes.
