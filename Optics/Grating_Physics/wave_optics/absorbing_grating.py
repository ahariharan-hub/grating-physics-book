"""Passive, free-standing scalar absorber with periodically repeated air openings.

The clear opening is centred at opening_shift, unlike the centred dielectric
ridge in scalar_grating.py. Illumination has unit amplitude at z=0 and travels
toward -z, with exp(-i omega t). Both exterior half-spaces are air.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .scalar_grating import ScalarGratingSolution, _real_scalar, _samples, _solve_layer


@dataclass(frozen=True)
class AbsorbingGratingConfiguration:
    """Illustrative lengths in micrometres; indices are not fitted chrome data."""

    wavelength: float = .633
    period: float = 1.5
    clear_width: float = .75
    thickness: float = .3
    n_absorber: float = 1.5
    kappa: float = .8
    opening_shift: float = 0.
    M: int = 24

    def __post_init__(self):
        for name in ('wavelength', 'period', 'thickness', 'n_absorber'):
            object.__setattr__(self, name, _real_scalar(getattr(self, name), name, positive=True))
        for name in ('clear_width', 'kappa', 'opening_shift'):
            object.__setattr__(self, name, _real_scalar(getattr(self, name), name))
        if not 0 <= self.clear_width <= self.period:
            raise ValueError('clear_width must lie in [0, period].')
        if self.kappa < 0:
            raise ValueError('kappa must be nonnegative (passive exp(-i omega t) convention).')
        if isinstance(self.M, (bool, np.bool_)) or not isinstance(self.M, (int, np.integer)) or self.M < 0:
            raise ValueError('M must be a nonnegative integer.')
        object.__setattr__(self, 'M', int(self.M))

    @property
    def open_fraction(self):
        return self.clear_width/self.period

    @property
    def absorber_index(self):
        return self.n_absorber+1j*self.kappa

    n_incident = n_output = 1.

    def permittivity_coefficients(self, p):
        """epsilon_p: absorber background plus centred/shifted clear aperture."""
        p = np.asarray(p)
        if not np.issubdtype(p.dtype, np.integer):
            raise ValueError('p must contain integer Fourier indices.')
        eps = self.absorber_index**2
        d = self.open_fraction
        if d in (0., 1.):
            return np.asarray((p == 0)*(eps if d == 0 else 1.), dtype=complex)
        opening = d*np.sinc(p*d)*np.exp(-2j*np.pi*p*self.opening_shift/self.period)
        return np.asarray(eps*(p == 0)+(1-eps)*opening, dtype=complex)

    def refractive_index(self, x_samples, z_samples):
        x, z = _samples(x_samples, 'x_samples'), _samples(z_samples, 'z_samples')
        wrapped = np.remainder((x-self.opening_shift)/self.period+.5, 1.)-.5
        opening = (wrapped >= -self.open_fraction/2) & (wrapped < self.open_fraction/2)
        layer = np.where(opening, 1., self.absorber_index)
        return np.where((z[:, None] >= 0) | (z[:, None] <= -self.thickness), 1., layer)


@dataclass(frozen=True)
class AbsorbingGratingSolution(ScalarGratingSolution):
    """R/T from exterior flux; A from an independent volume-loss integral."""

    absorption: float = 0.
    absorption_quadrature_error: float = 0.

    @property
    def reflectance(self):
        return float(np.sum(self.reflection_efficiency))

    @property
    def transmittance(self):
        return float(np.sum(self.transmission_efficiency))

    @property
    def energy_balance(self):
        return self.power_balance+self.absorption

    def downward_flux(self, z_samples):
        """Period-averaged -Im(U* dU/dz) divided by incident beta_0."""
        u = self.harmonics(z_samples)
        du = self.harmonics(z_samples, derivative=1)
        return -np.imag(np.sum(u.conj()*du, axis=1))/self.beta_incident[self.configuration.M].real


def _volume_absorption(solution):
    """Integrate k0² Im(epsilon)|U|²/(incident beta_0), averaged over x.

    The exact material convolution integrates the truncated field over x.
    Its anti-Hermitian part (not elementwise imaginary part) also handles a
    shifted aperture correctly. Only the depth integral needs quadrature.
    """
    c, m = solution.configuration, solution.orders
    if c.kappa == 0 or c.open_fraction == 1:
        return 0., 0.
    epsilon = c.permittivity_coefficients(m[:, None]-m[None, :])
    loss = (epsilon-epsilon.conj().T)/(2j)
    scale = solution._k0**2/solution.beta_incident[c.M].real

    def integrand(depth):
        u = solution.harmonics(-depth)[0]
        return float(scale*np.vdot(u, loss @ u).real)

    skin = min(c.thickness/2, 1/(solution._k0*max(np.max(solution._q.imag), 1e-12)))
    return quad(integrand, 0., c.thickness, epsabs=2e-10, epsrel=2e-9,
                points=sorted(set([skin, c.thickness-skin])), limit=250)


def solve_absorbing_grating(configuration=None):
    """Solve passive complex Helmholtz scattering and verify volume absorption.

    Reuses the lossless solver's bounded two-boundary basis and equilibrated
    matching. Complex layers use a general eigensolver, not Hermitian eigh.
    This remains a scalar U/dU-dz continuity model, not a general Maxwell solver.
    """
    c = AbsorbingGratingConfiguration() if configuration is None else configuration
    if not isinstance(c, AbsorbingGratingConfiguration):
        raise TypeError('configuration must be an AbsorbingGratingConfiguration.')
    base = _solve_layer(c, absorbing=c.kappa > 0 and c.open_fraction < 1)
    absorption, error = _volume_absorption(base)
    solution = AbsorbingGratingSolution(**vars(base), absorption=absorption,
                                       absorption_quadrature_error=error)
    if absorption < -1e-9 or abs(solution.energy_balance-1) > max(1e-7, 10*error):
        raise RuntimeError('Absorbing scalar flux/volume-loss check failed; inspect convergence and conditioning.')
    return solution
