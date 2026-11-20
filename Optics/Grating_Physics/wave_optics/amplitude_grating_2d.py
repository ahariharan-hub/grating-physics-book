"""Physical 2D absorbing screen and ideal rectangular-opening exit prescription."""
from dataclasses import dataclass, field

import numpy as np

from .scalar_grating import _exterior_branch, _real_scalar, _samples
from .scalar_grating_2d import (
    ScalarGrating2DConfiguration, ScalarGrating2DSolution, solve_scalar_grating_2d,
)


def absorbing_screen_configuration(**kwargs):
    """Air openings in a passive background; illustrative indices, not fitted data.

    fill_x/fill_y are the clear opening fractions, not absorber fractions.
    Mx=My=3 is an initial cutoff, not a convergence guarantee.
    """
    parameters = dict(n_inclusion=1., n_background=1.5+.8j, Mx=3, My=3)
    parameters.update(kwargs)
    return ScalarGrating2DConfiguration(**parameters)


def _validate_openings(c):
    if not isinstance(c, ScalarGrating2DConfiguration):
        raise TypeError('configuration must be a ScalarGrating2DConfiguration.')
    if c.n_inclusion != c.n_incident or c.n_output != c.n_incident:
        raise ValueError('Ideal opening comparison requires equal real opening and exterior indices.')


def _orders(c):
    m, n = np.meshgrid(np.arange(-c.Mx, c.Mx+1), np.arange(-c.My, c.My+1), indexing='ij')
    return np.column_stack((m.ravel(), n.ravel()))


def _aperture_coefficients(c, orders):
    m, n = orders.T
    result = c.fill_x*c.fill_y*np.sinc(m*c.fill_x)*np.sinc(n*c.fill_y)
    if c.fill_x == 1:
        result = np.where(m == 0, result, 0.)
    if c.fill_y == 1:
        result = np.where(n == 0, result, 0.)
    return result


def _reconstruct(coefficients, kx, ky, x, y):
    return np.einsum('za,ay,ax->zyx', coefficients,
                     np.exp(1j*ky[:, None]*y), np.exp(1j*kx[:, None]*x), optimize=True)


@dataclass(frozen=True)
class IdealAmplitudeGrating2D:
    """Thin exit-plane prescription; predicts neither reflection nor absorption.

    Same incident phase U_i(0,0,0)=1 as FMM, and the same exit z=-h.
    No phase fitting or peak/power normalization is applied.
    """
    configuration: ScalarGrating2DConfiguration
    orders: np.ndarray = field(init=False, repr=False)
    kx: np.ndarray = field(init=False, repr=False)
    ky: np.ndarray = field(init=False, repr=False)
    beta_output: np.ndarray = field(init=False, repr=False)
    output_kinds: np.ndarray = field(init=False, repr=False)
    transmission: np.ndarray = field(init=False, repr=False)
    transmission_efficiency: np.ndarray = field(init=False, repr=False)
    incident_reference: complex = field(init=False)

    def __post_init__(self):
        c = self.configuration
        _validate_openings(c)
        orders = _orders(c)
        k0 = 2*np.pi/c.wavelength
        kx = c.kx_incident+2*np.pi*orders[:, 0]/c.period_x
        ky = c.ky_incident+2*np.pi*orders[:, 1]/c.period_y
        beta, kinds = _exterior_branch(c.n_output, np.hypot(kx, ky)/k0)
        beta0 = np.sqrt((k0*c.n_incident)**2-c.kx_incident**2-c.ky_incident**2)
        reference = np.exp(1j*beta0*c.thickness)
        coefficients = reference*_aperture_coefficients(c, orders)
        values = dict(orders=orders, kx=kx, ky=ky, beta_output=k0*beta, output_kinds=kinds,
                      transmission=coefficients, incident_reference=reference,
                      transmission_efficiency=np.where(kinds == 'propagating',
                                                       k0*beta.real/beta0*abs(coefficients)**2, 0.))
        for name, value in values.items():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    def exact_exit_field(self, x_samples, y_samples):
        """Unsummed binary boundary prescription, shape (ny,nx)."""
        c = self.configuration
        x, y = _samples(x_samples, 'x'), _samples(y_samples, 'y')
        wx = np.remainder(x/c.period_x+.5, 1.)-.5
        wy = np.remainder(y/c.period_y+.5, 1.)-.5
        mask = ((wy[:, None] >= -c.fill_y/2) & (wy[:, None] < c.fill_y/2)
                & (wx[None, :] >= -c.fill_x/2) & (wx[None, :] < c.fill_x/2))
        return mask*self.incident_reference*np.exp(1j*(c.kx_incident*x[None, :]+c.ky_incident*y[:, None]))

    def harmonics(self, z_samples):
        z = _samples(z_samples, 'z_samples')
        if np.any(z > -self.configuration.thickness):
            raise ValueError('The ideal exit prescription propagates only at or below z=-thickness.')
        return self.transmission*np.exp(-1j*(z[:, None]+self.configuration.thickness)*self.beta_output)

    def reconstruct(self, x_samples, y_samples, z_samples):
        return _reconstruct(self.harmonics(z_samples), self.kx, self.ky,
                            _samples(x_samples, 'x'), _samples(y_samples, 'y'))


def field_fourier_coefficients_2d(field_values, x_samples, y_samples, configuration, orders=None):
    """FFT2 of the complex Bloch-periodic field on one endpoint-excluded cell.

    Remove the incident carrier before FFT; retain absolute coordinate-origin
    phase. Input shape is (ny,nx). This checks representation, not scattering.
    """
    c = configuration
    if not isinstance(c, ScalarGrating2DConfiguration):
        raise TypeError('configuration must be a ScalarGrating2DConfiguration.')
    x, y = _samples(x_samples, 'x'), _samples(y_samples, 'y')
    u = np.asarray(field_values, dtype=complex)
    if u.shape != (y.size, x.size) or min(x.size, y.size) < 2 or not np.all(np.isfinite(u)):
        raise ValueError('field_values must be finite with shape (ny,nx), with at least two samples per axis.')
    for samples, period in ((x, c.period_x), (y, c.period_y)):
        if not np.allclose((samples-samples[0])/period, np.arange(samples.size)/samples.size,
                           rtol=1e-10, atol=1e-12):
            raise ValueError('Use ascending uniform samples over one period, excluding the repeated endpoint.')
    pairs = _orders(c) if orders is None else np.asarray(orders)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0
            or not np.issubdtype(pairs.dtype, np.integer)
            or np.any(abs(pairs[:, 0]) >= x.size/2) or np.any(abs(pairs[:, 1]) >= y.size/2)):
        raise ValueError('orders must have integer shape (K,2), strictly below Nyquist in each axis.')
    periodic = u*np.exp(-1j*(c.kx_incident*x[None, :]+c.ky_incident*y[:, None]))
    spectrum = np.fft.fft2(periodic)/u.size
    m, n = pairs.T
    return spectrum[n % y.size, m % x.size]*np.exp(-2j*np.pi*(m*x[0]/c.period_x+n*y[0]/c.period_y))


class AbsorbingComparison2DData:
    """One solved absorber and its ideal opening prescription, reusable by views."""

    def __init__(self, configuration=None, *, solution=None):
        if solution is not None:
            if not isinstance(solution, ScalarGrating2DSolution):
                raise TypeError('solution must be a ScalarGrating2DSolution.')
            if configuration is not None and configuration != solution.configuration:
                raise ValueError('configuration must match the supplied solution.')
            configuration = solution.configuration
        self.configuration = absorbing_screen_configuration() if configuration is None else configuration
        _validate_openings(self.configuration)
        self.solution = solve_scalar_grating_2d(self.configuration) if solution is None else solution
        self.ideal = IdealAmplitudeGrating2D(self.configuration)

    def coefficients(self, distance=0.):
        """Return physical and ideal coefficients at z=-h-distance."""
        distance = _real_scalar(distance, 'distance')
        if distance < 0:
            raise ValueError('distance must be nonnegative.')
        z = -self.configuration.thickness-distance
        return self.solution.at_reference_plane(z), self.ideal.harmonics(z)[0]
