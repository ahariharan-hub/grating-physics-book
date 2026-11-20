"""Ideal local binary transmission functions and exact Fourier coefficients.

Two complex transmission levels occupy a centred rectangular region and its
complement. These are prescribed boundary fields under unit illumination,
not solutions for reflection, absorption or a material boundary problem.
"""

import numpy as np

from .scalar_grating import ScalarGratingConfiguration, _real_scalar, _samples


def _binary_parameters(duty_cycle, t_r, t_g):
    f = _real_scalar(duty_cycle, 'duty_cycle')
    if not 0 <= f <= 1:
        raise ValueError('duty_cycle must lie in [0, 1].')
    levels = []
    for value, name in ((t_r, 't_r'), (t_g, 't_g')):
        try:
            array = np.asarray(value, dtype=complex)
            if isinstance(value, (bool, np.bool_)) or array.ndim != 0 or not np.isfinite(array):
                raise ValueError
            levels.append(complex(array))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f'{name} must be a finite complex scalar.') from error
    return f, *levels


def binary_transmission_coefficients(orders, duty_cycle, t_r, t_g):
    """Period-average coefficients C_m=t_g*delta_m0+(t_r-t_g)*f*sinc(m*f).

    ``orders`` is an integer scalar or nonempty 1D array; sinc(v)=sin(pi*v)/(pi*v).
    The centred origin fixes the lateral phase. No FFT, power normalization
    or truncation-dependent correction is applied. Choose t_r=1,t_g=0 for
    the ideal transparent/opaque amplitude mask; unit-modulus levels give a
    phase mask. The squared coefficients are not generally power efficiencies.
    """
    f, tr, tg = _binary_parameters(duty_cycle, t_r, t_g)
    m = np.asarray(orders)
    if m.ndim > 1 or m.size == 0 or not np.issubdtype(m.dtype, np.integer):
        raise ValueError('orders must be an integer scalar or nonempty 1D integer array.')
    with np.errstate(over='raise', invalid='raise'):
        try:
            if f in (0., 1.):
                return np.asarray((m == 0)*(tg if f == 0 else tr), dtype=complex)
            return np.asarray(tg*(m == 0)+(tr-tg)*f*np.sinc(m*f), dtype=complex)
        except (FloatingPointError, OverflowError) as error:
            raise ValueError('Binary transmission levels exceed the supported numerical range.') from error


def binary_transmission_field(x_samples, period, duty_cycle, t_r, t_g):
    """Exact periodic local transmission, shape (nx,), before Fourier truncation.

    R(x)=1 in [-f*d/2,f*d/2), repeated with period d; t=t_g+(t_r-t_g)*R.
    The half-open ridge convention matches the scalar grating geometry.
    """
    f, tr, tg = _binary_parameters(duty_cycle, t_r, t_g)
    d = _real_scalar(period, 'period', positive=True)
    x = _samples(x_samples, 'x_samples')
    with np.errstate(over='raise', invalid='raise'):
        try:
            wrapped = np.remainder(x/d+.5, 1.)-.5
            ridge = (wrapped >= -f/2) & (wrapped < f/2)
            return np.where(ridge, tr, tg)
        except (FloatingPointError, OverflowError) as error:
            raise ValueError('Transmission samples exceed the supported numerical range.') from error


def dielectric_phase_levels(configuration):
    """Return (t_r,t_g)=exp(i*k0*n*h), retaining the absolute exit-plane phase.

    Unit illumination is referenced to z=0; the exit is z=-h, with exp(-i*omega*t).
    This local approximation has no Fresnel factors or internal round trips.
    """
    if not isinstance(configuration, ScalarGratingConfiguration):
        raise TypeError('configuration must be a ScalarGratingConfiguration.')
    c = configuration
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            levels = np.exp(1j*(2*np.pi/c.wavelength)*c.thickness
                            * np.array([c.n_ridge, c.n_groove]))
        except (FloatingPointError, OverflowError) as error:
            raise ValueError('Thin-mask parameters exceed the supported numerical range.') from error
    if not np.all(np.isfinite(levels)):
        raise ValueError('Nonfinite dielectric phase levels.')
    return tuple(levels)
