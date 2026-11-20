"""Prescribed periodic complex fields for the Chapter 9 sampling lessons.

Coordinates are in nm; Fourier frequency is in cycles/nm. No mask scattering
or propagation is calculated here.
"""
import numpy as np


DEFAULT_ORDERS = np.array([-3, 0, 2, 5])
DEFAULT_COEFFICIENTS = np.array([
    .25*np.exp(.6j), .8*np.exp(-.3j), .4*np.exp(.9j), .2*np.exp(-.7j),
])
DEFAULT_ORDERS.setflags(write=False)
DEFAULT_COEFFICIENTS.setflags(write=False)


def _field_parameters(period_nm, orders, coefficients):
    period = float(period_nm)
    modes = np.asarray(orders)
    values = np.asarray(coefficients, dtype=complex)
    if not np.isfinite(period) or period <= 0:
        raise ValueError('period_nm must be finite and positive.')
    if (modes.ndim != 1 or modes.size == 0 or not np.issubdtype(modes.dtype, np.integer)
            or np.unique(modes).size != modes.size):
        raise ValueError('orders must be distinct integers in a nonempty 1D array.')
    if values.shape != modes.shape or not np.all(np.isfinite(values)):
        raise ValueError('coefficients must contain one finite complex value per order.')
    return period, modes, values


def periodic_field(x_nm, period_nm=1000., orders=DEFAULT_ORDERS,
                   coefficients=DEFAULT_COEFFICIENTS):
    """Evaluate the prescribed finite sum at arbitrary 1D spatial coordinates."""
    p, modes, values = _field_parameters(period_nm, orders, coefficients)
    x = np.asarray(x_nm, dtype=float)
    if x.ndim > 1 or not np.all(np.isfinite(x)):
        raise ValueError('x_nm must be a finite scalar or 1D array.')
    return np.exp(2j*np.pi*x[..., None]*modes/p) @ values


def sampling_experiment(samples_per_period=32, periods=1, *, period_nm=1000.,
                        start_fraction=-.5, orders=DEFAULT_ORDERS,
                        coefficients=DEFAULT_COEFFICIENTS):
    """Sample a fixed field, FFT it, and reconstruct its DFT interpolant.

    The window contains an integer number of physical periods. The start is
    start_fraction times the window length; no spatial shift of the field is
    introduced. Even-N Nyquist is represented by its negative-frequency bin,
    following np.fft.fftfreq. Resolved means all nonzero specified harmonics
    lie strictly inside the Nyquist interval.
    """
    for name, value in (('samples_per_period', samples_per_period), ('periods', periods)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if samples_per_period < 2:
        raise ValueError('Use at least two samples per period.')
    p, modes, values = _field_parameters(period_nm, orders, coefficients)
    start = float(start_fraction)
    if not np.isfinite(start):
        raise ValueError('start_fraction must be finite.')
    n = int(samples_per_period*periods)
    length = p*periods
    dx = p/samples_per_period
    x_start = start*length
    x = x_start+np.arange(n)*dx
    sampled = periodic_field(x, p, modes, values)
    raw = np.fft.fft(sampled)
    frequencies = np.fft.fftfreq(n, d=dx)
    fft_coefficients = raw/n*np.exp(-2j*np.pi*frequencies*x_start)
    fine_x = np.linspace(x_start, x_start+length, 1601)
    exact_fine = periodic_field(fine_x, p, modes, values)
    reconstructed = np.exp(2j*np.pi*fine_x[:, None]*frequencies) @ fft_coefficients
    required_bins = modes*periods
    resolved = bool(np.all(np.abs(required_bins[abs(values) > 0]) < n/2))
    expected = np.zeros(n, dtype=complex)
    if resolved:
        expected[required_bins % n] = values
        coefficient_error = float(np.max(abs(fft_coefficients-expected)))
    else:
        coefficient_error = None
    return dict(period_nm=p, samples_per_period=samples_per_period, periods=periods,
                N=n, L_nm=length, dx_nm=dx, x_start_nm=x_start, x_nm=x,
                field=sampled, raw_fft=raw, frequencies=frequencies,
                coefficients=fft_coefficients, known_orders=modes,
                known_coefficients=values, fine_x_nm=fine_x, exact_fine=exact_fine,
                reconstructed_fine=reconstructed, resolved=resolved,
                coefficient_error=coefficient_error,
                roundtrip_error=float(np.max(abs(np.fft.ifft(raw)-sampled))),
                field_error=float(np.max(abs(reconstructed-exact_fine))))
