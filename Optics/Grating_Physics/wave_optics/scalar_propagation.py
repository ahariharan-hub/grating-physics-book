"""Fourier-space scalar propagation for the structured ASM theory notebook.

Use one length unit for wavelength, sample spacings and propagation distance;
spatial frequencies then have reciprocal units. Fields use exp(-i omega t),
and NumPy's forward Fourier transform uses exp(-2 pi i f x). The FFT example
represents a finite periodic grid. It does not guarantee adequate sampling,
padding, or suppression of wraparound for an arbitrary diffraction problem.
"""

import numpy as np


def _scalar(value, name, *, positive=False, nonnegative=False):
    try:
        if np.iscomplexobj(value):
            raise ValueError
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a finite real scalar.') from error
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f'{name} must be a finite real scalar.')
    value = float(array)
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive.')
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be nonnegative.')
    return value


def _frequencies(fx, fy):
    arrays = []
    for value, name in ((fx, 'fx'), (fy, 'fy')):
        try:
            if np.iscomplexobj(value):
                raise ValueError
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name} must contain finite real frequencies.') from error
        if not np.all(np.isfinite(array)):
            raise ValueError(f'{name} must contain finite real frequencies.')
        arrays.append(array)
    try:
        return np.broadcast_arrays(*arrays)
    except ValueError as error:
        raise ValueError('fx and fy must have broadcast-compatible shapes.') from error


def _transverse_geometry(fx, fy, wavelength):
    """Validated transverse geometry without unused angle/component arrays."""
    wavelength = _scalar(wavelength, 'wavelength', positive=True)
    fx, fy = _frequencies(fx, fy)
    k = 2*np.pi/wavelength
    alpha, beta = wavelength*fx, wavelength*fy
    with np.errstate(over='ignore', invalid='ignore'):
        q = alpha**2+beta**2
    if not np.isfinite(k) or not np.all(np.isfinite(q)):
        raise ValueError('Wavelength and frequencies exceed the supported numerical range.')
    return fx, fy, k, alpha, beta, q


def spectrum_geometry(fx, fy, wavelength):
    """Map transverse frequencies to the outgoing Helmholtz spectral branch.

    ``fx`` and ``fy`` broadcast together. Returned ``fx``, ``fy``, ``kx``,
    ``ky``, ``kz``, ``alpha``, ``beta``, ``gamma``, ``q``, ``propagating``,
    ``theta_deg`` and ``azimuth_deg`` all have that shape; ``k`` is a scalar.
    Here alpha=lambda*fx, beta=lambda*fy, gamma=kz/k, and q=alpha**2+beta**2.
    The complex square root chooses Re(kz)>=0 and Im(kz)>=0. ``propagating``
    includes grazing (q=1). Polar and azimuthal angles are NaN for evanescent
    components; the on-axis azimuth is also NaN because it is undefined.
    """
    fx, fy, k, alpha, beta, q = _transverse_geometry(fx, fy, wavelength)
    gamma = np.sqrt((1-q).astype(complex))
    propagating = q <= 1
    theta = np.where(propagating,
                     np.rad2deg(np.arctan2(np.sqrt(q), gamma.real)), np.nan)
    azimuth = np.where(propagating, np.rad2deg(np.arctan2(beta, alpha)), np.nan)
    azimuth = np.where(q == 0, np.nan, azimuth)
    return dict(fx=fx, fy=fy, k=k, kx=k*alpha, ky=k*beta, kz=k*gamma,
                alpha=alpha, beta=beta, gamma=gamma, q=q,
                propagating=propagating, theta_deg=theta, azimuth_deg=azimuth)


def transfer_function(fx, fy, wavelength, distance, method='asm'):
    """Return the ASM or paraxial Fresnel multiplier for propagation into +z.

    ASM is exp(i*kz*distance), including evanescent decay. Fresnel is
    exp(i*k*distance-i*pi*wavelength*distance*(fx**2+fy**2)); it is an
    approximation for small transverse frequencies and does not describe
    evanescent waves. Both include the carrier phase. Negative distance is
    rejected, avoiding exponential amplification by backward evanescent
    propagation. At zero distance the multiplier is exactly one.
    """
    distance = _scalar(distance, 'distance', nonnegative=True)
    if method not in ('asm', 'fresnel'):
        raise ValueError("method must be 'asm' or 'fresnel'.")
    _, _, k, _, _, q = _transverse_geometry(fx, fy, wavelength)
    if distance == 0:
        return np.ones(q.shape, dtype=complex)
    if method == 'asm':
        kz = k*np.sqrt((1-q).astype(complex))
        return np.exp(1j*kz*distance)
    return np.exp(1j*k*distance*(1-q/2))


def fresnel_phase_error(fx, fy, wavelength, distance):
    """Return exact ASM phase minus Fresnel phase, in radians (not wrapped).

    The stable exact expression is -k*z*q**2/[2*(1+sqrt(1-q))**2], where
    q=lambda**2*(fx**2+fy**2). ``leading`` is its small-q limit -k*z*q**2/8.
    ``exact`` and ``leading`` are NaN for evanescent components because the
    propagating-wave comparison no longer applies. Returned keys are
    ``exact``, ``leading`` and ``q``. At q=0 both errors are zero.
    """
    distance = _scalar(distance, 'distance', nonnegative=True)
    _, _, k, _, _, q = _transverse_geometry(fx, fy, wavelength)
    propagating_q = np.where(q <= 1, q, np.nan)
    root = np.sqrt(1-propagating_q)
    numerator = -k*distance*propagating_q**2
    exact = numerator/(2*(1+root)**2)
    leading = numerator/8
    return dict(exact=exact, leading=leading, q=q)


def propagate_field(field, dx, dy, wavelength, distance, method='asm'):
    """Propagate a sampled scalar field on a finite periodic 2D FFT grid.

    ``field`` is indexed [y, x], is finite, and has at least two samples on
    each axis. ``dx`` and ``dy`` are positive sample spacings. This teaching
    helper performs no padding, resampling or band limiting: choose a wide,
    sufficiently sampled grid before drawing conclusions about an isolated
    physical aperture. ``spectrum`` is the raw unshifted NumPy FFT (not an
    area-normalized continuous Fourier transform).

    The result has keys ``field`` (propagated complex field), ``spectrum``,
    ``spectrum_out``, ``transfer`` (all [y,x]), and ``fx``, ``fy`` (1D unshifted
    frequency axes). The carrier phase is retained for both methods.
    """
    dx = _scalar(dx, 'dx', positive=True)
    dy = _scalar(dy, 'dy', positive=True)
    try:
        field = np.asarray(field, dtype=complex)
    except (TypeError, ValueError) as error:
        raise ValueError('field must be a finite two-dimensional numeric array.') from error
    if field.ndim != 2 or min(field.shape) < 2 or not np.all(np.isfinite(field)):
        raise ValueError('field must be finite and two-dimensional, with at least 2x2 samples.')
    ny, nx = field.shape
    fx, fy = np.fft.fftfreq(nx, dx), np.fft.fftfreq(ny, dy)
    transfer = transfer_function(fx[None, :], fy[:, None], wavelength, distance, method)
    spectrum = np.fft.fft2(field)
    spectrum_out = spectrum*transfer
    propagated = np.fft.ifft2(spectrum_out)
    return dict(field=propagated, spectrum=spectrum, spectrum_out=spectrum_out,
                transfer=transfer, fx=fx, fy=fy)
