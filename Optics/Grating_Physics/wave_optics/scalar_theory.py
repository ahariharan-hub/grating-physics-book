"""Small scalar-wave models for the foundational theory notebook.

Lengths within each Helmholtz model use one common unit, and phasors use the
time convention exp(-i omega t). The slit examples label their units and
distinguish outgoing angular-spectrum propagation near the slit from the
paraxial Fraunhofer approximation at the distant detector.
"""

import numpy as np


def _finite_scalar(value, name, *, positive=False, nonnegative=False):
    """Validate a real scalar before using it as a model parameter."""
    try:
        if np.iscomplexobj(value):
            raise ValueError('Complex values are not real parameters.')
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


def _finite_coordinates(values, name):
    try:
        if np.iscomplexobj(values):
            raise ValueError('Complex values are not spatial coordinates.')
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must contain finite real coordinates.') from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain finite real coordinates.')
    return array


def _parameters(wavelength, amplitude, phase_deg):
    wavelength = _finite_scalar(wavelength, 'wavelength', positive=True)
    amplitude = _finite_scalar(amplitude, 'amplitude', nonnegative=True)
    phase = np.deg2rad(_finite_scalar(phase_deg, 'phase_deg'))
    return wavelength, amplitude, phase


def helmholtz_trial(x, wavelength=2, spatial_ratio=1, amplitude=1, phase_deg=0):
    """Evaluate a trial exponential and its one-dimensional Helmholtz residual.

    The medium sets k=2*pi/wavelength, while the trial has q=spatial_ratio*k.
    The returned analytic second derivative is d²U/dx². The residual is
    d²U/dx²+k²U, so both q=k and q=-k are allowed travelling-wave solutions.
    """
    x = _finite_coordinates(x, 'x')
    wavelength, amplitude, phase = _parameters(wavelength, amplitude, phase_deg)
    spatial_ratio = _finite_scalar(spatial_ratio, 'spatial_ratio')
    k = 2*np.pi/wavelength
    q = spatial_ratio*k
    field = amplitude*np.exp(1j*(q*x+phase))
    second = -q*q*field
    return dict(k=k, q=q, field=field, second=second, residual=second+k*k*field)


def scalar_solution(points, wavelength=2, kind='plane', amplitude=1, phase_deg=0):
    """Return an exact homogeneous scalar Helmholtz solution where defined.

    ``points`` has shape (..., 3). ``plane`` propagates along +z. ``standing``
    is the sum of equal +z and -z waves, each with amplitude ``amplitude/2``.
    ``spherical`` is outgoing and equals A*wavelength*exp(i(kr+phase))/r;
    its magnitude is A at r=wavelength. Its source at the origin is excluded
    from the homogeneous equation and is returned as complex NaN, even if A=0.
    """
    points = _finite_coordinates(points, 'points')
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError('points must have a final coordinate axis of length 3.')
    wavelength, amplitude, phase = _parameters(wavelength, amplitude, phase_deg)
    if kind not in ('plane', 'standing', 'spherical'):
        raise ValueError("kind must be 'plane', 'standing', or 'spherical'.")
    k = 2*np.pi/wavelength
    if kind == 'plane':
        return amplitude*np.exp(1j*(k*points[..., 2]+phase))
    if kind == 'standing':
        return amplitude*np.cos(k*points[..., 2])*np.exp(1j*phase)
    radius = np.hypot.reduce(points, axis=-1)
    radial_scale = np.divide(wavelength, radius,
                             out=np.full_like(radius, np.nan), where=radius != 0)
    return amplitude*radial_scale*np.exp(1j*(k*radius+phase))


def boundary_family(z, wavelength=2, backward_fraction=0, phase_deg=0, tau=0):
    """Mix forward/backward solutions that all share U(0)=exp(i phase).

    ``backward_fraction`` b lies in [0, 1]; the component amplitudes are 1-b
    and b. ``derivative`` is dU/dz, which generally differs at z=0 even though
    the field there is identical. ``tau`` is time divided by the period.
    """
    z = _finite_coordinates(z, 'z')
    wavelength, _, phase = _parameters(wavelength, 1, phase_deg)
    backward = _finite_scalar(backward_fraction, 'backward_fraction')
    if not 0 <= backward <= 1:
        raise ValueError('backward_fraction must lie between 0 and 1.')
    tau = _finite_scalar(tau, 'tau')
    forward = 1-backward
    k = 2*np.pi/wavelength
    forward_wave = forward*np.exp(1j*k*z)
    backward_wave = backward*np.exp(-1j*k*z)
    phase_factor = np.exp(1j*phase)
    field = phase_factor*(forward_wave+backward_wave)
    derivative = 1j*k*phase_factor*(forward_wave-backward_wave)
    return dict(field=field, derivative=derivative,
                real_field=np.real(field*np.exp(-2j*np.pi*tau)),
                forward_amplitude=forward, backward_amplitude=backward, k=k)


def slit_far_field(x_mm, width_um=20, wavelength_um=.6, distance_mm=1000):
    """Evaluate the paraxial Fraunhofer pattern of a uniformly lit slit.

    The aperture field has unit amplitude. ``amplitude_scaled`` is the slit
    Fourier integral in micrometres, with the common propagation prefactor
    omitted. ``relative_intensity`` is normalized by this slit's own central
    intensity. It therefore does not retain the width-squared peak scaling.

    NumPy sinc(s)=sin(pi*s)/(pi*s). ``fresnel_number`` uses the full opening width W_open:
    a²/(lambda*L), after converting L from millimetres to micrometres. These
    formulas do not establish the approximation's validity at arbitrary inputs.
    """
    x_mm = _finite_coordinates(x_mm, 'x_mm')
    width = _finite_scalar(width_um, 'width_um', positive=True)
    wavelength = _finite_scalar(wavelength_um, 'wavelength_um', positive=True)
    distance = _finite_scalar(distance_mm, 'distance_mm', positive=True)
    argument = width*x_mm/(wavelength*distance)
    sinc = np.sinc(argument)
    return dict(amplitude_scaled=width*sinc, relative_intensity=sinc*sinc,
                first_zero_mm=wavelength*distance/width,
                fresnel_number=width*width/(wavelength*distance*1000),
                argument=argument)


def slit_propagation(width_um=20, wavelength_um=.6, *, dx_um=.0625,
                     window_um=4096, distances_um=None):
    """Outgoing angular-spectrum propagation of the prescribed long-slit field.

    This is a 1D transverse FFT calculation, uniform along y, using the shared
    Helmholtz spectral branch. It is not an aperture scattering solution.
    The finite FFT window represents periodically repeated copies; a wide
    zero-padded window keeps their contribution small in the displayed crop.
    Edge samples use cell-averaged transmission, retaining the requested width.

    Return only x=-80..80 um (0.5 um spacing) and z=0..1000 um by default.
    The field is indexed [z,x]; intensity is |U|² relative to unit incident
    intensity, without normalizing each plane. One FFT plane is held at a time.
    Sampling parameters allow convergence checks, not arbitrary simulation grids.
    """
    from .scalar_propagation import spectrum_geometry

    width = _finite_scalar(width_um, 'width_um', positive=True)
    wavelength = _finite_scalar(wavelength_um, 'wavelength_um', positive=True)
    dx = _finite_scalar(dx_um, 'dx_um', positive=True)
    window = _finite_scalar(window_um, 'window_um', positive=True)
    count, stride = round(window/dx), round(.5/dx)
    if (window < 320 or width >= window or count < 2 or count > 262144
            or stride < 1 or not np.isclose(count*dx, window)
            or not np.isclose(stride*dx, .5) or count % (2*stride)):
        raise ValueError('Use a centred FFT window and a spacing that divides 0.5 um.')
    distances = (np.linspace(0, 1000, 241) if distances_um is None
                 else _finite_coordinates(distances_um, 'distances_um'))
    if distances.ndim != 1 or not len(distances) or np.any(distances < 0):
        raise ValueError('distances_um must be a nonempty 1D array of nonnegative distances.')
    x = (np.arange(count)-count//2)*dx
    aperture = np.clip((width/2-np.abs(x))/dx+.5, 0, 1)
    spectrum = np.fft.fft(aperture)
    kz = spectrum_geometry(np.fft.fftfreq(count, dx), 0., wavelength)['kz']
    selected = np.arange(count//2-160*stride, count//2+160*stride+1, stride)
    field = np.empty((len(distances), len(selected)), dtype=complex)
    steps = np.diff(distances)
    regular = len(steps) and steps[0] > 0 and np.allclose(steps, steps[0], rtol=1e-12, atol=0)
    # The displayed planes are equally spaced. Reuse their phase increment
    # instead of evaluating a large complex exponential for every plane.
    step_factor = np.exp(1j*kz*steps[0]) if regular else None
    transfer = np.exp(1j*kz*distances[0])
    for index, distance in enumerate(distances):
        if index:
            transfer = transfer*step_factor if regular else np.exp(1j*kz*distance)
        if distance == 0:
            field[index] = aperture[selected]
        else:
            field[index] = np.fft.ifft(spectrum*transfer)[selected]
    return dict(x_um=x[selected], z_um=distances, field=field,
                intensity=np.abs(field)**2, boundary_field=aperture[selected],
                dx_um=dx, window_um=window)
