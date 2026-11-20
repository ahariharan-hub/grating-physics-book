"""Scalar Helmholtz families and a separate paraxial Gaussian beam.

All fields use exp(-i omega t). SciPy special functions are imported only
when an exact family needs them; the Gaussian beam uses NumPy alone.
"""

import numpy as np

from .scalar_theory import _finite_coordinates, _finite_scalar, _parameters


def helmholtz_family(points, wavelength=2, kind='cylindrical', amplitude=1,
                     phase_deg=0, transverse_ratio=.6, evanescent_ratio=1.25,
                     degree=1):
    """Evaluate an exact scalar Helmholtz family on coordinates (..., 3).

    ``cylindrical`` is outgoing about the y axis: H_0^(1)(k*rho), where
    rho=sqrt(x²+z²). Its normalization makes |U|=A at rho=wavelength.
    The singular source axis returns complex NaN, including when A=0.

    ``bessel`` has its axis along +z, transverse wavenumber k*transverse_ratio
    and longitudinal wavenumber k*sqrt(1-transverse_ratio²). The ratio is in
    [0, 1]; A is its on-axis amplitude. It is an ideal infinitely extended
    solution, not a claim about finite-aperture propagation.

    ``multipole`` is the regular axisymmetric solution
    A*j_l(k*r)*P_l(z/r)*exp(i*phase), proportional to j_l*Y_l0 without the
    spherical-harmonic normalization. Here A is a coefficient, not a peak
    amplitude. Integer degree l is restricted to 0 through 8. At the origin
    the regular value is A*exp(i*phase) for l=0 and zero otherwise.

    ``evanescent`` is A*exp[i(q*x+phase)-alpha*z], with q=k*evanescent_ratio
    and alpha=k*sqrt(evanescent_ratio²-1). The ratio must exceed 1. This helper
    models only the decaying half-space z>=0; values at negative z are rejected.
    Global parameters and coordinates are validated for every family; a family
    parameter is validated only for the family that uses it.
    """
    points = _finite_coordinates(points, 'points')
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError('points must have a final coordinate axis of length 3.')
    wavelength, amplitude, phase = _parameters(wavelength, amplitude, phase_deg)
    if kind not in ('cylindrical', 'bessel', 'multipole', 'evanescent'):
        raise ValueError("kind must be 'cylindrical', 'bessel', 'multipole', or 'evanescent'.")
    k = 2*np.pi/wavelength
    x, y, z = (points[..., index] for index in range(3))

    if kind == 'cylindrical':
        from scipy.special import hankel1

        radius = np.hypot(x, z)
        argument = np.where(radius > 0, k*radius, 1.)
        field = amplitude*hankel1(0, argument)/abs(hankel1(0, 2*np.pi))*np.exp(1j*phase)
        return np.where(radius > 0, field, np.nan+1j*np.nan)

    if kind == 'bessel':
        from scipy.special import j0

        ratio = _finite_scalar(transverse_ratio, 'transverse_ratio')
        if not 0 <= ratio <= 1:
            raise ValueError('transverse_ratio must lie between 0 and 1.')
        radius = np.hypot(x, y)
        kz = k*np.sqrt(1-ratio*ratio)
        return amplitude*j0(k*ratio*radius)*np.exp(1j*(kz*z+phase))

    if kind == 'multipole':
        from scipy.special import eval_legendre, spherical_jn

        order = _finite_scalar(degree, 'degree')
        if order != int(order) or not 0 <= order <= 8:
            raise ValueError('degree must be an integer between 0 and 8.')
        order = int(order)
        radius = np.hypot.reduce(points, axis=-1)
        cosine = np.divide(z, radius, out=np.zeros_like(radius), where=radius != 0)
        return amplitude*spherical_jn(order, k*radius)*eval_legendre(order, cosine)*np.exp(1j*phase)

    ratio = _finite_scalar(evanescent_ratio, 'evanescent_ratio')
    if ratio <= 1:
        raise ValueError('evanescent_ratio must exceed 1.')
    if np.any(z < 0):
        raise ValueError('The decaying evanescent half-space requires z >= 0.')
    q, alpha = k*ratio, k*np.sqrt(ratio*ratio-1)
    return amplitude*np.exp(1j*(q*x+phase)-alpha*z)


def gaussian_beam(points, wavelength=2, amplitude=1, phase_deg=0, waist_ratio=2):
    """Evaluate the fundamental paraxial Gaussian beam along +z.

    Coordinates have shape (..., 3), with the waist centered at z=0. This
    field solves the paraxial envelope equation, not the full Helmholtz
    equation exactly. The time convention is exp(-i omega t).

    ``amplitude`` is the on-axis amplitude at the waist. The waist radius
    w0 = waist_ratio*wavelength is the 1/e field-amplitude radius. The
    teaching model restricts finite waist_ratio >= 2 to moderately or
    weakly focused beams. Both positive and negative z are supported.
    """
    points = _finite_coordinates(points, 'points')
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError('points must have a final coordinate axis of length 3.')
    wavelength, amplitude, phase = _parameters(wavelength, amplitude, phase_deg)
    ratio = _finite_scalar(waist_ratio, 'waist_ratio')
    if ratio < 2:
        raise ValueError('waist_ratio must be at least 2 for this paraxial teaching model.')

    k = 2*np.pi/wavelength
    waist = ratio*wavelength
    rayleigh = k*waist**2/2
    x, y, z = (points[..., index] for index in range(3))
    radius_squared = x*x+y*y
    width = waist*np.sqrt(1+(z/rayleigh)**2)
    reciprocal_curvature = z/(z*z+rayleigh*rayleigh)
    envelope = amplitude*waist/width*np.exp(-radius_squared/width**2)
    spatial_phase = (k*z+k*radius_squared*reciprocal_curvature/2
                     -np.arctan(z/rayleigh)+phase)
    return envelope*np.exp(1j*spatial_phase)
