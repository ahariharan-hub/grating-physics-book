"""Coordinate descriptions of real plane waves in an isotropic medium."""

import numpy as np

from .waves import direction_cosines, wavenumber


def wavevector_coordinates(vacuum_wavelength, refractive_index=1.0,
                           theta_deg=40.0, phi_deg=30.0,
                           numerical_aperture=0.8):
    """Describe one wave in Cartesian, spherical, cylindrical and pupil form.

    Wavelength is in vacuum, unlike ``waves.wavevector`` which takes the
    wavelength in the medium. All lengths use the same arbitrary unit.
    Theta is polar from +z (0..180 degrees); phi is azimuth from +x toward +y.
    Pupil coordinates (sigma_x, sigma_y) use same-side NA and an ideal circular +z collection
    cone. They apply as physical pupil coordinates under a sine mapping.
    At either pole, azimuth is undefined and is returned as None.
    """
    k0 = wavenumber(vacuum_wavelength)
    n = float(refractive_index)
    na = float(numerical_aperture)
    theta = float(theta_deg)
    phi = float(phi_deg)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("Refractive index must be finite and positive.")
    if not np.isfinite(na) or not 0 < na <= n:
        raise ValueError("Numerical aperture must satisfy 0 < NA <= n.")
    if not np.isfinite(theta) or not 0 <= theta <= 180:
        raise ValueError("Polar angle must lie between 0 and 180 degrees.")
    if not np.isfinite(phi):
        raise ValueError("Azimuth must be finite.")
    phi = (phi + 180) % 360 - 180
    direction = direction_cosines(theta, phi)
    direction[np.abs(direction) < 1e-14] = 0.0
    k = n * k0
    cartesian = k * direction
    transverse = float(np.hypot(*cartesian[:2]))
    azimuth = None if transverse == 0 else phi
    pupil = cartesian[:2] / (k0 * na)
    pupil_radius = float(np.linalg.norm(pupil))
    hemisphere = ('+z' if direction[2] > 0 else
                  '-z' if direction[2] < 0 else 'grazing')
    return dict(
        vacuum_wavelength=float(vacuum_wavelength), refractive_index=n,
        numerical_aperture=na, medium_wavelength=float(vacuum_wavelength) / n,
        k0=k0, k=k, cartesian=cartesian,
        spherical=(k, theta, azimuth),
        cylindrical=(transverse, azimuth, float(cartesian[2])),
        direction_cosines=direction, spatial_frequency=cartesian / (2 * np.pi),
        pupil=pupil, pupil_radius=pupil_radius, hemisphere=hemisphere,
        collected=bool(direction[2] >= 0 and pupil_radius <= 1 + 1e-12),
    )
