"""Scalar plane waves, using angular wavevectors (k = 2 pi / wavelength)."""

import numpy as np


def wavenumber(wavelength):
    """Return the angular wavenumber for a positive wavelength in the medium."""
    wavelength = float(wavelength)
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("Wavelength must be finite and positive.")
    return 2 * np.pi / wavelength


def direction_cosines(theta_deg, phi_deg=0.0):
    """Return (x, y, z) components of a unit direction.

    Polar theta is measured from +z; azimuth phi goes from +x toward +y.
    With phi=0, signed theta also describes the notebook's x-z plane.
    """
    angles = np.asarray([theta_deg, phi_deg], dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Angles must be finite.")
    theta, phi = np.deg2rad(angles)
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def wavevector(wavelength, theta_deg=0.0, phi_deg=0.0):
    """Return a 3D wavevector, in radians per wavelength length unit."""
    return wavenumber(wavelength) * direction_cosines(theta_deg, phi_deg)


def _spatial_phase(points, k_vector):
    """Compute k dot r for coordinate arrays (..., dimensions)."""
    points = np.asarray(points, dtype=float)
    k_vector = np.asarray(k_vector, dtype=float)
    if k_vector.ndim != 1 or k_vector.size == 0:
        raise ValueError("k_vector must be a nonempty one-dimensional vector.")
    if points.ndim == 0 or points.shape[-1] != k_vector.size:
        raise ValueError("The last coordinate axis must match k_vector's dimension.")
    return points @ k_vector


def scalar_plane_wave(points, k_vector, time=0.0, omega=0.0, amplitude=1.0, phase=0.0):
    """Evaluate A cos(k dot r - omega t + phase).

    ``points`` has shape (..., dimensions), and ``k_vector`` has shape
    (dimensions,). For a 1D grid x, pass x[:, None] and [kx]. Time and omega
    must use consistent units; the initial phase is in radians.
    """
    return amplitude * np.cos(_spatial_phase(points, k_vector) - omega * time + phase)


def plane_wave_phasor(points, k_vector, amplitude=1.0, phase=0.0):
    """Return the time-independent phasor A exp(i(k dot r + phase)).

    Use coordinates (..., dimensions), as in ``scalar_plane_wave``. This uses
    the exp(-i omega t) time convention in every spatial dimension.
    """
    return amplitude * np.exp(1j * (_spatial_phase(points, k_vector) + phase))


def field_from_phasor(phasor, time=0.0, omega=0.0):
    """Restore the real field as Re{phasor exp(-i omega t)}."""
    return np.real(np.asarray(phasor) * np.exp(-1j * omega * time))
