"""Two coherent scalar waves using exp(-i omega t) and fixed field units.

Relative scalar intensity is |U1 + U2|^2 with a unit-amplitude reference.
Phase difference is always phase 2 minus phase 1. This module assigns
amplitudes; it does not solve a grating scattering or power-budget problem.
"""

import numpy as np

from .diffraction import diffraction_orders
from .waves import field_from_phasor, plane_wave_phasor, wavenumber


def _parameters(amplitude1, amplitude2, phase_difference_deg, phase1_deg):
    values = np.asarray([amplitude1, amplitude2, phase_difference_deg, phase1_deg], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values[:2] < 0):
        raise ValueError('Amplitudes must be nonnegative and all parameters finite.')
    return values


def two_wave_cycle(amplitude1=1.0, amplitude2=1.0, phase_difference_deg=0.0,
                   *, phase1_deg=0.0, sample_count=400):
    """Evaluate the two contributions and their sum at one fixed position.

    Cycle means sample [0, 1) uniformly, without a duplicated endpoint.
    The phase of a zero resultant is reported as None.
    """
    a1, a2, delta, phi1 = _parameters(amplitude1, amplitude2,
                                     phase_difference_deg, phase1_deg)
    if not isinstance(sample_count, (int, np.integer)) or sample_count < 8:
        raise ValueError('Use at least eight samples per complete cycle.')
    phasors = np.array([a1*np.exp(1j*np.deg2rad(phi1)),
                       a2*np.exp(1j*np.deg2rad(phi1+delta))])
    total = phasors.sum()
    tau = np.linspace(0, 1, sample_count, endpoint=False)
    fields = field_from_phasor(phasors[:, None], time=tau, omega=2*np.pi)
    field_sum = fields.sum(axis=0)
    result_phase = (None if abs(total) <= 1e-12*max(a1+a2, 1.0)
                    else float(np.rad2deg(np.angle(total))))
    return dict(tau=tau, phasors=phasors, total_phasor=total, fields=fields,
                field_sum=field_sum, amplitude=float(abs(total)),
                phase_deg=result_phase, mean_field=float(np.mean(field_sum)),
                mean_square=float(np.mean(field_sum**2)),
                relative_intensity=float(abs(total)**2),
                separate_intensities=float(a1*a1+a2*a2),
                interference_term=float(2*np.real(phasors[0]*phasors[1].conjugate())))


def symmetric_wavevectors(wavelength=2.0, half_angle_deg=30.0):
    """Return (k1, k2) at angles (-theta, +theta) about -z; ky=0."""
    k = wavenumber(wavelength)
    theta = float(half_angle_deg)
    if not np.isfinite(theta) or not 0 <= theta < 90:
        raise ValueError('Half-angle must be finite, from 0 to less than 90 degrees.')
    sine, cosine = np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))
    return k*np.array([[-sine, 0, -cosine], [sine, 0, -cosine]])


def grating_pair_wavevectors():
    """Fixed m=(-1,+1), lambda=2 um, d=4 um, normal incidence, outgoing -z."""
    directions = diffraction_orders(2.0, 4.0, 0.0, 0.0, orders=[-1, 1])['directions'].copy()
    directions[:, 2] *= -1
    return wavenumber(2.0)*directions


def two_wave_field(points, wavevectors, amplitude1=1.0, amplitude2=1.0,
                   phase_difference_deg=0.0, *, phase1_deg=0.0, tau=0.0):
    """Fields at (..., 3) points for two real, equal-wavenumber plane waves.

    tau is t/T. Both components share the same time factor. The relative
    scalar intensity is independent of tau, including when the instantaneous
    sum happens to vanish everywhere on a particular observation plane.
    """
    a1, a2, delta, phi1 = _parameters(amplitude1, amplitude2,
                                     phase_difference_deg, phase1_deg)
    vectors = np.asarray(wavevectors, dtype=float)
    if vectors.shape != (2, 3) or not np.all(np.isfinite(vectors)):
        raise ValueError('Provide two finite real wavevectors with shape (2, 3).')
    lengths = np.linalg.norm(vectors, axis=1)
    if lengths[0] == 0 or not np.isclose(lengths[0], lengths[1], rtol=1e-12, atol=0):
        raise ValueError('The two waves must have the same positive wavenumber.')
    if not np.isfinite(tau):
        raise ValueError('Normalized time must be finite.')
    u1 = plane_wave_phasor(points, vectors[0], amplitude=a1, phase=np.deg2rad(phi1))
    u2 = plane_wave_phasor(points, vectors[1], amplitude=a2, phase=np.deg2rad(phi1+delta))
    components = np.stack((u1, u2))
    total = u1+u2
    return dict(phasors=components, total_phasor=total,
                fields=field_from_phasor(components, time=tau, omega=2*np.pi),
                field_sum=field_from_phasor(total, time=tau, omega=2*np.pi),
                relative_intensity=np.abs(total)**2,
                phase_difference=np.asarray(points) @ (vectors[1]-vectors[0])+np.deg2rad(delta))


def ideal_camera_profile(x, detector_z=-1.0, amplitude2=1.0,
                         phase_difference_deg=0.0, *, tau=0.0,
                         amplitude1=1.0, phase1_deg=0.0,
                         dark_signal=10.0, reference_signal=100.0):
    """Expected small-pixel camera readings for the fixed symmetric order pair.

    The detector lies in z < 0, parallel to the grating. Exposure, pixel area,
    responsivity and the common incidence-angle factor are absorbed into the
    calibrated reference_signal (DN above dark for a unit-amplitude wave). Sampling
    assumes pixels much narrower than a fringe; noise, saturation and finite
    pixel averaging are omitted. DN denotes digital numbers, not photons.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError('Provide a finite one-dimensional array of detector positions.')
    if not np.isfinite(detector_z) or detector_z >= 0:
        raise ValueError('The detector must lie in the transmitted half-space z < 0.')
    if (not np.isfinite(dark_signal) or dark_signal < 0 or
        not np.isfinite(reference_signal) or reference_signal <= 0):
        raise ValueError('Use a nonnegative dark signal and a positive reference signal.')
    points = np.column_stack((x, np.zeros_like(x), np.full_like(x, detector_z)))
    data = two_wave_field(points, grating_pair_wavevectors(), amplitude1=amplitude1,
                          amplitude2=amplitude2, phase_difference_deg=phase_difference_deg,
                          phase1_deg=phase1_deg, tau=tau)
    data['x'] = x
    data['signal_dn'] = dark_signal + reference_signal*data['relative_intensity']
    return data
