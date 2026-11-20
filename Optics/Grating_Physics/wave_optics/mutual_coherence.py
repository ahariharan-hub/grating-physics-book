"""Notebook 7 V2: prescribed scalar correlations, not a stochastic source solver.

Fields use exp(-i omega t), phase 2 minus phase 1, and the existing relative
scalar-intensity convention. An arm of zero intensity has undefined normalized
coherence; that does not prevent calculation of the detector intensity.
"""

import numpy as np

from .diffraction import diffraction_orders
from .superposition import two_wave_field
from .waves import wavenumber


def _weights(weights, count):
    values = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    if (values.shape != (count,) or not np.isfinite(values).all()
            or np.any(values < 0) or not np.isfinite(np.sum(values)) or not np.sum(values) > 0):
        raise ValueError('Use finite nonnegative sample weights with positive total.')
    return values / np.sum(values)


def mutual_statistics(field1, field2, *, weights=None):
    """One-point finite ensemble/exposure average; samples and weights are explicit.

    gamma is None if either arm is dark. intensity averages |U1+U2|^2,
    independently of the expanded cross-term calculation.
    """
    first, second = np.asarray(field1, complex), np.asarray(field2, complex)
    if (first.ndim != 1 or first.size == 0 or first.shape != second.shape
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError('Provide matching nonempty finite one-dimensional field samples.')
    weight = _weights(weights, first.size)
    i1, i2 = float(weight @ abs(first)**2), float(weight @ abs(second)**2)
    correlation = complex(weight @ (first.conjugate() * second))
    gamma = correlation / (np.sqrt(i1) * np.sqrt(i2)) if i1 > 0 and i2 > 0 else None
    return dict(intensity1=i1, intensity2=i2, correlation=correlation, gamma=gamma,
                cross_term=2*correlation.real,
                intensity=float(weight @ abs(first + second)**2))


def phase_average(phases_deg, *, weights=None):
    """Weighted mean exp(i beta); deterministic modulation is not intrinsic incoherence."""
    phase = np.asarray(phases_deg, dtype=float)
    if phase.ndim != 1 or phase.size == 0 or not np.isfinite(phase).all():
        raise ValueError('Provide nonempty finite phase samples in degrees.')
    return complex(_weights(weights, phase.size) @ np.exp(1j*np.deg2rad(phase)))


def averaged_intensity(intensity1, intensity2, gamma=1+0j, phase=0.):
    """I1+I2+2 sqrt(I1 I2) Re(gamma exp(i phase)); phase is in radians.

    gamma is a prescribed reference correlation, before the supplied spatial
    or controlled phase. None is permitted only when an arm is dark.
    """
    i1, i2 = float(intensity1), float(intensity2)
    phase = np.asarray(phase, dtype=float)
    if not np.isfinite([i1, i2]).all() or min(i1, i2) < 0 or not np.isfinite(phase).all():
        raise ValueError('Use finite nonnegative arm intensities and finite phase.')
    if gamma is None:
        if i1 > 0 and i2 > 0:
            raise ValueError('Nonzero arms require a specified normalized correlation.')
        return np.full_like(phase, i1+i2)
    gamma = complex(gamma)
    if not np.isfinite(gamma) or abs(gamma) > 1+1e-12:
        raise ValueError('Normalized correlation must have magnitude at most one.')
    return i1+i2+2*np.sqrt(i1)*np.sqrt(i2)*np.real(gamma*np.exp(1j*phase))


def fringe_visibility(intensity1, intensity2, magnitude=1.):
    """Resolved phase-scan visibility; zero for one dark arm, None for both dark."""
    i1, i2, magnitude = map(float, (intensity1, intensity2, magnitude))
    if not np.isfinite([i1, i2, magnitude]).all() or min(i1, i2) < 0 or not 0 <= magnitude <= 1:
        raise ValueError('Use nonnegative finite intensities and coherence magnitude in [0,1].')
    return 2*np.sqrt(i1)*np.sqrt(i2)/(i1+i2)*magnitude if i1+i2 > 0 else None


def two_order_basis(x_um, *, detector_z_um=-1.5):
    """Prescribed m=(0,+1), lambda=2 um, d=4 um, normal incidence, into -z.

    Both |k|=pi/um. C0=C1=.4 are example field coefficients, not a scattering
    solution or flux efficiencies. Geometry is independent of pedagogical phase.
    """
    x = np.asarray(x_um, float)
    if x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(detector_z_um) or detector_z_um >= 0:
        raise ValueError('Use finite detector x positions and z < 0 in micrometres.')
    directions = diffraction_orders(2., 4., 0., 0., orders=[0, 1])['directions'].copy()
    directions[:, 2] *= -1
    vectors = wavenumber(2.) * directions
    points = np.column_stack((x, np.zeros_like(x), np.full_like(x, detector_z_um)))
    data = two_wave_field(points, vectors, .4, .4)
    return dict(x=x, vectors=vectors, field1=data['phasors'][0], field2=data['phasors'][1],
                spatial_phase=data['phase_difference'], period_um=4.)


def two_order_intensity(basis, amplitude_ratio=1., phase_deg=0., magnitude=1.):
    """Reuse fixed phase geometry; vary prescribed coefficient ratio/phase/correlation."""
    magnitude = float(magnitude)
    if not np.isfinite(magnitude) or not 0 <= magnitude <= 1:
        raise ValueError('Coherence magnitude must lie in [0,1].')
    ratio = float(amplitude_ratio)
    if not np.isfinite(ratio) or ratio < 0 or not np.isfinite(phase_deg):
        raise ValueError('Use a finite nonnegative amplitude ratio and finite phase.')
    return averaged_intensity(.4**2, (.4*ratio)**2, magnitude,
                              basis['spatial_phase']+np.deg2rad(phase_deg))
