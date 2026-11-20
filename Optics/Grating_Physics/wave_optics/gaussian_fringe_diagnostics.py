"""Detector-line phase-gradient diagnostics; no new propagation model."""
import numpy as np


def fringe_diagnostics(first, second, x, nominal_spacing):
    """Compare nominal carrier spacing with local 2π/|d arg(U1* U2)/dx|.

    A useful region needs envelope intensity >=1% of its line peak, cross-term
    amplitude >=0.1% of that peak, and local amplitude-balance visibility >=0.1.
    These are display masks, not coherence or experimental detection criteria.
    The local gradient length need not equal adjacent intensity-maxima spacing
    when envelopes or phase gradients vary across a fringe.
    """
    x = np.asarray(x, float)
    a, b = np.asarray(first, complex), np.asarray(second, complex)
    if (x.ndim != 1 or len(x) < 3 or a.shape != x.shape or b.shape != x.shape
            or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0)
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
            or not np.isfinite(nominal_spacing) or nominal_spacing <= 0):
        raise ValueError('Require finite matching fields, increasing x and positive nominal spacing.')
    intensity = abs(a)**2+abs(b)**2
    cross_amplitude = 2*abs(a)*abs(b)
    peak = intensity.max()
    visibility = np.divide(cross_amplitude, intensity, out=np.zeros_like(x), where=intensity > 0)
    phase = np.unwrap(np.angle(np.conj(a)*b))
    gradient = np.gradient(phase, x, edge_order=2)
    mask = ((peak > 0) & (intensity >= .01*peak)
            & (cross_amplitude >= .001*peak) & (visibility >= .1)
            & (abs(gradient) > 1e-9))
    mask[[0, -1]] = False
    spacing = np.divide(2*np.pi, abs(gradient), out=np.full_like(x, np.nan), where=mask)
    center = int(np.argmax(np.where(mask, cross_amplitude, -1.))) if mask.any() else None
    return dict(x=x, phase=phase, gradient=gradient, spacing=spacing, mask=mask,
                nominal=float(nominal_spacing), center=center,
                local=float(spacing[center]) if center is not None else np.nan,
                envelope=intensity, cross_amplitude=cross_amplitude,
                visibility=visibility)
