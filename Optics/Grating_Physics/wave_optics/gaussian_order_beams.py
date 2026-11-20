"""Prescribed rotated Gaussian beams for Notebook 7.3; no grating solver.

Lengths are in micrometres, wavelength in the medium, time factor exp(-iωt).
Each waist centre is the origin; its plane is perpendicular to that beam's
axis, not the common grating plane. The axisymmetric helper is sampled at y=0
for the xz illustration. Detector overlap below is a line diagnostic at y=0.
"""
from dataclasses import dataclass
import numpy as np
from .helmholtz_families import gaussian_beam


@dataclass(frozen=True)
class GaussianOrderConfiguration:
    wavelength: float = 1.
    period: float = 4.
    waist: float = 6.

    def __post_init__(self):
        values = np.asarray((self.wavelength, self.period, self.waist), float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError('Wavelength, period and waist must be finite and positive.')
        if self.waist/self.wavelength < 3:
            raise ValueError('Require w0/λ >= 3 for this weak-divergence teaching model.')

    @property
    def rayleigh(self):
        return np.pi*self.waist**2/self.wavelength


class GaussianOrderExperiment:
    """Normal incidence; assign equal coefficients unless explicitly supplied.

    Offered axes satisfy |sin θ| <= 0.5 (30 degrees from -z). This is a
    detector-geometry restriction, separate from local paraxiality w0/λ >= 3.
    No binary-mask efficiencies or output profiles are inferred.
    """
    def __init__(self, configuration=None):
        self.configuration = configuration or GaussianOrderConfiguration()
        c = self.configuration
        self.k = 2*np.pi/c.wavelength
        self.G = 2*np.pi/c.period
        count = int(np.floor(.5*c.period/c.wavelength + 1e-12))
        self.orders = tuple(range(-count, count+1))

    def basis(self, order):
        if not np.isscalar(order) or not np.isfinite(order) or order != int(order):
            raise ValueError('Order must be a finite integer.')
        sine = int(order)*self.configuration.wavelength/self.configuration.period
        if abs(sine) >= 1:
            raise ValueError('Order is nonpropagating or grazing.')
        if abs(sine) > .5+1e-12:
            raise ValueError('This explorer restricts axes to 30 degrees from -z.')
        cosine = np.sqrt(1-sine*sine)
        # e_q × e_y = e_l: a right-handed orthonormal local coordinate system.
        return np.array([-cosine, 0., -sine]), np.array([sine, 0., -cosine])

    def local_coordinates(self, points, order):
        points = np.asarray(points, float)
        if points.ndim == 0 or points.shape[-1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError('Require finite points (..., 3).')
        transverse, direction = self.basis(order)
        return np.stack((points@transverse, points[..., 1], points@direction), axis=-1)

    def field(self, points, order, coefficient=1.+0j):
        if not np.isscalar(coefficient) or not np.isfinite(coefficient):
            raise ValueError('Coefficient must be a finite complex scalar.')
        c = self.configuration
        local = self.local_coordinates(points, order)
        return coefficient*gaussian_beam(local, wavelength=c.wavelength,
                                         waist_ratio=c.waist/c.wavelength)

    def sample(self, x, depths, pair=(0, 1), coefficients=(1., 1.)):
        x, depths = np.asarray(x, float), np.asarray(depths, float)
        if (x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x))
                or np.any(np.diff(x) <= 0)):
            raise ValueError('x must contain increasing finite detector coordinates.')
        if (depths.ndim != 1 or not len(depths) or not np.all(np.isfinite(depths))
                or np.any(depths < 0)):
            raise ValueError('Depths must be a nonempty finite nonnegative vector.')
        if len(pair) != 2 or pair[0] == pair[1] or len(coefficients) != 2:
            raise ValueError('Require two distinct orders and two coefficients.')
        xx, ss = np.meshgrid(x, depths)
        points = np.stack((xx, np.zeros_like(xx), -ss), axis=-1)
        fields = np.stack([self.field(points, m, a) for m, a in zip(pair, coefficients)])
        intensities = abs(fields)**2
        total = fields.sum(axis=0)
        cross = 2*np.real(np.conj(fields[0])*fields[1])
        return dict(x=x, depth=depths, fields=fields, intensities=intensities,
                    total=total, intensity=abs(total)**2, cross=cross,
                    overlap=line_overlap(fields[0], fields[1], x))


def line_overlap(first, second, x):
    """Normalized ∫|U1||U2|dx on the declared detector line; zero => NaN.

    Normalize magnitudes before products to avoid strength-induced underflow.
    This is neither full XY overlap, mutual coherence nor fringe visibility.
    """
    a, b = np.broadcast_arrays(abs(np.asarray(first)), abs(np.asarray(second)))
    x = np.asarray(x, float)
    if (a.ndim == 0 or a.shape[-1] != x.size or x.ndim != 1 or len(x) < 2
            or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0)
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
        raise ValueError('Finite fields must match increasing detector coordinates.')
    an, bn = a.max(axis=-1, keepdims=True), b.max(axis=-1, keepdims=True)
    a = np.divide(a, an, out=np.zeros_like(a, dtype=float), where=an > 0)
    b = np.divide(b, bn, out=np.zeros_like(b, dtype=float), where=bn > 0)
    # NumPy 2.4 removed trapz; keep older course environments supported too.
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    norm = np.sqrt(integrate(a*a, x, axis=-1)*integrate(b*b, x, axis=-1))
    value = np.divide(integrate(a*b, x, axis=-1), norm,
                      out=np.full_like(norm, np.nan), where=norm > 0)
    return np.clip(value, 0., 1.)
