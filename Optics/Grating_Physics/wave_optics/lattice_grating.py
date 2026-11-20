"""2D prescribed thin masks; vacuum wavelength, lossless media, outgoing -z.

Lengths share one unit. Lattice vectors are columns; sampled cells use [y, x].
No material scattering or lens field transformation is computed.
"""
import numpy as np
from .diffraction import GRAZING_TOLERANCE


def reciprocal_basis(direct):
    """Return columns b_j satisfying a_i dot b_j = 2*pi*delta_ij."""
    a = np.asarray(direct, float)
    if a.shape != (2, 2) or not np.all(np.isfinite(a)):
        raise ValueError('direct must be a finite 2 by 2 matrix of columns.')
    if abs(np.linalg.det(a)) <= np.finfo(float).eps*np.linalg.norm(a)**2:
        raise ValueError('Lattice vectors must be independent.')
    return 2*np.pi*np.linalg.inv(a).T


def rectangular_coefficients(labels, duty=(.5, .5), shift_fraction=(0., 0.)):
    """Centered clear rectangle shifted by fractions of its physical periods."""
    mn = np.asarray(labels)
    d, shift = np.asarray(duty, float), np.asarray(shift_fraction, float)
    if (mn.ndim != 2 or mn.shape[1] != 2 or not np.all(np.isfinite(mn))
            or not np.all(mn == np.rint(mn))):
        raise ValueError('labels must have integer shape (N, 2).')
    if (d.shape != (2,) or shift.shape != (2,) or not np.all(np.isfinite(shift))
            or not np.all(np.isfinite(d)) or np.any((d < 0) | (d > 1))):
        raise ValueError('Use two finite duty cycles in [0, 1] and two shifts.')
    return np.prod(d*np.sinc(mn*d), axis=1)*np.exp(-2j*np.pi*(mn@shift))


def sampled_coefficients(samples, labels, origin=(-.5, -.5)):
    """Unit-cell mean of t*exp(-i2pi(mx/px+ny/py)).

    Samples exclude repeated endpoints: x/px=origin[0]+j/Nx,
    y/py=origin[1]+l/Ny. FFT aliases harmonics; use resolved labels only.
    """
    t = np.asarray(samples, complex)
    mn = np.asarray(labels)
    o = np.asarray(origin, float)
    if t.ndim != 2 or min(t.shape) < 1 or not np.all(np.isfinite(t)):
        raise ValueError('samples must be a finite nonempty 2D array.')
    if (mn.ndim != 2 or mn.shape[1] != 2 or not np.all(np.isfinite(mn))
            or not np.all(mn == np.rint(mn))):
        raise ValueError('labels must have integer shape (N, 2).')
    if o.shape != (2,) or not np.all(np.isfinite(o)):
        raise ValueError('origin must have two finite fractional coordinates.')
    ny, nx = t.shape
    if np.any(abs(mn[:, 0]) >= nx/2) or np.any(abs(mn[:, 1]) >= ny/2):
        raise ValueError('Retain labels strictly inside the sampled Nyquist limits.')
    mn = mn.astype(int)
    f = np.fft.fft2(t)/t.size
    return f[mn[:, 1] % ny, mn[:, 0] % nx]*np.exp(-2j*np.pi*(mn@o))


def lattice_orders(direct, labels, wavelength0=.193, n_in=1., n_out=1.,
                   incident=(0., 0.), NA_obj=None):
    """All supplied labels, including zero-amplitude and evanescent candidates.

    incident=(alpha_inc,beta_inc) uses n_in*k0. Geometry needs no objective.
    Supply NA_obj only for compatibility with earlier callers; it delegates to
    collect_orders after geometry. Prefer the two explicit steps in new work.
    """
    b = reciprocal_basis(direct)
    mn = np.asarray(labels)
    if (mn.ndim != 2 or mn.shape[1] != 2 or not np.all(np.isfinite(mn))
            or not np.all(mn == np.rint(mn))):
        raise ValueError('labels must have integer shape (N, 2).')
    values = np.asarray([wavelength0, n_in, n_out], float)
    inc = np.asarray(incident, float)
    if (not np.all(np.isfinite(values)) or np.any(values <= 0)
            or inc.shape != (2,) or not np.all(np.isfinite(inc)) or inc@inc >= 1):
        raise ValueError('Positive wavelength/indices and |incident| < 1 required.')
    k0 = 2*np.pi/wavelength0
    transverse = n_in*k0*inc + mn@b.T
    rad = n_out**2 - np.sum((transverse/k0)**2, axis=1)
    grazing = abs(rad) <= GRAZING_TOLERANCE*n_out**2
    rad[grazing] = 0.
    kz = -k0*np.sqrt(rad.astype(complex))
    status = np.where(grazing, 'grazing', np.where(rad > 0, 'propagating', 'evanescent'))
    k = np.column_stack([transverse, kz])
    data = dict(labels=mn.astype(int), k=k, f=k/(2*np.pi),
                components=k/(n_out*k0), status=status, k0=k0,
                wavelength0=float(wavelength0), n_in=float(n_in), n_out=float(n_out),
                incident=inc.copy(), reciprocal=b)
    return collect_orders(data, NA_obj) if NA_obj is not None else data


def collect_orders(data, NA_obj):
    """Return angular sigma coordinates and collection, retaining every label/vector.

    The sigma key is k_parallel/(k0*NA_obj). The q key is a compatibility alias;
    the teaching notation uses sigma for these angular coordinates.

    NA is in the outgoing medium. Grazing does not reach a downstream lens.
    Source identity or coefficient entries in data are preserved unchanged.
    """
    if not np.isfinite(NA_obj) or not 0 < NA_obj <= data['n_out']:
        raise ValueError('Use finite 0 < NA_obj <= n_out.')
    q = data['k'][:, :2].real/(data['k0']*NA_obj)
    collected = ((data['status'] == 'propagating') &
                 (np.sum(q*q, axis=1) <= 1+GRAZING_TOLERANCE))
    return dict(data, sigma=q, q=q, collected=collected, NA_obj=float(NA_obj))
