"""Three-dimensional orders for the notebooks' outgoing negative-z side.

The grating is periodic along x and uniform along y. Wavelength is in the
common lossless exterior medium. Components and fields do not determine
scattering amplitudes or efficiencies.
"""

import numpy as np

from .diffraction import GRAZING_TOLERANCE


def conical_orders(wavelength=2., period=4., alpha_i=.3, beta_i=0.,
                   selected_order=1, orders=None):
    """Return real directions and complex components, retaining the selection.

    alpha_i and beta_i are incident kx/k and ky/k. Their squared sum must be
    strictly less than one. ``components`` contains km/k for every candidate;
    ``directions`` and ``changes`` contain only real (including grazing) orders.
    With no explicit ``orders``, include all real orders, one evanescent neighbour
    on each side, and the selected integer even if it lies farther outside.
    """
    values = np.asarray([wavelength, period, alpha_i, beta_i], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values[:2] <= 0):
        raise ValueError('Use finite parameters and positive wavelength and period.')
    wavelength, period, alpha_i, beta_i = values
    if alpha_i**2 + beta_i**2 >= 1:
        raise ValueError('Incident components must satisfy alpha_i² + beta_i² < 1.')
    if not np.isfinite(selected_order) or selected_order != np.rint(selected_order):
        raise ValueError('The selected order must be a finite integer.')
    chosen = int(selected_order)
    spacing = wavelength / period
    rho = np.sqrt(1 - beta_i**2)
    if orders is None:
        first = int(np.ceil((-rho - alpha_i) / spacing - GRAZING_TOLERANCE))
        last = int(np.floor((rho - alpha_i) / spacing + GRAZING_TOLERANCE))
        labels = np.unique(np.r_[np.arange(first - 1, last + 2), chosen])
    else:
        labels = np.asarray(orders)
        if (labels.ndim != 1 or not labels.size or not np.all(np.isfinite(labels))
                or not np.all(labels == np.rint(labels))):
            raise ValueError('orders must be a nonempty one-dimensional integer sequence.')
        labels = np.unique(labels.astype(int))
        if chosen not in labels:
            raise ValueError('Explicit orders must include selected_order.')
    incident = np.array([alpha_i, beta_i, -np.sqrt(1-alpha_i**2-beta_i**2)])
    a = alpha_i + labels * spacing
    b = np.full_like(a, beta_i)
    rad = 1-a*a-b*b
    grazing = np.abs(rad) <= GRAZING_TOLERANCE
    rad[grazing] = 0.
    valid = rad >= 0
    gamma = -np.sqrt(rad.astype(complex))
    components = np.column_stack([a, b, gamma])
    directions = components[valid].real
    states = np.where(grazing, 'grazing', np.where(valid, 'propagating', 'evanescent'))
    return dict(wavelength=float(wavelength), period=float(period),
                k=2*np.pi/wavelength, spacing=spacing, incident=incident,
                rho=rho, labels=labels, a=a, b=b, rad=rad, valid=valid,
                grazing=grazing, components=components, states=states,
                allowed=labels[valid], directions=directions,
                changes=directions-incident, chosen=chosen,
                selected_index=int(np.flatnonzero(labels == chosen)[0]))


def detector_intersections(data, distance=3., half_width=6.):
    """Intersect real outgoing rays with z=-distance (all lengths in µm).

    The plane is unbounded; ``in_patch`` only says which hits fit in the drawn
    square. Grazing and evanescent orders have no finite ray intersection and
    retain NaN coordinates, rather than acquiring a fictitious detector spot.
    """
    if not np.all(np.isfinite([distance, half_width])) or min(distance, half_width) <= 0:
        raise ValueError('Detector distance and patch half-width must be finite and positive.')
    reaches = data['states'] == 'propagating'
    points = np.full((len(data['labels']), 3), np.nan)
    directions = data['components'][reaches].real
    points[reaches] = (-distance/directions[:, 2])[:, None]*directions
    in_patch = reaches & np.all(abs(points[:, :2]) <= half_width, axis=1)
    return dict(points=points, reaches=reaches, in_patch=in_patch,
                distance=float(distance), half_width=float(half_width))


def conical_phasor(x, y, z, data):
    """Selected component U/A = exp(i km.r), for real coordinates with z <= 0."""
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float),
                                  np.asarray(z, float))
    if any(not np.all(np.isfinite(v)) for v in (x, y, z)) or np.any(z > 0):
        raise ValueError('Coordinates must be finite with z <= 0.')
    kx, ky, kz = data['k'] * data['components'][data['selected_index']]
    return np.exp(1j*(kx*x + ky*y + kz*z))
