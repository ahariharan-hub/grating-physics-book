"""Transmission-order kinematics for a 1D surface grating.

The period runs along x, grooves along y, and transmitted waves travel toward
+z. Both sides have the same homogeneous isotropic medium. Wavelength means
wavelength in that medium; wavelength and period must use the same units.
This module determines allowed directions, without calculating efficiencies.
"""

import numpy as np

from .waves import direction_cosines

GRAZING_TOLERANCE = 1e-12


def diffraction_orders(wavelength, period, theta_deg, phi_deg, orders=None, selected_order=1):
    """Return direction-cosine and normalized Ewald coordinates.

    If ``orders`` is omitted, include all real orders and the nearest
    evanescent candidate on each side. Explicit order labels may instead be
    supplied as a nonempty 1D integer sequence.

    ``labels``, ``a``, ``b``, ``rad``, ``valid``, and ``grazing`` include all
    candidates. ``allowed``, ``directions`` (= km/k), and ``changes`` (= Δk/k)
    contain only real directions, including grazing orders. Here
    Δk = km - ki. Evanescent candidates have no real direction on the sphere.
    """
    inputs = np.asarray([wavelength, period, theta_deg, phi_deg], dtype=float)
    if not np.all(np.isfinite(inputs)):
        raise ValueError("All inputs must be finite.")
    wavelength, period, theta_deg, phi_deg = inputs
    if wavelength <= 0 or period <= 0:
        raise ValueError("Wavelength and period must be positive.")
    if not 0 <= theta_deg < 90:
        raise ValueError("Use 0 <= theta_deg < 90 for incidence from z < 0.")

    incident = direction_cosines(theta_deg, phi_deg)
    spacing = wavelength / period
    rho = np.sqrt(max(0.0, 1 - incident[1]**2))
    if orders is None:
        first = int(np.ceil((-rho - incident[0]) / spacing - GRAZING_TOLERANCE))
        last = int(np.floor((rho - incident[0]) / spacing + GRAZING_TOLERANCE))
        labels = np.arange(first - 1, last + 2)
    else:
        labels = np.asarray(orders)
        if labels.ndim != 1 or labels.size == 0 or not np.all(np.isfinite(labels)):
            raise ValueError("orders must be a nonempty one-dimensional array of integers.")
        if not np.all(labels == np.rint(labels)):
            raise ValueError("Diffraction-order labels must be integers.")
        labels = labels.astype(int)

    a = incident[0] + labels * spacing
    b = np.full(len(labels), incident[1])
    rad = 1 - a*a - b*b
    valid = rad >= -GRAZING_TOLERANCE
    grazing = np.abs(rad) <= GRAZING_TOLERANCE
    g = np.sqrt(np.maximum(rad, 0))
    directions = np.column_stack([a[valid], b[valid], g[valid]])
    changes = directions - incident
    allowed = labels[valid]
    if selected_order in allowed:
        chosen = int(selected_order)
    elif 0 in allowed:
        chosen = 0
    else:
        chosen = int(allowed[0]) if allowed.size else None
    return dict(incident=incident, spacing=spacing, rho=rho, labels=labels,
                a=a, b=b, rad=rad, valid=valid, grazing=grazing,
                directions=directions, changes=changes, allowed=allowed,
                chosen=chosen)


def sphere_residuals(data):
    """Return maximum (Harvey, Ewald) sphere residuals, or None if no real orders."""
    if not np.any(data["valid"]):
        return None
    harvey = np.max(np.abs(np.sum(data["directions"]**2, axis=1) - 1))
    ewald = np.max(np.abs(np.sum((data["changes"] + data["incident"])**2, axis=1) - 1))
    return float(harvey), float(ewald)
