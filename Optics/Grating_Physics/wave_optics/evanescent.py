"""Individual diffraction harmonics in the homogeneous half-space z <= 0.

The period is along x, ky=0, and wavelength is measured in the lossless
medium. The time convention is exp(-i omega t). A unit surface amplitude is
prescribed; this module does not solve scattering or diffraction efficiency.
"""

import numpy as np

from .diffraction import GRAZING_TOLERANCE
from .waves import wavenumber


def evanescent_order(wavelength=2., period=2., angle_deg=30., order=1):
    """Classify one integer order and choose its outgoing/decaying kz branch.

    q = -kz is the component along s=-z, the distance into the outgoing side.
    q is positive real for propagation, zero at grazing, and +i*kappa for
    evanescence. The dispersion relation uses squares, not absolute squares.
    """
    k = wavenumber(wavelength)
    if not np.isfinite(period) or period <= 0:
        raise ValueError('Period must be finite and positive.')
    if not np.isfinite(angle_deg) or not -90 < angle_deg < 90:
        raise ValueError('Use a finite incidence angle between -90 and 90 degrees.')
    if not np.isfinite(order) or order != np.rint(order):
        raise ValueError('The diffraction-order number must be an integer.')
    order = int(order)
    kix = k*np.sin(np.deg2rad(angle_deg))
    kx = kix + order*2*np.pi/period
    alpha = kx/k
    rad = 1-alpha**2
    if abs(rad) <= GRAZING_TOLERANCE:
        status, q = 'grazing', 0j
    elif rad > 0:
        status, q = 'propagating', complex(k*np.sqrt(rad))
    else:
        status, q = 'evanescent', complex(0, k*np.sqrt(-rad))
    kappa = q.imag
    angle = (None if status == 'evanescent'
             else float(np.rad2deg(np.arcsin(np.clip(alpha, -1, 1)))))
    return dict(wavelength=float(wavelength), period=float(period),
                angle_deg=float(angle_deg), order=order, k=k, kix=kix, kx=kx,
                alpha=alpha, radicand=rad, q=q, kz=-q, kappa=kappa,
                decay_length=1/kappa if kappa > 0 else np.inf,
                status=status, outgoing_angle=angle)


def order_phasor(x, z, data):
    """Return U/A = exp(i*kx*x + i*kz*z) on broadcast coordinates, z <= 0.

    Take Re{U/A * exp(-2*pi*i*tau)} for the signed field at tau=t/T.
    Do not pass complex wavevectors to the older real-wave helpers.
    """
    x, z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(z, dtype=float))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(z)) or np.any(z > 0):
        raise ValueError('Coordinates must be finite and on the outgoing side z <= 0.')
    return np.exp(1j*(data['kx']*x + data['kz']*z))


def decay_envelope(distance, data):
    """Amplitude relative to the prescribed surface amplitude; not power."""
    distance = np.asarray(distance, dtype=float)
    if not np.all(np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError('Distance from the grating must be finite and nonnegative.')
    return np.exp(-data['kappa']*distance)
