"""Wave calculations for the optics notebooks.

Plotting and Jupyter controls are available in ``plotting`` and ``widgets``.
Importing this package only loads the numerical functions and NumPy.
"""

from .waves import (
    direction_cosines, field_from_phasor, plane_wave_phasor,
    scalar_plane_wave, wavenumber, wavevector,
)
from .diffraction import diffraction_orders, sphere_residuals
from .coordinates import wavevector_coordinates

__all__ = [
    "direction_cosines",
    "field_from_phasor",
    "plane_wave_phasor",
    "scalar_plane_wave",
    "wavenumber",
    "wavevector",
    "diffraction_orders",
    "sphere_residuals",
    "wavevector_coordinates",
]
