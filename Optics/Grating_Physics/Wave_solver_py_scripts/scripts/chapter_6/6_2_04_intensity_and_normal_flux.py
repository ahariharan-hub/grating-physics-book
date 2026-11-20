# -*- coding: utf-8 -*-
"""6.2, worked example 4: Compare field intensity with flux normal to a detector.

Companion topic: Intensity versus normal energy flow.
Lossless homogeneous scalar field, fixed medium. The example is a scalar flux comparison, not a cross-medium irradiance calibration.
Run the entire file in Python or Spyder, or execute its # %% sections in order.
Change INPUT PARAMETERS to investigate the result. No figures are saved automatically.
"""
from pathlib import Path
import sys
import numpy as np

# Locate only the plotting/path helper; the numerical implementation is below.
_scripts_directory = str(Path(__file__).resolve().parents[1])
if _scripts_directory not in sys.path:
    sys.path.insert(0, _scripts_directory)
from _common import pyplot, finish

# %% INPUT PARAMETERS
wavelength = 2.0
amplitude = 1.0
angles_degrees = np.linspace(0, 80, 161)
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
angles = np.deg2rad(angles_degrees)

# %% NUMERICAL CALCULATION
relative_intensity = np.full(angles.size, amplitude**2)
q = k * np.cos(angles)
normal_scalar_flux = q * amplitude**2 / k

# %% CHECK AND INTERPRET
assert np.isclose(normal_scalar_flux[0], relative_intensity[0])
assert np.all(np.diff(normal_scalar_flux) <= 0)
print("A single plane wave can have fixed magnitude but a smaller normal flux at oblique incidence.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(angles_degrees, relative_intensity, label='Relative |U| squared')
    ax.plot(angles_degrees, normal_scalar_flux, label='Normal scalar flux / k')
    ax.set(xlabel='Angle from detector normal (deg)', ylabel='Relative quantity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('6.2 - Compare field intensity with flux normal to a detector')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
