# -*- coding: utf-8 -*-
"""3.3, worked example 3: Recover a diffraction angle from an integer path increment.

Companion topic: From path difference to the order angle.
Lengths are micrometres. Wavelength is in the medium. Time convention: exp(-i omega t).
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
wavelength = 1.0
period = 4.0
incidence_degrees = 10.0
orders = np.arange(-4, 5)
show_plot = True

# %% SETUP: coordinates, sampling and constants
incident_sine = np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
extra_paths = orders * wavelength
candidate_sines = incident_sine + extra_paths / period
valid = abs(candidate_sines) < 1
angles = np.arcsin(candidate_sines[valid])
reconstructed_paths = period * (np.sin(angles) - incident_sine)

# %% CHECK AND INTERPRET
assert np.allclose(reconstructed_paths, extra_paths[valid])
print("Angles exist only where the path-based direction cosine is physical.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(orders[valid], np.rad2deg(angles), 'o-')
    ax.set(xlabel='Extra path in wavelengths / order m', ylabel='Outgoing angle (deg)')

    # %% NUMERICAL SUMMARY
    print('3.3 - Recover a diffraction angle from an integer path increment')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
