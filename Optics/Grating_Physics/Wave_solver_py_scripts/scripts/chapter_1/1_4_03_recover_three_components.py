# -*- coding: utf-8 -*-
"""1.4, worked example 3: Retrieve all three wavevector components from phase samples.

Companion topic: Spatial phase in three dimensions.
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
wavelength = 2.0
polar_degrees = 35.0
azimuth_degrees = 60.0
step = 0.05
show_plot = True

# %% SETUP: coordinates, sampling and constants
theta, phi = np.deg2rad([polar_degrees, azimuth_degrees])
k = 2 * np.pi / wavelength
positions = np.vstack([np.zeros(3), step * np.eye(3)])

# %% NUMERICAL CALCULATION
vector = k * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
field_samples = np.exp(1j * (positions @ vector))
recovered = np.angle(field_samples[1:] / field_samples[0]) / step

# %% CHECK AND INTERPRET
assert k * step < np.pi
assert np.allclose(recovered, vector)
print("Small phase increments avoid wrapping ambiguity; recovered k:", recovered)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.plot(['x', 'y', 'z'], vector, 'o-', label='Specified')
    ax.plot(['x', 'y', 'z'], recovered, 'x--', label='Recovered')
    ax.set(ylabel='Wavevector component (rad/um)')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('1.4 - Retrieve all three wavevector components from phase samples')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
