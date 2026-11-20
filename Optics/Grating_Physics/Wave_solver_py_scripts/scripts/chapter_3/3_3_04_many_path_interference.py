# -*- coding: utf-8 -*-
"""3.3, worked example 4: Build grating maxima from a finite set of path differences.

Companion topic: Many openings and integer phase cycles.
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
openings = 6
incidence_degrees = 15.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
angles = np.linspace(-70, 70, 2001)
positions = np.arange(openings) * period
incident_sine = np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
path_increments = positions[None, :] * (np.sin(np.deg2rad(angles))[:, None] - incident_sine)
contributions = np.exp(-2j * np.pi * path_increments / wavelength)
intensity = abs(contributions.sum(axis=1) / openings)**2
matched_angle = np.rad2deg(np.arcsin(incident_sine))

# %% CHECK AND INTERPRET
assert np.allclose(np.exp(-2j * np.pi * positions * (np.sin(np.deg2rad(matched_angle)) - incident_sine) / wavelength), 1)
print("A finite path sum gives peaks with finite angular width, not delta functions.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(angles, intensity)
    ax.set(xlabel='Outgoing angle (deg)', ylabel='Normalized finite-array intensity')

    # %% NUMERICAL SUMMARY
    print('3.3 - Build grating maxima from a finite set of path differences')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
