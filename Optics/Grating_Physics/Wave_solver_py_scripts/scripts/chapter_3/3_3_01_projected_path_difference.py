# -*- coding: utf-8 -*-
"""3.3, worked example 1: Convert incident and outgoing path projections into phase.

Companion topic: Connect phase cycles to path length.
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
period = 4.0
incidence_degrees = 0.0
outgoing_degrees = 30.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(0, period, 401)
sine_incident, sine_outgoing = np.sin(np.deg2rad([incidence_degrees, outgoing_degrees]))

# %% NUMERICAL CALCULATION
incident_path = x * sine_incident
outgoing_path = x * sine_outgoing
path_difference = outgoing_path - incident_path
phase_difference = 2 * np.pi * path_difference / wavelength

# %% CHECK AND INTERPRET
assert np.isclose(phase_difference[-1] / (2 * np.pi), period * (sine_outgoing - sine_incident) / wavelength)
print("Extra path per pitch / wavelength:", path_difference[-1] / wavelength)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(x, incident_path, label='Incident projection')
    ax.plot(x, outgoing_path, label='Outgoing projection')
    ax.plot(x, path_difference, '--', label='Extra path')
    ax.set(xlabel='Surface displacement (um)', ylabel='Projected path (um)')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('3.3 - Convert incident and outgoing path projections into phase')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
