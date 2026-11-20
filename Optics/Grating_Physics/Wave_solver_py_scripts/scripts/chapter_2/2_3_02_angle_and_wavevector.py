# -*- coding: utf-8 -*-
"""2.3, worked example 2: Recover the grating-angle law from vector components.

Companion topic: Recover the angle equation.
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
period = 3.0
incidence_degrees = 15.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
orders = np.arange(-5, 6)
incident_sine = np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
outgoing_sines = incident_sine + orders * wavelength / period
valid = abs(outgoing_sines) < 1
angles = np.rad2deg(np.arcsin(outgoing_sines[valid]))
recovered_orders = period / wavelength * (np.sin(np.deg2rad(angles)) - incident_sine)

# %% CHECK AND INTERPRET
assert np.allclose(recovered_orders, orders[valid])
print("Propagating order angles (deg):", np.column_stack((orders[valid], angles)))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(orders[valid], angles, 'o-')
    ax.set(xlabel='Order m', ylabel='Outgoing angle from -z (deg)')

    # %% NUMERICAL SUMMARY
    print('2.3 - Recover the grating-angle law from vector components')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
