# -*- coding: utf-8 -*-
"""2.3, worked example 3: Locate a diffraction-order cutoff in wavelength.

Companion topic: Allowed and nonpropagating orders.
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
period = 3.0
order = 2
incidence_degrees = 0.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
wavelengths = np.linspace(0.5, 2.0, 501)
incident_sine = np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
alpha = incident_sine + order * wavelengths / period
q_over_k = np.sqrt((1 - alpha**2).astype(complex))
threshold = period * (1 - incident_sine) / order

# %% CHECK AND INTERPRET
assert np.isclose(incident_sine + order * threshold / period, 1.0)
print("Positive-order grazing wavelength (um):", threshold)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(wavelengths, q_over_k.real, label='Real part')
    ax.plot(wavelengths, q_over_k.imag, label='Imaginary part')
    ax.axvline(threshold, color='gray', linestyle=':')
    ax.set(xlabel='Medium wavelength (um)', ylabel='q/k')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('2.3 - Locate a diffraction-order cutoff in wavelength')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
