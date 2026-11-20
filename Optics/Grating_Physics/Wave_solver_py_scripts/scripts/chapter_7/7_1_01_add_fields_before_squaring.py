# -*- coding: utf-8 -*-
"""7.1, worked example 1: Evaluate the cross term in two-wave addition.

Companion topic: Add fields before squaring.
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
amplitude_one = 1.0
amplitude_two = 0.7
phases_degrees = np.linspace(-180, 180, 721)
show_plot = True

# %% SETUP: coordinates, sampling and constants
phase = np.deg2rad(phases_degrees)

# %% NUMERICAL CALCULATION
field_one = amplitude_one + 0j
field_two = amplitude_two * np.exp(1j * phase)
total_intensity = abs(field_one + field_two)**2
separate_intensities = abs(field_one)**2 + abs(field_two)**2
cross_term = 2 * np.real(np.conj(field_one) * field_two)

# %% CHECK AND INTERPRET
assert np.allclose(total_intensity, separate_intensities + cross_term)
print("The missing interference term is:", "2 Re(conj(U1) U2)")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(phases_degrees, total_intensity, label='Intensity after field addition')
    ax.plot(phases_degrees, separate_intensities, color='gray', linestyle='--', label='Separate intensities added')
    ax.set(xlabel='Relative phase (deg)', ylabel='Relative intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('7.1 - Evaluate the cross term in two-wave addition')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
