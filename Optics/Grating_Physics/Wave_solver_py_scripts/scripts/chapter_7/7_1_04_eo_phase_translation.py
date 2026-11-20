# -*- coding: utf-8 -*-
"""7.1, worked example 4: Translate fringes by rotating a complex correlation.

Companion topic: EO phase control is not decorrelation.
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
coherence_magnitude = 0.6
coherence_phase_degrees = 30.0
eo_phase_degrees = 90.0
fringe_period = 2.5
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-5, 5, 1001)
gradient = 2 * np.pi / fringe_period
gamma = coherence_magnitude * np.exp(1j * np.deg2rad(coherence_phase_degrees))

# %% NUMERICAL CALCULATION
eo_phase = np.deg2rad(eo_phase_degrees)
shifted_gamma = gamma * np.exp(1j * eo_phase)
original = 2 + 2 * np.real(gamma * np.exp(1j * gradient * x))
shifted = 2 + 2 * np.real(shifted_gamma * np.exp(1j * gradient * x))
displacement = -eo_phase / gradient
translated_reference = 2 + 2 * np.real(gamma * np.exp(1j * gradient * (x - displacement)))

# %% CHECK AND INTERPRET
assert np.isclose(abs(shifted_gamma), abs(gamma))
assert np.allclose(shifted, translated_reference)
print("Fringe displacement (um):", displacement, "; coherence magnitude stays fixed.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, original, label='Original correlation')
    ax.plot(x, shifted, label='EO phase applied')
    ax.set(xlabel='Detector x (um)', ylabel='Relative intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('7.1 - Translate fringes by rotating a complex correlation')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
