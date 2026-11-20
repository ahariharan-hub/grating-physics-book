# -*- coding: utf-8 -*-
"""9.1, worked example 3: Separate periodic transmission coefficients from a tilted exit field.

Companion topic: The field and its reference-plane convention.
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
period = 4.0
incident_frequency = 0.13
samples = 256
orders = np.array([-2, 0, 1])
coefficients = np.array([0.2j, 0.7, 0.3 * np.exp(0.5j)])
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.arange(samples) * period / samples
frequencies = np.fft.fftfreq(samples, period / samples)

# %% NUMERICAL CALCULATION
periodic_transmission = np.exp(2j * np.pi * x[:, None] * orders / period) @ coefficients
exit_field = periodic_transmission * np.exp(2j * np.pi * incident_frequency * x)
reduced = exit_field * np.exp(-2j * np.pi * incident_frequency * x)
recovered = np.fft.fft(reduced)[orders % samples] / samples
raw_exit_spectrum = np.fft.fft(exit_field) / samples

# %% CHECK AND INTERPRET
assert np.allclose(recovered, coefficients)
print("An ordinary one-period FFT applies directly to the periodic carrier-removed field, not an arbitrary Bloch field.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(np.fft.fftshift(frequencies), abs(np.fft.fftshift(raw_exit_spectrum)), label='Raw tilted exit-field FFT')
    ax.plot(np.fft.fftshift(frequencies), abs(np.fft.fftshift(np.fft.fft(reduced) / samples)), label='Incident carrier removed')
    ax.set(xlabel='FFT frequency (cycles/um)', ylabel='Coefficient magnitude', xlim=(-1, 1))
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('9.1 - Separate periodic transmission coefficients from a tilted exit field')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
