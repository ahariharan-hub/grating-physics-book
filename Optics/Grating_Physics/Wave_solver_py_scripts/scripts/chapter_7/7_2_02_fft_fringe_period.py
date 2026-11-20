# -*- coding: utf-8 -*-
"""7.2, worked example 2: Synthesize grating-order fringes and measure their period with an FFT.

Companion topic: Pitch is not always fringe period.
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
orders = np.array([-1, 1])
samples = 2048
window = 32.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.arange(samples) * window / samples
k = 2 * np.pi / wavelength
kx = 2 * np.pi * orders / period

# %% NUMERICAL CALCULATION
q = np.sqrt(k**2 - kx**2)
components = np.exp(1j * kx[:, None] * x + 1j * q[:, None] * 3.0)
intensity = abs(components.sum(axis=0))**2
spectrum = abs(np.fft.rfft(intensity - intensity.mean()))
frequency = np.fft.rfftfreq(samples, window / samples)
measured_period = 1 / frequency[np.argmax(spectrum[1:]) + 1]
predicted_period = period / abs(orders[1] - orders[0])

# %% CHECK AND INTERPRET
assert np.isclose(measured_period, predicted_period)
print("Measured/predicted fringe period (um):", measured_period, predicted_period)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(x, intensity)
    axes[0].set(xlabel='Detector x (um)', ylabel='Relative intensity', xlim=(0, 8))
    axes[1].plot(frequency, spectrum)
    axes[1].set(xlabel='Fringe spatial frequency (cycles/um)', ylabel='DFT magnitude', xlim=(0, 1))

    # %% NUMERICAL SUMMARY
    print('7.2 - Synthesize grating-order fringes and measure their period with an FFT')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
