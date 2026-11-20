# -*- coding: utf-8 -*-
"""0.4, worked example 2: Choose the outgoing square-root branch for each component.

Companion topic: How does each component propagate?.
Lengths um; medium wavelength; downstream distance s=-z >= 0; exp(-i omega t).
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
from wave_optics.scalar_propagation import transfer_function

# %% INPUT PARAMETERS
wavelength = 1.0
distance = 2.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
normalized_frequency = np.linspace(-1.5, 1.5, 601)
k = 2 * np.pi / wavelength
fx = normalized_frequency / wavelength

# %% NUMERICAL CALCULATION
q = k * np.sqrt((1 - normalized_frequency**2).astype(complex))
multiplier = np.exp(1j * q * distance)
backend = transfer_function(fx, 0.0, wavelength, distance, 'asm')

# %% CHECK AND INTERPRET
assert np.allclose(multiplier, backend)
assert np.all(abs(multiplier) <= 1 + 1e-12)
print('Real q advances phase; positive-imaginary q gives downstream decay.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(normalized_frequency, q.real / k, label='Re(q/k)')
    axes[0].plot(normalized_frequency, q.imag / k, label='Im(q/k)')
    axes[0].set(xlabel='lambda fx', ylabel='Longitudinal component')
    axes[0].legend()
    axes[1].plot(normalized_frequency, abs(multiplier))
    axes[1].set(xlabel='lambda fx', ylabel='Multiplier magnitude')

    # %% NUMERICAL SUMMARY
    print('0.4 - Choose the outgoing square-root branch for each component')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
