# -*- coding: utf-8 -*-
"""9.2, worked example 4: Recover a two-dimensional complex order lattice with FFT2.

Companion topic: Repeat the coefficient check in two dimensions.
Coordinates nm; two-dimensional prescribed periodic field. Samples use array order [y,x]; labels use (m,n).
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
period_x = 1000.0
period_y = 800.0
Nx = 32
Ny = 24
labels = np.array([[-2, 1], [0, 0], [1, -2], [3, 2]])
coefficients = np.array([0.2 * np.exp(0.4j), 0.75, 0.35 * np.exp(-0.6j), 0.25 * np.exp(1.1j)])
show_plot = True

# %% SETUP: coordinates, sampling and constants
x0, y0 = -period_x / 2, -period_y / 2
x = x0 + np.arange(Nx) * period_x / Nx
y = y0 + np.arange(Ny) * period_y / Ny
xx, yy = np.meshgrid(x, y)
fx = np.fft.fftfreq(Nx, period_x / Nx)
fy = np.fft.fftfreq(Ny, period_y / Ny)

# %% NUMERICAL CALCULATION
field = np.zeros((Ny, Nx), dtype=complex)
for (m, n), coefficient in zip(labels, coefficients):
    field += coefficient * np.exp(2j * np.pi * (m * xx / period_x + n * yy / period_y))
raw = np.fft.fft2(field) / field.size
origin_correction = np.exp(-2j * np.pi * (fy[:, None] * y0 + fx[None, :] * x0))
corrected = raw * origin_correction
recovered = corrected[labels[:, 1] % Ny, labels[:, 0] % Nx]

# %% CHECK AND INTERPRET
assert np.allclose(recovered, coefficients)
print("Maximum complex coefficient error:", np.max(abs(recovered - coefficients)))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    image = axes[0].imshow(abs(field), origin='lower', extent=[x[0], x[-1], y[0], y[-1]])
    axes[0].set(xlabel='x (nm)', ylabel='y (nm)', title='Prescribed field amplitude')
    figure.colorbar(image, ax=axes[0])
    axes[1].scatter(labels[:, 0], labels[:, 1], s=400 * abs(coefficients), label='Specified')
    axes[1].scatter(labels[:, 0], labels[:, 1], s=400 * abs(recovered), marker='x', label='FFT2 recovered')
    axes[1].set(xlabel='Order m', ylabel='Order n', aspect='equal')
    axes[1].legend()

    # %% NUMERICAL SUMMARY
    print('9.2 - Recover a two-dimensional complex order lattice with FFT2')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
