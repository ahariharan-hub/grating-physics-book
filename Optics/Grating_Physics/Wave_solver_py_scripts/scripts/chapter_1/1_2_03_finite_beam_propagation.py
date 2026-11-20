# -*- coding: utf-8 -*-
"""1.2, worked example 3: Compare finite-beam propagation with a uniform plane wave.

Companion topic: When is the plane-wave approximation useful?.
Medium wavelength and lengths um. Scalar 1D ASM with a prescribed input envelope; downstream s=-z.
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
waist = 4.0
distance = 30.0
window = 64.0
samples = 2048
show_plot = True

# %% SETUP: coordinates, sampling and constants
dx = window / samples
x = (np.arange(samples) - samples // 2) * dx
fx = np.fft.fftfreq(samples, dx)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
gaussian = np.exp(-(x / waist)**2)
transfer = np.exp(1j * k * distance * np.sqrt((1 - (wavelength * fx)**2).astype(complex)))
finite_output = np.fft.ifft(np.fft.fft(gaussian) * transfer)
uniform_output = np.fft.ifft(np.fft.fft(np.ones(samples)) * transfer)

# %% CHECK AND INTERPRET
assert np.allclose(abs(uniform_output), 1)
print("Plane-wave intensity stays uniform; the finite beam diffracts. Check FFT padding.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, abs(gaussian)**2, '--', label='Finite input')
    ax.plot(x, abs(finite_output)**2, label='Finite propagated beam')
    ax.plot(x, abs(uniform_output)**2, ':', label='Uniform plane wave')
    ax.set(xlabel='x (um)', ylabel='Relative intensity', xlim=(-15, 15))
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('1.2 - Compare finite-beam propagation with a uniform plane wave')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
