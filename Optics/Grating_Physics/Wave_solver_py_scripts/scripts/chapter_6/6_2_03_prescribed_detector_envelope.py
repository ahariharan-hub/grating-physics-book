# -*- coding: utf-8 -*-
"""6.2, worked example 3: Compare a uniform order with a finite detector-plane envelope.

Companion topic: Ideal plane waves and finite recorded profiles.
A prescribed detector-plane envelope is compared with a plane wave. This is not a finite-beam propagation solve.
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
waist = 3.0
angle_degrees = 25.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-10, 10, 1001)
kx = 2 * np.pi / wavelength * np.sin(np.deg2rad(angle_degrees))

# %% NUMERICAL CALCULATION
uniform = np.exp(1j * kx * x)
envelope = np.exp(-(x / waist)**2)
finite_field = envelope * uniform
finite_intensity = abs(finite_field)**2

# %% CHECK AND INTERPRET
assert np.allclose(finite_intensity, envelope**2)
print("The finite intensity profile comes from amplitude variation, not from the single carrier phase.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, finite_field.real, label='Finite instantaneous field')
    ax.plot(x, finite_intensity, label='Finite intensity')
    ax.plot(x, abs(uniform)**2, '--', label='Ideal plane-wave intensity')
    ax.set(xlabel='Detector x (um)', ylabel='Relative quantity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('6.2 - Compare a uniform order with a finite detector-plane envelope')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
