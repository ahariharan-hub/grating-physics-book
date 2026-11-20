# -*- coding: utf-8 -*-
"""4.1, worked example 2: Construct and sample an evanescent field in space.

Companion topic: An imaginary component produces a real decaying field.
Medium wavelength and lengths um. Downstream s=-z; U=exp(i kx x+i q s); positive-imaginary q decays.
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
transverse_ratio = 1.5
time_fraction = 0.25
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-4, 4, 241)
s = np.linspace(0, 2, 161)
k = 2 * np.pi / wavelength
kappa = k * np.sqrt(transverse_ratio**2 - 1)

# %% NUMERICAL CALCULATION
field = np.exp(1j * k * transverse_ratio * x[None, :] - kappa * s[:, None])
real_field = np.real(field * np.exp(-2j * np.pi * time_fraction))
envelope = np.exp(-kappa * s)

# %% CHECK AND INTERPRET
assert np.allclose(abs(field), envelope[:, None])
print("Amplitude decay length (um):", 1 / kappa)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    image = axes[0].imshow(real_field, origin='lower', extent=[x[0], x[-1], s[0], s[-1]], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set(xlabel='x (um)', ylabel='s=-z (um)', title='Real evanescent field')
    figure.colorbar(image, ax=axes[0])
    axes[1].plot(s, envelope)
    axes[1].set(xlabel='Depth (um)', ylabel='Amplitude / surface amplitude')

    # %% NUMERICAL SUMMARY
    print('4.1 - Construct and sample an evanescent field in space')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
