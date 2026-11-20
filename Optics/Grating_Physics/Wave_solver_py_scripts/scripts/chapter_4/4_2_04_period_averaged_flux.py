# -*- coding: utf-8 -*-
"""4.2, worked example 4: Check constant average normal flux while the near-field intensity changes.

Companion topic: Decay and energy flow for several orders.
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
wavelength = 1.0
period = 3.0
orders = np.array([0, 1, 4])
coefficients = np.array([1.0, 0.5, 0.4j])
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.arange(512) * period / 512
s = np.linspace(0, 1, 101)
k = 2 * np.pi / wavelength
kx = 2 * np.pi * orders / period
q = np.sqrt((k**2 - kx**2).astype(complex))

# %% NUMERICAL CALCULATION
basis = np.exp(1j * kx[:, None] * x)
advanced = coefficients[None, :] * np.exp(1j * s[:, None] * q)
field = advanced @ basis
derivative = (1j * q * advanced) @ basis
mean_flux = np.mean(np.imag(np.conj(field) * derivative), axis=1) / k
mean_intensity = np.mean(abs(field)**2, axis=1)
order_flux = np.sum(q.real * abs(coefficients)**2) / k

# %% CHECK AND INTERPRET
assert np.allclose(mean_flux, order_flux)
assert mean_intensity[-1] < mean_intensity[0]
print("Mean scalar normal flux:", order_flux, "; mean intensity is a different quantity.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(s, mean_intensity, label='Mean intensity')
    ax.plot(s, mean_flux, label='Mean normal flux / k')
    ax.set(xlabel='Depth (um)', ylabel='Relative quantity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('4.2 - Check constant average normal flux while the near-field intensity changes')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
