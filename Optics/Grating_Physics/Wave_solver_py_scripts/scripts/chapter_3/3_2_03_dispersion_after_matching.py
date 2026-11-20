# -*- coding: utf-8 -*-
"""3.2, worked example 3: Complete the longitudinal component after tangential matching.

Companion topic: Phase matching versus a real outgoing direction.
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
wavelength = 2.0
period = 4.0
incidence_degrees = 20.0
orders = np.arange(-4, 5)
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
incident_kx = k * np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
kx = incident_kx + orders * 2 * np.pi / period
q = np.sqrt((k**2 - kx**2).astype(complex))
surface_matched = np.exp(1j * (kx - incident_kx) * period)

# %% CHECK AND INTERPRET
assert np.allclose(surface_matched, 1)
assert np.allclose(kx**2 + q**2, k**2)
print("All retained labels match translation; only some have real normal components.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(orders, q.real / k, 'o-', label='Real q/k')
    ax.plot(orders, q.imag / k, 's-', label='Imaginary q/k')
    ax.set(xlabel='Integer order m', ylabel='Normal component')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('3.2 - Complete the longitudinal component after tangential matching')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
