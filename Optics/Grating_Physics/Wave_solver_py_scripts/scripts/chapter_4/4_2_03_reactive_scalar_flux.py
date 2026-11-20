# -*- coding: utf-8 -*-
"""4.2, worked example 3: Separate real normal scalar flux from a reactive component.

Companion topic: Does decay mean energy is absorbed?.
Medium wavelength and lengths um. Downstream s=-z; U=exp(i kx x+i q s); positive-imaginary q decays. The flux is a scalar Helmholtz diagnostic, not a general vector Poynting calculation.
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
transverse_ratio = 1.3
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
q = 1j * k * np.sqrt(transverse_ratio**2 - 1)
s = np.linspace(0, 2, 401)

# %% NUMERICAL CALCULATION
field = np.exp(1j * q * s)
derivative = 1j * q * field
complex_scalar_flux = -1j * np.conj(field) * derivative
normal_flux = complex_scalar_flux.real

# %% CHECK AND INTERPRET
assert np.allclose(normal_flux, 0)
assert np.all(complex_scalar_flux.imag >= 0)
print("An isolated decaying harmonic has zero real normal scalar flux in this lossless medium.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(s, complex_scalar_flux.real / k, label='Real normal flux / k')
    ax.plot(s, complex_scalar_flux.imag / k, label='Reactive scalar diagnostic / k')
    ax.set(xlabel='Depth (um)', ylabel='Scalar flux diagnostic')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('4.2 - Separate real normal scalar flux from a reactive component')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
