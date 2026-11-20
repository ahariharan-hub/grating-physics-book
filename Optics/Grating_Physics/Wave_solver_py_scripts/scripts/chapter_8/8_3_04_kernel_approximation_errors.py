# -*- coding: utf-8 -*-
"""8.3, worked example 4: Compare full RS I and approximate spatial kernels at common points.

Companion topic: What does the kernel contribute at P?.
A meridional section of full 3D spatial kernels, not a one-dimensional slit propagator. Medium wavelength and lengths um.
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
from wave_optics.propagation_kernels import kernel_values

# %% INPUT PARAMETERS
wavelength = 1.0
source_patch_x = 1.0
distances = [2.0, 8.0, 32.0]
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-5, 5, 501)
k = 2 * np.pi / wavelength
errors = []
maximum_errors = []

# %% NUMERICAL CALCULATION
for distance in distances:
    radius = np.sqrt(distance**2 + (x - source_patch_x)**2)
    full_reduced = (distance / radius)**2 * (1 + 1j / (k * radius)) * np.exp(1j * k * (radius - distance))
    full_phase = k * (radius - distance) + np.arctan(1 / (k * radius))
    fresnel_phase = k * (x - source_patch_x)**2 / (2 * distance)
    error = fresnel_phase - full_phase
    errors.append(error)
    maximum_errors.append(np.max(abs(error)))
    reference = kernel_values(x, source_patch_x, distance=distance, wavelength=wavelength)
    assert np.allclose(full_reduced, reference['reduced'][0])

# %% CHECK AND INTERPRET
assert maximum_errors[-1] < maximum_errors[0]
print("Maximum unwrapped Fresnel/RS I phase errors:", maximum_errors)
print("Kernel accuracy depends on wavelength, aperture and observation geometry, not distance alone.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for distance, error in zip(distances, errors):
        ax.plot(x, error, label=f's={distance:g} um')
    ax.set(xlabel='Observation x (um)', ylabel='Unwrapped Fresnel minus RS I phase (rad)')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('8.3 - Compare full RS I and approximate spatial kernels at common points')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
