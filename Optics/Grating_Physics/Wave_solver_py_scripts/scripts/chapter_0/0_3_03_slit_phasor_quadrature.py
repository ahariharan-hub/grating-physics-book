# -*- coding: utf-8 -*-
"""0.3, worked example 3: Integrate a slit far field and recover the sinc pattern.

Companion topic: What happens when a plane wave encounters an aperture?.
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
wavelength = 0.6
slit_width = 20.0
distance = 1000000.0
aperture_samples = 2001
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-3, 3, 401) * wavelength * distance / slit_width
xi = -slit_width / 2 + (np.arange(aperture_samples) + 0.5) * slit_width / aperture_samples
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
phasors = np.exp(-1j * k * (x[:, None] / distance) * xi)
numerical_amplitude = phasors.mean(axis=1)
analytic_amplitude = np.sinc(slit_width * x / (wavelength * distance))

# %% CHECK AND INTERPRET
error = np.max(abs(numerical_amplitude - analytic_amplitude))
assert error < 2e-6
print('Maximum amplitude error:', error, '; paraxial Fraunhofer shape, peak normalized.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(x / 1000, abs(numerical_amplitude)**2, label='Midpoint integral')
    ax.plot(x / 1000, analytic_amplitude**2, '--', label='sinc squared')
    ax.set(xlabel='Detector x (mm)', ylabel='Peak-normalized intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.3 - Integrate a slit far field and recover the sinc pattern')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
