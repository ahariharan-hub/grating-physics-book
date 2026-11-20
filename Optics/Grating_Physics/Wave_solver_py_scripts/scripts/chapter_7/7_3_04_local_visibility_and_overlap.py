# -*- coding: utf-8 -*-
"""7.3, worked example 4: Distinguish local fringe visibility from integrated beam overlap.

Companion topic: Visibility and finite-beam overlap.
Prescribed paraxial Gaussian order beams; wavelength 1 um, pitch 4 um, outgoing -z. No material scattering or recombination optics are solved.
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
from wave_optics.gaussian_order_beams import GaussianOrderConfiguration, GaussianOrderExperiment
from scipy.integrate import trapezoid

# %% INPUT PARAMETERS
waist = 6.0
depth = 40.0
amplitude_ratio = 0.5
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-40, 50, 1801)
model = GaussianOrderExperiment(GaussianOrderConfiguration(waist=waist))

# %% NUMERICAL CALCULATION
sample = model.sample(x, np.array([depth]), coefficients=(1.0, amplitude_ratio))
first, second = sample['fields'][:, 0]
intensity_sum = abs(first)**2 + abs(second)**2
visibility = np.divide(2 * abs(first) * abs(second), intensity_sum, out=np.zeros_like(x), where=intensity_sum > 1e-14)
weighted_visibility = trapezoid(visibility * intensity_sum, x) / trapezoid(intensity_sum, x)
line_overlap = trapezoid(abs(first) * abs(second), x) / np.sqrt(trapezoid(abs(first)**2, x) * trapezoid(abs(second)**2, x))

# %% CHECK AND INTERPRET
assert np.all(visibility <= 1 + 1e-12)
print("Intensity-weighted local visibility:", weighted_visibility, "; normalized line overlap:", line_overlap)
print("High local visibility in a very dim tail need not imply strong total interference.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(x, intensity_sum)
    axes[0].set(xlabel='x (um)', ylabel='Sum of component intensities')
    axes[1].plot(x, visibility)
    axes[1].set(xlabel='x (um)', ylabel='Local phase-scan visibility', ylim=(0, 1.05))

    # %% NUMERICAL SUMMARY
    print('7.3 - Distinguish local fringe visibility from integrated beam overlap')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
