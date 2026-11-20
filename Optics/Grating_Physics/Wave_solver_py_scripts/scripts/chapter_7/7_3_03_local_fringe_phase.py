# -*- coding: utf-8 -*-
"""7.3, worked example 3: Calculate finite-beam fringe phase and its local spatial frequency.

Companion topic: Coherent interference of finite beams.
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

# %% INPUT PARAMETERS
waist = 6.0
depth = 25.0
coefficient_phase_degrees = 45.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-12, 16, 1401)
model = GaussianOrderExperiment(GaussianOrderConfiguration(waist=waist))

# %% NUMERICAL CALCULATION
sample = model.sample(x, np.array([depth]), coefficients=(1.0, np.exp(1j * np.deg2rad(coefficient_phase_degrees))))
first = sample['fields'][0, 0]
second = sample['fields'][1, 0]
relative_phase = np.unwrap(np.angle(np.conj(first) * second))
local_frequency = np.gradient(relative_phase, x) / (2 * np.pi)
expanded_intensity = abs(first)**2 + abs(second)**2 + 2 * np.real(np.conj(first) * second)

# %% CHECK AND INTERPRET
assert np.allclose(expanded_intensity, sample["intensity"][0])
print("Gaussian phase curvature can change local fringe spacing from the ideal plane-wave prediction.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(x, expanded_intensity)
    axes[0].set(xlabel='Detector x (um)', ylabel='Relative intensity')
    axes[1].plot(x, local_frequency)
    axes[1].axhline(1 / model.configuration.period, color='gray', linestyle='--')
    axes[1].set(xlabel='Detector x (um)', ylabel='Local fringe frequency (cycles/um)')

    # %% NUMERICAL SUMMARY
    print('7.3 - Calculate finite-beam fringe phase and its local spatial frequency')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
