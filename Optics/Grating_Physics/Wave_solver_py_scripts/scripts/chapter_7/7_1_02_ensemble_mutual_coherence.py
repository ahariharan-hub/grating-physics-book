# -*- coding: utf-8 -*-
"""7.1, worked example 2: Compute mutual coherence from a sampled phase ensemble.

Companion topic: Mutual coherence and fringe visibility.
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
phase_standard_deviation = 0.9
ensemble_samples = 10000
random_seed = 12345
show_plot = True

# %% SETUP: coordinates, sampling and constants
rng = np.random.default_rng(random_seed)
random_phases = rng.normal(0, phase_standard_deviation, ensemble_samples)
controlled_phase = np.linspace(-np.pi, np.pi, 121)

# %% NUMERICAL CALCULATION
gamma = np.mean(np.exp(1j * random_phases))
measured = []
for phase in controlled_phase:
    intensity_samples = abs(1 + np.exp(1j * (phase + random_phases)))**2
    measured.append(np.mean(intensity_samples))
prediction = 2 + 2 * np.real(gamma * np.exp(1j * controlled_phase))
gaussian_ensemble_limit = np.exp(-phase_standard_deviation**2 / 2)

# %% CHECK AND INTERPRET
assert np.allclose(measured, prediction)
assert abs(abs(gamma) - gaussian_ensemble_limit) < 0.03
print("Sampled coherence magnitude / Gaussian ensemble limit:", abs(gamma), gaussian_ensemble_limit)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(controlled_phase, measured, label='Average of sampled intensities')
    ax.plot(controlled_phase, prediction, '--', label='Prediction from sampled gamma')
    ax.set(xlabel='Controlled phase (rad)', ylabel='Ensemble-averaged intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('7.1 - Compute mutual coherence from a sampled phase ensemble')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
