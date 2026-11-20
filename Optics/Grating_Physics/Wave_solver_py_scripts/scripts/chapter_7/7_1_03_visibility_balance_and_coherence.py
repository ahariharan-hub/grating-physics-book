# -*- coding: utf-8 -*-
"""7.1, worked example 3: Separate amplitude imbalance from reduced mutual coherence.

Companion topic: The same visibility can have different causes.
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
amplitude_ratios = np.linspace(0.0, 3.0, 121)
coherences = [1.0, 0.6, 0.3]
show_plot = True

# %% SETUP: coordinates, sampling and constants
phase = np.linspace(0, 2 * np.pi, 2001)
measured_visibilities = []

# %% NUMERICAL CALCULATION
for coherence in coherences:
    intensity = 1 + amplitude_ratios[:, None]**2 + 2 * coherence * amplitude_ratios[:, None] * np.cos(phase)
    maximum = intensity.max(axis=1)
    minimum = intensity.min(axis=1)
    visibility = (maximum - minimum) / (maximum + minimum)
    predicted = 2 * amplitude_ratios * coherence / (1 + amplitude_ratios**2)
    assert np.allclose(visibility, predicted)
    measured_visibilities.append(visibility)

# %% CHECK AND INTERPRET
assert all(np.all(v <= 1 + 1e-12) for v in measured_visibilities)
print("Visibility alone cannot uniquely identify amplitude imbalance and coherence.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for coherence, visibility in zip(coherences, measured_visibilities):
        ax.plot(amplitude_ratios, visibility, label=f'|gamma|={coherence:g}')
    ax.set(xlabel='Second / first amplitude', ylabel='Phase-scan visibility')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('7.1 - Separate amplitude imbalance from reduced mutual coherence')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
