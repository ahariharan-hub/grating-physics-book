# -*- coding: utf-8 -*-
"""1.1, worked example 3: Recover amplitude and phase from quadrature samples.

Companion topic: Complex phase and a real measurement.
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
amplitude = 1.4
phase_degrees = 65.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
phase = np.deg2rad(phase_degrees)
time_fractions = np.array([0.0, 0.25])

# %% NUMERICAL CALCULATION
samples = amplitude * np.cos(phase - 2 * np.pi * time_fractions)
recovered_phasor = samples[0] + 1j * samples[1]
recovered_amplitude = abs(recovered_phasor)
recovered_phase = np.angle(recovered_phasor)

# %% CHECK AND INTERPRET
assert np.isclose(recovered_amplitude, amplitude)
assert np.isclose(recovered_phase, phase)
print("Recovered amplitude and phase:", recovered_amplitude, recovered_phase)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(5, 5), layout='constrained')
    ax.plot([0, recovered_phasor.real], [0, recovered_phasor.imag], 'o-')
    ax.set(xlabel='In-phase sample', ylabel='Quadrature sample', aspect='equal')

    # %% NUMERICAL SUMMARY
    print('1.1 - Recover amplitude and phase from quadrature samples')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
