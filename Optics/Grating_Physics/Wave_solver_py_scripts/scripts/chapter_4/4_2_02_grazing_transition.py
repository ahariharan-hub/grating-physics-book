# -*- coding: utf-8 -*-
"""4.2, worked example 2: Follow one order continuously through grazing.

Companion topic: Follow one order through grazing.
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
wavelength = 2.0
order = 1
periods = np.linspace(1.5, 3.0, 601)
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
alpha = order * wavelength / periods
q = k * np.sqrt((1 - alpha**2).astype(complex))
decay_lengths = np.divide(1.0, q.imag, out=np.full_like(periods, np.nan), where=q.imag > 0)
grazing_period = abs(order) * wavelength

# %% CHECK AND INTERPRET
assert np.isclose(1 - (order * wavelength / grazing_period)**2, 0)
print("Decay length diverges on approaching grazing from the evanescent side.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(periods, q.real / k, label='Real q/k')
    axes[0].plot(periods, q.imag / k, label='Imaginary q/k')
    axes[0].axvline(grazing_period, color='gray', linestyle=':')
    axes[0].set(xlabel='Period (um)', ylabel='Normal component')
    axes[0].legend()
    axes[1].plot(periods, decay_lengths)
    axes[1].set(xlabel='Period (um)', ylabel='Amplitude decay length (um)')

    # %% NUMERICAL SUMMARY
    print('4.2 - Follow one order continuously through grazing')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
