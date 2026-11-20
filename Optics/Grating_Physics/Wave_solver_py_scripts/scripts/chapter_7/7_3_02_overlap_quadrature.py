# -*- coding: utf-8 -*-
"""7.3, worked example 2: Integrate detector-line overlap for two spreading order beams.

Companion topic: Separation spreading and spatial overlap.
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
depths = np.linspace(0, 100, 51)
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-80, 80, 1601)
model = GaussianOrderExperiment(GaussianOrderConfiguration(waist=waist))

# %% NUMERICAL CALCULATION
sample = model.sample(x, depths)
first, second = sample['fields']
numerator = trapezoid(abs(first) * abs(second), x, axis=1)
denominator = np.sqrt(trapezoid(abs(first)**2, x, axis=1) * trapezoid(abs(second)**2, x, axis=1))
overlap = numerator / denominator

# %% CHECK AND INTERPRET
assert np.allclose(overlap, sample["overlap"])
assert np.all((overlap >= 0) & (overlap <= 1 + 1e-12))
print("This normalized line-overlap measure is not mutual coherence or full-plane overlap.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(depths, overlap)
    ax.set(xlabel='Detector depth (um)', ylabel='Normalized detector-line amplitude overlap')

    # %% NUMERICAL SUMMARY
    print('7.3 - Integrate detector-line overlap for two spreading order beams')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
