# -*- coding: utf-8 -*-
"""8.2, worked example 3: Rebuild a spherical Green response from propagating and evanescent spectra.

Companion topic: Weyl bridge from patches to plane waves.
3D outgoing Green function. Medium wavelength and lengths um; downstream s=-z; source at s=-source_distance.
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
from scipy.special import j0

# %% INPUT PARAMETERS
wavelength = 1.0
source_distance = 1.5
observation_depth = 0.5
quadrature_order = 96
evanescent_decay_cutoff = 32.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-3, 3, 201)
k = 2 * np.pi / wavelength
depth = source_distance + observation_depth
nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
mu = (nodes + 1) / 2
w_mu = weights / 2
kappa_max = evanescent_decay_cutoff / depth
kappa = (nodes + 1) * kappa_max / 2
w_kappa = weights * kappa_max / 2

# %% NUMERICAL CALCULATION
# Azimuth has been integrated analytically, giving J0; the remaining quadratures are explicit.
propagating_terms = j0(abs(x[:, None]) * k * np.sqrt(1 - mu**2)) * np.exp(1j * k * mu * depth)
propagating = 1j * k / (4 * np.pi) * (propagating_terms @ w_mu)
evanescent_terms = j0(abs(x[:, None]) * np.sqrt(k**2 + kappa**2)) * np.exp(-kappa * depth)
evanescent = (evanescent_terms @ w_kappa) / (4 * np.pi)
reconstructed = propagating + evanescent
radius = np.sqrt(x**2 + depth**2)
reference = np.exp(1j * k * radius) / (4 * np.pi * radius)
relative_error = np.linalg.norm(reconstructed - reference) / np.linalg.norm(reference)

# %% CHECK AND INTERPRET
assert relative_error < 1e-8
print("Relative complex Weyl reconstruction error:", relative_error)
print("The evanescent integral is truncated where its exponential factor is very small.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(x, reconstructed.real, label='Weyl real field')
    axes[0].plot(x, reference.real, '--', label='Spherical Green reference')
    axes[0].set(xlabel='x (um)', ylabel='Real Green field (inverse um)')
    axes[0].legend()
    axes[1].plot(x, abs(propagating), label='Propagating contribution')
    axes[1].plot(x, abs(evanescent), label='Evanescent contribution')
    axes[1].set(xlabel='x (um)', ylabel='Contribution magnitude')
    axes[1].legend()

    # %% NUMERICAL SUMMARY
    print('8.2 - Rebuild a spherical Green response from propagating and evanescent spectra')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
