# -*- coding: utf-8 -*-
"""5.4, worked example 4: Compare an assumed sine-condition pupil map with a flat propagation plane.

Companion topic: Supply the optical mapping to a physical pupil.
Air, lengths um. The example assumes an aplanatic sine-condition objective with r=f sin(theta); it is not a lens-design simulation.
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
focal_length = 20000.0
angles_degrees = np.linspace(-60, 60, 301)
show_plot = True

# %% SETUP: coordinates, sampling and constants
angles = np.deg2rad(angles_degrees)

# %% NUMERICAL CALCULATION
sine_pupil = focal_length * np.sin(angles)
flat_plane = focal_length * np.tan(angles)
paraxial = focal_length * angles

# %% CHECK AND INTERPRET
assert np.allclose(sine_pupil[149:152], paraxial[149:152], rtol=1e-4, atol=1e-12)
print("The physical pupil map requires an optical assumption; a flat detector follows a different geometry.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(angles_degrees, sine_pupil / 1000, label='Assumed sine-condition pupil')
    ax.plot(angles_degrees, flat_plane / 1000, label='Flat propagation plane')
    ax.plot(angles_degrees, paraxial / 1000, '--', label='Paraxial approximation')
    ax.set(xlabel='Direction angle (deg)', ylabel='Transverse position (mm)')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('5.4 - Compare an assumed sine-condition pupil map with a flat propagation plane')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
