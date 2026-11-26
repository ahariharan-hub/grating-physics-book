"""9.3 worked example: compare both fields at a common observation plane.
Lengths um; scalar/TE, air on both sides, outgoing -z, time factor exp(-i omega t).
The incident field has unit amplitude and zero phase at x=0, z=0.
Edit INPUT PARAMETERS and run the file or its # %% sections in order.
Use --no-show for a noninteractive run; --save-dir DIRECTORY saves the figure.
"""
from pathlib import Path
import sys
import numpy as np

_scripts_directory = str(Path(__file__).resolve().parents[1])
if _scripts_directory not in sys.path:
    sys.path.insert(0, _scripts_directory)
from _common import pyplot, finish
from wave_optics.amplitude_grating_2d import absorbing_screen_configuration
from wave_optics.scalar_grating_2d import solve_scalar_grating_2d
from wave_optics.exit_field_comparison import spatial_phase

# %% INPUT PARAMETERS
lambda_0 = 0.633  # Vacuum wavelength; all lengths are micrometres.
d = 1.5
W_open = 0.75
h = 0.3
theta_deg = 0.0  # Measured from -z toward +x.
M = 24  # Numerical harmonic cutoff, separate from the physical inputs.
samples = 768
show_plot = "--no-show" not in sys.argv
absorber_index = 1.5+0.8j  # Illustrative material, not fitted chrome data.

# %% SETUP: solve the freestanding absorbing layer once.
k0 = 2*np.pi/lambda_0
configuration = absorbing_screen_configuration(
    wavelength=lambda_0, period_x=d, period_y=d, thickness=h,
    fill_x=W_open/d, fill_y=1.0, n_inclusion=1.0,
    n_background=absorber_index, kx_incident=k0*np.sin(np.deg2rad(theta_deg)),
    Mx=M, My=0)
solution = solve_scalar_grating_2d(configuration)
x = np.linspace(-d/2, d/2, samples, endpoint=False)
keep = solution.output_kinds == 'propagating'
m = solution.orders[keep, 0]
beta = solution.beta_output[keep]
basis = np.exp(1j*solution.kx[keep, None]*x)

# %% OBSERVATION-PLANE INPUT
ell = 1.0  # Distance below the physical exit, in micrometres.

# %% NUMERICAL CALCULATION
assert ell >= 0
z_ref = -h-ell
thick_equivalent = solution.transmission[keep]*np.exp(-1j*beta*h)
thin_object = (W_open/d)*np.sinc(m*W_open/d)
transfer = np.exp(1j*beta*(h+ell))
thick_replay = (thick_equivalent*transfer) @ basis
thin_field = (thin_object*transfer) @ basis
# Independent backend query at the observation plane, restricted to the same orders.
direct_coefficients = solution.at_reference_plane(z_ref)[keep]
thick_direct = direct_coefficients @ basis
relative_replay_error = np.linalg.norm(thick_replay-thick_direct)/np.linalg.norm(thick_direct)
relative_thin_error = np.linalg.norm(thin_field-thick_direct)/np.linalg.norm(thick_direct)

# %% CHECK AND INTERPRET
assert relative_replay_error < 1e-12
assert abs(solution.energy_balance-1) < 1e-8
print(f'z_ref = -h-ell = {z_ref:g} um; propagating orders: {m}')
print(f'Replay versus direct thick field: {relative_replay_error:.3e}')
print(f'Thin versus thick relative complex-field difference: {relative_thin_error:.3e}')
print('The replay checks reference-plane algebra, not scattering convergence.')
print('All curves use propagating orders, even at ell=0; the full near field also has evanescent terms.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    curves = [(thick_replay, 'Thick via object plane', '-'),
              (thin_field, 'Thin mask from z=0', '--'),
              (thick_direct, 'Thick directly from exit', 'x')]
    for field, label, style in curves:
        options = {'markevery': 32, 'ms': 4} if style == 'x' else {}
        axes[0].plot(x, abs(field), style, label=label, **options)
        axes[1].plot(x, spatial_phase(field), style, **options)
        axes[2].plot(x, abs(field)**2, style, **options)
    for axis, label in zip(axes, ['Amplitude', 'Wrapped phase (rad)', 'Relative intensity']):
        axis.set(xlabel='x (um)', ylabel=label)
    axes[1].set_ylim(-3.4, 3.4)
    axes[0].legend(fontsize=8)
    figure.suptitle(f'Common observation plane z_ref={z_ref:g} um')
    finish(figure, show_plot)
