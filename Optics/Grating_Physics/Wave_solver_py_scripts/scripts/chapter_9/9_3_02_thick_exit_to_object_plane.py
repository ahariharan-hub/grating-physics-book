"""9.3 worked example: bring the thick exit field back to the object plane.
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

# %% NUMERICAL CALCULATION
exit_coefficients = solution.transmission[keep]
# Apply each order's own inverse propagation factor, only for propagating orders.
equivalent_coefficients = exit_coefficients*np.exp(-1j*beta*h)
t_m = (W_open/d)*np.sinc(m*W_open/d)
equivalent_field = equivalent_coefficients @ basis
thin_projected = t_m @ basis
exit_propagating = exit_coefficients @ basis
all_basis = np.exp(1j*solution.kx[:, None]*x)
exit_complete = solution.transmission @ all_basis

# %% CHECK AND INTERPRET
replayed_exit = (equivalent_coefficients*np.exp(1j*beta*h)) @ basis
relative_error = np.linalg.norm(replayed_exit-exit_propagating)/np.linalg.norm(exit_propagating)
assert relative_error < 1e-12
assert abs(solution.energy_balance-1) < 1e-8
print(f'Propagating orders: {m}; exit replay difference: {relative_error:.3e}')
print('The equivalent outgoing field is not the actual total field at the material top.')
print('The complete exit field contains evanescent orders; the object comparison excludes them.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    rows = [[(exit_complete, 'Physical exit: all solved orders', '-'),
             (exit_propagating, 'Physical exit: propagating part', '--')],
            [(equivalent_field, 'Thick equivalent at object plane', '-'),
             (thin_projected, 'Ideal thin mask: same propagating orders', '--')]]
    for row, curves in enumerate(rows):
        for field, label, style in curves:
            axes[row, 0].plot(x, abs(field), style, label=label)
            axes[row, 1].plot(x, spatial_phase(field), style, label=label)
        axes[row, 0].set(ylabel='Amplitude')
        axes[row, 1].set(ylabel='Wrapped phase (rad)', ylim=(-3.4, 3.4))
        axes[row, 0].legend(fontsize=8)
    axes[0, 0].set_title('Physical exit z_e=-h')
    axes[1, 0].set_title('Common object plane z_obj=0')
    for axis in axes.flat:
        axis.set_xlabel('x (um)')
    finish(figure, show_plot)
