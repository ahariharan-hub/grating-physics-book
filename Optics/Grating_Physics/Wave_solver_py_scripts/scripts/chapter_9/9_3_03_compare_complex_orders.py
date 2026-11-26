"""9.3 worked example: compare complex coefficients at the object plane.
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
thick = solution.transmission[keep]*np.exp(-1j*beta*h)
thin = (W_open/d)*np.sinc(m*W_open/d)
difference = thick-thin
valid = (abs(thick) > 1e-6) & (abs(thin) > 1e-6)
phase_difference = np.full(m.shape, np.nan)
phase_difference[valid] = np.angle(thick[valid]*np.conj(thin[valid]))
# Recover these same object coefficients from the sampled equivalent field.
equivalent_field = thick @ basis
periodic_envelope = equivalent_field*np.exp(-1j*configuration.kx_incident*x)
from_fft = np.fft.fft(periodic_envelope)[m % samples]/samples
from_fft *= np.exp(-2j*np.pi*m*x[0]/d)

# %% CHECK AND INTERPRET
fft_error = np.max(abs(from_fft-thick))
assert fft_error < 1e-12
assert abs(solution.energy_balance-1) < 1e-8
print('Both coefficient sets use z_obj=0 and the same incident phase.')
print(' m       thick equivalent C_m          ideal t_m           |difference|   delta phase')
for order, thick_c, thin_c, error, phase in zip(m, thick, thin, abs(difference), phase_difference):
    phase_text = f'{phase:+.5f}' if np.isfinite(phase) else 'undefined'
    print(f'{order:+3d}   {thick_c.real:+.6f}{thick_c.imag:+.6f}j   '
          f'{thin_c:+.6f}   {error:.6f}   {phase_text}')
print(f'Maximum FFT versus direct complex-coefficient difference: {fft_error:.3e}')
print('Undefined phase at an ideal zero is omitted; coefficients are amplitudes, not efficiencies.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(m, abs(thick), 'o-', label='Thick equivalent')
    axes[0].plot(m, abs(thin), 'x--', label='Ideal thin mask')
    axes[0].set(ylabel='Coefficient magnitude at z_obj=0')
    axes[1].plot(m, phase_difference, 'o')
    axes[1].axhline(0, color='gray', lw=0.8)
    axes[1].set(ylabel='Wrapped thick-minus-thin phase (rad)')
    for axis in axes:
        axis.set(xlabel='Order m', xticks=m)
    axes[0].legend()
    finish(figure, show_plot)
