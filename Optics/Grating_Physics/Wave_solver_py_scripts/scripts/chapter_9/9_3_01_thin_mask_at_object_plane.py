"""9.3 worked example: prescribe the thin mask at the object plane.
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

# %% INPUT PARAMETERS
lambda_0 = 0.633  # Vacuum wavelength; all lengths are micrometres.
d = 1.5
W_open = 0.75
theta_deg = 0.0  # Measured from -z toward +x.
M = 24  # Numerical harmonic cutoff, separate from the physical inputs.
samples = 768
show_plot = "--no-show" not in sys.argv

# %% NUMERICAL CALCULATION
k0 = 2*np.pi/lambda_0
kx_i = k0*np.sin(np.deg2rad(theta_deg))
D = W_open/d
# Midpoint samples avoid assigning a phase to an aperture edge.
x = -d/2+(np.arange(samples)+0.5)*d/samples
mask = (abs(x) < W_open/2).astype(float)
incident = np.exp(1j*kx_i*x)
thin_exact = mask*incident
orders = np.arange(-M, M+1)
kx = kx_i+2*np.pi*orders/d
beta = np.sqrt((k0**2-kx**2).astype(complex))
keep = (abs(beta.imag) < 1e-12) & (beta.real > 1e-12)
t_m = D*np.sinc(orders*D)
thin_projected = t_m[keep] @ np.exp(1j*kx[keep, None]*x)
# Remove the incident carrier before applying a one-cell FFT.
envelope = thin_exact*np.exp(-1j*kx_i*x)
t_fft = np.fft.fft(envelope)[orders % samples]/samples
# The first midpoint is x[0], not zero.
t_fft *= np.exp(-2j*np.pi*orders*x[0]/d)

# %% CHECK AND INTERPRET
assert 0 < W_open <= d and samples > 2*M
# Sampling error decreases when samples is increased; it is not solver error.
fft_error = np.max(abs(t_fft-t_m))
assert fft_error < 2/samples
print(f'D = W_open/d = {D:g}; propagating orders: {orders[keep]}')
print(f'Maximum sampled versus analytical mask-coefficient difference: {fft_error:.3e}')
print('The exact boundary is binary. Its propagating projection is smooth and can ring.')

# %% GRAPHICS
if __name__ == '__main__':
    from wave_optics.exit_field_comparison import spatial_phase
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for field, label, style in [(thin_exact, 'Exact thin boundary', ':'),
                                 (thin_projected, 'Same propagating projection used in 9.3', '-')]:
        axes[0].plot(x, abs(field), style, label=label)
        axes[1].plot(x, spatial_phase(field), style, label=label)
    axes[0].set(xlabel='x (um)', ylabel='Amplitude at z_obj=0')
    axes[1].set(xlabel='x (um)', ylabel='Wrapped phase (rad)', ylim=(-3.4, 3.4))
    axes[0].legend(fontsize=8)
    finish(figure, show_plot)
