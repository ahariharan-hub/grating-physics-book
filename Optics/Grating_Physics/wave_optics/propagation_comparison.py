"""Full 2D scalar ASM versus Fresnel and paraxial Fraunhofer, at y=0.

exp(-i omega t); lengths in micrometres; s=-z>0. The boundary is a circular
Gaussian envelope times a thin sinusoidal amplitude mask, period 8 um. The
angular spectrum is analytic; both transverse wavevectors are integrated.
The numerical Fourier sum has a finite spectral cutoff and periodization:
window/spacing are quadrature parameters, not an aperture or detector size.
"""
from functools import lru_cache
from io import StringIO
from pathlib import Path

import numpy as np
import ipywidgets as widgets
from matplotlib.figure import Figure
from matplotlib import rc_context
from scipy.interpolate import CubicSpline

from .scalar_propagation import transfer_function
from .green_kirchhoff import _owned_panel, _slider, _png
from .propagation_pictures import BLUE, ORANGE, GREEN, GREY, INK

PERIOD = 8.
ORDERS = ((-1, .25), (0, .5), (1, .25))
METHODS = ('ASM', 'Fresnel', 'Fraunhofer')
COLORS = (BLUE, ORANGE, GREEN)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f'{name} must be finite, real and positive.')
    try:
        a = np.asarray(value, float)
        if a.ndim or not np.isfinite(a) or a <= 0:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be finite, real and positive.') from error
    return float(a)


def _coordinates(x, y):
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ValueError('Observation coordinates must be finite and real.')
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Observation coordinates must be finite and real.')
    return x, y


def boundary_field(x, y=0., width=6.):
    """A prescribed field, not a material-scattering or Gaussian-beam solver."""
    width = _positive(width, 'width'); x, y = _coordinates(x, y)
    return np.exp(-(x*x+y*y)/width**2)*(.5+.5*np.cos(2*np.pi*x/PERIOD))


def boundary_spectrum(kx, ky=0., width=6.):
    """Forward 2D Fourier transform, with inverse measure dkx dky/(2pi)^2."""
    width = _positive(width, 'width'); kx, ky = _coordinates(kx, ky)
    return np.pi*width**2*sum(c*np.exp(-width**2*((kx-m*2*np.pi/PERIOD)**2+ky**2)/4)
                            for m, c in ORDERS)


def approximate_field(x, y=0., *, wavelength=1., width=6., distance=120., method='fresnel'):
    """Analytic Fresnel solution or usual paraxial Fraunhofer field in 3D.

    The carrier, quadratic observation phase and absolute amplitude are kept.
    This avoids resampling an FFT onto a different Fraunhofer output grid.
    """
    wavelength = _positive(wavelength, 'wavelength'); width = _positive(width, 'width')
    distance = _positive(distance, 'distance'); x, y = _coordinates(x, y)
    k = 2*np.pi/wavelength; carrier = np.exp(1j*k*distance)
    if method == 'fraunhofer':
        return (carrier*np.exp(1j*k*(x*x+y*y)/(2*distance))/(1j*wavelength*distance)*
                boundary_spectrum(k*x/distance, k*y/distance, width))
    if method != 'fresnel':
        raise ValueError("method must be 'fresnel' or 'fraunhofer'.")
    denominator = 1+1j*wavelength*distance/(np.pi*width**2)
    out = np.zeros(x.shape, complex)
    for m, c in ORDERS:
        q = m*2*np.pi/PERIOD
        out += c*np.exp(-((x-distance*q/k)**2+y*y)/(width**2*denominator))*np.exp(1j*q*x-1j*q*q*distance/(2*k))
    return carrier*out/denominator


@lru_cache(maxsize=6)
def _asm_weights(wavelength, width, distance, window=2048., spacing=.5):
    """Collapse the ky integral at y=0; retain kx for arbitrary output points.

    Uniform quadrature is equivalent to a periodic Fourier representation,
    with delta_k=2pi/window and cutoff pi/spacing. Gaussian tails below 1e-14
    are omitted. Chunking bounds memory; no 2D observation grid is required.
    The existing ASM helper receives positive distance, exactly as for s=-z.
    """
    for value, name in [(wavelength, 'wavelength'), (width, 'width'), (distance, 'distance'),
                        (window, 'window'), (spacing, 'spacing')]:
        _positive(value, name)
    size = int(round(window/spacing))
    if size < 64 or size > 32768 or not np.isclose(size*spacing, window):
        raise ValueError('Use an integral Fourier grid with 64..32768 samples.')
    kx = 2*np.pi*np.fft.fftfreq(size, spacing)
    ax = sum(c*np.exp(-width**2*(kx-m*2*np.pi/PERIOD)**2/4) for m, c in ORDERS)
    ay = np.exp(-width**2*kx*kx/4)
    keep_x = ax > 1e-14; keep_y = ay > 1e-14
    ky, ay = kx[keep_y], ay[keep_y]
    kx, ax = kx[keep_x], ax[keep_x]
    result = np.zeros(len(kx), complex)
    for start in range(0, len(ky), 128):
        h = transfer_function(kx[None, :]/(2*np.pi), ky[start:start+128, None]/(2*np.pi),
                              wavelength, distance, 'asm')
        result += np.sum(ay[start:start+128, None]*h, axis=0)
    weights = np.pi*width**2*ax*result/window**2
    kx.setflags(write=False); weights.setflags(write=False)
    return kx, weights


def asm_cut(x, *, wavelength=1., width=6., distance=120., window=2048., spacing=.5):
    """Full 2D angular-spectrum integral at (x,0,s), with finite quadrature."""
    x, _ = _coordinates(x, 0.)
    kx, weights = _asm_weights(wavelength, width, distance, window, spacing)
    # A dense inverse Fourier series permits accurate interpolation at the same
    # arbitrary output points as the analytic approximations. No output grid
    # is replaced by a differently scaled Fraunhofer FFT grid.
    count = 64*int(round(window/spacing))
    coefficients = np.zeros(count, complex)
    indices = np.rint(kx*window/(2*np.pi)).astype(int) % count
    coefficients[indices] = weights
    field = np.fft.fftshift(np.fft.ifft(coefficients))*count
    grid = (np.arange(count)-count//2)*(window/count)
    interpolator = CubicSpline(np.r_[grid, window/2], np.r_[field, field[0]], bc_type='periodic')
    return interpolator((x+window/2) % window-window/2)


def relative_cut_error(value, reference, x):
    """Complex L2 difference over this cut; no intensity-only or piston fit."""
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(np.sqrt(integrate(abs(value-reference)**2, x)/integrate(abs(reference)**2, x)))


@lru_cache(maxsize=6)
def comparison(wavelength=1., width=6., distance=120., window=2048., spacing=.5):
    """All methods at identical physical points and with one amplitude reference."""
    wavelength = _positive(wavelength, 'wavelength'); width = _positive(width, 'width')
    distance = _positive(distance, 'distance')
    if not .5 <= wavelength <= 1.5 or not 2 <= width <= 12 or not 20-1e-9 <= distance <= 800+1e-9:
        raise ValueError('Use wavelength .5..1.5, width 2..12 and distance 20..800 um.')
    beam_width = np.hypot(width, wavelength*distance/(np.pi*width))
    direction = wavelength/PERIOD
    half = max(3*width, distance*direction/np.sqrt(1-direction**2)+3*beam_width)
    # Include fine sampling of narrow Fraunhofer peaks even where that model
    # is invalid; coarse sampling must not hide its incorrect prediction.
    x = np.linspace(-half, half, int(np.ceil(2*half/.25))+1)
    far_width = wavelength*distance/(np.pi*width)
    for m, _ in ORDERS:
        x = np.r_[x, np.linspace(m*wavelength*distance/PERIOD-4*far_width,
                                m*wavelength*distance/PERIOD+4*far_width, 241)]
    x = np.unique(x[(x >= -half) & (x <= half)])
    fields = np.stack([asm_cut(x, wavelength=wavelength, width=width, distance=distance,
                              window=window, spacing=spacing)]+[
        approximate_field(x, wavelength=wavelength, width=width, distance=distance, method=m)
        for m in ('fresnel', 'fraunhofer')])
    errors = [relative_cut_error(u, fields[0], x) for u in fields[1:]]
    x.setflags(write=False); fields.setflags(write=False)
    return dict(x=x, fields=fields, errors=tuple(errors), width_scale=width**2/(wavelength*distance),
                wavelength=wavelength, width=width, distance=distance, window=window, spacing=spacing)


def phase_difference(value, reference):
    """Wrapped phase difference; omit dark points and break branch-cut lines."""
    phase = np.angle(value*reference.conj())
    valid = (abs(value) > .01*max(abs(value))) & (abs(reference) > .01*max(abs(reference)))
    phase = np.where(valid, phase, np.nan)
    jumps = np.r_[False, abs(np.diff(phase)) > np.pi]
    return np.where(jumps, np.nan, phase)


def comparison_figure(data, view='Intensity'):
    if view not in ('Intensity', 'Amplitude', 'Phase difference'):
        raise ValueError('Unknown comparison view.')
    fig = Figure(figsize=(10.5, 6.1), layout='constrained', facecolor='white')
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.7])
    plane = fig.add_subplot(grid[0, 0]); spectrum = fig.add_subplot(grid[0, 1])
    output = fig.add_subplot(grid[1, :])
    width, wavelength, distance = data['width'], data['wavelength'], data['distance']
    xi = np.linspace(-36, 36, 1001)
    plane.plot(xi, np.exp(-xi*xi/width**2), ':', color=GREY, label='Illumination envelope')
    plane.plot(xi, boundary_field(xi, width=width), color=BLUE, lw=1.5, label=r'$U_{\rm ref}(\xi,0)$')
    plane.axvline(-width, color=GREY, lw=.6, alpha=.5); plane.axvline(width, color=GREY, lw=.6, alpha=.5)
    plane.set(xlim=(-36, 36), ylim=(0, 1.05), xlabel=r'Plane coordinate $\xi$ (um)',
              ylabel='Field amplitude', title='One prescribed finite-grating field')
    plane.legend(fontsize=8, loc='upper right')
    alpha = np.linspace(-1.2, 1.2, 1201); k = 2*np.pi/wavelength
    spectrum.plot(alpha, boundary_spectrum(k*alpha, 0., width)/(np.pi*width**2), color=BLUE)
    for edge in (-1, 1): spectrum.axvline(edge, color=GREY, ls='--', lw=.8)
    spectrum.axvspan(-1.2, -1, color=GREY, alpha=.12); spectrum.axvspan(1, 1.2, color=GREY, alpha=.12)
    spectrum.set(xlim=(-1.2, 1.2), ylim=(0, 1.05), xlabel=r'$k_x/k$ at $k_y=0$',
                 ylabel=r'Spectrum / $(\pi w^2)$', title='Finite illumination broadens the spectrum')
    spectrum.text(.02, .95, 'Dashed: propagation boundary', transform=spectrum.transAxes, va='top', fontsize=8, color=GREY)
    for name, color, field in zip(METHODS, COLORS, data['fields']):
        if view == 'Intensity': values = abs(field)**2
        elif view == 'Amplitude': values = abs(field)
        else: values = phase_difference(field, data['fields'][0])
        output.plot(data['x'], values, color=color, label=name,
                    ls='-' if name == 'ASM' else '--' if name == 'Fresnel' else ':', lw=1.8)
    output.set(xlim=(data['x'][0], data['x'][-1]), xlabel=r'Observation coordinate $x$ at $y=0$ (um)',
               title=f'Same observation plane: s = {distance:g} um')
    if view == 'Phase difference':
        output.set(ylabel='Phase relative to ASM (rad)', ylim=(-np.pi, np.pi))
        output.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
    else:
        output.set(ylabel='Intensity / incident peak' if view == 'Intensity' else 'Amplitude / incident peak', ylim=(0, None))
        output.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useMathText=True)
    output.legend(fontsize=9); output.grid(alpha=.15)
    for ax in (plane, spectrum, output): ax.tick_params(labelsize=9)
    fig.suptitle('ASM, Fresnel and Fraunhofer: compare the same complex field', fontsize=14, color=INK)
    return fig


def comparison_explorer():
    controls = dict(distance=widgets.FloatLogSlider(value=120., base=10, min=np.log10(20), max=np.log10(800),
                        step=.025, description='Distance s', continuous_update=False, readout_format='.1f',
                        layout=widgets.Layout(width='290px', max_width='100%'), style={'description_width':'initial'}),
                    width=_slider(6., 2., 12., .5, 'Illumination w'),
                    wavelength=_slider(1., .5, 1.5, .1, 'Wavelength'),
                    view=widgets.ToggleButtons(options=['Intensity', 'Amplitude', 'Phase difference'],
                        description='Compare', layout=widgets.Layout(width='100%', max_width='100%'),
                        style={'button_width':'135px'}))
    def render(panel, values):
        physical = {key: values[key] for key in ('wavelength', 'width', 'distance')}
        data = comparison(**physical)
        panel.image.value = _png(comparison_figure(data, values['view'])); panel.data = data
        panel.note.value = ('<p>All methods use the same boundary field, physical observation coordinates and incident amplitude. '
                            'There is no per-curve peak normalization. The output axes follow the predictions. '+
                            ('Phase is compared directly with ASM; points below 1% of either field peak are omitted.'
                             if values['view'] == 'Phase difference' else 'Switch to Phase difference to reveal disagreements that intensity can hide.')+'</p>')
        ef, eff = data['errors']
        panel.equation.value = (
            f'<div><b>Gaussian width indicator w²/(λs):</b> {data["width_scale"]:.3g} (not a universal validity threshold)</div>'
            '<details><summary>Complex-field differences and numerical model</summary>'
            f'<div>Relative L2 difference from ASM over this cut: Fresnel {ef:.3g}; Fraunhofer {eff:.3g}.</div>'
            '<div>No phase alignment or intensity-only fit. Full 2D angular-spectrum quadrature includes evanescent decay; '
            'Fresnel and Fraunhofer are evaluated analytically at the same points. Both transverse coordinates are integrated, '
            'although the plot shows y=0. All lengths are micrometres; wavelength is in the medium.</div></details>')
    return _owned_panel(controls, render)


def write_static(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({'svg.fonttype':'path', 'svg.hashsalt':'propagation_comparison_8_3', 'font.family':'DejaVu Sans'}):
        fig = comparison_figure(comparison())
        try:
            stream = StringIO(); fig.savefig(stream, format='svg', metadata={'Date':None, 'Title':'ASM, Fresnel and Fraunhofer comparison'})
            path.write_text('\n'.join(line.rstrip() for line in stream.getvalue().splitlines())+'\n', encoding='utf8')
        finally: fig.clear()
    return path
