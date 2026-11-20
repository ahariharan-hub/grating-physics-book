"""Fourier-space kernel pictures for Section 6 of notebook 8.3.

Reuse scalar_propagation's transfer functions with positive downstream s=-z.
The plots select ky=0 sections of the 3D theory. No boundary field or
spectrum amplitude is prescribed. Widget readouts contain no TeX.
"""
import ipywidgets as widgets
import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from . import scalar_propagation
from .green_kirchhoff import _owned_panel, _png, _slider
from .propagation_geometry import draw_planes
from .propagation_kernels import _positive, _real
from .propagation_pictures import BLUE, ORANGE, GREEN, GREY


def multiplier_values(kx, ky=0., *, distance=4., wavelength=1.):
    """Full and Fresnel multipliers for angular spatial frequencies.

    H retains the carrier; reduced removes the SAME exp(i*k*s) from both.
    phase_error is Fresnel minus full ASM, analytically unwrapped, and NaN
    outside the propagating band. The existing backend uses the opposite
    error sign, which is converted here to match Section 5's convention.
    """
    s = _real(distance, 'distance')
    if s.ndim or s < 0:
        raise ValueError('distance must be a nonnegative scalar.')
    s = float(s)
    lam = _positive(wavelength, 'wavelength')
    kx, ky = np.broadcast_arrays(_real(kx, 'kx'), _real(ky, 'ky'))
    fx, fy = kx/(2*np.pi), ky/(2*np.pi)
    geometry = scalar_propagation.spectrum_geometry(fx, fy, lam)
    h = np.stack([scalar_propagation.transfer_function(fx, fy, lam, s, method)
                  for method in ('asm', 'fresnel')])
    k = geometry['k']
    error = -scalar_propagation.fresnel_phase_error(fx, fy, lam, s)['exact']
    return dict(H=h, reduced=h*np.exp(-1j*k*s), kappa=geometry['kz'],
                kappa_f=k*(1-geometry['q']/2), phase_error=error,
                propagating=geometry['propagating'], k=k,
                distance=s, wavelength=lam)


def multiplier_state(distance=4., component=.3):
    """A signed kx/k section; selected and profile use identical operators."""
    s = _positive(distance, 'distance')
    u = _real(component, 'component')
    if u.ndim or not 1 <= s <= 100 or abs(u) > 1.25:
        raise ValueError('Use s/lambda=1..100 and kx/k=-1.25..1.25.')
    u = float(u)
    grid = np.linspace(-1.25, 1.25, 1001)
    return dict(distance=s, component=u, grid=grid,
                profile=multiplier_values(2*np.pi*grid, distance=s),
                selected=multiplier_values(2*np.pi*u, distance=s))


def multiplier_figure(data):
    """Wavevector geometry, selected phasor, magnitude and signed phase error."""
    with rc_context({'font.size':10, 'axes.titlesize':11, 'axes.labelsize':10,
                     'xtick.labelsize':9, 'ytick.labelsize':9}):
        fig = Figure(figsize=(9., 6.5), layout='constrained', facecolor='white')
        dispersion, phasor, magnitude, error = fig.subplots(2, 2).flat
        u = data['grid']; chosen = data['component']
        p, selected = data['profile'], data['selected']
        gamma = p['kappa']/p['k']; gamma_f = p['kappa_f']/p['k']
        band = p['propagating']; evanescent = ~band
        # Real-circle and imaginary branches are explicitly different curves.
        dispersion.plot(u, np.where(band, gamma.real, np.nan), color=BLUE,
                        lw=1.8, label='ASM: real part')
        dispersion.plot(u, np.where(evanescent, gamma.imag, np.nan), color=BLUE,
                        lw=1.5, ls=':', label='ASM: imaginary part')
        dispersion.plot(u, gamma_f, color=ORANGE, lw=1.8, ls='--', label='Fresnel')
        g = complex(selected['kappa']/selected['k'])
        if bool(selected['propagating']):
            dispersion.annotate('', (chosen, g.real), (0, 0),
                                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.3))
            height = g.real
        else:
            height = g.imag
        dispersion.plot(chosen, height, 'o', color=BLUE, ms=5)
        dispersion.plot(chosen, selected['kappa_f']/selected['k'], 'o', color=ORANGE, ms=5)
        dispersion.set(title='Wavevector circle and Fresnel parabola',
                       ylabel=r'Longitudinal component / $k$', ylim=(-.05, 1.17))
        dispersion.set_aspect('equal', adjustable='box')
        handles, labels = dispersion.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=3,
                   fontsize=9, frameon=False)

        phasor.add_patch(Circle((0, 0), 1., fill=False, color=GREY, lw=.8, ls=':'))
        phasor.axhline(0, color=GREY, lw=.6); phasor.axvline(0, color=GREY, lw=.6)
        for weight, color, style in zip(selected['reduced'], (BLUE, ORANGE), ('-', '--')):
            phasor.annotate('', (weight.real, weight.imag), (0, 0),
                            arrowprops=dict(arrowstyle='->', color=color, lw=2., linestyle=style))
            phasor.plot(weight.real, weight.imag, 'o', color=color, ms=4)
        phasor.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect='equal',
                   title='Selected component: carrier removed',
                   xlabel=r'$\mathrm{Re}\,\overline{H}$', ylabel=r'$\mathrm{Im}\,\overline{H}$')
        phasor.set_xticks([-1, 0, 1]); phasor.set_yticks([-1, 0, 1])

        for i, color, style in zip(range(2), (BLUE, ORANGE), ('-', '--')):
            magnitude.plot(u, abs(p['H'][i]), color=color, ls=style, lw=1.8)
            magnitude.plot(chosen, abs(selected['H'][i]), 'o', color=color, ms=5)
        magnitude.set(title='Propagating phase or evanescent decay?',
                      ylabel=r'Multiplier magnitude $|H|$', ylim=(0, 1.12))
        error.plot(u, p['phase_error'], color=ORANGE, lw=1.8)
        if bool(selected['propagating']):
            error.plot(chosen, selected['phase_error'], 'o', color=ORANGE, ms=5)
        error.set(title='Accumulated Fresnel phase error',
                  ylabel='Unwrapped Fresnel − ASM (rad)',
                  ylim=(-.02, max(.5, 1.07*float(np.nanmax(p['phase_error'])))))
        for ax in (dispersion, magnitude, error):
            for lo, hi in [(-1.25, -1.), (1., 1.25)]:
                ax.axvspan(lo, hi, color=GREY, alpha=.08)
            ax.axvline(chosen, color=GREY, ls=':', lw=.7)
            ax.set(xlim=(-1.25, 1.25), xlabel=r'Selected section: $k_x/k$; $k_y=0$')
            ax.grid(alpha=.12)
        return fig


def multiplier_explorer(distance=4., component=.3):
    """Two local controls; fixed phasor and magnitude scales, owned cleanup."""
    initial = multiplier_state(distance, component)
    controls = dict(
        distance=widgets.FloatLogSlider(value=initial['distance'], base=10, min=0., max=2., step=.02,
            description='Separation s / λ', continuous_update=False, readout_format='.2f',
            layout=widgets.Layout(width='290px', max_width='100%'),
            style={'description_width':'initial'}),
        component=_slider(initial['component'], -1.25, 1.25, .01, 'Component kx / k'))

    def render(panel, values):
        data = multiplier_state(**values); panel.data = data
        panel.image.value = _png(multiplier_figure(data))
        selected = data['selected']
        propagating = bool(selected['propagating'])
        g = complex(selected['kappa']/selected['k'])
        branch = 'evanescent' if not propagating else ('grazing' if abs(data['component']) == 1 else 'propagating')
        panel.note.value = (
            '<div>Blue: full ASM. Orange: Fresnel. Move the component to select a different wavevector; '
            'change separation to rotate its response or watch it decay. '
            'Grey bands are evanescent: the orange extension there is outside Fresnel’s validity.</div>')
        error = (f'{float(selected["phase_error"]):+.4g} rad (Fresnel − ASM; unwrapped)'
                 if propagating else 'not compared: this component is evanescent.')
        panel.equation.value = (
            f'<div><b>Selected component:</b> {branch}; κ/k = {g.real:.4f} + {g.imag:.4f}i.</div>'
            f'<div><b>Magnitude |H|:</b> ASM {abs(selected["H"][0]):.4g}; '
            f'Fresnel {abs(selected["H"][1]):.4g}.</div>'
            f'<div><b>Phase error:</b> {error}</div>'
            '<div><b>Arrow scale:</b> the same carrier exp(iks) is removed from both; '
            'the arrow lengths retain the actual magnitudes.</div>')
    return _owned_panel(controls, render)


def far_field_coordinate(x, y=0., *, distance=30., wavelength=1.):
    """Paraxial Fraunhofer observation-to-input-spectrum coordinate mapping.

    A coordinate selection, not a third same-frequency multiplier and not a
    spectrum amplitude. No boundary field is required to draw this map.
    """
    s = _positive(distance, 'distance'); lam = _positive(wavelength, 'wavelength')
    x, y = np.broadcast_arrays(_real(x, 'x'), _real(y, 'y'))
    k = 2*np.pi/lam
    return np.stack([k*x/s, k*y/s])


def mapping_state(direction=.08):
    """Fix s=30 lambda and ky=0; frequency is the selected (kx/k, ky/k)."""
    direction = _real(direction, 'direction')
    if direction.ndim or abs(direction) > .12+1e-12:
        raise ValueError('Use the paraxial direction x/s=-0.12..0.12.')
    direction = float(direction); s = 30.
    x = s*direction
    return dict(distance=s, observation=x, boundary=0., direction=direction,
                frequency=far_field_coordinate(x, distance=s)/(2*np.pi))


def mapping_figure(data):
    with rc_context({'font.size':10, 'axes.titlesize':11, 'axes.labelsize':10,
                     'xtick.labelsize':9, 'ytick.labelsize':9}):
        fig = Figure(figsize=(9., 3.35), layout='constrained', facecolor='white')
        geometry, frequency = fig.subplots(1, 2, gridspec_kw={'width_ratios':[1.3, 1.]})
        draw_planes(geometry, data, stage='fraunhofer')
        geometry.set_title('Move P to choose an observation direction')
        ux, uy = data['frequency']
        frequency.axhline(0, color=GREY, lw=.8)
        frequency.axvline(0, color=GREY, lw=.8)
        frequency.annotate('', (ux, uy), (0, 0),
                           arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8))
        frequency.plot(ux, uy, 'o', color=GREEN, ms=7)
        frequency.annotate(r'$k_x^{\rm obs}/k=x/s$', (ux, uy), xytext=(0, 12),
                           textcoords='offset points', ha='right' if ux >= 0 else 'left', fontsize=10)
        frequency.text(.5, .07, 'Frequency coordinate only;\nno spectrum amplitude assigned',
                       transform=frequency.transAxes, ha='center', fontsize=8)
        frequency.set(xlim=(-.16, .16), ylim=(-.16, .16), aspect='equal',
                      xlabel=r'$k_x/k$', ylabel=r'$k_y/k$',
                      title='Selected input-spectrum coordinate')
        frequency.set_xticks([-.12, 0, .12]); frequency.set_yticks([-.12, 0, .12])
        frequency.grid(alpha=.12)
        return fig


def mapping_explorer(direction=.08):
    initial = mapping_state(direction)
    controls = dict(direction=_slider(initial['direction'], -.12, .12, .01, 'P direction x / s'))

    def render(panel, values):
        data = mapping_state(**values); panel.data = data
        panel.image.value = _png(mapping_figure(data))
        panel.note.value = (
            '<div>Move P sideways. Its direction selects a Fourier weight across Q. '
            'The dot marks a frequency coordinate, not a measured amplitude. '
            'This is the paraxial Fraunhofer mapping; the validity conditions in Section 4 still apply.</div>')
        panel.equation.value = (
            f'<div><b>Observation:</b> s = 30 λ, x = {data["observation"]:.2f} λ, y = 0.</div>'
            f'<div><b>Selected Fourier coordinate:</b> kx/k = x/s = {data["direction"]:+.3f}; ky/k = 0.</div>')
    return _owned_panel(controls, render)
