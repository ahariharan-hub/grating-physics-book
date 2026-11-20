"""Spatial propagation kernels for 8.3, before any exit-field calculation.

Time convention exp(-i omega t), positive distance s towards -z. The
meridional plots are sections of 3D kernels, never 1D slit propagators.
Explorer coordinates are measured in medium wavelengths (lambda=1).
"""
from functools import lru_cache
from io import StringIO
from pathlib import Path

import ipywidgets as widgets
import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from .green_kirchhoff import _owned_panel, _png, _slider
from .propagation_pictures import BLUE, ORANGE, GREEN, GREY


NAMES = ('Full RS I / ASM', 'Fresnel', 'Fraunhofer')
COLORS = (BLUE, ORANGE, GREEN)
STYLES = ('-', '--', ':')
VIEWS = ('Patch response', 'Compare shapes', 'Absolute weights')


def phase_front_radii(max_radius):
    """Full-kernel phase fronts in wavelength units, without a far-field phase.

    For s>0, arg K = kR - atan(kR). Solve that unwrapped phase = 2*pi*j.
    Display at most twelve fronts; the returned stride labels any thinning.
    """
    radius = _positive(max_radius, 'max_radius')
    k = 2*np.pi
    count = int(np.floor((k*radius-np.arctan(k*radius))/(2*np.pi)))
    stride = max(1, int(np.ceil(count/12)))
    levels = 2*np.pi*np.arange(1, count+1, stride)
    u = levels+np.pi/2
    for _ in range(5):
        u -= (u-np.arctan(u)-levels)/(1-1/(1+u*u))
    return u/k, stride


def _real(value, name):
    if np.iscomplexobj(value) or np.asarray(value).dtype.kind == 'b':
        raise ValueError(f'{name} must be finite and real.')
    try:
        value = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f'{name} must be finite and real.') from error
    if not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be finite and real.')
    return value


def _positive(value, name):
    value = _real(value, name)
    if value.ndim or value <= 0:
        raise ValueError(f'{name} must be a positive scalar.')
    return float(value)


def kernel_values(x, xi, *, y=0., eta=0., distance=4., wavelength=1.):
    """Full RS I, Fresnel and usual paraxial Fraunhofer at identical P,Q.

    Coordinates broadcast; kernels have inverse-area units. ``reduced`` is
    i*lambda*s*exp(-iks)*K, using one shared complex factor for all methods.
    ``phase`` continuously tracks its phase analytically, without wrapped
    phase subtraction or sampled unwrapping. Thus phase errors retain whole
    turns. Full ``kernels`` retain the carrier and absolute amplitude.
    """
    s = _positive(distance, 'distance')
    lam = _positive(wavelength, 'wavelength')
    x, xi, y, eta = np.broadcast_arrays(*[
        _real(value, name) for value, name in [(x, 'x'), (xi, 'xi'), (y, 'y'), (eta, 'eta')]])
    k = 2*np.pi/lam
    rho2 = (x-xi)**2+(y-eta)**2
    R = np.sqrt(s*s+rho2)
    # R-s = rho^2/(R+s) avoids subtracting nearly equal path lengths.
    delta = k*rho2/(R+s)
    full = (s/R)**2*(1+1j/(k*R))*np.exp(1j*delta)
    fresnel_phase = k*rho2/(2*s)
    far_phase = k*(x*x+y*y-2*(x*xi+y*eta))/(2*s)
    reduced = np.stack([full, np.exp(1j*fresnel_phase), np.exp(1j*far_phase)])
    phase = np.stack([delta+np.arctan(1/(k*R)), fresnel_phase, far_phase])
    kernels = np.exp(1j*k*s)/(1j*lam*s)*reduced
    return dict(kernels=kernels, reduced=reduced, phase=phase,
                phase_error=phase-phase[0], R=R, rho2=rho2,
                distance=s, wavelength=lam)


@lru_cache(maxsize=8, typed=True)
def kernel_state(distance=4., observation=-1., boundary=2.):
    """One selected pair and a fixed [-3,3] lambda section of the same kernels."""
    s = _positive(distance, 'distance')
    x, xi = _real(observation, 'observation'), _real(boundary, 'boundary')
    if x.ndim or xi.ndim or not 1-1e-10 <= s <= 100+1e-10 or abs(x) > 3 or abs(xi) > 3:
        raise ValueError('Use s/lambda=1..100 and transverse coordinates -3..3.')
    x, xi = float(x), float(xi)
    grid = np.linspace(-3, 3, 1201)
    profile = kernel_values(x, grid, distance=s)
    selected = kernel_values(x, xi, distance=s)
    time_cycles = np.linspace(0, 2, 601)
    # U(Q) has unit amplitude and zero phase. This is the real contribution
    # per U(Q)*dA/lambda^2, not the field of a finite patch or full aperture.
    real_response = np.real(selected['kernels'][:, None]*np.exp(-2j*np.pi*time_cycles))
    for data in (profile, selected):
        for value in data.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
    grid.setflags(write=False)
    time_cycles.setflags(write=False); real_response.setflags(write=False)
    return dict(distance=s, observation=x, boundary=xi, xi=grid,
                profile=profile, selected=selected,
                time_cycles=time_cycles, real_response=real_response)


def kernel_figure(data, view='Patch response'):
    """Four linked views; one normalization and fixed magnitude/phasor scales."""
    if view not in VIEWS:
        raise ValueError('Unknown kernel display.')
    with rc_context({'font.size':11, 'axes.titlesize':12, 'axes.labelsize':11,
                     'xtick.labelsize':10, 'ytick.labelsize':10}):
        fig = Figure(figsize=(9., 6.7), layout='constrained', facecolor='white')
        geometry, phasor, magnitude, phase = fig.subplots(2, 2).flat
        s, x, xi = data['distance'], data['observation'], data['boundary']
        selected, profile = data['selected'], data['profile']
        reduced = view == 'Compare shapes'
        key = 'reduced' if reduced else 'kernels'
        weights, curves = selected[key], profile[key]

        from .propagation_geometry import draw_planes
        draw_planes(geometry, data, stage='full', fronts=True)
        geometry.set_title('Reference and observation planes')

        if view == 'Patch response':
            for color, style, response in zip(COLORS, STYLES, data['real_response']):
                phasor.plot(data['time_cycles'], response, color=color, ls=style, lw=1.8)
                phasor.plot(0, response[0], 'o', color=color, ms=4)
            phasor.axhline(0, color=GREY, lw=.6, alpha=.4)
            phasor.set(xlim=(0, 2), ylim=(-1.15, 1.15), title='One patch: real response at P',
                       xlabel=r'Time $t/T$', ylabel=r'$\mathrm{Re}[\lambda^2 K e^{-i\omega t}]$')
            phasor.set_xticks([0, .5, 1, 1.5, 2]); phasor.set_yticks([-1, 0, 1])
            phasor.grid(alpha=.15)
        else:
            phasor.add_patch(Circle((0, 0), 1, fill=False, color=GREY, alpha=.35, ls=':'))
            phasor.axhline(0, color=GREY, lw=.6, alpha=.4)
            phasor.axvline(0, color=GREY, lw=.6, alpha=.4)
            for name, color, style, weight in zip(NAMES, COLORS, STYLES, weights):
                phasor.annotate('', (weight.real, weight.imag), (0, 0),
                                arrowprops=dict(arrowstyle='->', lw=1.7, color=color, linestyle=style))
            phasor.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect='equal',
                       title='Selected pair: complex weights',
                       xlabel=r'$\mathrm{Re}\,\widehat K$' if reduced else r'$\mathrm{Re}\,(\lambda^2 K)$',
                       ylabel=r'$\mathrm{Im}\,\widehat K$' if reduced else r'$\mathrm{Im}\,(\lambda^2 K)$')
            phasor.set_xticks([-1, 0, 1]); phasor.set_yticks([-1, 0, 1])

        for i, (name, color, style) in enumerate(zip(NAMES, COLORS, STYLES)):
            magnitude.plot(data['xi'], abs(curves[i]), color=color, ls=style, lw=1.8, label=name)
            magnitude.plot(xi, abs(weights[i]), 'o', color=color, ms=5)
            phase.plot(data['xi'], profile['phase_error'][i], color=color, ls=style, lw=1.8)
            phase.plot(xi, selected['phase_error'][i], 'o', color=color, ms=5)
        for ax in (magnitude, phase):
            ax.axvline(xi, color=GREY, lw=.8, ls=':', alpha=.65)
            ax.set(xlim=(-3, 3), xlabel=r'Boundary coordinate $\xi/\lambda$')
            ax.grid(alpha=.15)
        magnitude.set(ylim=(0, 1.15), title='Magnitude across the reference plane',
                      ylabel=r'$|\widehat K| = \lambda s |K|$' if reduced else r'$\lambda^2 |K|$')
        # Leave a fixed central band: agreement must flatten towards zero,
        # rather than being enlarged arbitrarily by automatic vertical scaling.
        lo = min(-.5, float(np.min(profile['phase_error'])))
        hi = max(.5, float(np.max(profile['phase_error'])))
        pad = .08*(hi-lo)
        phase.set(ylim=(lo-pad, hi+pad), title='Phase error relative to full RS I',
                  ylabel=r'$\Delta\phi=\phi_{\rm approx}-\phi_{\rm full}$ (rad)')
        handles, labels = magnitude.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False, fontsize=10)
        return fig


def kernel_explorer(state=None):
    """Live kernels and their one-patch interpretation; no full-field solve."""
    initial = dict(distance=4., observation=-1., boundary=2.) if state is None else {
        name:getattr(state, name) for name in ('distance', 'observation', 'boundary')}
    controls = dict(
        distance=widgets.FloatLogSlider(value=initial['distance'], base=10, min=0., max=2., step=.02,
            description='Separation s / λ', continuous_update=False, readout_format='.2f',
            layout=widgets.Layout(width='290px', max_width='100%'),
            style={'description_width':'initial'}),
        observation=_slider(initial['observation'], -3., 3., .1, 'P: x / λ'),
        boundary=_slider(initial['boundary'], -3., 3., .1, 'Q: ξ / λ'),
        view=widgets.ToggleButtons(options=VIEWS, value=VIEWS[0], description='Display',
            layout=widgets.Layout(width='100%', max_width='100%'),
            style={'button_width':'155px'}))

    def render(panel, values):
        data = kernel_state(**{key:values[key] for key in ('distance', 'observation', 'boundary')})
        panel.data = data
        panel.image.value = _png(kernel_figure(data, values['view']))
        panel.note.value = (
            '<p>Imagine one tiny boundary patch Q oscillating with unit amplitude and zero phase. '
            'Patch response shows its real contribution at P per unit patch strength, including actual phase and spreading. '
            'Blue fronts show the phase of the full response; P samples a different part when moved. '
            'The curves compare kernels, not a summed aperture field.</p>')
        selected = data['selected']
        magnitudes = abs(selected['kernels'])
        ef, eff = selected['phase_error'][1:]
        radius = float(selected['R'])
        angle = np.rad2deg(np.arctan2(abs(data['observation']-data['boundary']), data['distance']))
        if values['view'] == 'Patch response':
            display_note = 'Real oscillation per U(Q)dA/λ², with U(Q) real and positive; all response axes stay fixed.'
        elif values['view'] == 'Compare shapes':
            display_note = 'One common factor iλs exp(−iks) multiplies every kernel; arrow and magnitude axes stay fixed.'
        else:
            display_note = 'Absolute kernels in inverse-wavelength-squared units; no carrier or spreading factor is removed. Axes stay fixed.'
        panel.equation.value = (
            '<div><b>Display:</b> '+display_note+'</div>'
            f'<div><b>Path:</b> R / λ = {radius:.3f} wavelengths; '
            f'θ = {angle:.1f}° from the plane normal.</div>'
            f'<div><b>Full magnitude:</b> 1/(λR) spreading × cos θ × near-field correction; '
            f'cos θ = {data["distance"]/radius:.3f}.</div>'
            f'<div><b>Full response phase:</b> kR − atan(kR) = '
            f'{2*np.pi*radius-np.arctan(2*np.pi*radius):.3f} rad before display-factor removal.</div>'
            f'<div><b>Absolute weights λ²|K|:</b> full {magnitudes[0]:.4g}; '
            f'Fresnel {magnitudes[1]:.4g}; Fraunhofer {magnitudes[2]:.4g}.</div>'
            f'<div><b>Unwrapped phase error (approximation − full):</b> Fresnel {ef:+.3f} rad; '
            f'Fraunhofer {eff:+.3f} rad.</div>'
            f'<div><b>Extra Fraunhofer phase error:</b> {eff-ef:+.3f} rad '
            '(FF − F: negative omitted curvature phase).</div>')
    from .propagation_geometry import bind_geometry
    return bind_geometry(_owned_panel(controls, render), state)


def write_static(path):
    """Export a default state for offline inspection on explicit request."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({'svg.fonttype':'path', 'svg.hashsalt':'propagation_kernels_8_3',
                     'font.family':'DejaVu Sans'}):
        fig = kernel_figure(kernel_state())
        try:
            stream = StringIO()
            fig.savefig(stream, format='svg', metadata={'Date':None, 'Title':'Propagation kernel comparison'})
            path.write_text('\n'.join(line.rstrip() for line in stream.getvalue().splitlines())+'\n', encoding='utf-8')
        finally:
            fig.clear()
    return path
