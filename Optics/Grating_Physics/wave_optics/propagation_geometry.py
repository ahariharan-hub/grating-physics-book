"""Linked two-plane geometry beside the kernel derivations in notebook 8.3."""
import numpy as np
import ipywidgets as widgets
from traitlets import Float, HasTraits, TraitError, validate
from matplotlib import rc_context
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Arc

from .green_kirchhoff import _owned_panel, _png, _slider
from .propagation_pictures import BLUE, ORANGE, GREEN, GREY
from . import propagation_kernels


GEOMETRY_NAMES = ('distance', 'observation', 'boundary')
STAGES = ('planes', 'full', 'fresnel', 'fraunhofer')


class GeometryState(HasTraits):
    """Shared numerical geometry; every displayed panel owns its own widgets."""
    distance = Float(4.)
    observation = Float(-1.)
    boundary = Float(2.)

    @validate(*GEOMETRY_NAMES)
    def _bounded(self, proposal):
        value = proposal['value']
        lo, hi = (1., 100.) if proposal['trait'].name == 'distance' else (-3., 3.)
        if not np.isfinite(value) or not lo-1e-10 <= value <= hi+1e-10:
            raise TraitError('Geometry must use s/lambda=1..100 and transverse coordinates -3..3.')
        return value


def bind_geometry(panel, state):
    """Synchronize local controls; close detaches only this panel's observers."""
    if state is None:
        return panel
    # A recreated panel adopts the shared state, never resets its siblings.
    for name in GEOMETRY_NAMES:
        expected = getattr(state, name)
        if panel.controls[name].value != expected:
            panel.controls[name].value = expected

    def from_control(change):
        for name in GEOMETRY_NAMES:
            if change['owner'] is panel.controls[name]:
                setattr(state, name, change['new'])
                break

    def from_state(change):
        control = panel.controls[change['name']]
        if control.value != change['new']:
            control.value = change['new']

    close_owned = panel.close

    def close():
        if not getattr(panel, '_closed', False):
            state.unobserve(from_state, names=GEOMETRY_NAMES)
            for name in GEOMETRY_NAMES:
                panel.controls[name].unobserve(from_control, names='value')
        close_owned()

    panel.close = close
    panel.geometry_state = state
    state.observe(from_state, names=GEOMETRY_NAMES)
    for name in GEOMETRY_NAMES:
        panel.controls[name].observe(from_control, names='value')
    return panel


def geometry_controls(state):
    controls = dict(
        distance=widgets.FloatLogSlider(value=state.distance, base=10, min=0., max=2., step=.02,
            description='s / λ', continuous_update=False, readout_format='.2f',
            layout=widgets.Layout(width='230px', max_width='100%'),
            style={'description_width':'initial'}),
        observation=_slider(state.observation, -3., 3., .1, 'P: x / λ'),
        boundary=_slider(state.boundary, -3., 3., .1, 'Q: ξ / λ'))
    for control in controls.values():
        control.layout.width = '230px'
    return controls


def draw_planes(ax, data, stage='planes', fronts=False):
    """Project 3D plane windows; selected P,Q lie in the undistorted y=0 cut.

    Horizontal is s and vertical is x. The y direction is projected diagonally.
    Window extent grows with the camera, never representing an aperture edge.
    """
    s, x, xi = data['distance'], data['observation'], data['boundary']
    extent = max(3.4, .13*s)
    def project(longitudinal, transverse, other):
        return np.array([longitudinal+.35*other, transverse+.25*other])
    corners = [(-extent, -extent), (extent, -extent), (extent, extent), (-extent, extent)]
    for position, color in [(0., BLUE), (s, ORANGE)]:
        vertices = np.array([project(position, a, b) for a, b in corners])
        ax.add_patch(Polygon(vertices, facecolor=color, edgecolor=color, alpha=.12, lw=1.2))
        ax.plot([position, position], [-extent, extent], color=color, lw=1.)
    if fronts:
        radii, stride = propagation_kernels.phase_front_radii(1.2*max(10., s+3))
        angle = np.linspace(-np.pi/2+.005, np.pi/2-.005, 241)
        segments = [np.column_stack([r*np.cos(angle), xi+r*np.sin(angle)]) for r in radii]
        ax.add_collection(LineCollection(segments, colors=BLUE, linewidths=.6,
                                        alpha=.22, zorder=0), autolim=False)
        ax.text(.02, .97, f'Full phase fronts: every {stride} cycle'+('s' if stride != 1 else ''),
                transform=ax.transAxes, fontsize=9.5, va='top')
    # The true connecting segment is unchanged by either approximation.
    ax.plot([0, s], [xi, x], color=GREY, lw=1.4)
    ax.plot([0, s], [0, 0], color=GREY, lw=.65, ls=':')
    if stage in ('planes', 'fresnel'):
        ax.plot([0, s, s], [xi, xi, x], color=ORANGE, lw=1., ls='--', alpha=.8)
    if stage == 'fraunhofer':
        ax.plot([0, s], [0, x], color=GREEN, lw=1.3)
        ax.plot([0, 0], [0, xi], color=GREEN, lw=3, alpha=.6)
    ax.scatter([0], [xi], marker='s', color=BLUE, s=36, zorder=4)
    ax.scatter([s], [x], color=ORANGE, s=30, zorder=4)
    ax.annotate(r'$Q=O$' if abs(xi) < 1e-12 else r'$Q$', (0, xi),
                xytext=(-8, 8), textcoords='offset points', ha='right')
    ax.annotate(r'$P$', (s, x), xytext=(8, 8), textcoords='offset points')
    ax.annotate(r'$R$', (s/2, (x+xi)/2), xytext=(0, 8), textcoords='offset points')
    if abs(xi) > .3:
        ax.plot(0, 0, '.', color=GREY, ms=4)
        ax.annotate(r'$O$', (0, 0), xytext=(3, 5) if stage == 'planes' else (-12, -12),
                    textcoords='offset points', fontsize=9)
    if abs(x) > .3:
        ax.plot(s, 0, '.', color=GREY, ms=4)
    if stage == 'full':
        angle = np.rad2deg(np.arctan2(x-xi, s))
        ax.annotate('', (1.6, xi), (0, xi), arrowprops=dict(arrowstyle='->', lw=.7, color=GREY))
        if abs(angle) > 5:
            ax.add_patch(Arc((0, xi), 1.25, 1.25, theta1=min(0, angle), theta2=max(0, angle),
                             color=GREY, lw=.7))
            a = np.deg2rad(angle/2)
            ax.text(.9*np.cos(a), xi+.9*np.sin(a), r'$\theta$', ha='center', va='center', fontsize=9)
    # Explicit plane labels stay readable as the physical separation increases.
    ax.annotate('Reference plane\n'+r'$s=0:\ (\xi,\eta)$', project(0, extent, -extent),
                xytext=(-5, 12), textcoords='offset points', ha='right', va='bottom', fontsize=10)
    ax.annotate('Observation plane\n'+r'$s>0:\ (x,y)$', project(s, extent, extent),
                xytext=(5, 8), textcoords='offset points', ha='left', va='bottom', fontsize=10)
    ydim = -1.65*extent
    ax.annotate('', (s, ydim), (0, ydim),
                arrowprops=dict(arrowstyle='<->', lw=.8, color=GREY))
    ax.text(s/2, ydim-.17*extent, f's = {s:g} λ  (towards −z)', ha='center', va='top', fontsize=10)
    if stage == 'planes':
        # Introduce both transverse axes once, on the same plane windows that
        # are reused in later sections. Avoid the selected meridional points.
        for position, coordinate, other, color, labels in [
            (0., 0. if xi < 0 else -.7, -.7, BLUE, (r'$\xi$', r'$\eta$')),
            (s, 0. if x < 0 else -.7, .4, ORANGE, (r'$x$', r'$y$'))]:
            anchor = project(position, coordinate*extent, other*extent)
            tips = [anchor+project(0, .45*extent, 0), anchor+project(0, 0, .45*extent)]
            for tip, label in zip(tips, labels):
                ax.annotate('', tip, anchor, arrowprops=dict(arrowstyle='->', lw=.7, color=color))
                ax.annotate(label, tip, xytext=(-7, 4), textcoords='offset points', fontsize=10)
    vertices = np.vstack([np.array([project(q, a, b) for a, b in corners]) for q in (0., s)])
    ax.update_datalim(vertices)
    ax.update_datalim([[-2.8-.35*extent, -2.15*extent], [s+2.8+.35*extent, 2.1*extent]])
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_axis_off()


def geometry_figure(data, stage='planes'):
    if stage not in STAGES:
        raise ValueError('Unknown geometry stage.')
    with rc_context({'font.size':10, 'axes.titlesize':11, 'axes.labelsize':10,
                     'xtick.labelsize':9, 'ytick.labelsize':9}):
        fig = Figure(figsize=(9., 5.0 if stage == 'fresnel' else 3.35),
                     layout='constrained', facecolor='white')
        if stage == 'fresnel':
            grid = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.])
            geometry = fig.add_subplot(grid[:, 0])
            focus = fig.add_subplot(grid[0, 1])
            magnitude = fig.add_subplot(grid[1, 1], sharex=focus)
        else:
            geometry, focus = fig.subplots(1, 2, gridspec_kw={'width_ratios':[1.3, 1.]})
        draw_planes(geometry, data, stage, fronts=stage == 'full')
        xi, s, x = data['xi'], data['distance'], data['observation']
        profile = data['profile']
        if stage == 'planes':
            # A meridional right triangle makes the 3D separation formula visible.
            delta = x-data['boundary']
            focus.plot([0, s, s, 0], [0, 0, delta, 0], color=GREY, lw=1.5)
            focus.scatter([0, s], [0, delta], color=[BLUE, ORANGE], s=28)
            focus.annotate(r'$s$', (s/2, 0), xytext=(0, 9 if delta <= 0 else -15), textcoords='offset points', ha='center')
            focus.annotate(r'$\rho$', (s, delta/2), xytext=(9, 0), textcoords='offset points')
            focus.annotate(r'$R$', (s/2, delta/2), xytext=(-12, -12 if delta <= 0 else 9), textcoords='offset points')
            focus.update_datalim([[-.2*s-1, -abs(delta)-2], [1.2*s+1, abs(delta)+2]])
            focus.autoscale_view(); focus.set_aspect('equal', adjustable='datalim'); focus.set_axis_off()
            focus.set_title(r'The selected connection: $R^2=s^2+\rho^2$')
            focus.text(.5, .03, r'$\rho=|x-\xi|$ in the $y=\eta=0$ section',
                       transform=focus.transAxes, ha='center', fontsize=10)
        elif stage == 'full':
            focus.plot(xi, abs(profile['kernels'][0]), color=BLUE, lw=1.8)
            focus.plot(data['boundary'], abs(data['selected']['kernels'][0]), 'o', color=BLUE, ms=5)
            focus.set(title='Full kernel: contribution strength',
                      ylabel=r'$\lambda^2 |K_{\rm RS1}|$', ylim=(0, 1.15))
        elif stage == 'fresnel':
            exact = profile['rho2']/(profile['R']+s)
            quadratic = profile['rho2']/(2*s)
            focus.plot(xi, exact, color=BLUE, lw=1.8, label=r'Full path: $R-s$')
            focus.plot(xi, quadratic, color=ORANGE, lw=1.8, ls='--', label=r'Quadratic: $\rho^2/(2s)$')
            for a, color in [(data['selected']['rho2']/(data['selected']['R']+s), BLUE),
                             (data['selected']['rho2']/(2*s), ORANGE)]:
                focus.plot(data['boundary'], a, 'o', color=color, ms=5)
            focus.set(title='Extra path: exact vs quadratic', ylabel=r'Extra path / $\lambda$', ylim=(0, None))
            focus.tick_params(labelbottom=False)
            magnitude.plot(xi, abs(profile['kernels'][0]), color=BLUE, lw=1.8)
            magnitude.plot(xi, abs(profile['kernels'][1]), color=ORANGE, lw=1.8, ls='--')
            for i, color in enumerate((BLUE, ORANGE)):
                magnitude.plot(data['boundary'], abs(data['selected']['kernels'][i]),
                               'o', color=color, ms=5)
            magnitude.axvline(data['boundary'], color=GREY, ls=':', lw=.7)
            magnitude.set(title=r'Magnitude: $|K_{\rm RS1}|$ vs $|K_{\rm F}|$',
                          ylabel=r'$\lambda^2 |K|$', ylim=(0, 1.15),
                          xlabel=r'Boundary coordinate $\xi/\lambda$')
            magnitude.grid(alpha=.12)
            fig.legend(focus.lines[:2], ['Full RS I', 'Fresnel'], loc='outside upper center',
                       ncol=2, frameon=False, fontsize=10)
        else:
            # Both expressions are relative to Q=O, so the common observation
            # phase cancels. Their difference is precisely the omitted phase.
            fresnel = np.pi*(xi*xi-2*x*xi)/s
            far = -2*np.pi*x*xi/s
            focus.plot(xi, fresnel, color=ORANGE, lw=1.8, ls='--', label='Fresnel: cross + boundary phase')
            focus.plot(xi, far, color=GREEN, lw=1.8, ls=':', label='Fraunhofer: cross phase')
            focus.fill_between(xi, far, fresnel, color=ORANGE, alpha=.12)
            q = data['boundary']
            focus.plot([q, q], [-2*np.pi*x*q/s, np.pi*(q*q-2*x*q)/s], color=GREY, lw=1.2)
            focus.plot(q, np.pi*(q*q-2*x*q)/s, 'o', color=ORANGE, ms=5)
            focus.plot(q, -2*np.pi*x*q/s, 'o', color=GREEN, ms=5)
            lo, hi = min(-.5, min(far), min(fresnel)), max(.5, max(far), max(fresnel))
            focus.set(title='Fraunhofer omits the shaded phase', ylabel=r'$\phi(Q)-\phi(O)$ (rad)',
                      ylim=(lo-.08*(hi-lo), hi+.08*(hi-lo)))
            focus.legend(fontsize=9)
        if stage != 'planes':
            focus.axvline(data['boundary'], color=GREY, ls=':', lw=.7)
            focus.set(xlim=(-3, 3))
            if stage != 'fresnel':
                focus.set_xlabel(r'Boundary coordinate $\xi/\lambda$')
            focus.grid(alpha=.12)
        return fig


def geometry_explorer(stage='planes', state=None):
    if stage not in STAGES:
        raise ValueError('Unknown geometry stage.')
    state = GeometryState() if state is None else state
    controls = geometry_controls(state)
    notes = {
        'planes':'Q lies on the reference plane; P lies on the observation plane. The outlines are viewing windows on infinite planes, not apertures. Dashed legs identify separation s and transverse offset.',
        'full':'The full kernel retains the true path, directional weighting and near-field term. Blue fronts are its phase contours in the y = η = 0 section. ASM describes the same propagation in a plane-wave basis.',
        'fresnel':'The planes and straight separation R do not change. Upper right: exact versus quadratic extra path. Lower right: magnitudes of the full and Fresnel kernels on one fixed scale. The dashed triangle labels geometry, not a bent propagation path.',
        'fraunhofer':'Move P to change the observation direction and phase tilt. Fresnel retains curvature; Fraunhofer keeps the local tilt. The green phase line is tangent to the Fresnel parabola at Q = O; the shaded gap is the omitted curvature phase.'}

    def render(panel, values):
        data = propagation_kernels.kernel_state(**values)
        panel.data = data
        panel.image.value = _png(geometry_figure(data, stage))
        panel.note.value = '<div>'+notes[stage]+'</div>'
        r = float(data['selected']['R'])
        s, q = data['distance'], data['boundary']
        if stage == 'fresnel':
            pref = (s/r)**2*(1+1j/(2*np.pi*r))
            panel.equation.value = (
                f'<div><b>Selected magnitudes λ²|K|:</b> full '
                f'{abs(data["selected"]["kernels"][0]):.4g}; '
                f'Fresnel {abs(data["selected"]["kernels"][1]):.4g}.</div>'
                f'<div><b>Selected extra path:</b> full {data["selected"]["rho2"]/(r+s):.4g} λ; '
                f'quadratic {data["selected"]["rho2"]/(2*s):.4g} λ.</div>'
                f'<div><b>Complex prefactor:</b> {pref.real:.4f} + {pref.imag:.4f}i → 1 in Fresnel.</div>')
        elif stage == 'fraunhofer':
            panel.equation.value = f'<div><b>Selected omitted phase:</b> kξ²/(2s) = {np.pi*q*q/s:.4g} rad. This vanishes at Q = O.</div>'
        elif stage == 'full':
            panel.equation.value = (
                f'<div><b>Selected:</b> λ²|K<sub>s</sub>| = '
                f'{abs(data["selected"]["kernels"][0]):.4g}; '
                f'R = {r:.4g} λ; cos θ = s/R = {s/r:.4g}.</div>')
        else:
            panel.equation.value = f'<div><b>Selected connection:</b> R = {r:.4g} λ; transverse offset = {abs(data["observation"]-q):.4g} λ. Geometry controls are linked across sections.</div>'
    return bind_geometry(_owned_panel(controls, render), state)
