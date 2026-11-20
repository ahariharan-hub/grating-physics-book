"""Small, separate diagrams for the introductory diffraction-order lesson.

Use the coordinates of notebook two: +x right, +z up, incidence toward
-z. The surface-phase explorer follows one incident wave; the earlier
path diagrams use normal illumination. No field summation or power model.
Figures and their SVG/PNG representations are generated in memory only.
"""

import io

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc

from .plotting import figure_png


BLUE = '#2866a3'
ORANGE = '#b85c17'
GREY = '#8c959f'
INK = '#243444'


def _figure(title, *, size=(7.4, 4.6)):
    figure = Figure(figsize=size, dpi=110, facecolor='white')
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    figure.subplots_adjust(left=.06, right=.96, bottom=.07, top=.88)
    axis.set_title(title, fontsize=12, pad=14)
    axis.set_aspect('equal')
    axis.axis('off')
    return figure, axis


def _openings(axis, positions):
    positions = np.asarray(positions)
    axis.plot([positions[0] - .6, positions[-1] + .6], [0, 0], color=GREY, lw=4)
    for index, xx in enumerate(positions):
        axis.plot([xx - .13, xx + .13], [0, 0], color='white', lw=7,
                  solid_capstyle='butt')
        axis.plot(xx, 0, 'o', color=BLUE, markersize=5)
        axis.text(xx, .16, chr(ord('A') + index), ha='center', color=INK, fontsize=11)


def illumination_figure():
    """Two openings on the same incident plane-wave crest."""
    figure, axis = _figure('What is the same at A and B?')
    _openings(axis, [0, 2])
    # Crest spacing is one length unit; d=2 units, as in the later d=2*lambda example.
    for height in (.55, 1.55):
        axis.plot([-.75, 2.75], [height, height], color=BLUE, lw=1.5)
    for xx in (0, 2):
        axis.annotate('', (xx, .34), (xx, 2.10),
                      arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.6))
    axis.text(3.0, 1.15, 'Incident crests\nparallel to the grating', fontsize=10,
              ha='left', va='center', color=BLUE)
    axis.annotate('', (0, -.48), (2, -.48),
                  arrowprops=dict(arrowstyle='<->', color=INK, shrinkA=0, shrinkB=0))
    axis.text(1, -.63, 'period d', ha='center', va='top', fontsize=11)
    axis.text(3.0, 0, 'grating', fontsize=10, va='center', color=INK)
    axis.text(1, -1.1, 'A and B are driven at the same phase.',
              ha='center', fontsize=11, color=INK)
    axis.set(xlim=(-.9, 5.0), ylim=(-1.35, 2.35))
    return figure


def equal_paths_figure():
    """Finite on-axis observation point: equality is exact by symmetry."""
    figure, axis = _figure('Look directly in front of the midpoint')
    _openings(axis, [0, 2])
    point = np.array([1, -2.7])
    axis.plot([1, 1], [0, point[1]], '--', color=GREY, lw=.9)
    for xx in (0, 2):
        axis.plot([xx, point[0]], [0, point[1]], color=BLUE, lw=1.8)
    axis.plot(*point, 'o', color=INK, markersize=6)
    axis.text(1, -3.04, 'P', fontsize=12, ha='center')
    axis.text(-.55, -1.4, 'AP', fontsize=11, color=BLUE)
    axis.text(2.40, -1.4, 'BP', fontsize=11, color=BLUE)
    axis.text(3, -1.0, 'AP = BP', fontsize=13, color=INK)
    axis.text(3, -1.55, 'Equal travel lengths\npreserve phase agreement.',
              fontsize=10, color=INK, va='center')
    axis.set(xlim=(-.9, 5.5), ylim=(-3.35, .6))
    return figure


def path_geometry(period=2.0, angle_deg=30.0):
    """Right-triangle construction for a positive far-field direction.

    A=(0,0), B=(d,0). Q lies on the ray from A and on the line through B
    perpendicular to the outgoing direction. AQ=d*sin(theta), QP~BP.
    The upper bound avoids the degenerate grazing triangle in this lesson.
    """
    if (not np.all(np.isfinite([period, angle_deg])) or period <= 0
            or not 0 <= angle_deg < 90):
        raise ValueError('Use a positive finite period and 0 <= angle < 90 degrees.')
    theta = np.deg2rad(angle_deg)
    direction = np.array([np.sin(theta), -np.cos(theta)])
    a, b = np.array([0., 0.]), np.array([float(period), 0.])
    extra = float((b - a) @ direction)
    q = a + extra * direction
    return dict(a=a, b=b, q=q, direction=direction, extra=extra,
                period=float(period), angle_deg=float(angle_deg))


def _ray(axis, start, direction, length, *, color=BLUE, linewidth=1.6):
    end = start + length * direction
    axis.annotate('', end, start,
                  arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                  shrinkA=0, shrinkB=0))
    return end


def phase_connection(wavelength=1.0, period=2.0, angle_deg=30.0):
    """The normal-incidence link between extra path and transverse phase.

    Lengths use a common unit; wavevector components are radians per unit.
    Arrival phase uses A-minus-B, with exp(-i*omega*t). Spatial plane-wave
    phase uses B-minus-A at equal time. Both give +kx*d in this geometry.
    """
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError('Wavelength must be positive and finite.')
    data = path_geometry(period, angle_deg)
    k = 2*np.pi/wavelength
    vector = k*data['direction']
    return dict(data, wavelength=float(wavelength), k=k, vector=vector,
                phase_from_path=k*data['extra'], phase_from_step=vector[0]*period)


def wavevector_connection_figure():
    """The existing d=2*lambda, theta=30 example, now in wavevector space."""
    figure, axis = _figure('The same outgoing direction, described by its wavevector',
                          size=(8.2, 5.0))
    data = phase_connection()
    kx, kz = data['vector']
    origin = np.zeros(2)
    _ray(axis, origin, data['direction'], data['k'], color=BLUE, linewidth=2)
    _ray(axis, origin, np.array([1., 0.]), kx, color=ORANGE, linewidth=2)
    axis.plot([kx, kx], [0, kz], '--', color=GREY, lw=1)
    axis.plot([0, 0], [0, -3.6], '--', color=GREY, lw=.9)
    axis.add_patch(Arc((0, 0), 4.0, 4.0, theta1=-90, theta2=-60, color=INK, lw=1))
    axis.text(.63, -2.28, 'θ = 30°', fontsize=11, ha='center')
    axis.text(-.18, -3.9, '−z direction', fontsize=10, color=GREY, ha='right')
    axis.text(kx/2, .38, r'$k_x=k\sin\theta$', color=ORANGE, fontsize=13, ha='center')
    axis.text(kx/2+.38, kz/2-.2, r'$\mathbf{k}$', fontsize=16, color=BLUE)
    axis.text(kx+.25, kz/2, r'$k_z<0$', fontsize=11, color=GREY)
    axis.text(kx+.25, .02, '+x', fontsize=11, color=INK, va='center')
    axis.text(5.0, -.65, 'Same example', fontsize=12, color=INK)
    axis.text(5.0, -1.35, 'λ = 1 µm; d = 2 µm', fontsize=11, color=INK)
    axis.text(5.0, -2.15, r'$k=2\pi\ \mathrm{rad}/\mu\mathrm{m}$', fontsize=12, color=BLUE)
    axis.text(5.0, -3.0, r'$k_x=\pi\ \mathrm{rad}/\mu\mathrm{m}$', fontsize=12, color=ORANGE)
    axis.text(5.0, -4.1, r'$k_xd=2\pi\ \mathrm{rad}$', fontsize=13, color=INK)
    axis.text(3.8, -6.15, 'Horizontal phase change per unit distance × one period = phase change.',
              ha='center', fontsize=10, color=INK)
    axis.set(xlim=(-2, 10.1), ylim=(-6.7, 1.2))
    return figure


def extra_path_figure():
    """One extra segment, isolated using a perpendicular reference line."""
    figure, axis = _figure('Look toward a distant point to the right', size=(7.4, 5.4))
    data = path_geometry()
    a, b, q, direction = (data[key] for key in ('a', 'b', 'q', 'direction'))
    _openings(axis, [0, 2])
    _ray(axis, a, direction, 4)
    _ray(axis, b, direction, 3)
    axis.plot([q[0], b[0]], [q[1], b[1]], '--', color=GREY, lw=1.1)
    axis.plot([a[0], q[0]], [a[1], q[1]], color=ORANGE, lw=4, solid_capstyle='butt')
    axis.plot(*q, 'o', color=ORANGE, markersize=4)
    axis.annotate('Q', q, xytext=(-14, -10), textcoords='offset points', fontsize=11)
    axis.text(-.45, -.48, 'extra path\nA to Q', color=ORANGE, fontsize=10,
              ha='right', va='center')
    axis.annotate('', (.19, -.35), (-.4, -.43),
                  arrowprops=dict(arrowstyle='-', color=ORANGE, lw=.8))
    perpendicular = np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
    corner = np.array([q - .17*direction,
                       q - .17*direction + .17*perpendicular,
                       q + .17*perpendicular])
    axis.plot(corner[:, 0], corner[:, 1], color=GREY, lw=.9)
    axis.plot([0, 0], [0, -1.8], '--', color=GREY, lw=.8)
    axis.add_patch(Arc((0, 0), 2.65, 2.65, theta1=-90, theta2=-60, color=INK, lw=1))
    axis.text(.42, -1.57, 'θ', fontsize=12)
    axis.text(-.12, -1.95, 'normal (−z)', ha='right', fontsize=10, color=GREY)
    axis.annotate('', (0, .58), (2, .58),
                  arrowprops=dict(arrowstyle='<->', color=INK, shrinkA=0, shrinkB=0))
    axis.text(1, .70, 'd', ha='center', fontsize=12)
    axis.text(3.35, -1.15, 'Parallel local paths\ntoward the same distant P',
              fontsize=10, va='center', color=BLUE)
    axis.text(3.8, -2.05, 'BQ is perpendicular\nto the outgoing paths.',
              fontsize=10, va='center', color=GREY)
    axis.text(1.1, -3.9, 'From Q and B onward, the remaining lengths to P are approximately equal.',
              ha='center', fontsize=10, color=INK)
    axis.set(xlim=(-1.5, 5.8), ylim=(-4.3, 1.0))
    return figure


def repeated_paths_figure():
    """The same extra segment repeats at every neighboring pair."""
    figure, axis = _figure('Add two more identical, equally spaced openings', size=(8.2, 4.8))
    data = path_geometry()
    direction, shift = data['direction'], data['q']
    positions = 2*np.arange(4)
    _openings(axis, positions)
    for xx in positions:
        start = np.array([xx, 0.])
        _ray(axis, start, direction, 2.7)
    for xx in positions[:-1]:
        start = np.array([xx, 0.])
        q = start + shift
        next_opening = start + np.array([2., 0.])
        axis.plot([q[0], next_opening[0]], [q[1], next_opening[1]],
                  '--', color=GREY, lw=.9)
        axis.plot([start[0], q[0]], [start[1], q[1]],
                  color=ORANGE, lw=4, solid_capstyle='butt')
        axis.text(start[0] - .10, -.65, 'λ', color=ORANGE, ha='right', fontsize=12)
        axis.annotate('', (xx, .60), (xx+2, .60),
                      arrowprops=dict(arrowstyle='<->', color=INK, shrinkA=0, shrinkB=0))
        axis.text(xx+1, .72, 'd', fontsize=12, ha='center')
    axis.text(3.3, -3.00, 'One wavelength of extra path at every neighboring pair.',
              fontsize=11, ha='center', color=INK)
    axis.text(3.3, -3.38, 'Example: d = 2λ and θ = 30°; all arrows point toward the same distant P.',
              fontsize=10, ha='center', color=INK)
    axis.set(xlim=(-.7, 8.0), ylim=(-3.7, 1.1))
    return figure


def surface_phase_figure(wavelength=2.0, period=2.0, angle_deg=30.0, *, figure=None):
    """Incident plane-wave phase at z=0, t=0 over one grating period.

    Reuse notebook two's example: lambda=d=2 micrometres, theta_i=30 deg.
    This is one wave sampled at two positions, with no source summation.
    """
    if (not np.all(np.isfinite([wavelength, period, angle_deg]))
            or wavelength <= 0 or period <= 0 or not -90 < angle_deg < 90):
        raise ValueError('Use positive lengths and an incidence angle between -90 and 90 degrees.')
    kx = (2*np.pi/wavelength)*np.sin(np.deg2rad(angle_deg))
    x = np.linspace(0, period, 101)
    phase = kx*x
    if figure is None:
        figure = Figure(figsize=(11, 4.7), dpi=110, facecolor='white')
        FigureCanvasAgg(figure)
    else:
        figure.clear()
    geometry, axis = figure.subplots(1, 2)
    figure.subplots_adjust(left=.065, right=.97, bottom=.18, top=.82, wspace=.32)
    figure.suptitle('One incident plane wave: compare A and B at t = 0', fontsize=13)

    # Physical positions keep a fixed scale over the slider's period range.
    xmax = max(4.8, period+.8)
    geometry.axhline(0, color=GREY, lw=3)
    geometry.plot([0, 0], [0, 2.85], '--', color=GREY, lw=1)
    geometry.text(.15, 2.9, 'normal', color=GREY, fontsize=10)
    theta = np.deg2rad(angle_deg)
    tail = np.array([-2.3*np.sin(theta), 2.3*np.cos(theta)])
    geometry.annotate('', (0, 0), tail,
                      arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.2))
    geometry.annotate('incident k', tail, xytext=(0, 14),
                      textcoords='offset points', ha='center', color=BLUE, fontsize=10,
                      bbox=dict(facecolor='white', edgecolor='none', pad=1))
    if abs(angle_deg) > 1e-8:
        arc = np.linspace(0, theta, 40)
        geometry.plot(-.8*np.sin(arc), .8*np.cos(arc), color=BLUE, lw=1.2)
    geometry.set_title(f'Positions on the grating; θᵢ = {angle_deg:g}°', fontsize=11)
    geometry.text(xmax-.15, .15, 'surface z = 0', fontsize=10, ha='right', color=GREY)
    for position, label, color in ((0, 'A', BLUE), (period, 'B', ORANGE)):
        geometry.plot(position, 0, 'o', color=color, markersize=7, zorder=4)
        geometry.annotate(label, (position, 0), xytext=(0, -17),
                          textcoords='offset points', ha='center', color=color, fontsize=11)
    geometry.annotate('', (period, -.7), (0, -.7),
                      arrowprops=dict(arrowstyle='<->', color=INK, shrinkA=0, shrinkB=0))
    geometry.text(period/2, -.85, f'd = {period:g} µm', ha='center', va='top', fontsize=10)
    geometry.set(xlim=(-2.8, xmax), ylim=(-1.35, 3.65), aspect='equal',
                 xlabel='Position x (µm)', ylabel='Height z (µm)')
    geometry.spines[['top', 'right']].set_visible(False)

    axis.plot(x, phase, color=BLUE, linewidth=2)
    for position, value, label, color in ((0, 0, 'A', BLUE),
                                         (period, phase[-1], 'B', ORANGE)):
        axis.plot(position, value, 'o', color=color, markersize=6)
        axis.annotate(label, (position, value), xytext=(-10, 10),
                      textcoords='offset points', color=color, fontsize=11)
    axis.axhline(0, color=GREY, lw=.8)
    # Keep the introductory exercises on the same vertical scale. Expand
    # symmetrically only for phase differences beyond +/-2*pi.
    limit = max(2, np.ceil(abs(phase[-1])/np.pi))*np.pi
    if abs(phase[-1]) > 1e-8:
        axis.plot([period, period+.25], [phase[-1], phase[-1]], color=GREY, lw=.8)
        axis.annotate('', (period+.25, phase[-1]), (period+.25, 0),
                      arrowprops=dict(arrowstyle='<->', color=INK, lw=1.3,
                                      shrinkA=0, shrinkB=0))
        axis.text(period+.4, phase[-1]/2, f'Δφ = {phase[-1]/np.pi:.2f}π rad',
                  rotation=90, fontsize=10, va='center', color=INK)
    else:
        axis.text(period/2, .3*np.pi, 'Δφ = 0', fontsize=11, ha='center', color=INK)
    tick_step = max(1, np.ceil(limit/np.pi/3))
    count = int(np.floor(limit/np.pi/tick_step))
    ticks = np.arange(-count, count+1)*tick_step
    axis.set(xlim=(-.2, max(5.1, period+1.1)),
             ylim=(-1.15*limit, 1.15*limit),
             yticks=ticks*np.pi,
             yticklabels=['0' if value == 0 else f'{value:g}π' for value in ticks],
             xlabel='Position x along the grating (µm)', ylabel='Phase φ (radians)',
             title='Surface phase: φ(x) = kₓx')
    axis.title.set_fontsize(11)
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', color='#e2e5e9', lw=.7)
    axis.set_axisbelow(True)
    return figure


class SurfacePhaseExplorer:
    """Three local controls for the same-time difference phi(B)-phi(A)=kx*d."""

    def __init__(self):
        import ipywidgets as widgets

        self._closed = False
        self._resetting = False
        self.figure = None
        self._defaults = (2., 2., 30.)

        def slider(value, minimum, maximum, step, description):
            return widgets.FloatSlider(value=value, min=minimum, max=maximum,
                step=step, description=description, readout_format='.1f',
                continuous_update=False, style={'description_width': 'initial'},
                layout=widgets.Layout(width='310px'))

        self.wavelength = slider(2, 1, 4, .1, 'Wavelength λ (µm)')
        self.period = slider(2, 1, 4, .1, 'Period d (µm)')
        self.angle = slider(30, -60, 60, 5, 'Incidence θᵢ (°)')
        self._controls = (self.wavelength, self.period, self.angle)
        self.reset = widgets.Button(description='Reset example')
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1210px', height='auto'))
        row = widgets.Box([*self._controls, self.reset], layout=widgets.Layout(
            display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([row, self.readout, self.picture])
        for control in self._controls:
            control.observe(self._update, names='value')
        self.reset.on_click(self._reset)
        self._update()

    def _update(self, change=None):
        if self._closed or self._resetting:
            return
        wavelength, period, angle = (control.value for control in self._controls)
        kx = (2*np.pi/wavelength)*np.sin(np.deg2rad(angle))
        phase_step = kx*period
        self.data = dict(wavelength=wavelength, period=period, angle_deg=angle,
                         kx=kx, phase_a=0., phase_b=phase_step, phase_step=phase_step)
        self.figure = surface_phase_figure(wavelength, period, angle, figure=self.figure)
        self.picture.value = figure_png(self.figure)
        self.readout.value = (
            '<div style="padding:10px 0;line-height:1.7">'
            f'<b>k<sub>x</sub></b> = (2π / {wavelength:.1f}) sin({angle:.0f}°)'
            f' = <b>{kx:.3f} rad/µm</b><br>'
            f'<b>Δφ = k<sub>x</sub>d</b> = ({kx:.3f}) × {period:.1f}'
            f' = <b>{phase_step:.3f} rad = {phase_step/np.pi:.3f}π rad</b><br>'
            f'A: x = 0, φ = 0. &nbsp; B: x = {period:.1f} µm, '
            f'φ = {phase_step:.3f} rad. &nbsp; Both at z = 0, t = 0.</div>')

    def _reset(self, button=None):
        if self._closed:
            return
        self._resetting = True
        try:
            for control, value in zip(self._controls, self._defaults):
                control.value = value
        finally:
            self._resetting = False
        self._update()

    def close(self):
        """Disconnect and release this display; repeated calls are harmless."""
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.reset.on_click(self._reset, remove=True)
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        from .grating_intro import _close_widget_tree
        _close_widget_tree(self.ui)


def phase_match_data(incident_kx=np.pi/2, outgoing_kx=np.pi/2, period=2.):
    """Compare spatial phase advances, without testing propagation or power."""
    if (not np.all(np.isfinite([incident_kx, outgoing_kx, period])) or period <= 0):
        raise ValueError('Use finite wavevector components and a positive period.')
    incident_step, outgoing_step = incident_kx*period, outgoing_kx*period
    cycles = (outgoing_step-incident_step)/(2*np.pi)
    nearest = int(np.rint(cycles))
    matches = abs(cycles-nearest) < 1e-9
    return dict(incident_kx=incident_kx, outgoing_kx=outgoing_kx, period=period,
                incident_step=incident_step, outgoing_step=outgoing_step,
                difference=outgoing_step-incident_step, cycles=cycles,
                matches=matches, order=nearest if matches else None)


def phase_match_figure(incident_kx=np.pi/2, outgoing_kx=np.pi/2, period=2., *, figure=None):
    """Full phase advances and their positions within one cycle at x=d.

    Each curve independently subtracts its phase at x=0. Coincident dial
    arrows therefore compare translation phases, not absolute field phases.
    """
    data = phase_match_data(incident_kx, outgoing_kx, period)
    if figure is None:
        figure = Figure(figsize=(11, 4.8), dpi=110, facecolor='white')
        FigureCanvasAgg(figure)
    else:
        figure.clear()
    phase, dial = figure.subplots(1, 2, gridspec_kw={'width_ratios': [1.25, 1]})
    figure.subplots_adjust(left=.075, right=.96, bottom=.23, top=.80, wspace=.32)
    figure.suptitle('Do the phase changes agree up to whole cycles?', fontsize=13)
    x = np.linspace(0, period, 101)
    for kx, label, color, style, width in (
            (incident_kx, 'Incident', BLUE, '-', 3.5),
            (outgoing_kx, 'Outgoing', ORANGE, '--', 2.2)):
        phase.plot(x, kx*x, color=color, linestyle=style, lw=width, label=label)
        phase.plot(period, kx*period, 'o', color=color, markersize=5)
    phase.axhline(0, color=GREY, lw=.8)
    phase.axvline(period, color=GREY, lw=.8, linestyle=':')
    limit = max(6., np.ceil(max(abs(data['incident_step']),
                               abs(data['outgoing_step']))/np.pi))*np.pi
    phase.set(xlim=(-.06*period, 1.35*period), ylim=(-1.08*limit, 1.08*limit),
              xticks=[0, period/2, period], xticklabels=['A: 0', f'{period/2:g}', f'B: d = {period:g}'],
              xlabel='Position x along the surface (µm)',
              ylabel='Phase change from A (radians)',
              title='Each curve starts at zero at A')
    ticks = np.arange(-int(limit/np.pi)//2*2, int(limit/np.pi)+1, 2)
    phase.set_yticks(ticks*np.pi, ['0' if t == 0 else f'{t:g}π' for t in ticks])
    phase.title.set_fontsize(11)
    phase.legend(loc='upper left', fontsize=10)
    phase.grid(axis='y', alpha=.2)
    phase.spines[['top', 'right']].set_visible(False)
    if abs(data['difference']) > 1e-9:
        phase.annotate('', (period*1.1, data['outgoing_step']),
                       (period*1.1, data['incident_step']),
                       arrowprops=dict(arrowstyle='<->', color=INK, shrinkA=0, shrinkB=0))

    circle = np.linspace(0, 2*np.pi, 241)
    dial.plot(np.cos(circle), np.sin(circle), color=GREY, lw=1.1)
    for angle, label in ((0, '0 / 2π'), (np.pi/2, 'π/2'),
                         (np.pi, 'π'), (3*np.pi/2, '3π/2')):
        dial.text(1.24*np.cos(angle), 1.24*np.sin(angle), label,
                  ha='center', va='center', fontsize=11, color=INK)
    for advance, color, style, width in (
            (data['incident_step'], BLUE, '-', 5),
            (data['outgoing_step'], ORANGE, '--', 2.2)):
        endpoint = np.array([np.cos(advance), np.sin(advance)])
        dial.annotate('', endpoint, (0, 0),
                      arrowprops=dict(arrowstyle='->', color=color, lw=width,
                                      linestyle=style, shrinkA=0, shrinkB=0))
    dial.set(xlim=(-1.6, 1.7), ylim=(-1.5, 1.5), aspect='equal',
             title='Compare the phase changes at B on one cycle')
    dial.title.set_fontsize(11)
    dial.axis('off')
    figure.text(.31, .08,
                f'Δφ(out) − Δφ(i) = {data["difference"]/np.pi:+.3f}π rad'
                f' = {data["cycles"]:+.3f} full cycles', ha='center', fontsize=11)
    status = (f'Same cycle position: order m = {data["order"]:+d}' if data['matches']
              else 'Different cycle positions: no phase match')
    figure.text(.77, .08, status, ha='center', fontsize=11,
                color='#23734d' if data['matches'] else ORANGE)
    return figure


class PhaseMatchExplorer:
    """Compare incident/outgoing phase advances at fixed d=2 micrometres."""

    def __init__(self):
        import ipywidgets as widgets
        from functools import partial

        self._closed = False
        self.figure = None
        self.period = 2.
        self.incident = widgets.Dropdown(
            options=[('Point 3 example: Δφᵢ = π', np.pi/2), ('Normal incidence: Δφᵢ = 0', 0.)],
            value=np.pi/2, description='Incident',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='320px'))
        self.outgoing = widgets.FloatSlider(value=np.pi/2, min=-8, max=8, step=.05,
            description='Outgoing kₓ (rad/µm)', readout_format='.3f', continuous_update=False,
            style={'description_width': 'initial'}, layout=widgets.Layout(width='440px'))
        self._controls = (self.incident, self.outgoing)
        self.order_buttons = {}
        self._button_callbacks = []
        for order in (-1, 0, 1):
            button = widgets.Button(description=f'Set m = {order:+d}',
                                    layout=widgets.Layout(width='125px'))
            callback = partial(self._set_order, order)
            button.on_click(callback)
            self.order_buttons[order] = button
            self._button_callbacks.append((button, callback))
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1210px', height='auto'))
        controls = widgets.Box(list(self._controls), layout=widgets.Layout(
            display='flex', flex_flow='row wrap', align_items='center'))
        buttons = widgets.HBox(list(self.order_buttons.values()))
        self.ui = widgets.VBox([controls, buttons, self.readout, self.picture])
        for control in self._controls:
            control.observe(self._update, names='value')
        self._update()

    def _set_order(self, order, button=None):
        if not self._closed:
            self.outgoing.value = self.incident.value+order*2*np.pi/self.period

    def _update(self, change=None):
        if self._closed:
            return
        self.data = phase_match_data(self.incident.value, self.outgoing.value, self.period)
        self.figure = phase_match_figure(self.incident.value, self.outgoing.value,
                                         self.period, figure=self.figure)
        self.picture.value = figure_png(self.figure)
        data = self.data
        status = (f'Whole-cycle agreement: m = {data["order"]:+d}.' if data['matches']
                  else 'The difference is not a whole number of cycles: no phase match.')
        self.readout.value = (
            '<div style="padding:8px 0;line-height:1.7">'
            f'd = 2 µm; k<sub>ix</sub> = {data["incident_kx"]:.3f} rad/µm; '
            f'k<sub>out,x</sub> = {data["outgoing_kx"]:.3f} rad/µm.<br>'
            f'Δφ<sub>i</sub> = {data["incident_step"]/np.pi:.3f}π rad; '
            f'Δφ<sub>out</sub> = {data["outgoing_step"]/np.pi:.3f}π rad.<br>'
            f'(Δφ<sub>out</sub> − Δφ<sub>i</sub>) / 2π = '
            f'<b>{data["cycles"]:.3f}</b>. <b>{status}</b></div>')

    def close(self):
        """Release only this explorer, including callbacks and image buffers."""
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        for button, callback in self._button_callbacks:
            button.on_click(callback, remove=True)
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        from .grating_intro import _close_widget_tree
        _close_widget_tree(self.ui)


def lesson_svg(scene):
    """Render one lesson figure in memory and release the figure afterward."""
    builders = dict(illumination=illumination_figure, equal_paths=equal_paths_figure,
                    extra_path=extra_path_figure, repeated_paths=repeated_paths_figure,
                    wavevector_connection=wavevector_connection_figure,
                    surface_phase=surface_phase_figure)
    if scene not in builders:
        raise ValueError(f'Unknown lesson scene: {scene}')
    figure = builders[scene]()
    try:
        output = io.StringIO()
        figure.savefig(output, format='svg', facecolor='white')
        return output.getvalue()
    finally:
        figure.clear()
