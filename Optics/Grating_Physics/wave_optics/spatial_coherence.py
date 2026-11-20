"""Spatial coherence sampled by two openings, with propagation along -z.

A centred, uniform, spatially incoherent line source illuminates Q1=(-d/2,0,0)
and Q2=(d/2,0,0). Source and detector distances are positive R and L, with
planes z=R and z=-L. Narrowband scalar light, paraxial propagation and equal
local arm intensities isolate the spatial-coherence contribution to contrast.
"""

import numpy as np


def uniform_line_coherence(separation_mm, source_width_mm, *,
                           wavelength_nm=632.8, source_distance_mm=500.):
    """Normalized <U(Q1)* U(Q2)> for a centred uniform incoherent line.

    np.sinc(q) means sin(pi*q)/(pi*q), including its continuous limit at q=0.
    The result is real but can be negative: its sign carries a pi phase shift.
    Width zero denotes a point-source limit at fixed normalized brightness.
    """
    d = np.asarray(separation_mm, dtype=float)
    parameters = np.asarray([source_width_mm, wavelength_nm, source_distance_mm], dtype=float)
    if (not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(d))
            or np.any(d < 0) or source_width_mm < 0
            or wavelength_nm <= 0 or source_distance_mm <= 0):
        raise ValueError('Use nonnegative separation/width and finite positive wavelength/distances.')
    return np.sinc(d*source_width_mm/(wavelength_nm*1e-6*source_distance_mm))


def young_spatial_data(separation_mm=.5, source_width_mm=0., *,
                       wavelength_nm=632.8, source_distance_mm=500.,
                       detector_distance_mm=1000.):
    """Detector intensity divided by the sum of the two arm intensities.

    The displayed source-point curves are illustrative samples. The recorded
    intensity uses the analytic continuous-source integral, not their finite sum.
    Finite-opening diffraction envelopes and spectral averaging are omitted.
    """
    if (not np.isfinite(separation_mm) or separation_mm <= 0
            or not np.isfinite(detector_distance_mm) or detector_distance_mm <= 0):
        raise ValueError('Use finite positive opening separation and detector distance.')
    gamma = float(uniform_line_coherence(separation_mm, source_width_mm,
        wavelength_nm=wavelength_nm, source_distance_mm=source_distance_mm))
    wavelength_mm = wavelength_nm*1e-6
    x = np.linspace(-4., 4., 641)
    y = np.linspace(-2., 2., 65)
    propagation_phase = -2*np.pi*separation_mm*x/(wavelength_mm*detector_distance_mm)
    profile = 1 + gamma*np.cos(propagation_phase)
    positions = np.linspace(-source_width_mm/2, source_width_mm/2, 5)
    illumination_phase = -2*np.pi*separation_mm*positions/(wavelength_mm*source_distance_mm)
    components = 1 + np.cos(propagation_phase[None, :] + illumination_phase[:, None])
    baselines = np.linspace(0., max(1., separation_mm), 401)
    curve = uniform_line_coherence(baselines, source_width_mm,
        wavelength_nm=wavelength_nm, source_distance_mm=source_distance_mm)
    return dict(
        separation_mm=float(separation_mm), source_width_mm=float(source_width_mm),
        wavelength_nm=float(wavelength_nm), source_distance_mm=float(source_distance_mm),
        detector_distance_mm=float(detector_distance_mm),
        coherence=gamma, visibility=abs(gamma),
        coherence_phase_deg=(None if abs(gamma) < 1e-12 else (0. if gamma > 0 else 180.)),
        fringe_spacing_mm=wavelength_mm*detector_distance_mm/separation_mm,
        first_zero_width_mm=wavelength_mm*source_distance_mm/separation_mm,
        first_zero_separation_mm=(wavelength_mm*source_distance_mm/source_width_mm
                                  if source_width_mm > 0 else None),
        x_mm=x, y_mm=y, propagation_phase=propagation_phase, profile=profile,
        image=np.broadcast_to(profile, (len(y), len(x))),
        source_positions_mm=positions, source_profiles=components,
        curve_separation_mm=baselines, curve_visibility=np.abs(curve))


class SpatialCoherenceExplorer:
    """Two aperture/source controls with retained Matplotlib artists."""

    def __init__(self, *, wavelength_nm=632.8, source_distance_mm=500.,
                 detector_distance_mm=1000.):
        import ipywidgets as widgets
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        self.parameters = dict(wavelength_nm=wavelength_nm,
            source_distance_mm=source_distance_mm, detector_distance_mm=detector_distance_mm)
        young_spatial_data(**self.parameters)  # Validate before allocating widgets.
        self._closed = False
        self.separation = widgets.FloatSlider(
            value=.5, min=.1, max=1., step=.025, readout_format='.3f',
            description='Opening separation d (mm)', continuous_update=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='650px', max_width='100%'))
        self.source_width = widgets.FloatSlider(
            value=0., min=0., max=2., step=.01, readout_format='.3f',
            description='Incoherent source width a (mm)', continuous_update=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='650px', max_width='100%'))
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1300px', height='auto'))
        self.readout = widgets.HTML()
        specifications = widgets.HTML(
            f'λ = {wavelength_nm:g} nm &nbsp; · &nbsp; source distance R = {source_distance_mm:g} mm'
            f' &nbsp; · &nbsp; detector distance L = {detector_distance_mm:g} mm.<br>'
            'Equal arm intensities; source brightness is normalized as its width changes.')
        self.ui = widgets.VBox([
            self.separation, self.source_width, specifications, self.picture, self.readout])
        self.figure = Figure(figsize=(12.8, 8.2), dpi=105, facecolor='white')
        FigureCanvasAgg(self.figure)
        grid = self.figure.add_gridspec(2, 2, left=.065, right=.95, bottom=.12,
                                      top=.94, wspace=.35, hspace=.42)
        self.setup_axis = self.figure.add_subplot(grid[0, 0])
        self.coherence_axis = self.figure.add_subplot(grid[0, 1])
        self.detector_axis = self.figure.add_subplot(grid[1, 0])
        self.profile_axis = self.figure.add_subplot(grid[1, 1])
        self._make_setup()
        ink, purple = '#243444', '#8054a1'
        ax = self.coherence_axis
        ax.set(xlim=(0, 1), ylim=(-.035, 1.07),
               xlabel='Opening separation d (mm)', ylabel='Visibility',
               title='Spatial coherence between the openings')
        ax.grid(alpha=.2)
        self.coherence_curve, = ax.plot([], [], color=purple, lw=2)
        self.coherence_point, = ax.plot([], [], 'o', color='#df792a', ms=7)
        self.coherence_guide = ax.axvline(.5, color='#df792a', ls=':', lw=1)
        self.first_zero_guide = ax.axvline(1, color='#7d8994', ls='--', lw=1, visible=False)
        self.first_zero_label = ax.text(0, .15, 'First zero', color='#667580',
                                       fontsize=9, ha='center', visible=False)
        self.detector_image = self.detector_axis.imshow(
            np.ones((65, 641)), origin='lower', cmap='gray', vmin=0, vmax=2,
            extent=(-4, 4, -2, 2), aspect='equal', interpolation='nearest')
        self.detector_axis.set(xlabel='x (mm)', ylabel='y (mm)',
                               title='Detector: averaged intensity')
        colourbar = self.figure.colorbar(self.detector_image, ax=self.detector_axis,
                                        fraction=.035, pad=.03)
        colourbar.set_label(r'$I/(I_1+I_2)$')
        ax = self.profile_axis
        ax.set(xlim=(-4, 4), ylim=(-.05, 2.12), xlabel='Detector x (mm)',
               ylabel=r'$I/(I_1+I_2)$', title='Shifted patterns and their average')
        ax.grid(alpha=.16)
        self.source_curves = [
            ax.plot([], [], color='#df792a', alpha=.25, lw=1,
                    label='Sample source points' if i == 0 else None)[0]
            for i in range(5)]
        self.average_curve, = ax.plot([], [], color=ink, lw=2.3,
                                      label='Continuous-source average')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17),
                  ncol=1, frameon=False, fontsize=9)
        for control in (self.separation, self.source_width):
            control.observe(self._update, names='value')
        self._update()

    def _make_setup(self):
        ax = self.setup_axis
        ink, blue, orange, teal = '#243444', '#287bb5', '#df792a', '#17806b'
        ax.set(xlim=(-2.15, 2.75), ylim=(-2.9, 3.15),
               title='Sample the illuminating field at Q₁ and Q₂')
        ax.set_axis_off()
        self.source_line, = ax.plot([], [], color=teal, lw=6, solid_capstyle='round')
        self.source_points, = ax.plot([], [], 'o', ms=4, color=teal)
        self.source_span = ax.annotate('', xy=(0, 2.7), xytext=(0, 2.7),
            arrowprops=dict(arrowstyle='<->', color=teal, lw=1))
        self.source_label = ax.text(0, 2.79, '', ha='center', fontsize=10, color=teal)
        self.incident_rays = [ax.plot([], [], color=teal, alpha=.22, lw=1)[0] for _ in range(6)]
        self.mask = [ax.plot([], [], color=ink, lw=4)[0] for _ in range(3)]
        self.samples = [ax.plot([], [], 'o', color=colour, ms=5)[0]
                        for colour in (blue, orange)]
        self.q1_text = ax.text(0, .15, r'$Q_1$', ha='right', fontsize=11, color=blue)
        self.q2_text = ax.text(0, .15, r'$Q_2$', ha='left', fontsize=11, color=orange)
        self.separation_span = ax.annotate('', xy=(.25, -.48), xytext=(-.25, -.48),
            arrowprops=dict(arrowstyle='<->', color=ink, lw=1))
        ax.text(0, -.62, '$d$', ha='center', va='top', fontsize=11, color=ink)
        self.outgoing_rays = [ax.plot([], [], color=colour, lw=1.5)[0] for colour in (blue, orange)]
        ax.plot([-1.35, 1.35], [-2.5, -2.5], color='#8d9daa', lw=5)
        ax.plot(0, -2.5, 'o', color=teal, ms=4)
        ax.text(.08, -2.72, '$P$', color=teal)
        for z, label in ((2.3, '$z=R$'), (0., '$z=0$'), (-2.5, '$z=-L$')):
            ax.text(1.50, z, label, va='center', fontsize=10, color=ink)
        for start, end, label in ((0., 2.3, '$R$'), (-2.5, 0., '$L$')):
            ax.annotate('', xy=(-1.7, end), xytext=(-1.7, start),
                arrowprops=dict(arrowstyle='<->', color='#7d8994', lw=1))
            ax.text(-1.82, (start+end)/2, label, ha='right', va='center',
                    color='#667580', fontsize=11)
        ax.annotate('$-z$', xy=(2.35, -1.65), xytext=(2.35, -.65), ha='center',
                    arrowprops=dict(arrowstyle='->', color='#667580'), color='#667580')
        ax.text(.5, -.04, 'Sketch: longitudinal distances compressed.',
                transform=ax.transAxes, ha='center', fontsize=9, color='#667580')

    def _update(self, change=None):
        if self._closed:
            return
        from .plotting import figure_png

        data = young_spatial_data(self.separation.value, self.source_width.value, **self.parameters)
        self.data = data
        d, a = data['separation_mm'], data['source_width_mm']
        q1, q2 = -d/2, d/2
        self.source_line.set_data([-a/2, a/2], [2.3, 2.3])
        self.source_points.set_data(data['source_positions_mm'], np.full(5, 2.3))
        self.source_span.xy = (a/2, 2.7)
        self.source_span.set_position((-a/2, 2.7))
        self.source_span.set_visible(a > 0)
        self.source_label.set_text('Incoherent source: $a$' if a > 0 else 'Point-source limit')
        for line, (xi, q) in zip(self.incident_rays,
                [(xi, q) for xi in (-a/2, 0., a/2) for q in (q1, q2)], strict=True):
            line.set_data([xi, q], [2.3, 0.])
        gap = .035
        for line, ends in zip(self.mask, ((-1.35, q1-gap), (q1+gap, q2-gap),
                                         (q2+gap, 1.35)), strict=True):
            line.set_data(ends, [0., 0.])
        for sample, q in zip(self.samples, (q1, q2), strict=True):
            sample.set_data([q], [0.])
        self.q1_text.set_x(q1-.035)
        self.q2_text.set_x(q2+.035)
        self.separation_span.xy = (q2, -.48)
        self.separation_span.set_position((q1, -.48))
        for line, q in zip(self.outgoing_rays, (q1, q2), strict=True):
            line.set_data([q, 0.], [0., -2.5])
        self.coherence_curve.set_data(data['curve_separation_mm'], data['curve_visibility'])
        self.coherence_point.set_data([d], [data['visibility']])
        self.coherence_guide.set_xdata([d, d])
        zero = data['first_zero_separation_mm']
        show_zero = zero is not None and 0 < zero <= 1.
        self.first_zero_guide.set_visible(show_zero)
        self.first_zero_label.set_visible(show_zero)
        if show_zero:
            self.first_zero_guide.set_xdata([zero, zero])
            self.first_zero_label.set_x(zero)
        self.detector_image.set_data(data['image'])
        self.average_curve.set_data(data['x_mm'], data['profile'])
        for index, (line, profile) in enumerate(zip(self.source_curves, data['source_profiles'], strict=True)):
            line.set_data(data['x_mm'], profile)
            line.set_visible(a > 0 or index == 0)
        phase_note = ('The correlation changes sign: the averaged fringes shift by half a period.'
                      if data['coherence'] < -1e-12 else
                      'The averaged interference term vanishes.' if abs(data['coherence']) < 1e-12 else '')
        self.readout.value = (
            f'<b>Visibility = |γ<sub>Q₁Q₂</sub>| = {data["visibility"]:.3f}</b>'
            f' &nbsp; · &nbsp; γ<sub>Q₁Q₂</sub> = {data["coherence"]:+.3f}'
            f' &nbsp; · &nbsp; fringe spacing = {data["fringe_spacing_mm"]:.3f} mm.<br>'
            f'At this separation, the first visibility zero is at source width '
            f'a = {data["first_zero_width_mm"]:.3f} mm. {phase_note}<br>'
            'Orange curves illustrate a few source points, each normalized to its own mean; '
            'the dark curve averages the entire uniform source.')
        self.picture.value = figure_png(self.figure)

    def close(self):
        """Detach observers and release the display; safe to call twice."""
        if self._closed:
            return
        from .grating_intro import _close_widget_tree

        self._closed = True
        for control in (self.separation, self.source_width):
            control.unobserve(self._update, names='value')
        self.picture.value = b''
        self.data = None
        self.figure.clear()
        _close_widget_tree(self.ui)
