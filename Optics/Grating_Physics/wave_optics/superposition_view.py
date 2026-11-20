"""Progressive, independent two-wave views for notebook seven."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .grating_intro import _close_widget_tree, _uniform_field_image
from .plotting import figure_png
from .superposition import (
    grating_pair_wavevectors, ideal_camera_profile, symmetric_wavevectors,
    two_wave_cycle, two_wave_field,
)
from .waves import wavenumber
from ._view_cache import ViewCache


BLUE, ORANGE, INK, GREY = '#2866a3', '#b85c17', '#243444', '#727c86'
STAGES = ('addition', 'phase', 'phasor', 'intensity', 'wavefronts',
          'local_phase', 'spacing', 'directions', 'orders')


def _style(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(alpha=.14)
    axis.tick_params(labelsize=10)


def _arrow(axis, start, end, color, *, dashed=False):
    if abs(end-start) < 1e-12:
        axis.plot(end.real, end.imag, 'o', color=color, ms=5)
    else:
        axis.annotate('', xy=(end.real, end.imag), xytext=(start.real, start.imag),
            arrowprops=dict(arrowstyle='->', color=color, lw=2.2,
                            linestyle='--' if dashed else '-'))


def _clip_front_z(vertices, boundary, keep_above):
    """Clip a drawn wavefront patch at the observation window's z limits."""
    result = []
    if not len(vertices):
        return np.empty((0, 3))
    previous = vertices[-1]
    previous_inside = previous[2] >= boundary if keep_above else previous[2] <= boundary
    for current in vertices:
        inside = current[2] >= boundary if keep_above else current[2] <= boundary
        if inside != previous_inside:
            fraction = (boundary-previous[2])/(current[2]-previous[2])
            result.append(previous+fraction*(current-previous))
        if inside:
            result.append(current)
        previous, previous_inside = current, inside
    return np.asarray(result).reshape(-1, 3)


def _wavefront_patches(vector, phase_deg, tau):
    """Finite drawing patches of k.r + phi - omega*t = 2*pi*n planes.

    The patch boundaries only limit the illustration, never the optical field.
    All fronts are clipped to the incoming side of the detector at z=-1 um.
    """
    k = np.linalg.norm(vector)
    normal = vector/k
    transverse = np.array([-normal[2], 0, normal[0]])
    across_y = np.array([0, 1.25, 0])
    target = np.array([0., 0., -1.])
    patches = []
    for order in range(-10, 11):
        distance = (2*np.pi*(order+tau)-np.deg2rad(phase_deg)-vector@target)/k
        centre = target+distance*normal
        polygon = np.array([centre+sx*1.25*transverse+sy*across_y
                            for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
        polygon = _clip_front_z(polygon, -1, True)
        polygon = _clip_front_z(polygon, 5, False)
        if len(polygon) >= 3:
            patches.append(polygon)
    return patches


class TwoWaveExplorer:
    """Independent amplitude and phase controls for both waves in each section."""

    def __init__(self, stage='addition'):
        import ipywidgets as widgets

        if stage not in STAGES:
            raise ValueError(f'Stage must be one of {STAGES}.')
        self.stage, self._closed, self._resetting = stage, False, False
        self._cache = ViewCache()
        self._time_panel_artists = self._order_time_artists = None
        self._local_time_artists = None
        self.time = self.angle = self.detector_x = None
        self.detector_z = self.order_view = None
        controls = []

        def slider(description, value, minimum, maximum, step, fmt):
            control = widgets.FloatSlider(description=description, value=value,
                min=minimum, max=maximum, step=step, readout_format=fmt,
                continuous_update=False, style={'description_width': 'initial'},
                layout=widgets.Layout(width='340px', max_width='100%'))
            controls.append(control)
            return control

        self.amplitude1 = slider('Amplitude A₁', 1, 0, 1, .1, '.1f')
        self.phase1 = slider('Phase φ₁ (°)', 0, 0, 360, 15, '.0f')
        self.amplitude2 = slider('Amplitude A₂', 1, 0, 1, .1, '.1f')
        self.phase2 = slider('Phase φ₂ (°)', 90 if stage == 'phasor' else 0,
                             0, 360, 15, '.0f')
        wave_groups = []
        for index, amplitude, phase, color in (
                (1, self.amplitude1, self.phase1, BLUE),
                (2, self.amplitude2, self.phase2, ORANGE)):
            order = (' · order m = −1' if index == 1 else ' · order m = +1') if stage == 'orders' else ''
            wave_groups.append(widgets.VBox([
                widgets.HTML(f'<b style="color:{color}">Wave {index}{order}</b>'),
                amplitude, phase], layout=widgets.Layout(width='350px', max_width='100%')))
        if stage in ('addition', 'phase', 'wavefronts', 'local_phase', 'directions', 'orders'):
            self.time = slider('Time t/T', 0, 0, 1, .025, '.3f')
        if stage in ('spacing', 'directions'):
            self.angle = slider('Half-angle θ from −z (°)', 30, 5, 45, 1, '.0f')
        if stage in ('wavefronts', 'local_phase', 'directions'):
            self.time.description = 'Model time t/T'
        if stage == 'local_phase':
            self.detector_x = slider('Detector x (µm)', 0, -2, 2, .25, '.2f')
        if stage == 'orders':
            self.time.description = 'Model time t/T'
            self.detector_z = slider('Detector z (µm)', -1, -2.75, -.5, .25, '.2f')
            self.order_view = widgets.ToggleButtons(
                options=[('Experiment and camera', 'experiment'), ('Compare fields', 'fields')],
                value='experiment', style={'button_width': '190px'},
                layout=widgets.Layout(width='100%'))
            controls.append(self.order_view)
        self._controls = tuple(controls)
        self._defaults = tuple((control, control.value) for control in controls)
        self.reset_button = widgets.Button(description='Reset this example',
                                            layout=widgets.Layout(width='180px'))
        self.reset_button.on_click(self._reset)
        self.parameter_readout = widgets.HTML()
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1100px', height='auto'))
        self.ui = widgets.VBox([widgets.Box(wave_groups, layout=widgets.Layout(
            display='flex', flex_flow='row wrap', align_items='center')),
            widgets.Box([*controls[4:], self.reset_button], layout=widgets.Layout(
            display='flex', flex_flow='row wrap', align_items='center')),
            self.parameter_readout, self.readout, self.picture])
        height = (8.9 if stage == 'directions' else 8.3 if stage == 'orders'
                  else 8.7 if stage == 'local_phase'
                  else 5.4 if stage in ('wavefronts', 'spacing')
                  else 5.7 if stage in ('addition', 'phase') else 4.6)
        self.figure = Figure(figsize=(10, height), dpi=110, facecolor='white')
        FigureCanvasAgg(self.figure)
        for control in controls:
            control.observe(self._update, names='value')
        self._update()

    def _update(self, change=None):
        if self._closed or self._resetting:
            return
        a1, a2 = self.amplitude1.value, self.amplitude2.value
        phi1, phi2 = self.phase1.value, self.phase2.value
        delta = phi2-phi1
        self.phase_difference_deg = delta
        tau = self.time.value if self.time is not None else 0.0
        self.data = self._cache.get('cycle', (a1, a2, delta, phi1),
            lambda: two_wave_cycle(a1, a2, delta, phase1_deg=phi1))
        if change is not None and change.get('owner') is self.time:
            if self._local_time_artists is not None:
                self._refresh_local_phase(tau)
                self.picture.value = figure_png(self.figure)
                return
            if self._time_panel_artists is not None:
                self._refresh_time_panels(tau)
                self.picture.value = figure_png(self.figure)
                return
            if self._order_time_artists is not None:
                self._order_data(a1, a2, delta, tau)
                for artist, field_index in self._order_time_artists:
                    values = (self.spatial_data['field_sum'] if field_index is None
                              else self.spatial_data['fields'][field_index])
                    artist.set_data(values)
                self.picture.value = figure_png(self.figure)
                return
        self._time_panel_artists = self._order_time_artists = None
        self._local_time_artists = None
        self.figure.clear()
        self.figure.subplots_adjust(left=.09, right=.96, bottom=.19, top=.87,
                                    wspace=.40, hspace=.60)
        getattr(self, f'_draw_{self.stage}')(a1, a2, delta, tau)
        where = ('at the spatial origin' if self.stage in
                 ('wavefronts', 'local_phase', 'spacing', 'directions', 'orders')
                 else 'at the fixed point')
        absent = (' Both waves are absent; phase settings have no effect.' if a1 == a2 == 0 else
                  ' Wave 1 is absent; its phase setting has no effect.' if a1 == 0 else
                  ' Wave 2 is absent; its phase setting has no effect.' if a2 == 0 else '')
        self.parameter_readout.value = (
            f'Phase offsets {where}: φ₁ = {phi1:.0f}°, φ₂ = {phi2:.0f}°. '
            f'<b>Calculated difference δ = φ₂ − φ₁ = {delta:+.0f}°</b> '
            f'(equivalent to {delta % 360:.0f}° modulo 360°).{absent}')
        self.picture.value = figure_png(self.figure)

    @staticmethod
    def _at_time(base, tau):
        """Keep invariant complex fields and intensities; select the real instant."""
        factor = np.exp(-2j*np.pi*tau)
        return dict(base, fields=np.real(base['phasors']*factor),
                    field_sum=np.real(base['total_phasor']*factor))

    def _spatial_field(self, name, geometry_key, points, vectors, a1, a2, delta, tau=0.):
        key = (*geometry_key, a1, a2, delta, self.phase1.value)
        base = self._cache.get(name, key,
            lambda: two_wave_field(points, vectors, a1, a2, delta, phase1_deg=self.phase1.value))
        return self._at_time(base, tau)

    def _order_grid(self):
        def grid():
            x, z = np.linspace(-3, 3, 241), np.linspace(-3, 0, 121)
            xx, zz = np.meshgrid(x, z)
            return x, z, np.stack((xx, np.zeros_like(xx), zz), axis=-1), grating_pair_wavevectors()

        return self._cache.get('order_grid', (), grid)

    def _order_data(self, a1, a2, delta, tau):
        x, z, points, vectors = self._order_grid()
        key = (self.detector_z.value, a1, a2, delta, self.phase1.value)
        camera = self._cache.get('camera_profile', key,
            lambda: ideal_camera_profile(x, self.detector_z.value, a2, delta,
                                         amplitude1=a1, phase1_deg=self.phase1.value))
        self.camera_data = self._at_time(camera, tau)
        self.spatial_data = self._spatial_field('order_field', (), points, vectors, a1, a2, delta, tau)
        return x, z, vectors

    def _reset(self, button=None):
        if self._closed:
            return
        self._resetting = True
        try:
            for control, value in self._defaults:
                control.value = value
        finally:
            self._resetting = False
        self._update()

    def _draw_addition(self, a1, a2, delta, tau):
        self._time_panels(tau)
        self.figure.suptitle('Add readings at the same position and the same time', fontsize=14)

    def _draw_phase(self, a1, a2, delta, tau):
        self._time_panels(tau)
        self.figure.suptitle('Set each wave’s phase and compare their sum', fontsize=14)

    def _time_panels(self, tau):
        first, total = self.figure.subplots(2, 1)
        tt = np.append(self.data['tau'], 1.0)
        fields = np.column_stack((self.data['fields'], self.data['fields'][:, 0]))
        snapshot = np.real(self.data['phasors']*np.exp(-2j*np.pi*tau))
        cursors = []
        for axis in (first, total):
            axis.set(xlim=(0, 1), ylim=(-2.25, 2.25), xticks=[0, .25, .5, .75, 1],
                     xlabel='Time t/T at the fixed point', ylabel='Signed field (field units)')
            axis.axhline(0, color=GREY, lw=.8)
            cursors.append(axis.axvline(tau, color=GREY, ls=':', lw=1))
            _style(axis)
        first.plot(tt, fields[0], color=BLUE, lw=2.5, label='Wave 1')
        first.plot(tt, fields[1], color=ORANGE, ls='--', lw=1.8, label='Wave 2')
        first_marker, = first.plot(tau, snapshot[0], 'o', color=BLUE, ms=7)
        second_marker, = first.plot(tau, snapshot[1], 's', color=ORANGE, ms=5)
        first.legend(loc='upper center', bbox_to_anchor=(.5, 1.32), ncol=2, frameon=False)
        total.plot(tt, fields.sum(axis=0), color=INK, lw=2.3)
        sum_marker, = total.plot(tau, snapshot.sum(), 'o', color=INK, ms=7)
        total.set_title('Result: u = u₁ + u₂', fontsize=12)
        self.figure.subplots_adjust(top=.79, bottom=.11, hspace=.73)
        self._time_panel_artists = (cursors, (first_marker, second_marker, sum_marker))
        self._refresh_time_panels(tau)

    def _refresh_time_panels(self, tau):
        snapshot = np.real(self.data['phasors']*np.exp(-2j*np.pi*tau))
        cursors, markers = self._time_panel_artists
        for cursor in cursors:
            cursor.set_xdata([tau, tau])
        for marker, value in zip(markers, (*snapshot, snapshot.sum())):
            marker.set_data([tau], [value])
        self.readout.value = (f'At t/T = {tau:.3f}: <b>u₁ = {snapshot[0]:+.3f}</b>, '
            f'<b>u₂ = {snapshot[1]:+.3f}</b>; sum = <b>{snapshot.sum():+.3f}</b>. '
            f'The sum’s amplitude over a cycle is <b>{self.data["amplitude"]:.3f}</b>.')

    def _draw_phasor(self, a1, a2, delta, tau):
        arrows, cycle = self.figure.subplots(1, 2)
        u1, u2 = self.data['phasors']
        total = self.data['total_phasor']
        arrows.axhline(0, color=GREY, lw=.8)
        arrows.axvline(0, color=GREY, lw=.8)
        _arrow(arrows, 0j, total, INK)
        _arrow(arrows, 0j, u1, BLUE)
        _arrow(arrows, u1, total, ORANGE, dashed=True)
        for value, color in ((u1, BLUE), (total, INK)):
            arrows.plot(value.real, value.imag, 'o', color=color, ms=5)
        arrows.set(xlim=(-2.35, 2.35), ylim=(-2.35, 2.35), aspect='equal',
            xlabel='Real part (field units)', ylabel='Imaginary part (field units)',
            title='Place U₂ at the tip of U₁')
        for color, text, style in ((BLUE, 'U₁', '-'), (ORANGE, 'U₂ translated', '--'),
                                    (INK, 'Sum U', '-')):
            arrows.plot([], [], color=color, ls=style, label=text)
        arrows.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=3,
                      fontsize=8, frameon=False)
        _style(arrows)
        tt = np.append(self.data['tau'], 1)
        fields = np.column_stack((self.data['fields'], self.data['fields'][:, 0]))
        cycle.plot(tt, fields[0], color=BLUE, lw=2.5, label='u₁')
        cycle.plot(tt, fields[1], color=ORANGE, ls='--', lw=1.8, label='u₂')
        cycle.plot(tt,
                   np.append(self.data['field_sum'], self.data['field_sum'][0]),
                   color=INK, lw=2.3, label='u = u₁ + u₂')
        cycle.axhline(0, color=GREY, lw=.8)
        cycle.set(xlim=(0, 1), ylim=(-2.25, 2.25), xticks=[0, .25, .5, .75, 1],
            xlabel='Time t/T', ylabel='Signed field (field units)',
            title='The two waves and their sum')
        cycle.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=3,
                     fontsize=8, frameon=False)
        _style(cycle)
        self.figure.subplots_adjust(bottom=.24, wspace=.40)
        phase_text = ('undefined (zero resultant)' if self.data['phase_deg'] is None else
                      f'{self.data["phase_deg"]:+.1f}°')
        self.readout.value = (f'U₁ = {u1.real:+.3f} {u1.imag:+.3f}i; '
            f'U₂ = {u2.real:+.3f} {u2.imag:+.3f}i. '
            f'<b>U = {total.real:+.3f} {total.imag:+.3f}i</b>; '
            f'<b>|U| = {abs(total):.3f}</b>; phase of sum: {phase_text}.')

    def _draw_intensity(self, a1, a2, delta, tau):
        axis = self.figure.add_subplot(111)
        phases = np.linspace(0, 360, 361)
        curve = np.abs(a1+a2*np.exp(1j*np.deg2rad(phases)))**2
        axis.plot(phases, curve, color=INK, lw=2.3, label='Intensity of the combined field')
        axis.axhline(a1*a1+a2*a2, color=GREY, ls='--', lw=1.4, label='I₁ + I₂ (cross term omitted)')
        axis.plot(delta % 360, self.data['relative_intensity'], 'o', color=ORANGE, ms=8)
        axis.set(xlim=(-3, 363), ylim=(-.15, 4.35), xticks=[0, 90, 180, 270, 360],
            yticks=[0, 1, 2, 3, 4], xlabel='Phase difference at the point: φ₂ − φ₁ (degrees)',
            ylabel='Relative scalar intensity I / I_ref',
            title='Add complex amplitudes first, then take the squared magnitude')
        _style(axis)
        axis.legend(loc='upper center', bbox_to_anchor=(.5, -.24), frameon=False, ncol=2, fontsize=9)
        self.figure.subplots_adjust(bottom=.28, top=.86)
        self.readout.value = (f'<b>I / I_ref = {self.data["relative_intensity"]:.3f}</b> = '
            f'{self.data["separate_intensities"]:.3f} + '
            f'({self.data["interference_term"]:+.3f}). '
            f'Mean of (u₁ + u₂)² over one cycle = <b>{self.data["mean_square"]:.3f}</b>.')

    def _draw_wavefronts(self, a1, a2, delta, tau):
        axes = self.figure.subplots(1, 2)
        vectors = symmetric_wavevectors(2, 30)
        self.wavefronts, self.crest_rulers = [], []
        target = np.array([0., 0., -1.])
        for index, (axis, vector, amp, phase, color) in enumerate(zip(
                axes, vectors, (a1, a2), (self.phase1.value, self.phase2.value), (BLUE, ORANGE))):
            normal = vector/np.linalg.norm(vector)
            transverse = np.array([-normal[2], 0., normal[0]])
            patches = _wavefront_patches(vector, phase, tau) if amp > 0 else []
            self.wavefronts.append(patches)
            for patch in patches:
                ends = patch[[np.argmin(patch[:, 0]), np.argmax(patch[:, 0])]]
                axis.plot(ends[:, 0], ends[:, 2], color=color, lw=2, alpha=.25+.65*amp)
            start = target-3.6/(-normal[2])*normal
            _arrow(axis, complex(start[0], start[2]), complex(target[0], target[2]),
                   color if amp else GREY, dashed=amp == 0)
            axis.plot([-4, 4], [-1, -1], color='#79539e', lw=2)
            axis.text(0, -1.32, 'Detector: z = −1 µm', color='#79539e', ha='center', fontsize=9)
            # Endpoints lie on adjacent crest planes; the ruler runs along k.
            centers = []
            for order in range(-10, 11):
                distance = (2*np.pi*(order+tau)-np.deg2rad(phase)-vector@target)/np.linalg.norm(vector)
                center = target+distance*normal
                if -.4 <= center[2] <= 3.1:
                    centers.append(center)
            ruler = None
            if amp > 0 and len(centers) >= 2:
                ruler = np.asarray(centers[:2])+.9*transverse
                axis.annotate('', xy=ruler[1, [0, 2]], xytext=ruler[0, [0, 2]],
                              arrowprops=dict(arrowstyle='<->', color=INK, lw=1.3))
                middle = ruler.mean(axis=0)+.3*transverse
                axis.text(middle[0], middle[2], 'λ = 2 µm', fontsize=9, ha='center',
                          bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1))
            self.crest_rulers.append(ruler)
            axis.set(xlim=(-4, 4), ylim=(-1.8, 3.7), aspect='equal',
                     xticks=[-4, -2, 0, 2, 4], yticks=[-1, 0, 1, 2, 3],
                     xlabel='x (µm)', ylabel='z (µm)',
                     title=f'Wave {index+1}: crest planes in xz' + (' (off)' if amp == 0 else ''))
            _style(axis)
            axis.title.set_fontsize(11)
        self.figure.subplots_adjust(left=.08, right=.98, top=.92, bottom=.16, wspace=.28)
        self.readout.value = (f'λ = 2 µm; directions fixed at ±30° from −z; model t/T = {tau:.3f}. '
            f'Each crest has advanced λt/T = {2*tau:.2f} µm from its position at t = 0. '
            'Each propagation arrow is perpendicular to its crest lines.')

    def _draw_local_phase(self, a1, a2, delta, tau):
        layout = self.figure.add_gridspec(2, 2, height_ratios=(1.35, .8))
        geometry = self.figure.add_subplot(layout[0, 0])
        readings = self.figure.add_subplot(layout[0, 1])
        cycle = self.figure.add_subplot(layout[1, :])
        vectors = symmetric_wavevectors(2, 30)
        point = np.array([self.detector_x.value, 0., -1.])
        self.local_data = self._spatial_field('local_field', (self.detector_x.value,),
                                             point, vectors, a1, a2, delta, tau)
        self.local_readings = np.asarray(self.local_data['fields'])
        phase_offsets = np.deg2rad([self.phase1.value, self.phase2.value])

        # Draw both plane-wave crest patterns across the whole observation
        # window, so each available detector point samples both waves.
        front_lines = []
        for index, (vector, amp, offset, color) in enumerate(zip(
                vectors, (a1, a2), phase_offsets, (BLUE, ORANGE))):
            style = '--' if index else '-'
            # A fixed line per crest order lets fronts enter and leave the
            # window without allocating artists during a time update.
            front_lines.append([geometry.plot([], [], color=color, ls=style,
                lw=1.5, alpha=.2+.45*amp, visible=False)[0] for _ in range(-8, 9)])
            normal = vector/np.linalg.norm(vector)
            start = point-2.25*normal
            geometry.annotate('', xy=point[[0, 2]], xytext=start[[0, 2]],
                arrowprops=dict(arrowstyle='->', color=color if amp else GREY,
                                lw=2.2, linestyle=style if amp else ':'))
            geometry.text(start[0], start[2]+.15, f'k{index+1}' + (' (off)' if amp == 0 else ''),
                          color=color if amp else GREY, ha='center', fontsize=10)

        geometry.axhline(-1, color='#79539e', lw=2)
        geometry.plot(point[0], point[2], 'o', color='#79539e', markeredgecolor='white',
                      ms=8, zorder=5, label='Selected point P')
        geometry.text(point[0], -1.46, 'P', color='#79539e', ha='center', fontsize=11)
        geometry.set(xlim=(-3.25, 3.25), ylim=(-1.7, 3.35), aspect='equal',
                     xticks=[-3, -2, -1, 0, 1, 2, 3], yticks=[-1, 0, 1, 2, 3],
                     xlabel='x (µm)', ylabel='z (µm)',
                     title='Both waves at the same point P')
        geometry.text(.5, -.24, 'Crest lines and propagation directions', transform=geometry.transAxes,
                      ha='center', fontsize=9, color=GREY)

        values = np.r_[self.local_readings, self.local_readings.sum()]
        readings.axhline(0, color=GREY, lw=.8)
        bars = readings.bar([0, 1, 2], values, width=.55, color=(BLUE, ORANGE, INK), alpha=.85)
        bar_labels = []
        for index, value in enumerate(values):
            bar_labels.append(readings.text(index, value + (.12 if value >= 0 else -.12),
                f'{value:+.3f}', ha='center',
                va='bottom' if value >= 0 else 'top', fontsize=10))
        readings.set(xlim=(-.6, 2.6), ylim=(-2.5, 2.5),
                     xticks=[0, 1, 2], xticklabels=['Wave 1', 'Wave 2', 'Sum'],
                     yticks=[-2, -1, 0, 1, 2], ylabel='Signed field (field units)',
                     title='Add readings at P at the same time')
        readings.text(.5, -.24, 'u(P,t) = u₁(P,t) + u₂(P,t)',
                      transform=readings.transAxes, ha='center', fontsize=10, color=INK)
        # Both complete time traces refer to this same spatial point. The
        # model-time control selects one simultaneous reading on each curve;
        # it must not shift the traces themselves.
        tt = self._cache.get('local_cycle_grid', (), lambda: np.linspace(0., 1., 401))
        fields = self._cache.get('local_cycles', (self.detector_x.value, a1, a2, delta, self.phase1.value),
            lambda: np.real(self.local_data['phasors'][:, None]*np.exp(-2j*np.pi*tt)))
        self.local_cycle_data = dict(tau=tt, fields=fields)
        cycle.axhline(0, color=GREY, lw=.8)
        cursor = cycle.axvline(tau, color='#79539e', ls=':', lw=1.5, label='Selected time', zorder=4)
        cursor_tip, = cycle.plot(tau, 1.17, 'v', color='#79539e', ms=6, zorder=5, clip_on=False)
        markers = []
        for index, (curve, value, color) in enumerate(zip(
                fields, self.local_readings, (BLUE, ORANGE))):
            cycle.plot(tt, curve, color=color, ls='--' if index else '-',
                       lw=1.8 if index else 2.8, label=f'Wave {index+1} at P')
            marker, = cycle.plot(tau, value, 's' if index else 'o', color=color,
                        ms=5 if index else 8, zorder=5+index, clip_on=False,
                        label=f'Wave {index+1} at selected time')
            markers.append(marker)
        cycle.set(xlim=(0, 1), ylim=(-1.2, 1.2), xticks=[0, .25, .5, .75, 1],
                  yticks=[-1, 0, 1], xlabel='Time t/T at the same point P',
                  ylabel='Signed field (field units)', title='Two wave cycles at P — one shared time cursor')
        cycle_label = cycle.text(.5, -.38,
                   f'At the cursor: u₁(P,t) = {self.local_readings[0]:+.3f}; '
                   f'u₂(P,t) = {self.local_readings[1]:+.3f}',
                   transform=cycle.transAxes, ha='center', fontsize=10, color=INK)
        for axis in (geometry, readings, cycle):
            _style(axis)
            axis.title.set_fontsize(11)
        self.figure.subplots_adjust(left=.085, right=.97, top=.94, bottom=.14,
                                    wspace=.38, hspace=.68)
        self._local_time_artists = dict(point=point, vectors=vectors,
            amplitudes=(a1, a2), phase_offsets=phase_offsets, front_lines=front_lines,
            bars=bars, bar_labels=bar_labels, cursor=cursor, cursor_tip=cursor_tip,
            markers=markers, cycle_label=cycle_label)
        self._refresh_local_phase(tau)

    def _refresh_local_phase(self, tau):
        """Advance local readings and moving crests while retaining the scene."""
        artists = self._local_time_artists
        point, vectors = artists['point'], artists['vectors']
        a1, a2 = artists['amplitudes']
        phase_offsets = artists['phase_offsets']
        self.local_data = self._at_time(self.local_data, tau)
        self.local_readings = np.asarray(self.local_data['fields'])
        self.wavefronts = []
        for vector, amp, offset, lines in zip(vectors, (a1, a2), phase_offsets,
                                              artists['front_lines']):
            fronts = []
            for order, line in zip(range(-8, 9), lines):
                phase = 2*np.pi*(order+tau)-offset
                edge_x = (phase-vector[2]*np.array([-1., 3.35]))/vector[0]
                left, right = max(-3.25, edge_x.min()), min(3.25, edge_x.max())
                visible = amp > 0 and right > left
                line.set_visible(visible)
                if visible:
                    xx = np.array([left, right])
                    front = np.column_stack((xx, np.zeros(2),
                                             (phase-vector[0]*xx)/vector[2]))
                    fronts.append(front)
                    line.set_data(front[:, 0], front[:, 2])
            self.wavefronts.append(fronts)
        values = np.r_[self.local_readings, self.local_readings.sum()]
        for bar, label, value in zip(artists['bars'], artists['bar_labels'], values):
            bar.set_height(value)
            label.set_y(value + (.12 if value >= 0 else -.12))
            label.set_va('bottom' if value >= 0 else 'top')
            label.set_text(f'{value:+.3f}')
        artists['cursor'].set_xdata([tau, tau])
        artists['cursor_tip'].set_xdata([tau])
        for marker, value in zip(artists['markers'], self.local_readings):
            marker.set_data([tau], [value])
        artists['cycle_label'].set_text(
            f'At the cursor: u₁(P,t) = {self.local_readings[0]:+.3f}; '
            f'u₂(P,t) = {self.local_readings[1]:+.3f}')
        local_angles = vectors@point+phase_offsets-2*np.pi*tau
        phase_difference = float(np.rad2deg(self.local_data['phase_difference']))
        phase_labels = [f'{np.rad2deg(angle) % 360:.1f}°' if amp else 'undefined (wave off)'
                        for angle, amp in zip(local_angles, (a1, a2))]
        difference_text = (f'<b>ΔΦ(P) = {phase_difference:+.0f}°</b> '
                           f'(equivalent to {phase_difference % 360:.0f}°).'
                           if a1 > 0 and a2 > 0 else
                           '<b>ΔΦ(P) is undefined when either wave is off.</b>')
        self.readout.value = (f'P = ({point[0]:.2f}, 0, −1) µm; λ = 2 µm; θ = 30°; '
            f'model t/T = {tau:.3f}.<br>'
            f'Φ₁(P,t) = {phase_labels[0]}; Φ₂(P,t) = {phase_labels[1]}; {difference_text} '
            'The bars add the two signed readings selected by the shared time cursor.')

    def _draw_spacing(self, a1, a2, delta, tau):
        directions, profile = self.figure.subplots(1, 2)
        theta = self.angle.value
        vectors = symmetric_wavevectors(2, theta)
        k = wavenumber(2)
        delta_kx = vectors[1, 0]-vectors[0, 0]
        period = 2*np.pi/delta_kx
        for vector, color, name, amp in zip(vectors/k, (BLUE, ORANGE), ('k₁/k', 'k₂/k'), (a1, a2)):
            _arrow(directions, 0j, complex(vector[0], vector[2]), color if amp else GREY, dashed=amp == 0)
            directions.text(vector[0]+(-.08 if vector[0] < 0 else .08), vector[2]+.07, name,
                            color=color, ha='right' if vector[0] < 0 else 'left', fontsize=10)
        first, second = vectors[:, [0, 2]]/k
        _arrow(directions, complex(*first), complex(*second), '#79539e')
        directions.text(0, first[1]-.12, 'Δk / k', color='#79539e', ha='center', fontsize=11)
        directions.axvline(0, color=GREY, ls=':', lw=1)
        directions.axhline(0, color=GREY, lw=.8)
        directions.set(xlim=(-1.08, 1.08), ylim=(-1.32, .35), aspect='equal',
                       xticks=[-1, -.5, 0, .5, 1], yticks=[-1, -.5, 0],
                       xlabel='k_x / k', ylabel='k_z / k', title='Subtract the two wavevectors')
        directions.text(.5, .95, f'θ = {theta:.0f}°; angle between waves = {2*theta:.0f}°',
                        transform=directions.transAxes, ha='center', va='top', fontsize=9)
        x = self._cache.get('spacing_x', (), lambda: np.linspace(-6, 6, 901))
        points = self._cache.get('spacing_points', (),
            lambda: np.column_stack((x, np.zeros_like(x), -np.ones_like(x))))
        self.spatial_data = self._spatial_field('spacing_field', (theta,), points, vectors, a1, a2, delta)
        profile.plot(x, self.spatial_data['relative_intensity'], color=INK, lw=2)
        profile.axhline(a1*a1+a2*a2, color=GREY, ls='--', lw=1)
        bright_x = (2*np.pi*np.arange(-20, 21)-np.deg2rad(delta))/delta_kx
        bright_x = bright_x[(bright_x >= -6) & (bright_x <= 6)]
        if a1*a2 > 0 and len(bright_x) > 1:
            pair = np.argmin(np.abs((bright_x[:-1]+bright_x[1:])/2))
            left, right = bright_x[pair:pair+2]
            profile.annotate('', xy=(right, 4.12), xytext=(left, 4.12),
                             arrowprops=dict(arrowstyle='<->', color='#79539e', lw=1.5))
            profile.plot([left, left, np.nan, right, right], [3.88, 4.23, np.nan, 3.88, 4.23],
                         color='#79539e', lw=1, label='Fringe ruler')
            profile.text((left+right)/2, 4.26, 'Λ', color='#79539e', ha='center', va='bottom', fontsize=11)
        profile.set(xlim=(-6, 6), ylim=(-.15, 4.65), xticks=[-6, -3, 0, 3, 6],
                    xlabel='Detector x (µm)', ylabel='Relative intensity I / I_ref',
                    title=f'Fringe spacing Λ = {period:.3f} µm')
        for axis in (directions, profile):
            _style(axis)
            axis.title.set_fontsize(11)
        self.figure.subplots_adjust(left=.08, right=.98, top=.9, bottom=.22, wspace=.38)
        self.figure.text(.5, .055, r'$\Lambda=2\pi/\Delta k_x=\lambda/(2\sin\theta)$',
                         ha='center', fontsize=14)
        ruler_note = ('No visible fringes with one component absent.' if a1*a2 == 0 else
                      'Two bright fringes do not fit in this view.' if len(bright_x) < 2 else
                      'The ruler joins adjacent bright fringes.')
        self.readout.value = (f'λ = 2 µm; <b>Δk_x = {delta_kx:.3f} rad/µm</b>; '
            f'<b>Λ = {period:.3f} µm</b>. The purple arrow is k₂ − k₁, not another wave. {ruler_note}')

    def _draw_directions(self, a1, a2, delta, tau):
        scene, detector, vector_axis, intensity_axis = self.figure.subplots(2, 2).flat
        theta = self.angle.value
        vectors = symmetric_wavevectors(2, theta)
        k = wavenumber(2)
        delta_kx = vectors[1, 0]-vectors[0, 0]
        period = 2*np.pi/delta_kx
        x, y = self._cache.get('direction_axes', (),
            lambda: (np.linspace(-4, 4, 641), np.linspace(-1.5, 1.5, 61)))
        points = self._cache.get('direction_points', (),
            lambda: np.column_stack((x, np.zeros_like(x), -np.ones_like(x))))
        self.spatial_data = self._spatial_field('direction_field', (theta,), points, vectors, a1, a2, delta, tau)
        intensity = self.spatial_data['relative_intensity']
        self.screen_intensity = np.broadcast_to(intensity, (len(y), len(x)))
        self._draw_wavefront_scene(scene, vectors, (a1, a2),
                                  (self.phase1.value, self.phase2.value), tau)

        detector_image = _uniform_field_image(detector, x, y, self.screen_intensity, cmap='gray', vmin=0, vmax=4)
        detector.axhline(0, color='#79539e', ls='--', lw=1.3)
        detector.set(xlim=(-4, 4), ylim=(-1.5, 1.5), xlabel='x (µm)', ylabel='y (µm)',
                     title='2. Same detector, viewed face-on (xy)')
        detector.tick_params(labelsize=9)
        # Mark actual adjacent maxima when two fit in the displayed detector.
        bright_x = (2*np.pi*np.arange(-10, 11)-np.deg2rad(delta))/delta_kx
        bright_x = bright_x[(bright_x >= -4) & (bright_x <= 4)]
        if a1*a2 > 0 and len(bright_x) > 1:
            pair = np.argmin(np.abs((bright_x[:-1]+bright_x[1:])/2))
            left, right = bright_x[pair:pair+2]
            detector.annotate('', xy=(right, 1.0), xytext=(left, 1.0),
                              arrowprops=dict(arrowstyle='<->', color='#79539e', lw=1.5))
            detector.plot([left, left], [.87, 1.13], color='#79539e', lw=1.2)
            detector.plot([right, right], [.87, 1.13], color='#79539e', lw=1.2)
            detector.text((left+right)/2, 1.12, r'$\Lambda$', color='#79539e',
                          ha='center', va='bottom', fontsize=11,
                          bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
        detector.text(.5, -.25, 'Purple dashed line: row y = 0, plotted below',
                      transform=detector.transAxes, ha='center', fontsize=8)

        for vector, color, name, amplitude in zip(vectors/k, (BLUE, ORANGE), ('k₁', 'k₂'), (a1, a2)):
            _arrow(vector_axis, 0j, complex(vector[0], vector[2]),
                   color if amplitude > 0 else GREY, dashed=amplitude == 0)
            vector_axis.text(vector[0]+(-.10 if vector[0] < 0 else .10), vector[2], name,
                color=color, ha='right' if vector[0] < 0 else 'left', fontsize=12)
        vector_axis.axhline(0, color=GREY, lw=.8)
        vector_axis.axvline(0, color=GREY, lw=.8)
        vector_axis.set(xlim=(-1.08, 1.08), ylim=(-1.18, .98), aspect='equal',
            xlabel='k_x / k', ylabel='k_z / k', title='3. Propagation directions (wavevector space)')
        vector_axis.text(.5, .93, f'θ = {theta:.0f}°; angle between waves = {2*theta:.0f}°',
                         transform=vector_axis.transAxes, ha='center', va='top', fontsize=9)
        vector_axis.text(.5, .72, r'$\Lambda=\frac{\lambda}{2\sin\theta}$' + f' = {period:.3f} µm',
                         transform=vector_axis.transAxes, ha='center', va='center', fontsize=13,
                         bbox=dict(facecolor='white', edgecolor='#d0d6dd', boxstyle='round,pad=.35'))
        _style(vector_axis)
        intensity_axis.plot(x, intensity, color=INK, lw=2.2)
        intensity_axis.axhline(a1*a1+a2*a2, color=GREY, ls='--', lw=1)
        intensity_axis.set(xlim=(-4, 4), ylim=(-.15, 4.35), xticks=[-4, -2, 0, 2, 4],
            yticks=[0, 1, 2, 3, 4], xlabel='x (µm), detector row y = 0, z = −1 µm',
            ylabel='Relative scalar intensity I / I_ref', title='4. Intensity along the detector row')
        _style(intensity_axis)
        self.figure.subplots_adjust(left=.085, right=.97, bottom=.15, top=.94,
                                    wspace=.40, hspace=.57)
        for axis in (scene, detector, vector_axis, intensity_axis):
            axis.title.set_fontsize(10)
        color_axis = self.figure.add_axes([.60, .06, .32, .016])
        self.figure.colorbar(detector_image, cax=color_axis, orientation='horizontal',
                            ticks=[0, 2, 4], label='Detector brightness: I / I_ref (fixed scale)')
        self.readout.value = (f'λ = 2 µm; A₁ = {a1:.1f}, A₂ = {a2:.1f}; directions = ±{theta:.0f}° from −z. '
            f'<b>Δk_x = {delta_kx:.3f} rad/µm</b>; '
            f'<b>Λ = λ / (2 sin θ) = {period:.3f} µm</b> (phase-repeat spacing). '
            f'Detector: z = −1 µm; model t/T = {tau:.3f}. '
            f'Grey dashed profile line: I₁ + I₂ = {a1*a1+a2*a2:.2f}. '
            'Time advances the wavefronts; the detector averages over the optical cycle. '
            + ('No visible fringes: at least one component is absent.' if a1*a2 == 0 else ''))

    def _draw_wavefront_scene(self, axis, vectors, amplitudes, phases, tau):
        """Orthogonal xz projection: horizontal x, vertical z, equal axis scales."""
        purple = '#79539e'
        axis.plot([-4, 4], [-1, -1], color=purple, lw=2.5, zorder=4)
        axis.text(0, -1.30, 'Detector plane seen edge-on: z = −1 µm',
                  ha='center', va='top', fontsize=8, color=purple)
        axis.plot([0, 0], [-1, 3.1], color=GREY, ls=':', lw=1)

        self.wavefronts = []
        for index, (vector, amp, phase, color) in enumerate(zip(vectors, amplitudes, phases, (BLUE, ORANGE))):
            patches = _wavefront_patches(vector, phase, tau) if amp > 0 else []
            self.wavefronts.append(patches)
            for patch in patches:
                # ky=0: each crest plane projects to one line in xz.
                ends = patch[[np.argmin(patch[:, 0]), np.argmax(patch[:, 0])]]
                axis.plot(ends[:, 0], ends[:, 2], color=color, lw=2,
                          alpha=.25+.65*amp, zorder=3)
            normal = vector/np.linalg.norm(vector)
            # One representative propagation arrow per plane wave.
            end = np.array([0., 0., -1.])
            start = end-(2.6+1)/(-normal[2])*normal
            _arrow(axis, complex(start[0], start[2]), complex(end[0], end[2]),
                   color if amp > 0 else GREY, dashed=amp == 0)
            label = f'Wave {index+1}: k{chr(0x2081+index)}'
            if amp == 0:
                label += ' (off)'
            axis.text(.97 if index == 0 else .03, .97, label,
                      color=color if amp > 0 else GREY, transform=axis.transAxes,
                      fontsize=9, va='top', ha='right' if index == 0 else 'left',
                      bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))

        axis.set(xlim=(-5, 5), ylim=(-1.95, 3.7), aspect='equal',
                 xticks=[-4, -2, 0, 2, 4], yticks=[-1, 0, 1, 2, 3],
                 xlabel='x (µm)', ylabel='z (µm)', title='1. Plane-wave incidence: xz side view')
        _style(axis)

    def _draw_orders(self, a1, a2, delta, tau):
        self._order_data(a1, a2, delta, tau)
        if self.order_view.value == 'experiment':
            self._draw_experiment(a1, a2, delta, tau)
        else:
            self._draw_order_fields(a1, a2, delta, tau)

    def _draw_experiment(self, a1, a2, delta, tau):
        setup, model, camera, profile = self.figure.subplots(2, 2).flat
        x, z, _, vectors = self._order_grid()
        detector_z = self.detector_z.value
        purple = '#79539e'

        setup.axhline(0, color=INK, lw=3)
        setup.text(-2.85, .13, 'Grating / prepared ±1 pair', fontsize=9, va='bottom')
        setup.text(0, .96, 'Broad coherent illumination', ha='center', fontsize=10)
        for xpos in (-2, -.7, .7, 2):
            _arrow(setup, complex(xpos, .82), complex(xpos, .34), GREY)
        for start_x in (-1.4, 0, 1.4):
            for vector, color, amp in zip(vectors/np.pi, (BLUE, ORANGE), (a1, a2)):
                start = complex(start_x, -.12)
                _arrow(setup, start, start+1.15*complex(vector[0], vector[2]),
                       color if amp > 0 else GREY, dashed=amp == 0)
        setup.axhline(detector_z, color=purple, lw=2)
        setup.text(-2.85, detector_z-.10, f'Flat detector: z = {detector_z:.2f} µm',
                   color=purple, fontsize=9, va='top',
                   bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
        setup.text(.5, -.32, 'Arrows: directions of overlapping wide waves',
                   transform=setup.transAxes, ha='center', fontsize=8)
        setup.set(xlim=(-3, 3), ylim=(-3.15, 1.2), aspect='equal', xlabel='x (µm)', ylabel='z (µm)',
                  title='Ideal arrangement: side view (xz)')
        _style(setup)

        field_picture = _uniform_field_image(model, x, z, self.spatial_data['field_sum'],
                                             cmap='RdBu_r', vmin=-2, vmax=2)
        self._order_time_artists = [(field_picture, None)]
        model.axhline(detector_z, color=purple, ls='--', lw=1.8)
        model.set(xlim=(-3, 3), ylim=(-3, 0), xlabel='x (µm)', ylabel='z (µm)',
                  title='Calculated signed field: an optical snapshot')
        model.tick_params(labelsize=9)

        y = np.linspace(-1.5, 1.5, 61)
        signal = self.camera_data['signal_dn']
        camera_picture = _uniform_field_image(camera, x, y, np.broadcast_to(signal, (len(y), len(x))),
                                              cmap='gray', vmin=10, vmax=410)
        camera.axhline(0, color=purple, ls='--', lw=1)
        camera.set(xlim=(-3, 3), ylim=(-1.5, 1.5), xlabel='x (µm)', ylabel='y (µm)',
                   title='Predicted camera image: face-on view (xy)')
        camera.tick_params(labelsize=9)
        profile.plot(x, signal, color=INK, lw=2.2, label='Expected signal')
        profile.axhline(10, color=GREY, ls=':', lw=1.2, label='Dark offset: 10 DN')
        profile.axhline(10+100*a1*a1, color=BLUE, ls='--', lw=1.2,
                       label=f'Wave 1 alone: {10+100*a1*a1:.0f} DN')
        profile.set(xlim=(-3, 3), ylim=(-10, 440), xticks=[-3, -2, -1, 0, 1, 2, 3],
                    yticks=[10, 110, 210, 310, 410], xlabel='x (µm), detector row y = 0',
                    ylabel='Expected camera signal (DN)', title='Camera row: brightness becomes numbers')
        _style(profile)
        profile.legend(loc='upper center', bbox_to_anchor=(.5, -.26), ncol=1,
                       fontsize=8, frameon=False)
        for axis in (setup, model, camera, profile):
            axis.title.set_fontsize(10)
        self.figure.subplots_adjust(left=.08, right=.97, top=.92, bottom=.24,
                                    hspace=.65, wspace=.38)
        field_bar = self.figure.add_axes([.58, .575, .30, .015])
        self.figure.colorbar(field_picture, cax=field_bar, orientation='horizontal',
            ticks=[-2, 0, 2], label='Signed field (model units)')
        camera_bar = self.figure.add_axes([.125, .10, .30, .018])
        self.figure.colorbar(camera_picture, cax=camera_bar, orientation='horizontal',
            ticks=[10, 210, 410], label='Expected signal (DN; fixed display scale)')
        self.readout.value = (f'<b>Detector z = {detector_z:.2f} µm</b>; '
            f'A_−1 = {a1:.1f}, A_+1 = {a2:.1f}. '
            f'<b>Signal extrema: {10+100*(a1-a2)**2:.1f} to {10+100*(a1+a2)**2:.1f} DN</b>. '
            'Calibration: N = 10 + 100 I/I_ref, fixed exposure. '
            'Model time changes the signed field; an ordinary camera averages many optical cycles. '
            'Images are ideal predictions, with fine pixels and no noise or saturation.')

    def _draw_order_fields(self, a1, a2, delta, tau):
        axes = self.figure.subplots(2, 2)
        x, z, _, vectors = self._order_grid()
        fields = [*self.spatial_data['fields'], self.spatial_data['field_sum']]
        titles = ['Wave 1: m = −1', 'Wave 2: m = +1', 'Sum of the signed fields',
                  'Cycle-averaged scalar intensity of the sum']
        self._order_time_artists = []
        for index, (axis, field, title) in enumerate(zip(axes.flat, fields, titles)):
            signed_picture = _uniform_field_image(axis, x, z, field, cmap='RdBu_r', vmin=-2, vmax=2)
            self._order_time_artists.append((signed_picture, index if index < 2 else None))
        intensity_picture = _uniform_field_image(axes[1, 1], x, z, self.spatial_data['relative_intensity'],
                                                 cmap='viridis', vmin=0, vmax=4)
        for axis, title in zip(axes.flat, titles):
            axis.axhline(0, color=INK, lw=1, clip_on=False)
            axis.axhline(self.detector_z.value, color='#79539e', lw=1.4, ls='--')
            axis.set(xlim=(-3, 3), ylim=(-3, 0), aspect='equal', xlabel='x (µm)',
                     ylabel='z (µm)', title=title)
            axis.title.set_fontsize(11)
            axis.tick_params(labelsize=9)
        for axis, vector, color, amp in zip(axes[0], vectors/wavenumber(2), (BLUE, ORANGE), (a1, a2)):
            start = complex(0, -.3)
            _arrow(axis, start, start+complex(vector[0], vector[2]),
                   INK if amp > 0 else GREY, dashed=amp == 0)
        self.figure.subplots_adjust(left=.075, right=.96, top=.91, bottom=.23, hspace=.52, wspace=.28)
        signed_bar = self.figure.add_axes([.12, .085, .33, .023])
        self.figure.colorbar(signed_picture, cax=signed_bar, orientation='horizontal',
                            ticks=[-2, 0, 2], label='Signed field (same units in all three panels)')
        intensity_bar = self.figure.add_axes([.59, .085, .33, .023])
        self.figure.colorbar(intensity_picture, cax=intensity_bar, orientation='horizontal',
                            ticks=[0, 2, 4], label='Relative scalar intensity I / I_ref')
        self.readout.value = ('λ = 2 µm; d = 4 µm; normal incidence; only orders −1 and +1. '
            f'A_−1 = {a1:.1f}; assigned A_+1 = {a2:.1f}; φ_+1 − φ_−1 = {delta:.0f}°. '
            f'<b>Phase-repeat spacing: 2 µm</b>; intensity extrema: '
            f'<b>{(a1-a2)**2:.2f} to {(a1+a2)**2:.2f}</b>. '
            'Purple dashed line: detector plane. These are calculated xz maps. '
            'Switch to “Experiment and camera” to see the detector image and calibrated signal.')

    def close(self):
        """Release owned figures, image buffers and widget callbacks; safe twice."""
        if self._closed:
            return
        self._closed = True
        self.reset_button.on_click(self._reset, remove=True)
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        self._time_panel_artists = self._order_time_artists = None
        self._local_time_artists = None
        self._cache.clear()
        if hasattr(self, 'spatial_data'):
            self.spatial_data = None
        if hasattr(self, 'camera_data'):
            self.camera_data = None
        if hasattr(self, 'screen_intensity'):
            self.screen_intensity = None
        if hasattr(self, 'wavefronts'):
            self.wavefronts = None
        if hasattr(self, 'crest_rulers'):
            self.crest_rulers = None
        if hasattr(self, 'local_data'):
            self.local_data = None
        if hasattr(self, 'local_readings'):
            self.local_readings = None
        if hasattr(self, 'local_cycle_data'):
            self.local_cycle_data = None
        _close_widget_tree(self.ui)
