"""Steady EO phase control and optional averaging in notebook seven.

The input is the existing wavefront view's detector patch. Only beam 2's
phase changes. It can be held steady or alternate between two equally
weighted settings; geometry, arm intensities and the display scale stay fixed.
"""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .grating_intro import _close_widget_tree
from .plotting import figure_png


def phase_step_data(source, phase_step_deg=0., *, vary_phase=True):
    """Record a steady phase or average two settings with the apparatus phase."""
    if not np.isfinite(phase_step_deg) or not 0 <= phase_step_deg <= 180:
        raise ValueError('Use a finite phase step between 0 and 180 degrees.')
    beta = np.deg2rad(phase_step_deg)
    fields = source['fields']
    phase_factor = (1 + np.exp(1j*beta))/2 if vary_phase else np.exp(1j*beta)
    first = fields['intensity']
    second = np.abs(fields['field_1'] + fields['field_2']*np.exp(1j*beta))**2
    average = (first + second)/2
    recorded = average if vary_phase else second
    magnitude = float(abs(phase_factor))
    point = source['point']
    i1, i2 = float(point['intensity_1']), float(point['intensity_2'])
    balance = 2*np.sqrt(i1)*np.sqrt(i2)/(i1+i2) if i1+i2 > 0 else None
    # Normalized mutual coherence is undefined if either local arm is dark.
    coherence = magnitude if i1 > 0 and i2 > 0 else None
    visibility = balance*magnitude if balance is not None else None
    return dict(first=first, second=second, average=average, recorded=recorded,
                vary_phase=bool(vary_phase),
                phase_step_deg=float(phase_step_deg), phase_factor=phase_factor,
                coherence_magnitude=coherence, visibility_at_point=visibility,
                steady_visibility_at_point=balance,
                coordinates=source['coordinates'],
                intensity_limit=source['intensity_limit'])


def phase_switching_figures(geometry, phase_step_deg=0., *, source=None, selected=None, **parameters):
    """Two cached EO snapshots beside the same completed exposure average.

    The wavefronts freeze the common optical carrier, as in the steady view.
    Playback switches only the imposed EO phase, not the optical time.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from .interference_wavefronts import wavefront_data, wavefront_figure, phase_shifted_data

    if not np.isfinite(phase_step_deg) or not 0 <= phase_step_deg <= 180:
        raise ValueError('Use a finite phase step between 0 and 180 degrees.')
    source = wavefront_data(geometry, **parameters) if source is None else source
    selected = phase_shifted_data(source, phase_step_deg) if selected is None else selected
    snapshots = [wavefront_figure(data=sample.copy()) for sample in (source, selected)]
    data = phase_step_data(snapshots[0][1], phase_step_deg)
    frames = []
    for phase, (snapshot, source) in zip((0., phase_step_deg), snapshots, strict=True):
        figure = make_subplots(
            rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            column_widths=[.54, .46], horizontal_spacing=.10,
            subplot_titles=[f'Current pattern · EO at {phase:g}°',
                            'Detector exposure · equal-time average'])
        # Reuse the exact crest planes, detector slice and direction arrows.
        for trace in snapshot.data:
            if getattr(trace, 'scene', None) == 'scene':
                figure.add_trace(trace, row=1, col=1)
        figure.add_trace(go.Heatmap(
            x=data['coordinates'], y=data['coordinates'], z=data['average'],
            zmin=0, zmax=data['intensity_limit'],
            colorscale=[[0, '#000000'], [1, '#ffffff']],
            colorbar=dict(title=dict(text='Relative intensity', side='right'),
                          thickness=12, len=.78),
            meta={'role': 'exposure-average'},
            hovertemplate='x=%{x:.3f} μm<br>y=%{y:.3f} μm'
                          '<br>Average I=%{z:.3f}<extra></extra>'), row=1, col=2)
        figure.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers+text', text=['P'], textposition='top right',
            marker=dict(size=7, color='#18b7c3', symbol='cross'),
            textfont=dict(color='#18b7c3'), showlegend=False, hoverinfo='skip'),
            row=1, col=2)
        scene = snapshot.layout.scene.to_plotly_json()
        scene.pop('domain', None)
        extent = source['bounds'][0][1]
        figure.update_layout(
            template='plotly_white', height=480,
            font=dict(family='Arial', size=12, color='#243444'),
            margin=dict(l=25, r=25, t=55, b=75),
            uirevision='eo-exposure', scene=scene,
            legend=dict(orientation='h', x=.02, y=-.14,
                        itemclick=False, itemdoubleclick=False))
        figure.update_xaxes(title='x (μm)', range=[-extent, extent],
                           autorange=False, constrain='domain', row=1, col=2)
        figure.update_yaxes(title='y (μm)', range=[-extent, extent],
                           scaleanchor='x', scaleratio=1, autorange=False,
                           constrain='domain', row=1, col=2)
        frames.append(figure)
    return frames, data


class PhaseSwitchingExposureView:
    """Play two current patterns while keeping their exposure average visible."""

    def __init__(self, geometry, phase_step_deg=0., *, source=None, selected=None, **parameters):
        import ipywidgets as widgets
        import plotly.graph_objects as go

        self._closed = False
        self._refreshing = False
        self._source = source
        self._frames, self.data = phase_switching_figures(
            geometry, phase_step_deg, source=source, selected=selected, **parameters)
        self.figure = go.FigureWidget(self._frames[0])
        self.play = widgets.Play(
            value=0, min=0, max=1, step=1, interval=1000,
            repeat=True, show_repeat=False, playing=False)
        self.state = widgets.ToggleButtons(
            options=[('0°', 0), (f'Selected: {phase_step_deg:g}°', 1)], value=0,
            description='Current EO:', style={'description_width': 'initial'})
        self._clock_link = widgets.link((self.play, 'value'), (self.state, 'value'))
        self.readout = widgets.HTML()
        self.ui = widgets.VBox([
            widgets.HTML(
                '<b>Set EO phase to 180° above, then press Play.</b> '
                'The left pattern switches; the right panel shows the completed '
                'exposure with equal time at both settings.<br>'
                'Playback is slowed for illustration. The crest planes freeze '
                'the optical oscillation; only the EO phase switches.'),
            widgets.HBox([self.play, self.state],
                         layout=widgets.Layout(flex_flow='row wrap')),
            self.figure, self.readout])
        self.state.observe(self._show_state, names='value')
        self._show_state()

    def set_parameters(self, geometry, phase_step_deg=0., *, source=None, selected=None, **parameters):
        """Pause and recalculate both settings after an apparatus/EO change."""
        if self._closed:
            return
        from .interference_wavefronts import wavefront_data, phase_shifted_data, update_wavefront_scene
        from .conical_plotting import _trace_roles

        self.pause()
        source = wavefront_data(geometry, **parameters) if source is None else source
        selected = phase_shifted_data(source, phase_step_deg) if selected is None else selected
        change = 'phase' if source is self._source else 'geometry'
        self._source = source
        self.data = phase_step_data(source, phase_step_deg)
        for index, (frame, sample) in enumerate(zip(self._frames, (source, selected), strict=True)):
            with frame.batch_update():
                if index == 1 or change == 'geometry':
                    update_wavefront_scene(frame, sample, change=change)
                _trace_roles(frame)['exposure-average'].z = self.data['average']
                phase = phase_step_deg if index else 0.
                frame.layout.annotations[0].text = f'Current pattern · EO at {phase:g}°'
        self._refreshing = True
        try:
            self.state.options = [('0°', 0), (f'Selected: {phase_step_deg:g}°', 1)]
            self.state.value = 0
        finally:
            self._refreshing = False
        with self.figure.batch_update():
            update_wavefront_scene(self.figure, source, change=change)
            _trace_roles(self.figure)['exposure-average'].z = self.data['average']
        self._show_state()

    def _show_state(self, change=None):
        if self._closed or self._refreshing:
            return
        frame = self._frames[self.state.value]
        # Playback reuses the cached states; it does not recalculate fields or
        # send the unchanged average, blue planes and direction arrows again.
        moving = {'detector-surface', 'beam-2-crests', 'beam-2-rims'}
        with self.figure.batch_update():
            for old, new in zip(self.figure.data, frame.data, strict=True):
                if isinstance(new.meta, dict) and new.meta.get('role') in moving:
                    keys = ('surfacecolor',) if new.type == 'surface' else (
                        ('x', 'y', 'z', 'i', 'j', 'k', 'customdata') if new.type == 'mesh3d'
                        else ('x', 'y', 'z'))
                    old.update({key: getattr(new, key) for key in keys})
            self.figure.layout.annotations[0].text = frame.layout.annotations[0].text
        magnitude = self.data['coherence_magnitude']
        if magnitude is None:
            note = 'Normalized coherence is undefined at P: an arm is dark.'
        else:
            note = (f'Exposure: <b>|γ<sub>12</sub>| = {magnitude:.3f}</b>; '
                    f'local visibility at P = {self.data["visibility_at_point"]:.3f}.')
        self.readout.value = (
            'Both panels use the same detector patch and intensity scale. '
            'The average stays fixed while the current pattern switches.<br>' + note)

    def pause(self):
        self.play.playing = False

    def close(self, *, close_widgets=True):
        if self._closed:
            return
        self.pause()
        self._closed = True
        self.state.unobserve(self._show_state, names='value')
        self._clock_link.unlink()
        self._frames = None
        self._source = None
        self.data = None
        if close_widgets:
            _close_widget_tree(self.ui)


class PhaseStepExposureView:
    """EO phase control, with optional variation during a detector exposure."""

    def __init__(self, source, *, phase_step=None, vary_phase=None):
        import ipywidgets as widgets

        self._closed = False
        self._source = source
        self.phase_step = phase_step if phase_step is not None else widgets.FloatSlider(
            value=0, min=0, max=180, step=5, readout_format='.0f',
            description='EO phase β (°)', continuous_update=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='550px', max_width='100%'))
        self.readout = widgets.HTML()
        self.vary_phase = vary_phase if vary_phase is not None else widgets.Checkbox(
            value=False, description='Phase varies during exposure', indent=False,
            layout=widgets.Layout(width='auto'))
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1250px', height='auto'))
        self.results_ui = widgets.VBox([self.picture, self.readout])
        self.ui = widgets.VBox([self.phase_step, self.vary_phase, self.results_ui])
        self.figure = Figure(figsize=(12.4, 4.5), dpi=110, facecolor='white')
        FigureCanvasAgg(self.figure)
        self.axes = self.figure.subplots(1, 3, sharex=True, sharey=True)
        self.figure.subplots_adjust(left=.055, right=.89, bottom=.17,
                                    top=.80, wspace=.15)
        self._three_positions = [axis.get_position(original=True).bounds for axis in self.axes]
        self._vary_layout = None
        self.images = []
        for axis in self.axes:
            self.images.append(axis.imshow(
                np.zeros((2, 2)), origin='lower', cmap='gray', vmin=0,
                vmax=source['intensity_limit'], interpolation='nearest',
                extent=(-1, 1, -1, 1), aspect='equal'))
            axis.set_xlabel('x (μm)')
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.tick_params(labelsize=10)
        self.axes[0].set_ylabel('y (μm)')
        self.axes[0].set_title('EO at 0°\nReference pattern', fontsize=11)
        self.axes[2].set_title('Detector average\nEqual time at both EO settings', fontsize=11)
        self.colorbar_axis = self.figure.add_axes([.92, .215, .015, .54])
        colorbar = self.figure.colorbar(self.images[0], cax=self.colorbar_axis)
        colorbar.set_label('Relative scalar intensity', fontsize=10)
        colorbar.ax.tick_params(labelsize=9)
        self.figure.suptitle('EO phase control and the recorded detector pattern',
                             fontsize=14, y=.97, color='#243444')
        self.caption = self.figure.text(.055, .045, '', fontsize=10, color='#52616e')
        self.phase_step.observe(self._update, names='value')
        self.vary_phase.observe(self._update, names='value')
        self._update()

    def set_source(self, source):
        """Follow apparatus changes without resetting the phase-step slider."""
        if self._closed:
            return
        self._source = source
        self._update()

    def _update(self, change=None):
        if self._closed:
            return
        vary = self.vary_phase.value
        self.data = phase_step_data(self._source, self.phase_step.value, vary_phase=vary)
        if self._vary_layout != vary:
            self.figure.set_size_inches(12.4 if vary else 10.8, 4.5)
            self.axes[1].set_visible(vary)
            if vary:
                for axis, position in zip(self.axes, self._three_positions, strict=True):
                    axis.set_position(position)
            else:
                self.axes[0].set_position([.065, .17, .35, .63])
                self.axes[2].set_position([.475, .17, .35, .63])
            self.colorbar_axis.set_position([.92 if vary else .87, .215, .015, .54])
            self._vary_layout = vary
        low, high = self.data['coordinates'][[0, -1]]
        for image, key in zip(self.images, ('first', 'second', 'recorded'), strict=True):
            image.set_data(self.data[key])
            image.set_extent((low, high, low, high))
            image.set_clim(0, self.data['intensity_limit'])
        self.axes[1].set_title(
            f'EO at {self.phase_step.value:.0f}°\nPattern at the selected phase', fontsize=11)
        self.axes[2].set_title(
            'Detector average\nEqual time at both EO settings' if vary else
            f'Detector at EO phase {self.phase_step.value:.0f}°\nPhase held steady', fontsize=11)
        self.figure.suptitle(
            'Phase changes during exposure: average the patterns' if vary else
            'EO phase control: move the fringes', fontsize=14, y=.97, color='#243444')
        self.caption.set_text(
            f'{high-low:.3f} μm square centred at P; same patch and intensity scale as above.')
        magnitude = self.data['coherence_magnitude']
        visibility = self.data['visibility_at_point']
        if magnitude is None:
            note = '|γ<sub>12</sub>| is undefined at P: one or both arms have zero intensity.'
        else:
            note = (f'<b>|γ<sub>12</sub>| = {magnitude:.3f}</b> &nbsp; '
                    f'Local visibility at P: <b>V = {visibility:.3f}</b> '
                    f'(steady contrast: {self.data["steady_visibility_at_point"]:.3f}).')
        description = ('EO alternates between 0° and the selected phase, for equal times.'
                       if vary else 'EO phase is held fixed during the exposure.')
        self.readout.value = description + ' &nbsp; ' + note
        self.picture.value = figure_png(self.figure)

    def close(self, *, close_widgets=True):
        """Release resources; an enclosing explorer may own the widget tree."""
        if self._closed:
            return
        self._closed = True
        self.phase_step.unobserve(self._update, names='value')
        self.vary_phase.unobserve(self._update, names='value')
        self.picture.value = b''
        self._source = None
        self.data = None
        self.figure.clear()
        self.images.clear()
        self.axes = ()
        if close_widgets:
            _close_widget_tree(self.ui)
