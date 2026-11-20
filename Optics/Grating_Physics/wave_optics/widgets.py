"""Jupyter controls and callbacks for the wave-optics plots."""

from os.path import relpath
from pathlib import Path
from urllib.parse import quote

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go

from .diffraction import diffraction_orders
from .coordinates import wavevector_coordinates
from .plotting import (
    PlaneWavePlot, ScalarPlaneWave3DPlot, PlaneWavePhasorPlot,
    SpatialPhasorGridPlot, geometry_figures,
    update_geometry_highlight, export_geometry_html,
    coordinate_figure, coordinate_readout, update_coordinate_figure,
)


class WavevectorCoordinateExplorer:
    """Linked wavevector sphere and switchable optical coordinate projection."""

    def __init__(self):
        self._busy = False

        def slider(value, minimum, maximum, step, description):
            return widgets.FloatSlider(
                value=value, min=minimum, max=maximum, step=step,
                description=description, continuous_update=False,
                readout_format='.2f', style={'description_width': '85px'},
                layout=widgets.Layout(width='310px'))

        self.wavelength = slider(0.55, 0.3, 1.0, 0.01, 'λ₀ (μm)')
        self.index = slider(1.0, 1.0, 1.8, 0.05, 'Index n')
        self.theta = slider(40, 0, 180, 1, 'Polar θ (°)')
        self.phi = slider(30, -180, 180, 1, 'Azimuth φ (°)')
        self.na = slider(0.8, 0.1, 1.0, 0.05, 'NA (+z)')
        self.projection = widgets.Dropdown(
            options=[('Direction cosines', 'direction'), ('Cartesian kₓ, kᵧ', 'cartesian'),
                     ('Spatial frequencies', 'frequency'), ('Normalized pupil', 'pupil')],
            value='direction', description='Projection',
            style={'description_width': '85px'}, layout=widgets.Layout(width='310px'))
        self.data = self._calculate()
        self.figure = go.FigureWidget(coordinate_figure(self.data, self.projection.value))
        self._projection_value = self.projection.value
        self.readout = widgets.HTML(coordinate_readout(self.data))
        self.controls = (self.wavelength, self.index, self.theta,
                         self.phi, self.na, self.projection)
        control_box = widgets.Box(list(self.controls), layout=widgets.Layout(
            display='flex', flex_flow='row wrap'))
        self.ui = widgets.VBox([
            control_box,
            widgets.HTML('Drag the sphere to rotate; hover for coordinates. '
                         'Green marks the collection rim. The sphere axes rescale with k.'),
            self.figure, self.readout,
        ])
        for control in self.controls:
            control.observe(self._update, names='value')

    def _calculate(self):
        return wavevector_coordinates(
            self.wavelength.value, self.index.value, self.theta.value,
            self.phi.value, self.na.value)

    def _update(self, change=None):
        if self._busy:
            return
        self._busy = True
        try:
            # Shrinking n also shrinks the physically possible maximum NA.
            self.na.max = self.index.value
            previous = self.data
            self.data = self._calculate()
            update_coordinate_figure(self.figure, self.data, self.projection.value,
                                     previous, self._projection_value)
            self._projection_value = self.projection.value
            self.readout.value = coordinate_readout(self.data)
        finally:
            self._busy = False

    def close(self):
        """Release widget resources when rerunning the notebook cell."""
        for control in self.controls:
            control.unobserve(self._update, names='value')
        def close_tree(widget):
            for child in getattr(widget, 'children', ()):
                close_tree(child)
            widget.close()
        close_tree(self.ui)


class _PhasorPanel:
    """Position controls for a local phasor, sharing its parent wave's clock."""

    def __init__(self, axes, bound):
        self.axes = tuple(axes)
        self.k_vector = None
        self.fraction = 0.0
        self.plot = PlaneWavePhasorPlot()
        self.coordinates = {
            axis: widgets.FloatSlider(
                value=0.25 if axis == 'y' else 0.5,
                min=-bound, max=bound, step=0.05, description=f'P: {axis}',
                readout_format='.2f', continuous_update=False,
                layout=widgets.Layout(width='280px'))
            for axis in self.axes
        }
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1000px', height='auto'))
        coordinates = widgets.Box(list(self.coordinates.values()),
            layout=widgets.Layout(display='flex', flex_flow='row wrap'))
        self.ui = widgets.VBox([
            widgets.HTML(
                '<b>Choose a position P in real space.</b> '
                'The circle below has real and imaginary amplitude axes. '
                'Use the shared Play and Time controls above.'),
            coordinates, self.readout, self.picture])
        for control in self.coordinates.values():
            control.observe(self._redraw, names='value')

    @property
    def position(self):
        return np.array([self.coordinates[axis].value for axis in self.axes])

    def set_wave(self, k_vector, fraction, redraw=True):
        if getattr(self, '_closed', False):
            return
        self.k_vector = np.asarray(k_vector, dtype=float)
        self.fraction = float(fraction)
        # Keep the hidden view's parameters current and render when it is opened.
        if redraw or not self.picture.value:
            self._redraw()

    def _redraw(self, change=None):
        if getattr(self, '_closed', False) or self.k_vector is None:
            return
        self.picture.value = self.plot.render(self.k_vector, self.position, self.fraction)
        phasor = self.plot.phasor
        point = ', '.join(f'{axis}={value:+.2f}'
                          for axis, value in zip(self.axes, self.position))
        phase = float(self.k_vector @ self.position)
        self.readout.value = (
            f'<div style="padding:6px 0">P: ({point}) length units; '
            f'spatial phase = {phase:+.3f} rad<br>'
            f'Phasor at P: {phasor.real:+.3f} {phasor.imag:+.3f}i; '
            f'|phasor| = {abs(phasor):.3f}; '
            f'real field u(P,t) = {self.plot.real_value:+.3f}</div>')


class _SpatialPhasorPanel:
    """A grid of local complex values, driven by the wave explorer's controls."""

    def __init__(self, dimensions):
        self.plot = SpatialPhasorGridPlot(dimensions)
        self.k_vector = None
        self.fraction = 0.0
        self.include_time = widgets.Checkbox(value=True, description='Include time factor',
            indent=False, layout=widgets.Layout(width='200px'))
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1200px', height='auto'))
        geometry = ('Three parallel x-z slices show the 3D field at y = -0.5, 0, +0.5. '
                    if dimensions == 3 else '')
        self.ui = widgets.VBox([
            widgets.HTML('<b>One complex value per spatial point.</b> ' + geometry
                         + 'The little arrows represent complex amplitude; '
                         'their angles are phase angles, not directions of propagation.'),
            widgets.Box([self.include_time, self.readout],
                        layout=widgets.Layout(display='flex', flex_flow='row wrap',
                                              align_items='center')), self.picture])
        self.include_time.observe(self._redraw, names='value')

    def set_wave(self, k_vector, fraction, redraw=True):
        if getattr(self, '_closed', False):
            return
        self.k_vector = np.asarray(k_vector, dtype=float)
        self.fraction = float(fraction)
        if redraw or not self.picture.value:
            self._redraw()

    def _redraw(self, change=None):
        if getattr(self, '_closed', False) or self.k_vector is None:
            return
        self.picture.value = self.plot.render(self.k_vector, self.fraction, self.include_time.value)
        self.readout.value = (
            f't/T = {self.fraction:.2f}. Play rotates all arrows by the same angle.'
            if self.include_time.value else
            'Time suppressed: Play leaves the phasors fixed; wavelength and direction still change them.')


class _WaveExplorer:
    """Resource cleanup shared by the scalar-wave explorers."""

    def close(self):
        """Dispose this explorer without touching other explorers; safe to repeat."""
        if getattr(self, '_closed', False):
            return
        self._closed = True
        if self.phasor is not None:
            self.phasor._closed = True
            self.spatial_phasors._closed = True
        self.play.playing = False
        self.time_link.unlink()
        for control in self._controls:
            control.unobserve(self._update, names='value')
        if self.views is not None:
            self.views.unobserve(self._change_view, names='selected_index')
        self._restart_control.on_click(self._restart, remove=True)
        if self.phasor is not None:
            for control in self.phasor.coordinates.values():
                control.unobserve(self.phasor._redraw, names='value')
            self.spatial_phasors.include_time.unobserve(
                self.spatial_phasors._redraw, names='value')

        # Layouts and styles have their own comms, outside the children tree.
        closed = set()

        def close_tree(widget):
            if not isinstance(widget, widgets.Widget) or id(widget) in closed:
                return
            closed.add(id(widget))
            for child in getattr(widget, 'children', ()):
                close_tree(child)
            close_tree(getattr(widget, 'layout', None))
            close_tree(getattr(widget, 'style', None))
            widget.close()

        for panel in (self, self.phasor, self.spatial_phasors):
            if panel is None:
                continue
            panel.picture.value = b''
            panel.plot.figure.clear()
            # Clearing axes alone leaves the renderer, canvas and artists held
            # by the plot object. Drop our references even if the caller keeps
            # the disposed explorer or its controls around.
            panel.plot = None
        self.figure = None
        close_tree(self.ui)


def close_wave_explorer(explorer):
    """Close a wave explorer, including instances created before close() existed.

    Reloading this module updates classes for new instances, not classes of
    objects already in a notebook's namespace. Adapt the previous wave-explorer
    layout to the same cleanup path so those objects can also be replaced.
    """
    if explorer is None:
        return
    close = getattr(explorer, 'close', None)
    if callable(close):
        close()
        return
    if not hasattr(explorer, '_controls'):
        # In 3D, direction is a numerical property, not a widget control.
        names = (('wavelength', 'polar', 'azimuth', 'time',
                  'view_elevation', 'view_azimuth') if hasattr(explorer, 'polar')
                 else ('wavelength', 'direction', 'time'))
        explorer._controls = tuple(getattr(explorer, name) for name in names)
    if not hasattr(explorer, '_restart_control'):
        explorer._restart_control = (explorer.reset if hasattr(explorer, 'reset')
                                     else explorer.restart)
    _WaveExplorer.close(explorer)


_PLAYBACK_NOTE = (
    'Phase speed: c = 1 length unit per time unit; wavelength is in the medium. '
    'Playback advances in normalized time t/T; '
    'each wavelength takes the same on-screen time per cycle. '
    'Changing wavelength holds t/T fixed and updates physical time t = (t/T)T, '
    'where T = wavelength/c. The readout gives T and t.')


class PlaneWaveExplorer(_WaveExplorer):
    """Interactive travelling wave (dimensions=1) or scalar x-z wave (2)."""
    def __init__(self, dimensions, *, include_phasors=True):
        self.dimensions = dimensions
        self.c = 1.0
        self.frames_per_period = 60
        self.wavelength = widgets.FloatSlider(
            value=2.0, min=1.0, max=4.0, step=0.1,
            description='Wavelength', readout_format='.1f',
            continuous_update=False, layout=widgets.Layout(width='330px'))
        if dimensions == 1:
            self.direction = widgets.ToggleButtons(
                options=[('Right (+x)', 1), ('Left (-x)', -1)],
                value=1, description='Direction')
        else:
            self.direction = widgets.IntSlider(
                value=30, min=-180, max=180, step=5,
                description='Angle (deg)', continuous_update=False,
                layout=widgets.Layout(width='330px'))
        self.play = widgets.Play(
            value=0, min=0, max=self.frames_per_period, step=1,
            interval=160, repeat=True, show_repeat=True)
        self.time = widgets.IntSlider(
            value=0, min=0, max=self.frames_per_period,
            description='Time t/T', readout=False, continuous_update=False,
            layout=widgets.Layout(width='330px'))
        self.time_label = widgets.HTML()
        self.reset = widgets.Button(description='Restart', icon='undo',
                                    layout=widgets.Layout(width='100px'))
        self._restart_control = self.reset
        self.reset.on_click(self._restart)
        self.time_link = widgets.jslink((self.play, 'value'), (self.time, 'value'))
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='960px', height='auto'))
        self.plot = PlaneWavePlot(dimensions, self.wavelength.min)
        self.figure = self.plot.figure
        self.phasor = self.spatial_phasors = self.views = None
        if include_phasors:
            self.phasor = _PhasorPanel(('x',) if dimensions == 1 else ('x', 'z'),
                                      bound=5 if dimensions == 1 else 4)
            self.spatial_phasors = _SpatialPhasorPanel(dimensions)
            self.views = widgets.Tab(children=[self.picture, self.phasor.ui, self.spatial_phasors.ui])
            self.views.set_title(0, 'Travelling wave' if dimensions == 1 else 'Wave & wavevector')
            self.views.set_title(1, 'Phasor')
            self.views.set_title(2, 'Spatial phasors')
            self.views.observe(self._change_view, names='selected_index')
        controls = widgets.Box([self.wavelength, self.direction],
            layout=widgets.Layout(display='flex', flex_flow='row wrap'))
        time_controls = widgets.Box([self.play, self.time, self.time_label, self.reset],
            layout=widgets.Layout(display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([controls, time_controls, widgets.HTML(_PLAYBACK_NOTE),
                               self.readout, self.views if self.views is not None else self.picture],
            layout=widgets.Layout(width='100%', max_width='1000px'))
        self._controls = (self.wavelength, self.direction, self.time)
        for control in self._controls:
            control.observe(self._update, names='value')
        self._update()

    def _restart(self, _):
        if getattr(self, '_closed', False):
            return
        self.play.playing = False
        self.play.value = 0
        self.time.value = 0

    def _update(self, change=None):
        if getattr(self, '_closed', False):
            return
        self.picture.value = self.plot.render(
            self.wavelength.value, self.direction.value,
            self.time.value / self.frames_per_period, self.c)
        self.time_label.value = self.plot.time_label
        self.readout.value = self.plot.summary_html
        if self.phasor is None:
            return
        self.phasor.set_wave(
            [self.kx] if self.dimensions == 1 else [self.kx, self.kz],
            self.fraction, redraw=self.views.selected_index == 1)
        self.spatial_phasors.set_wave(
            [self.kx] if self.dimensions == 1 else [self.kx, self.kz],
            self.fraction, redraw=self.views.selected_index == 2)

    def _change_view(self, change):
        if getattr(self, '_closed', False):
            return
        if change['new'] == 1:
            self.phasor._redraw()
        elif change['new'] == 2:
            self.spatial_phasors._redraw()

    @property
    def k(self):
        return self.plot.k

    @property
    def kx(self):
        return self.plot.kx

    @property
    def kz(self):
        return self.plot.kz

    @property
    def omega(self):
        return self.plot.omega

    @property
    def period(self):
        return self.plot.period

    @property
    def fraction(self):
        return self.plot.fraction

    @property
    def field(self):
        return self.plot.field


class ScalarPlaneWave3D(_WaveExplorer):
    """Interactive scalar wavefront planes and their 3D wavevector."""
    def __init__(self, *, include_phasors=True):
        self.c = 1.0
        self.frames_per_period = 60
        self.wavelength = widgets.FloatSlider(
            value=2.0, min=1.0, max=4.0, step=0.1, description='Wavelength',
            continuous_update=False, layout=widgets.Layout(width='310px'))
        self.polar = widgets.IntSlider(
            value=50, min=0, max=180, step=5, description='Polar (deg)',
            continuous_update=False, layout=widgets.Layout(width='310px'))
        self.azimuth = widgets.IntSlider(
            value=30, min=-180, max=180, step=5, description='Azimuth (deg)',
            continuous_update=False, layout=widgets.Layout(width='310px'))
        self.view_elevation = widgets.IntSlider(
            value=25, min=-80, max=90, step=5, description='View elevation',
            style={'description_width': 'initial'}, continuous_update=False,
            layout=widgets.Layout(width='350px'))
        self.view_azimuth = widgets.IntSlider(
            value=-55, min=-180, max=180, step=5, description='View azimuth',
            style={'description_width': 'initial'}, continuous_update=False,
            layout=widgets.Layout(width='350px'))
        self.play = widgets.Play(value=0, min=0, max=self.frames_per_period,
                                 interval=220, repeat=True, show_repeat=True)
        self.time = widgets.IntSlider(value=0, min=0, max=self.frames_per_period,
            description='Time t/T', readout=False, continuous_update=False,
            layout=widgets.Layout(width='310px'))
        self.time_label = widgets.HTML()
        self.time_link = widgets.jslink((self.play, 'value'), (self.time, 'value'))
        self.restart = widgets.Button(description='Restart', icon='undo',
                                       layout=widgets.Layout(width='100px'))
        self._restart_control = self.restart
        self.restart.on_click(self._restart)
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1100px', height='auto'))
        self.plot = ScalarPlaneWave3DPlot(self.wavelength.min)
        self.figure = self.plot.figure
        self.phasor = self.spatial_phasors = self.views = None
        if include_phasors:
            self.phasor = _PhasorPanel(('x', 'y', 'z'), bound=2)
            self.spatial_phasors = _SpatialPhasorPanel(3)
            self.views = widgets.Tab(children=[self.picture, self.phasor.ui, self.spatial_phasors.ui])
            self.views.set_title(0, 'Wavefronts & wavevector')
            self.views.set_title(1, 'Phasor')
            self.views.set_title(2, 'Spatial phasors')
            self.views.observe(self._change_view, names='selected_index')
        flex = dict(display='flex', flex_flow='row wrap', align_items='center')
        controls = widgets.Box([self.wavelength, self.polar, self.azimuth], layout=widgets.Layout(**flex))
        camera = widgets.Box([self.view_elevation, self.view_azimuth], layout=widgets.Layout(**flex))
        clock = widgets.Box([self.play, self.time, self.time_label, self.restart],
                            layout=widgets.Layout(**flex))
        self.ui = widgets.VBox([controls, clock, widgets.HTML(_PLAYBACK_NOTE),
                               camera, self.readout, self.views if self.views is not None else self.picture],
                              layout=widgets.Layout(width='100%', max_width='1140px'))
        self._controls = (self.wavelength, self.polar, self.azimuth, self.time,
                          self.view_elevation, self.view_azimuth)
        for control in self._controls:
            control.observe(self._update, names='value')
        self._update()

    def _restart(self, _):
        if getattr(self, '_closed', False):
            return
        self.play.playing = False
        self.play.value = 0
        self.time.value = 0

    def field_at(self, points):
        """Evaluate the scalar field at coordinates (..., 3) in the current frame."""
        return self.plot.field_at(points)

    def _update(self, change=None):
        if getattr(self, '_closed', False):
            return
        self.picture.value = self.plot.render(
            self.wavelength.value, self.polar.value, self.azimuth.value,
            self.time.value / self.frames_per_period,
            self.view_elevation.value, self.view_azimuth.value, self.c)
        self.time_label.value = self.plot.time_label
        self.readout.value = self.plot.summary_html
        if self.phasor is None:
            return
        self.phasor.set_wave(self.k_vector, self.fraction,
                             redraw=self.views.selected_index == 1)
        self.spatial_phasors.set_wave(self.k_vector, self.fraction,
                                      redraw=self.views.selected_index == 2)

    def _change_view(self, change):
        if getattr(self, '_closed', False):
            return
        if change['new'] == 1:
            self.phasor._redraw()
        elif change['new'] == 2:
            self.spatial_phasors._redraw()

    @property
    def k(self):
        return self.plot.k

    @property
    def k_vector(self):
        return self.plot.k_vector

    @property
    def direction(self):
        return self.plot.direction

    @property
    def period(self):
        return self.plot.period

    @property
    def fraction(self):
        return self.plot.fraction

    @property
    def fronts(self):
        return self.plot.fronts

    @property
    def half_width(self):
        return self.plot.half_width


class DiffractionGeometryExplorer:
    """Linked direction-cosine and Ewald views of transmitted grating orders."""
    def __init__(self, wavelength, period, theta_deg, phi_deg):
        self.period = float(period)
        self._busy = False
        layout = widgets.Layout(width="330px")
        style = {"description_width": "90px"}
        self.wavelength = widgets.FloatSlider(
            value=wavelength, min=min(wavelength, 0.1*period), max=max(wavelength, 1.5*period),
            step=0.01*period, description="λ (nm)", readout_format=".0f",
            continuous_update=False, layout=layout, style=style)
        self.theta = widgets.FloatSlider(
            value=theta_deg, min=0, max=max(85, theta_deg), step=1,
            description="θi (°)", readout_format=".0f",
            continuous_update=False, layout=layout, style=style)
        self.phi = widgets.FloatSlider(
            value=((phi_deg+180) % 360)-180, min=-180, max=180, step=1,
            description="φi (°)", readout_format=".0f",
            continuous_update=False, layout=layout, style=style)
        self.order = widgets.Dropdown(description="Highlight m", options=[0], value=0,
                                      layout=layout, style=style)
        self.summary = widgets.HTML()
        initial = diffraction_orders(wavelength, period, theta_deg, phi_deg)
        directions, ewald = geometry_figures(initial)
        self.directions = go.FigureWidget(directions)
        self.ewald = go.FigureWidget(ewald)
        self.tabs = widgets.Tab(children=[self.directions, self.ewald])
        self.tabs.set_title(0, "Directions & Harvey")
        self.tabs.set_title(1, "Ewald sphere")
        self.save_button = widgets.Button(description="Save current view as HTML",
                                           icon="save", layout=widgets.Layout(width="230px"))
        self.export_status = widgets.HTML()
        self.save_button.on_click(lambda _: self.export_html())
        controls = widgets.VBox([
            widgets.HBox([self.wavelength, self.theta]),
            widgets.HBox([self.phi, self.order])])
        self.ui = widgets.VBox([
            widgets.HTML(f"<b>Period d = {period:g} nm.</b> "
                         "Drag a sphere to rotate; scroll to zoom; hover over an order for coordinates."),
            controls, self.summary, self.tabs,
            widgets.HBox([self.save_button, self.export_status])])
        for control in (self.wavelength, self.theta, self.phi, self.order):
            control.observe(self._update, names="value")
        self._update(initial_order=initial["chosen"])

    def _update(self, change=None, initial_order=None):
        if self._busy:
            return
        self._busy = True
        try:
            if change is not None and change["owner"] is self.order:
                self.data["chosen"] = int(self.order.value)
                with self.directions.batch_update(), self.ewald.batch_update():
                    update_geometry_highlight(self.directions, self.ewald, self.data)
                self._set_summary()
                self.export_status.value = "View changed; save again to update the HTML."
                return
            selected = self.order.value if initial_order is None else initial_order
            data = diffraction_orders(self.wavelength.value, self.period,
                                 self.theta.value, self.phi.value, selected_order=selected)
            self.data = data
            options = tuple((f"m={m:+d}", int(m)) for m in data["allowed"])
            if self.order.options != options:
                self.order.options = options
            self.order.value = data["chosen"]
            new_figures = geometry_figures(data)
            for live, fresh in zip((self.directions, self.ewald), new_figures):
                assert len(live.data) == len(fresh.data)
                fresh.layout.scene.camera = live.layout.scene.camera
                with live.batch_update():
                    for current, updated in zip(live.data, fresh.data):
                        current.update(updated.to_plotly_json())
                    live.update_layout(fresh.layout.to_plotly_json())
            self._set_summary()
            self.export_status.value = "View changed; save again to update the HTML."
        finally:
            self._busy = False

    def _set_summary(self):
        data = self.data
        chosen = int(data["chosen"])
        index = int(np.flatnonzero(data["allowed"] == chosen)[0])
        direction = data["directions"][index]
        change = data["changes"][index]
        def coordinates(vector):
            return "(" + ", ".join(
                f"{0.0 if abs(value) < 0.00005 else value:+.4f}" for value in vector) + ")"
        grazing = data["labels"][data["grazing"]]
        propagating = data["labels"][data["valid"] & ~data["grazing"]]
        values = ", ".join(f"{m:+d}" for m in propagating) or "none"
        grazing_text = ", ".join(f"{m:+d}" for m in grazing) or "none"
        zero_note = (" At m=0, q=0: the incident and outgoing vectors coincide."
                     if chosen == 0 else "")
        self.summary.value = (
            f"<b>Highlighted order: m={chosen:+d}</b> — large square in every panel."
            f"{zero_note}<br>"
            f"<span style='font-family:monospace'>km/k = {coordinates(direction)}"
            f" &nbsp; Δk/k = {coordinates(change)}</span><br>"
            f"<b>λ/d = {data['spacing']:.3f}</b> &nbsp; "
            f"βi = {data['incident'][1]:.3f} &nbsp; "
            f"section radius ρ = {data['rho']:.3f}<br>"
            f"<b>Propagating m:</b> {values} &nbsp; <b>Grazing m:</b> {grazing_text}<br>"
            "All real orders and the nearest evanescent candidate on each side are shown. "
            "Colours pair +m with -m; diamonds mark grazing; "
            "crosses in the disk mark evanescent candidates.")

    def export_html(self):
        """Save the current geometry; physical-parameter sliders require the notebook."""
        target = export_geometry_html(
            self.directions, self.ewald, self.wavelength.value, self.period,
            self.theta.value, self.phi.value, self.order.value)
        href = quote(Path(relpath(target)).as_posix(), safe="/")
        self.export_status.value = (
            "<span style='font-size:12px'>Saved: "
            f"<a href='{href}' target='_blank'>"
            "open interactive HTML</a></span>")
        return target

