"""Notebook one's independent lesson: a spatial point and its local phasor."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from .grating_intro import _close_widget_tree
from .plotting import SpatialPhasorGridPlot, figure_png
from .waves import field_from_phasor, plane_wave_phasor, wavevector, wavenumber

BLUE, PURPLE, ORANGE = '#2266a5', '#75429a', '#db6c24'


class LocalPhasorExplorer:
    """Own controls for one spatial point; optionally show a later sample grid.

    A = 1, origin phase = 0, and c = 1. Spatial coordinates choose a point;
    the separate complex-plane axes describe its normalized complex value.
    """

    def __init__(self, dimensions=1, *, grid=False):
        import ipywidgets as widgets

        if dimensions not in (1, 2, 3):
            raise ValueError('Choose one, two, or three spatial dimensions.')
        self.dimensions, self.grid = dimensions, grid
        self._closed = self._resetting = False
        self._controls = []

        def slider(name, value, low, high, step, fmt='.2f'):
            control = widgets.FloatSlider(description=name, value=value, min=low,
                max=high, step=step, readout_format=fmt, continuous_update=False,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='310px', max_width='100%'))
            self._controls.append(control)
            return control

        self.wavelength = slider('Wavelength λ', 2, 1, 4, .1, '.1f')
        self.direction = self.theta = self.azimuth = None
        if dimensions == 1:
            self.direction = widgets.ToggleButtons(options=[('Right (+x)', 1), ('Left (−x)', -1)],
                                                   value=1, description='Direction')
            self._controls.append(self.direction)
        else:
            self.theta = slider('Angle θ from +z (°)' if dimensions == 2 else 'Polar θ (°)',
                                0 if dimensions == 2 else 90,
                                -180 if dimensions == 2 else 0, 180, 5, '.0f')
            if dimensions == 3:
                self.azimuth = slider('Azimuth φ (°)', 90, -180, 180, 5, '.0f')
        wave_controls = list(self._controls)
        self.axes = {1: ('x',), 2: ('x', 'z'), 3: ('x', 'y', 'z')}[dimensions]
        self.coordinates = {} if grid else {
            axis: slider(f'Point P: {axis}', .5 if dimensions == 1 else 0, -2, 2, .05)
            for axis in self.axes}
        self.time = slider('Time t/T', 0, 0, 1, .025, '.3f')
        self.show_rotation = widgets.Checkbox(value=False,
            description='Include time factor' if grid else 'Show rotating arrow (time factor)',
            indent=False, layout=widgets.Layout(width='310px'))
        self._controls.append(self.show_rotation)
        self._defaults = [(control, control.value) for control in self._controls]
        self.reset = widgets.Button(description='Reset this example',
                                   layout=widgets.Layout(width='180px'))
        self.reset.on_click(self._reset)
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1200px', height='auto'))
        flex = dict(display='flex', flex_flow='row wrap', align_items='center')
        note = ('Optional extension: circle locations are spatial samples; each arrow is a local complex value.'
                if grid else 'Left: choose the physical point P. Right: its one complex value on real–imaginary axes. '
                'The blue phasor excludes time; the rotating arrow is optional.')
        self.ui = widgets.VBox([
            widgets.HTML(f'<b>{note}</b>'),
            widgets.Box(wave_controls, layout=widgets.Layout(**flex)),
            widgets.Box(list(self.coordinates.values()), layout=widgets.Layout(**flex)),
            widgets.Box([self.time, self.show_rotation, self.reset], layout=widgets.Layout(**flex)),
            self.readout, self.picture])
        self.grid_plot = SpatialPhasorGridPlot(dimensions) if grid else None
        if grid:
            self.figure = self.grid_plot.figure
        else:
            self.figure = Figure(figsize=(10, 4.8), dpi=110, facecolor='white')
            FigureCanvasAgg(self.figure)
        for control in self._controls:
            control.observe(self._update, names='value')
        self._update()

    @property
    def position(self):
        return np.array([self.coordinates[axis].value for axis in self.axes])

    def _update(self, change=None):
        if self._closed or self._resetting:
            return
        k = wavenumber(self.wavelength.value)
        self.k_vector = (np.array([self.direction.value*k]) if self.dimensions == 1 else
                        wavevector(self.wavelength.value, self.theta.value,
                                   self.azimuth.value if self.azimuth is not None else 0))
        if self.dimensions == 2:
            self.k_vector = self.k_vector[[0, 2]]
        self.fraction = self.time.value
        timing = (f'A = 1; origin phase = 0; c = 1. T = {self.wavelength.value:.2f}; '
                  f't = {self.fraction*self.wavelength.value:.3f}; t/T = {self.fraction:.3f}. ')
        if self.grid:
            self.picture.value = self.grid_plot.render(self.k_vector, self.fraction,
                                                        self.show_rotation.value)
            self.readout.value = timing + (
                'Time factor included: all arrows rotate by the same angle.' if self.show_rotation.value else
                'Time factor excluded: changing time leaves every displayed phasor fixed.')
            return
        self.phasor = complex(plane_wave_phasor(self.position, self.k_vector))
        self.rotating = self.phasor*np.exp(-2j*np.pi*self.fraction)
        self.real_value = float(field_from_phasor(self.phasor, self.fraction, 2*np.pi))
        self._draw_local()
        point = ', '.join(f'{axis}={value:+.2f}' for axis, value in zip(self.axes, self.position))
        self.readout.value = (
            f'<b>P: ({point})</b> in length units. '
            f'<b>Fixed phasor U(P) = {self.phasor.real:+.3f} {self.phasor.imag:+.3f}i</b>; '
            f'phase at t = 0: {np.rad2deg(self.k_vector @ self.position):+.1f}°. '
            f'<b>Current real field u(P,t) = {self.real_value:+.3f}</b>.<br>' + timing)
        self.picture.value = figure_png(self.figure)

    def _draw_local(self):
        self.figure.clear()
        spatial, complex_axis = self.figure.subplots(1, 2)
        self.figure.subplots_adjust(left=.08, right=.96, bottom=.26, top=.85, wspace=.42)
        self.spatial_axis, self.complex_axis = spatial, complex_axis
        samples = np.linspace(-2, 2, 321)
        if self.dimensions == 1:
            self.spatial_points = samples[:, None]
            self.spatial_field = field_from_phasor(
                plane_wave_phasor(self.spatial_points, self.k_vector), self.fraction, 2*np.pi)
            spatial.plot(samples, self.spatial_field, color=BLUE, lw=2)
            spatial.axvline(self.position[0], color='0.5', ls=':', lw=1)
            spatial.axhline(0, color='0.75', lw=.8)
            spatial.plot(self.position[0], self.real_value, 'o', color=ORANGE, ms=8, label='Selected point P')
            spatial.set(xlim=(-2, 2), ylim=(-1.25, 1.25), xlabel='Position x (length units)',
                        ylabel='Real field u(x,t) / A')
            spatial.legend(loc='lower left', fontsize=9)
            spatial.grid(alpha=.12)
        else:
            xx, zz = np.meshgrid(samples[::2], samples[::2])
            self.spatial_points = (np.stack((xx, zz), axis=-1) if self.dimensions == 2 else
                np.stack((xx, np.full_like(xx, self.position[1]), zz), axis=-1))
            self.spatial_field = field_from_phasor(
                plane_wave_phasor(self.spatial_points, self.k_vector), self.fraction, 2*np.pi)
            spatial.pcolormesh(samples[::2], samples[::2], self.spatial_field,
                cmap='RdBu_r', vmin=-1, vmax=1, shading='auto', rasterized=True)
            spatial.plot(self.position[0], self.position[-1], 'o', color=ORANGE,
                         mec='black', mew=1, ms=9)
            spatial.text(.03, .97, 'P = orange point', transform=spatial.transAxes,
                va='top', fontsize=9, bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
            spatial.set(xlim=(-2, 2), ylim=(-2, 2), aspect='equal',
                        xlabel='Position x (length units)', ylabel='Position z (length units)')
            self.figure.text(.25, .14, 'Signed field / A: blue = −1, white = 0, red = +1',
                             ha='center', fontsize=9)
        title = f'Physical space: wave at t/T = {self.fraction:.3f}'
        if self.dimensions == 3:
            title = f'Physical space: xz slice at y = {self.position[1]:+.2f}'
        spatial.set_title(title, fontsize=11, pad=12)

        complex_axis.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect='equal',
            xlabel='Real amplitude / A', ylabel='Imaginary amplitude / A',
            title='Complex plane: the phasor at P')
        complex_axis.title.set_fontsize(11)
        complex_axis.add_patch(Circle((0, 0), 1, fill=False, ec='.75', lw=1))
        complex_axis.axhline(0, color='.75', lw=.8)
        complex_axis.axvline(0, color='.75', lw=.8)
        complex_axis.set_xticks([-1, 0, 1])
        complex_axis.set_yticks([-1, 0, 1])
        complex_axis.annotate('', xy=(self.phasor.real, self.phasor.imag), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.3, linestyle='--'))
        complex_axis.plot(self.phasor.real, self.phasor.imag, 'o', mfc='white', mec=BLUE, ms=7)
        handles = [Line2D([], [], color=BLUE, ls='--', lw=2, label='Fixed phasor U(P)')]
        if self.show_rotation.value:
            complex_axis.annotate('', xy=(self.rotating.real, self.rotating.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2.4))
            complex_axis.plot([self.rotating.real]*2, [0, self.rotating.imag], ':', color=ORANGE)
            complex_axis.plot(self.real_value, 0, 'o', color=ORANGE, ms=7)
            handles.extend([
                Line2D([], [], color=PURPLE, lw=2, label='With time factor: U(P) exp(−iωt)'),
                Line2D([], [], color=ORANGE, marker='o', ls='none', label='Real part = current field at P')])
        for axis in (spatial, complex_axis):
            axis.spines[['top', 'right']].set_visible(False)
        self.figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .025),
                           frameon=False, ncol=len(handles), fontsize=9)

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

    def close(self):
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.reset.on_click(self._reset, remove=True)
        self.picture.value = b''
        self.figure.clear()
        self.figure = self.grid_plot = None
        self.spatial_axis = self.complex_axis = None
        _close_widget_tree(self.ui)
