"""Owned interactive views of angular sigma and assumed sine-pupil mappings.

Lesson 5.4 uses mask-side geometry in micrometres and a physical pupil in mm.
The numerical order geometry is supplied by lattice_grating; the physical
positions explicitly assume r_p=C*sin(theta) for the chosen pupil or relay.
"""

import ipywidgets as widgets
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from . import lattice_grating as lg
from .grating_intro import _close_widget_tree
from .plotting import figure_png


def _draw_coordinate_chain(figure, data):
    """Draw one linked representation of the already calculated coordinates."""
    figure.clear()
    axes = figure.subplots(1, 3)
    figure.subplots_adjust(left=.055, right=.985, bottom=.22, top=.80, wspace=.38)
    radius_mm = data['R_mm']
    source = data['source_sigma']
    tracked = np.flatnonzero(np.all(data['labels'] == [-1, 0], axis=1))[0]
    extent = 1.45*max(1., np.max(abs(data['sigma'])))
    panels = [
        (data['k'][:, :2].real, data['k0']*data['NA_obj'],
         '1. Wavevector space\nObjective angular cutoff',
         r'$k_x$ (rad/µm)', r'$k_y$ (rad/µm)'),
        (data['sigma'], 1., '2. Angular sigma space\nUnit collection circle',
         r'$\sigma_x$', r'$\sigma_y$'),
        (data['pupil_xy_mm'], radius_mm,
         f'3. Physical pupil mapping\nAssumed sine law; R = {radius_mm:g} mm',
         r'$x_p$ (mm)', r'$y_p$ (mm)'),
    ]
    for ax, (points, radius, title, xlabel, ylabel) in zip(axes, panels):
        for keep, color, marker, label in (
                (data['collected'], '#2674af', 'o', 'Collected'),
                (~data['collected'], '#d46b19', 'x', 'Outside objective aperture')):
            ax.scatter(*points[keep].T, color=color, marker=marker, s=42,
                       label=label, zorder=3)
        ax.add_patch(Circle((0, 0), radius, fill=False, color='#2674af', lw=1.5))
        ax.scatter(*points[tracked], s=180, facecolors='none', edgecolors='black',
                   linewidths=1.2, label='Tracked order (−1, 0)', zorder=4)
        for order, point in zip(data['labels'], points):
            ax.annotate(f'({order[0]},{order[1]})', point, xytext=(5, 5),
                        textcoords='offset points', fontsize=8)
        ax.axhline(0, color='.75', lw=.6)
        ax.axvline(0, color='.75', lw=.6)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, aspect='equal',
               xlim=(-extent*radius, extent*radius), ylim=(-extent*radius, extent*radius))
        ax.grid(alpha=.13)
    figure.text(.35, .55, '→', fontsize=24, ha='center')
    figure.text(.35, .67, 'normalize\n(definition)', fontsize=8, ha='center')
    figure.text(.69, .55, '→', fontsize=24, ha='center')
    figure.text(.69, .67, 'sine-map\n(optical assumption)', fontsize=8, ha='center')
    figure.suptitle(f'One source at ({source[0]:.2f}, {source[1]:.2f}): '
                   'the same orders in three representations', fontsize=12)
    figure.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
                  bbox_to_anchor=(.5, .055), ncol=3, fontsize=9)
    figure.text(.5, .025,
                'Orange crosses in the physical panel mark required positions outside the aperture.',
                ha='center', fontsize=9, color='#555555')


class SigmaPupilExplorer:
    """Control one fixed-NA lithography example without changing its mapping law.

    Source coordinates and pitches change diffraction geometry. Pupil radius
    changes only the physical relay scale. Wavelength (193 nm), mask-side NA
    (0.3375), and index (1) stay fixed. The available controls keep all nine
    displayed candidates propagating. Each instance owns its figure/widgets.
    """

    wavelength0 = .193
    n_mask = 1.
    NA_mask = .3375

    def __init__(self):
        self._closed = False
        self._suspend_updates = False
        self.figure = Figure(figsize=(13, 4.8), dpi=110, facecolor='white')
        FigureCanvasAgg(self.figure)
        style = {'description_width': 'initial'}
        self.source_x = widgets.FloatSlider(value=.3, min=-.8, max=.8, step=.05,
            description='Source σx', readout_format='.2f', continuous_update=False,
            style=style, layout=widgets.Layout(width='300px'))
        self.source_y = widgets.FloatSlider(value=0, min=-.8, max=.8, step=.05,
            description='Source σy', readout_format='.2f', continuous_update=False,
            style=style, layout=widgets.Layout(width='300px'))
        self.pitch_x = widgets.IntSlider(value=800, min=500, max=1600, step=50,
            description='Mask dx (nm)', continuous_update=False,
            style=style, layout=widgets.Layout(width='300px'))
        self.pitch_y = widgets.IntSlider(value=1000, min=500, max=2000, step=50,
            description='Mask dy (nm)', continuous_update=False,
            style=style, layout=widgets.Layout(width='300px'))
        self.pupil_radius = widgets.FloatSlider(value=3, min=1, max=5, step=.5,
            description='Pupil R (mm)', readout_format='.1f', continuous_update=False,
            style=style, layout=widgets.Layout(width='300px'))
        self.on_axis = widgets.Button(description='On-axis source',
                                      layout=widgets.Layout(width='150px'))
        self.reset = widgets.Button(description='Reset example',
                                    layout=widgets.Layout(width='150px'))
        self._controls = (self.source_x, self.source_y, self.pitch_x, self.pitch_y,
                          self.pupil_radius)
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1430px', height='auto'))
        row = widgets.Box([*self._controls, self.on_axis, self.reset],
            layout=widgets.Layout(display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([row, self.readout, self.picture])
        for control in self._controls:
            control.observe(self._update, names='value')
        self.on_axis.on_click(self._on_axis)
        self.reset.on_click(self._reset)
        self._update()

    def _set_values(self, controls, values):
        if self._closed:
            return
        self._suspend_updates = True
        try:
            for control, value in zip(controls, values):
                control.value = value
        finally:
            self._suspend_updates = False
        self._update()

    def _on_axis(self, button):
        self._set_values((self.source_x, self.source_y), (0., 0.))

    def _reset(self, button):
        self._set_values(self._controls, (.3, 0., 800, 1000, 3.))

    def _update(self, change=None):
        if self._closed or self._suspend_updates:
            return
        source = np.array([self.source_x.value, self.source_y.value])
        dx, dy = self.pitch_x.value/1000, self.pitch_y.value/1000
        mm, pp = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
        labels = np.column_stack([mm.ravel(), pp.ravel()])
        geometry = lg.lattice_orders(np.diag([dx, dy]), labels,
            wavelength0=self.wavelength0, n_in=self.n_mask, n_out=self.n_mask,
            incident=self.NA_mask*source/self.n_mask)
        collected = lg.collect_orders(geometry, self.NA_mask)
        radius = self.pupil_radius.value
        self.data = dict(collected, source_sigma=source, dx=dx, dy=dy,
                         R_mm=radius, pupil_xy_mm=radius*collected['q'])
        _draw_coordinate_chain(self.figure, self.data)
        self.picture.value = figure_png(self.figure)
        tracked = np.flatnonzero(np.all(labels == [-1, 0], axis=1))[0]
        sigma = self.data['sigma'][tracked]
        pupil = self.data['pupil_xy_mm'][tracked]
        state = 'collected' if self.data['collected'][tracked] else 'outside the aperture'
        self.readout.value = (
            '<div style="padding:8px 0">All geometry is <b>mask-side</b>; '
            'λ₀ = 193 nm; n = 1; projection NA = 0.3375. '
            '<b>The physical panel assumes the declared sine mapping.</b><br>'
            f'Order spacings: Δσx = {self.wavelength0/(dx*self.NA_mask):.4f}, '
            f'Δσy = {self.wavelength0/(dy*self.NA_mask):.4f}. '
            f'{np.count_nonzero(self.data["collected"])}/9 displayed orders collected.<br>'
            f'Tracked order (−1, 0): σ = ({sigma[0]:+.4f}, {sigma[1]:+.4f}); '
            f'pupil position = ({pupil[0]:+.4f}, {pupil[1]:+.4f}) mm — {state}. '
            'Changing R rescales physical pupil positions without changing angular sigma coordinates.</div>')

    def close(self):
        """Disconnect and release only this explorer; safe on notebook reruns."""
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.on_axis.on_click(self._on_axis, remove=True)
        self.reset.on_click(self._reset, remove=True)
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        _close_widget_tree(self.ui)
