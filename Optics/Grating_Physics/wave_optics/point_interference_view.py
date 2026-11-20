"""One-point field addition and cycle-averaged intensity for notebook seven.

Equal unit amplitudes and one relative-phase control isolate the next step
after overlap geometry. Importing creates no figures or widgets.
"""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .grating_intro import _close_widget_tree
from .plotting import figure_png
from .superposition import two_wave_cycle


class PointInterferenceExplorer:
    """Compare the signed field and normalized intensity at one fixed point."""

    def __init__(self):
        import ipywidgets as widgets

        self._closed = False
        self.phase = widgets.FloatSlider(
            value=0, min=0, max=360, step=15, readout_format='.0f',
            description='Relative phase ΔΦ (°)', continuous_update=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='520px', max_width='100%'))
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1100px', height='auto'))
        self.ui = widgets.VBox([self.phase, self.readout, self.picture])
        self.figure = Figure(figsize=(11, 4.4), dpi=110, facecolor='white')
        FigureCanvasAgg(self.figure)
        self.figure.subplots_adjust(left=.075, right=.98, bottom=.24,
                                    top=.86, wspace=.36)
        field_axis, intensity_axis = self.figure.subplots(
            1, 2, gridspec_kw={'width_ratios': [1.6, 1]})
        self.axes = (field_axis, intensity_axis)
        blue, orange, ink, grey = '#2866a3', '#b85c17', '#243444', '#727c86'
        self._lines = [field_axis.plot([], [], color=color, lw=width,
                       ls=style, label=label)[0]
                       for color, width, style, label in (
                           (blue, 2.5, '-', 'Beam 1'),
                           (orange, 1.8, '--', 'Beam 2'),
                           (ink, 2.2, '-', 'Sum'))]
        field_axis.axhline(0, color=grey, lw=.8)
        field_axis.set(xlim=(0, 1), ylim=(-2.25, 2.25),
                       xticks=[0, .25, .5, .75, 1],
                       xlabel='Model time t/T at point P',
                       ylabel='Signed field (reference field units)',
                       title='Add the fields at the same point and time')
        field_axis.legend(loc='upper center', bbox_to_anchor=(.5, -.23),
                          ncol=3, frameon=False, fontsize=10)
        self._bars = intensity_axis.bar(
            [0, 1, 2], [1, 1, 4], color=[blue, orange, ink], width=.62)
        self._bar_labels = [intensity_axis.text(
            x, 0, '', ha='center', va='bottom', fontsize=11, color=ink)
            for x in (0, 1, 2)]
        intensity_axis.axhline(2, color=grey, lw=1.1, ls='--',
                               label=r'Individual intensities added: $1+1=2$')
        intensity_axis.set(xlim=(-.6, 2.6), ylim=(0, 4.55),
                           xticks=[0, 1, 2], yticks=[0, 1, 2, 3, 4],
                           xticklabels=['Beam 1\nalone', 'Beam 2\nalone', 'Both\nbeams'],
                           ylabel=r'Relative scalar intensity $I/I_{\rm ref}$',
                           title='Average the squared total field')
        intensity_axis.legend(loc='upper center', bbox_to_anchor=(.5, -.23),
                              frameon=False, fontsize=9)
        for axis in self.axes:
            axis.spines[['top', 'right']].set_visible(False)
            axis.tick_params(labelsize=10)
            axis.grid(axis='y', alpha=.14)
            axis.set_axisbelow(True)
        self.phase.observe(self._update, names='value')
        self._update()

    def _update(self, change=None):
        if self._closed:
            return
        self.data = two_wave_cycle(phase_difference_deg=self.phase.value)
        tau = np.append(self.data['tau'], 1)
        for line, values in zip(self._lines,
                                (*self.data['fields'], self.data['field_sum'])):
            # Close the plotted cycle without duplicating its averaged endpoint.
            line.set_data(tau, np.append(values, values[0]))
        values = (1., 1., self.data['relative_intensity'])
        for bar, label, value in zip(self._bars, self._bar_labels, values):
            bar.set_height(value)
            label.set_position((bar.get_x() + bar.get_width()/2, value + .08))
            label.set_text(f'{value:.2f}')
        cross = self.data['interference_term']
        self.readout.value = (
            f'Equal local amplitudes: A<sub>1</sub> = A<sub>2</sub> = A<sub>ref</sub>. '
            f'ΔΦ = {self.phase.value:.0f}°. '
            f'<b>I/I<sub>ref</sub> = 1 + 1 + ({cross:+.2f}) '
            f'= {self.data["relative_intensity"]:.2f}</b>')
        self.picture.value = figure_png(self.figure)

    def close(self):
        """Detach the control and release the figure and widgets, idempotently."""
        if self._closed:
            return
        self._closed = True
        self.phase.unobserve(self._update, names='value')
        self.picture.value = b''
        self.data = None
        self.figure.clear()
        _close_widget_tree(self.ui)
