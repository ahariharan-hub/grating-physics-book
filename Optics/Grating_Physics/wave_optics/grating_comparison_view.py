"""Linked thin-mask/HBVP views of an already solved, fixed scalar grating."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .grating_intro import _close_widget_tree
from .plotting import figure_png
from .scalar_grating_view import BLUE, ORANGE, INK, CYAN, KINDS, visible_phase, _style
from .thin_grating import ThinPhaseGrating, compare_grating_models


def _phase_text(value):
    return f'{value:+.3f}' if np.isfinite(value) else '—'


def _complex_text(value):
    return f'{value.real:+.5f}{value.imag:+.5f}i'


class GratingComparisonExplorer:
    """Two focused tabs sharing one plane and one existing scattering solution.

    Geometry is fixed. Moving the plane updates the analytical propagation
    factors and field sums, without solving a BVP. Only the visible figure
    is rendered. Figures, axes, artists, x basis and controls are reused.
    Call close() before replacement; no files or saved outputs are created.
    """

    def __init__(self, solution):
        import ipywidgets as widgets
        self.solution = solution
        self.thin_model = ThinPhaseGrating(solution.configuration)
        self.config = c = solution.configuration
        self._closed = self._resetting = False
        self._artists = {}
        self.data = self.field_values = None
        self.x = np.linspace(-c.period/2, c.period/2, max(512, 8*(2*c.M+1)), endpoint=False)
        self.basis = np.exp(1j*solution.kx[:, None]*self.x)
        # Triangle inequality bounds every output plane, including z=-h.
        self.amplitude_limit = 1.05*max(np.sum(abs(solution.transmission)),
                                        np.sum(abs(self.thin_model.transmission)), 1.)
        self.coefficient_limit = 1.1*max(np.max(abs(solution.transmission)),
                                         np.max(abs(self.thin_model.transmission)), .1)
        self.difference_limit = 1.1*max(np.max(abs(solution.transmission-self.thin_model.transmission)), .01)
        style = {'description_width': 'initial'}
        self.z_ref = widgets.FloatSlider(description='Comparison plane z_ref (µm)',
            min=-c.thickness-1.5, max=-c.thickness, value=-c.thickness,
            step=.005, readout_format='.3f', continuous_update=False, style=style,
            layout=widgets.Layout(width='480px', max_width='100%'))
        self.order = widgets.Dropdown(options=[int(m) for m in solution.orders], value=min(1, c.M),
                                     description='Order m:', style=style)
        self.quantity = widgets.Dropdown(options=[('Amplitude |U|', 'amplitude'),
            ('Intensity |U|²', 'intensity'), ('Real field Re U', 'real')],
            value='amplitude', description='Field:', style=style)
        self.reset_button = widgets.Button(description='Reset comparison')
        self.readout, self.table = widgets.HTML(), widgets.HTML()
        self.field_picture, self.coefficient_picture = [widgets.Image(format='png',
            layout=widgets.Layout(width='100%', max_width='1150px', height='auto')) for _ in range(2)]
        self.field_ui = widgets.VBox([self.field_picture])
        self.coefficient_ui = widgets.VBox([self.coefficient_picture, self.table])
        self.tabs = widgets.Tab(children=[self.field_ui, self.coefficient_ui], selected_index=0)
        self.tabs.set_title(0, 'Complex fields')
        self.tabs.set_title(1, 'Diffraction coefficients')
        self._controls = (self.z_ref, self.order, self.quantity)
        self._defaults = [control.value for control in self._controls]
        controls = widgets.Box([*self._controls, self.reset_button],
            layout=widgets.Layout(display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([controls, self.readout, self.tabs])
        self.field_figure = Figure(figsize=(10.5, 3.6), dpi=110, facecolor='white', constrained_layout=True)
        self.coefficient_figure = Figure(figsize=(10.5, 6.6), dpi=110, facecolor='white', constrained_layout=True)
        for figure in (self.field_figure, self.coefficient_figure):
            FigureCanvasAgg(figure)
        self._build_figures()
        for control in self._controls:
            control.observe(self._update, names='value')
        self.tabs.observe(self._update, names='selected_index')
        self.reset_button.on_click(self._reset)
        self._update()

    def _build_figures(self):
        a, c = self._artists, self.config
        field_axes = self.field_figure.subplots(1, 2)
        for name, axis in zip(('field', 'field_phase'), field_axes):
            a[name+'_hbvp'], = axis.plot([], [], color=BLUE, lw=1.8, label='Scalar HBVP')
            a[name+'_thin'], = axis.plot([], [], '--', color=ORANGE, lw=1.6, label='Thin phase mask')
            axis.set(xlabel='x (µm)', xlim=(-c.period/2, c.period/2))
            if name == 'field':
                axis.legend(fontsize=8, loc='upper right')
            _style(axis)
        field_axes[1].set(title='Wrapped field phase', ylabel='arg U (rad)', ylim=(-np.pi-.2, np.pi+.2),
                          yticks=[-np.pi, 0, np.pi], yticklabels=['−π', '0', 'π'])
        magnitude, phase, argand, difference = self.coefficient_figure.subplots(2, 2).flat
        m = self.solution.orders
        for name, axis in (('magnitude', magnitude), ('phase', phase)):
            a[name+'_hbvp'], = axis.plot(m, np.zeros(m.size), 'o', color=BLUE, ms=4, label='Scalar HBVP')
            a[name+'_thin'], = axis.plot(m, np.zeros(m.size), 'x', color=ORANGE, ms=5, label='Thin phase mask')
            if name == 'magnitude':
                axis.legend(fontsize=8, loc='upper right')
        a['difference'], = difference.plot(m, np.zeros(m.size), 'o', color=INK, ms=4)
        magnitude.set(title='Coefficient magnitude', ylabel='|Cₘ|', ylim=(0, self.coefficient_limit))
        phase.set(title='Wrapped coefficient phase', ylabel='arg Cₘ (rad)', ylim=(-np.pi-.2, np.pi+.2),
                  yticks=[-np.pi, 0, np.pi], yticklabels=['−π', '0', 'π'])
        difference.set(title='Magnitude of the complex difference', ylabel='|ΔCₘ|', ylim=(0, self.difference_limit))
        a['selected_markers'] = []
        for axis in (magnitude, phase, difference):
            axis.set(xlabel='Order m', xlim=(-c.M-.75, c.M+.75))
            a['selected_markers'].append(axis.axvline(0, color=CYAN, lw=1, alpha=.6))
            for order, kind in zip(m, self.solution.output_kinds):
                axis.plot(order, .025, 's', color=KINDS[kind], ms=3,
                          transform=axis.get_xaxis_transform())
            _style(axis)
        argand.axhline(0, color=INK, alpha=.2, lw=.8)
        argand.axvline(0, color=INK, alpha=.2, lw=.8)
        a['argand_hbvp'], = argand.plot([], [], '-o', color=BLUE, ms=6, label='Scalar HBVP')
        a['argand_thin'], = argand.plot([], [], '--x', color=ORANGE, ms=7, label='Thin phase mask')
        a['delta_arrow'] = argand.annotate('', xy=(0, 0), xytext=(0, 0),
                                         arrowprops={'arrowstyle': '->', 'color': CYAN, 'lw': 2})
        a['argand_axis'] = argand
        argand.set(xlabel='Re Cₘ', ylabel='Im Cₘ', title='Selected order in the complex plane',
                   xlim=(-self.coefficient_limit, self.coefficient_limit),
                   ylim=(-self.coefficient_limit, self.coefficient_limit), aspect='equal')
        argand.legend(fontsize=8, loc='upper right')
        _style(argand)

    def _update(self, change=None):
        if self._closed or self._resetting:
            return
        owner = None if change is None else change.get('owner')
        plane_changed = self.data is None or owner is self.z_ref or change is None
        if plane_changed:
            self.data = compare_grating_models(self.solution, self.z_ref.value, thin_model=self.thin_model)
            self.field_values = {'hbvp': self.data.hbvp @ self.basis, 'thin': self.data.thin @ self.basis}
        a, p = self._artists, self.data
        if plane_changed or owner is self.quantity:
            for model, values in self.field_values.items():
                quantity = self.quantity.value
                field = values.real if quantity == 'real' else abs(values)**(2 if quantity == 'intensity' else 1)
                a['field_'+model].set_data(self.x, field)
                a['field_phase_'+model].set_data(self.x, visible_phase(values, break_wraps=True))
            label, limits = {'amplitude': ('|U|', (0, self.amplitude_limit)),
                             'intensity': ('|U|²', (0, self.amplitude_limit**2)),
                             'real': ('Re U', (-self.amplitude_limit, self.amplitude_limit))}[self.quantity.value]
            a['field_hbvp'].axes.set(title='Same output plane: '+label, ylabel=label, ylim=limits)
        if plane_changed:
            for model in ('hbvp', 'thin'):
                a['magnitude_'+model].set_ydata(abs(getattr(p, model)))
                a['phase_'+model].set_ydata(getattr(p, model+'_phase'))
            a['difference'].set_ydata(abs(p.difference))
        if plane_changed or owner is self.order:
            i = self.order.value+self.config.M
            for model in ('hbvp', 'thin'):
                value = getattr(p, model)[i]
                a['argand_'+model].set_data([0, value.real], [0, value.imag])
            a['delta_arrow'].xy = (p.hbvp[i].real, p.hbvp[i].imag)
            a['delta_arrow'].set_position((p.thin[i].real, p.thin[i].imag))
            a['argand_axis'].set_title(f'm = {self.order.value:+d}: {p.kinds[i]}\nCyan arrow: thin → HBVP (ΔCₘ)', fontsize=10)
            for marker in a['selected_markers']:
                marker.set_xdata([self.order.value]*2)
            self._update_table()
        self.quantity.disabled = self.tabs.selected_index == 1
        self.order.disabled = self.tabs.selected_index == 0
        self.readout.value = (f'<b>z_ref = {p.z_ref:.3f} µm; {2*self.config.M+1} retained harmonics.</b> '
            'Both models use unit illumination at z=0 and the same absolute phase reference. '
            'Blue: HBVP; orange: thin mask. Phase gaps hide wraps or amplitudes below 10⁻⁶. '
            'Fixed scales; moving the plane does not solve the grating again.')
        if self.tabs.selected_index == 0:
            self.field_picture.value = figure_png(self.field_figure)
        else:
            self.coefficient_picture.value = figure_png(self.coefficient_figure)

    def _update_table(self):
        p = self.data
        selected = self.order.value+self.config.M
        rows = []
        for i, m in enumerate(p.orders):
            if abs(m) > 4 and i != selected:
                continue
            values = (f'{m:+d}', p.kinds[i], f'{abs(p.thin[i]):.5f}', f'{abs(p.hbvp[i]):.5f}',
                      _phase_text(p.thin_phase[i]), _phase_text(p.hbvp_phase[i]),
                      _complex_text(p.difference[i]), f'{p.thin_efficiency[i]:.6f}', f'{p.hbvp_efficiency[i]:.6f}')
            weight = 'font-weight:bold;' if i == selected else ''
            rows.append(f'<tr style="{weight}">'+''.join(
                f'<td style="padding:4px 8px;white-space:nowrap;text-align:right">{value}</td>' for value in values)+'</tr>')
        headers = ('m', 'Type', '|C thin|', '|C HBVP|', 'arg thin', 'arg HBVP', 'ΔC = HBVP − thin', 'η thin', 'η HBVP')
        self.table.value = ('<p>Order markers: blue = propagating, orange = evanescent, purple = grazing. '
            'Showing |m| ≤ 4 plus the selected order. Phases in radians; — means too small to report.</p>'
            '<div style="overflow-x:auto"><table style="font-size:12px"><thead><tr>'+''.join(
                f'<th style="padding:4px 8px;white-space:nowrap">{h}</th>' for h in headers)+
            '</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
            f'<p><b>Selected m = {self.order.value:+d}:</b> ΔC = {_complex_text(p.difference[selected])}; '
            f'Δφ = {_phase_text(p.phase_difference[selected])} rad.</p>'
            f'<p><b>All-order scalar flux:</b> thin T = {p.thin_transmittance:.6f}; '
            f'HBVP R = {p.hbvp_reflectance:.6f}, T = {p.hbvp_transmittance:.6f}, '
            f'R + T = {p.hbvp_power_balance:.12f}. '
            'The thin-model deficit is not absorption or calculated reflection.</p>')

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
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.tabs.unobserve(self._update, names='selected_index')
        self.reset_button.on_click(self._reset, remove=True)
        self.field_picture.value = self.coefficient_picture.value = b''
        self.table.value = self.readout.value = ''
        for figure in (self.field_figure, self.coefficient_figure):
            figure.clear()
        self.field_figure = self.coefficient_figure = None
        self.solution = self.thin_model = self.data = self.field_values = self.basis = self.x = None
        self._artists.clear()
        _close_widget_tree(self.ui)
