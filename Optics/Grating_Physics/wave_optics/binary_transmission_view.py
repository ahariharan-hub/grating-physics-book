"""Ideal amplitude/phase transmission lesson, separate from dielectric HBVP views."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .binary_transmission import (
    binary_transmission_coefficients, binary_transmission_field, dielectric_phase_levels,
)
from .grating_intro import _close_widget_tree
from .plotting import figure_png
from .scalar_grating import ScalarGratingConfiguration
from .scalar_grating_view import BLUE, ORANGE, INK, CYAN, visible_phase, _style


class BinaryTransmissionExplorer:
    """Fixed geometry, two prescribed local masks and selectable Fourier truncation.

    The phase mask uses the same physical indices, thickness and absolute
    exit phase as ThinPhaseGrating. The amplitude mask is a different ideal
    device, with levels 1 and 0. Neither calls or modifies a scattering solver.
    Coefficients, exact masks and the Fourier basis are cached; order selection
    reuses the reconstructed fields. close() releases all owned widgets.
    """

    def __init__(self, configuration=None):
        import ipywidgets as widgets
        self.config = c = ScalarGratingConfiguration(M=20) if configuration is None else configuration
        self.levels = {'amplitude': (1.+0j, 0j), 'phase': dielectric_phase_levels(c)}
        self._closed = self._resetting = False
        self._artists = {}
        self.all_orders = np.arange(-c.M, c.M+1)
        self.x = np.unique(np.r_[np.linspace(-c.period/2, c.period/2, max(513, 8*(2*c.M+1)+1)),
                                 -c.duty_cycle*c.period/2, c.duty_cycle*c.period/2])
        self.basis = np.exp(2j*np.pi*self.all_orders[:, None]*self.x/c.period)
        self.exact = {name: binary_transmission_field(self.x, c.period, c.duty_cycle, *levels)
                      for name, levels in self.levels.items()}
        self.coefficients = {name: binary_transmission_coefficients(self.all_orders, c.duty_cycle, *levels)
                             for name, levels in self.levels.items()}
        self.mean_square = {name: c.duty_cycle*abs(tr)**2+(1-c.duty_cycle)*abs(tg)**2
                            for name, (tr, tg) in self.levels.items()}
        # Bound every selectable partial sum on the actual displayed x grid.
        peak = 1.
        for values in self.coefficients.values():
            partial = values[c.M]*self.basis[c.M].copy()
            peak = max(peak, float(np.max(abs(partial))))
            for m in range(1, c.M+1):
                partial += values[c.M-m]*self.basis[c.M-m]+values[c.M+m]*self.basis[c.M+m]
                peak = max(peak, float(np.max(abs(partial))))
        self.amplitude_limit = 1.06*peak
        style = {'description_width': 'initial'}
        self.model = widgets.Dropdown(options=[('Both ideal masks', 'both'), ('Amplitude mask', 'amplitude'),
                                               ('Dielectric phase mask', 'phase')], value='both',
                                     description='Show:', style=style)
        self.truncation = widgets.IntSlider(value=c.M, min=0, max=c.M, step=1,
            description='Retain −M…M:', continuous_update=False, style=style,
            layout=widgets.Layout(width='300px'))
        self.order = widgets.Dropdown(options=[int(m) for m in self.all_orders], value=min(1, c.M),
                                     description='Order m:', style=style)
        self.show_sum = widgets.Checkbox(value=False, description='Show finite Fourier sum', indent=False)
        self.reset_button = widgets.Button(description='Reset ideal masks')
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1250px'))
        self._controls = (self.model, self.truncation, self.order, self.show_sum)
        controls = widgets.Box([*self._controls, self.reset_button],
            layout=widgets.Layout(display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([controls, self.readout, self.picture])
        self.figure = Figure(figsize=(12, 6.8), dpi=110, facecolor='white', constrained_layout=True)
        self.figure.get_layout_engine().set(rect=(0, 0, 1, .92))
        FigureCanvasAgg(self.figure)
        self._build_figure()
        self.data = None
        for control in self._controls:
            control.observe(self._update, names='value')
        self.reset_button.on_click(self._reset)
        self._update()

    def _build_figure(self):
        a, c = self._artists, self.config
        amplitude, phase, argand, cmag, cphase, notes = self.figure.subplots(2, 3).flat
        header = self.figure.suptitle('Blue: amplitude mask   •   Orange: dielectric phase mask\n'
                            'Solid: exact local transmission   •   Dashed: finite Fourier reconstruction',
                            fontsize=10, y=.99)
        header.set_in_layout(False)
        a['argand_axis'], a['notes'] = argand, notes
        for name, color in (('amplitude', BLUE), ('phase', ORANGE)):
            for key, axis in (('ideal_amplitude', amplitude), ('ideal_phase', phase)):
                a[key+'_'+name], = axis.plot([], [], color=color, lw=1.8, drawstyle='steps-post')
            for key, axis in (('sum_amplitude', amplitude), ('sum_phase', phase)):
                a[key+'_'+name], = axis.plot([], [], '--', color=color, lw=1.3, alpha=.85)
            for key, axis in (('coefficient_amplitude', cmag), ('coefficient_phase', cphase)):
                a[key+'_'+name], = axis.plot([], [], 'o' if name == 'amplitude' else 'x', color=color, ms=4)
            a['argand_'+name], = argand.plot([], [], '-o' if name == 'amplitude' else '--x', color=color, ms=6)
        amplitude.set(title='Transmission amplitude', ylabel='|t(x)|', ylim=(0, self.amplitude_limit))
        phase.set(title='Transmission phase', ylabel='arg t(x) (rad)')
        for axis in (amplitude, phase):
            axis.set(xlabel='x (µm)', xlim=(-c.period/2, c.period/2))
            for boundary in (-c.duty_cycle*c.period/2, c.duty_cycle*c.period/2):
                axis.axvline(boundary, color=INK, lw=.7, alpha=.2)
        cmag.set(title='Retained coefficient magnitudes', ylabel='|Cₘ|', ylim=(0, 1.06))
        cphase.set(title='Retained coefficient phases', ylabel='arg Cₘ (rad)')
        a['selection'] = []
        for axis in (cmag, cphase):
            axis.set(xlabel='Order m', xlim=(-c.M-.75, c.M+.75))
            a['selection'].append(axis.axvline(0, color=CYAN, alpha=.6, lw=1))
        for axis in (phase, cphase):
            axis.set(ylim=(-np.pi-.2, np.pi+.2), yticks=[-np.pi, 0, np.pi], yticklabels=['−π', '0', 'π'])
        argand.set(xlabel='Re Cₘ', ylabel='Im Cₘ', xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect='equal')
        argand.axhline(0, color=INK, alpha=.2, lw=.8)
        argand.axvline(0, color=INK, alpha=.2, lw=.8)
        for axis in (amplitude, phase, cmag, cphase, argand):
            _style(axis)
        notes.set_axis_off()
        a['normalization'] = notes.text(0, 1, '', va='top', fontsize=9, linespacing=1.25,
                                        transform=notes.transAxes)

    def _update(self, change=None):
        if self._closed or self._resetting:
            return
        owner = None if change is None else change.get('owner')
        c, a = self.config, self._artists
        if self.data is None or owner is self.truncation or change is None:
            M = self.truncation.value
            selected = int(np.clip(self.order.value, -M, M))
            self._resetting = True
            try:
                self.order.options = list(range(-M, M+1))
                self.order.value = selected
            finally:
                self._resetting = False
            keep = abs(self.all_orders) <= M
            coefficients = {name: values[keep] for name, values in self.coefficients.items()}
            sums = {name: values @ self.basis[keep] for name, values in coefficients.items()}
            self.data = dict(orders=self.all_orders[keep], coefficients=coefficients, reconstruction=sums,
                             retained_square={name: float(np.sum(abs(values)**2)) for name, values in coefficients.items()})
        p = self.data
        i = self.order.value+self.truncation.value
        notes = ['Fourier normalization (unit input)', '']
        for name in ('amplitude', 'phase'):
            show = self.model.value in ('both', name)
            exact, reconstructed, coefficients = self.exact[name], p['reconstruction'][name], p['coefficients'][name]
            values = {'ideal_amplitude': (self.x, abs(exact)),
                      'ideal_phase': (self.x, visible_phase(exact, break_wraps=True)),
                      'sum_amplitude': (self.x, abs(reconstructed)),
                      'sum_phase': (self.x, visible_phase(reconstructed, break_wraps=True)),
                      'coefficient_amplitude': (p['orders'], abs(coefficients)),
                      'coefficient_phase': (p['orders'], visible_phase(coefficients)),
                      'argand': ([0, coefficients[i].real], [0, coefficients[i].imag])}
            for key, (x, y) in values.items():
                a[key+'_'+name].set_data(x, y)
                a[key+'_'+name].set_visible(show and (not key.startswith('sum_') or self.show_sum.value))
            if show:
                value = coefficients[i]
                phase = visible_phase(np.array([value]))[0]
                phase_text = f'{phase:+.3f} rad' if np.isfinite(phase) else 'undefined / negligible'
                notes.extend([f'{name.capitalize()} mask',
                              f'Exact mean |t|²: {self.mean_square[name]:.6f}',
                              f'Retained sum |Cₘ|²: {p["retained_square"][name]:.6f}',
                              f'C(m={self.order.value:+d}) = {value.real:+.4f}{value.imag:+.4f}i',
                              f'Phase: {phase_text}', ''])
        notes.append('These squared norms are not\nlongitudinal-flux efficiencies.')
        a['normalization'].set_text('\n'.join(notes))
        a['argand_axis'].set_title(f'Selected coefficient: m = {self.order.value:+d}', fontsize=10)
        for marker in a['selection']:
            marker.set_xdata([self.order.value]*2)
        self.readout.value = (f'<b>Two ideal devices; fixed d = {c.period:g} µm and f = {c.duty_cycle:g}.</b> '
            f'The sum retains {2*self.truncation.value+1} harmonics; solid curves remain the exact local models. '
            'Phase is absent below amplitude 10⁻⁶, including opaque regions and zero coefficients. '
            'The phase-mask levels use exp(i k₀ n h); only that dielectric model enters the HBVP comparison below.')
        self.picture.value = figure_png(self.figure)

    def _reset(self, button=None):
        if self._closed:
            return
        self._resetting = True
        try:
            self.model.value, self.truncation.value, self.show_sum.value = 'both', self.config.M, False
            self.order.options = [int(m) for m in self.all_orders]
            self.order.value = min(1, self.config.M)
        finally:
            self._resetting = False
        self._update()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.reset_button.on_click(self._reset, remove=True)
        self.picture.value = b''
        self.figure.clear()
        self.figure = self.data = self.basis = self.x = self.exact = self.coefficients = None
        self._artists.clear()
        _close_widget_tree(self.ui)
