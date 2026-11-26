"""Two linked views of an ideal amplitude mask and a physical scalar absorber."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .amplitude_grating import (
    AbsorbingComparisonData, ideal_amplitude_object_field, field_fourier_coefficients, FIELD_FFT_ATOL,
)
from .grating_intro import _close_widget_tree
from .grating_comparison_view import _phase_text, _complex_text
from .plotting import figure_png
from .scalar_grating_view import BLUE, ORANGE, INK, CYAN, visible_phase, _style, _order_display_mask


FFT_COLOR = '#25824b'


class AbsorbingGratingExplorer:
    """Bounded precomputed thickness cases; plane/order controls never re-solve.

    Geometry and field images are cached. Existing artists and axes are reused;
    only the visible main tab and an expanded thickness plot are rendered.
    No output files are written. Call close() before replacing the explorer.
    display_order_limit restricts coefficient plots and selection, keeping all
    solved harmonics in the fields, Fourier check and power calculations.
    """

    def __init__(self, data=None, *, display_order_limit=None):
        import ipywidgets as widgets
        self.data = AbsorbingComparisonData() if data is None else data
        self.config = c = self.data.configuration
        self._display_mask = _order_display_mask(self.data.solutions[0].orders, display_order_limit)
        self.display_orders = self.data.solutions[0].orders[self._display_mask]
        self._closed = self._resetting = False
        self._artists = {}
        self.x = np.linspace(-c.period/2, c.period/2, max(512, 4*(2*c.M+1)), endpoint=False)
        self.basis = np.exp(1j*self.data.solutions[0].kx[:, None]*self.x)
        self.map_x = np.linspace(-c.period/2, c.period/2, 256)
        self.map_z = np.linspace(-max(self.data.thicknesses)-1., .25, 360)
        self.maps = tuple(abs(s.reconstruct(self.map_x, self.map_z))**2 for s in self.data.solutions)
        self.materials = tuple(s.configuration.refractive_index(self.map_x, self.map_z).imag for s in self.data.solutions)
        self.planes = self.plane = self.fields = None
        self.fft_coefficients = None
        self.fft_error = None
        self.coefficient_limit = 1.12*max(.1, *(np.max(abs(s.transmission)) for s in self.data.solutions),
                                        *(np.max(abs(a.transmission)) for a in self.data.ideal_models))
        self.difference_limit = 1.12*max(.01, *(np.max(abs(s.transmission-a.transmission))
            for s, a in zip(self.data.solutions, self.data.ideal_models)))
        self.efficiency_limit = 1.12*max(.01, *(np.max(s.transmission_efficiency) for s in self.data.solutions),
                                        *(np.max(a.transmission_efficiency) for a in self.data.ideal_models))
        self.amplitude_limit = 1.05*max(1., *(np.sum(abs(s.transmission)) for s in self.data.solutions),
                                      *(np.sum(abs(a.transmission)) for a in self.data.ideal_models))
        style = {'description_width': 'initial'}
        self.thickness = widgets.SelectionSlider(options=[(f'{h:.2f}', i) for i, h in enumerate(self.data.thicknesses)],
            value=int(np.argmin(abs(self.data.thicknesses-c.thickness))), description='Thickness h (µm)',
            continuous_update=False, style=style, layout=widgets.Layout(width='370px', max_width='100%'))
        self.distance = widgets.FloatSlider(value=0., min=0., max=1., step=.01, readout_format='.2f',
            description='Distance below exit (µm)', continuous_update=False, style=style,
            layout=widgets.Layout(width='390px', max_width='100%'))
        self.order = widgets.Dropdown(options=[int(m) for m in self.display_orders], value=min(1, int(self.display_orders[-1])),
                                     description='Order m', style=style)
        self.quantity = widgets.Dropdown(options=[('Amplitude |U|', 'amplitude'), ('Intensity |U|²', 'intensity'),
            ('Real field Re U', 'real'), ('Imaginary field Im U', 'imag'), ('Wrapped phase arg U', 'phase')],
            value='amplitude', description='Profile', style=style)
        self.x_position = widgets.SelectionSlider(options=[(f'{x:+.4f}', float(x)) for x in self.x],
            value=float(self.x[len(self.x)//2]), description='Position x (µm)', continuous_update=False,
            style=style, layout=widgets.Layout(width='350px', max_width='100%'))
        self.compare_ideal = widgets.Checkbox(value=False, description='Compare with ideal mask', indent=False,
            layout=widgets.Layout(width='240px'))
        self.reset_button = widgets.Button(description='Reset absorber')
        self.readout, self.table, self.fft_readout = widgets.HTML(), widgets.HTML(), widgets.HTML()
        self.pictures = [widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1200px', height='auto')) for _ in range(3)]
        self.sweep = widgets.Accordion(children=[widgets.VBox([widgets.HTML(
            '<b>Thickness experiment.</b> Each point is a separately solved layer. '
            'The selected order and distance below its exit are shared with the plots above. '
            'Lines join sampled cases; phase uses separate points to avoid joining wraps.'), self.pictures[2]])], selected_index=None)
        self.sweep.set_title(0, 'Does increasing thickness approach the ideal mask?')
        self.tabs = widgets.Tab(children=[widgets.VBox([self.pictures[0]]),
            widgets.VBox([self.fft_readout, self.pictures[1], self.table, self.sweep])], selected_index=0)
        self.tabs.set_title(0, 'A · Structure and complex field')
        self.tabs.set_title(1, 'B · Field → coefficients')
        self._controls = (self.thickness, self.distance, self.order, self.quantity, self.x_position, self.compare_ideal)
        self._defaults = [control.value for control in self._controls]
        controls = widgets.Box([*self._controls, self.reset_button],
            layout=widgets.Layout(display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([controls, self.readout, self.tabs])
        self.figures = [Figure(figsize=(11.3, height), dpi=105, facecolor='white', constrained_layout=True)
                        for height in (7.6, 7.5, 6.1)]
        for figure in self.figures:
            FigureCanvasAgg(figure)
        self._build_figures()
        for control in self._controls:
            control.observe(self._update, names='value')
        self.tabs.observe(self._update, names='selected_index')
        self.sweep.observe(self._update, names='selected_index')
        self.reset_button.on_click(self._reset)
        self._update()

    def _build_figures(self):
        a, c = self._artists, self.config
        axes = self.figures[0].subplots(2, 3)
        geometry, field, mask, profile, phase, field_argand = axes.flat
        extent = [self.map_x[0], self.map_x[-1], self.map_z[0], self.map_z[-1]]
        a['material'] = geometry.imshow(self.materials[0], origin='lower', extent=extent, aspect='auto',
                                         cmap='Greys', vmin=0, vmax=max(c.kappa, .01))
        a['physical_field'] = field.imshow(self.maps[0], origin='lower', extent=extent, aspect='auto',
            cmap='magma', vmin=0, vmax=max(1., *(float(np.max(v)) for v in self.maps)))
        self.figures[0].colorbar(a['physical_field'], ax=field, label='|U HBVP|²', shrink=.85)
        geometry.set(title='Air opening in an absorbing layer', xlabel='x (µm)', ylabel='z (µm)')
        field.set(title='Physical field: incident + scattered', xlabel='x (µm)', ylabel='z (µm)')
        geometry.annotate('Incident −z', xy=(0, .01), xytext=(0, .18), ha='center', fontsize=9,
                          arrowprops={'arrowstyle': '->', 'color': BLUE}, color=BLUE)
        a['plane_lines'], a['exit_lines'] = [], []
        for axis in (geometry, field):
            a['plane_lines'].append(axis.axhline(-c.thickness, color=CYAN, lw=1.5))
            a['exit_lines'].append(axis.axhline(-c.thickness, color='gray', ls='--', lw=1))
            axis.axhline(0, color='gray', ls='--', lw=1)
        a['mask'], = mask.plot(self.x, abs(ideal_amplitude_object_field(c, self.x)), color=ORANGE, lw=2)
        mask.set(title='Ideal transmission at object plane z = 0', xlabel='x (µm)', ylabel='|t ideal(x)|', ylim=(-.08, 1.15))
        for name, axis in (('profile', profile), ('field_phase', phase)):
            a[name+'_hbvp'], = axis.plot([], [], color=BLUE, label='Physical absorber')
            a[name+'_thin'], = axis.plot([], [], '--', color=ORANGE, label='Ideal: propagated from z = 0')
            axis.set(xlabel='x (µm)', xlim=(-c.period/2, c.period/2))
            a[name+'_position'] = axis.axvline(0, color=CYAN, alpha=.6, lw=1)
            a[name+'_point'], = axis.plot([], [], 'o', color=CYAN, mec=INK, ms=7, zorder=5)
        profile.legend(fontsize=8, loc='upper right')
        phase.set(title='Wrapped phase on the same plane', ylabel='arg U (rad)', ylim=(-3.4, 3.4),
                  yticks=[-np.pi, 0, np.pi], yticklabels=['−π', '0', 'π'])
        a['field_argand_thin'], = field_argand.plot([], [], '--', color=ORANGE, alpha=.55, lw=1,
                                                   label='Ideal: propagated from z = 0')
        a['field_argand'] = field_argand.scatter(np.zeros(self.x.size), np.zeros(self.x.size),
            c=self.x, cmap='viridis', vmin=-c.period/2, vmax=c.period/2, s=9, linewidths=0, zorder=3)
        self.figures[0].colorbar(a['field_argand'], ax=field_argand, label='HBVP position x (µm)', shrink=.8)
        a['field_argand_point'], = field_argand.plot([], [], 'o', mfc='none', mec=CYAN, mew=2, ms=11, zorder=5)
        field_argand.axhline(0, color=INK, alpha=.2)
        field_argand.axvline(0, color=INK, alpha=.2)
        field_argand.set(title='Local complex field U(x)', xlabel='Re U', ylabel='Im U', aspect='equal',
            xlim=(-self.amplitude_limit, self.amplitude_limit), ylim=(-self.amplitude_limit, self.amplitude_limit))
        a['field_argand_legend'] = field_argand.legend(fontsize=7, loc='upper right')
        for axis in (mask, profile, phase, field_argand):
            _style(axis)

        coefficient_figure = self.figures[1]
        coefficient_figure.suptitle(
            'Order type below each axis:  P = propagating   E = evanescent   G = grazing',
            fontsize=11, color=INK)
        grid = coefficient_figure.add_gridspec(2, 3)
        coefficient_axes, type_axes = [], []
        for i in range(6):
            if i == 2:  # The Argand diagram has no order-number axis.
                coefficient_axes.append(coefficient_figure.add_subplot(grid[i // 3, i % 3]))
                continue
            panels = grid[i // 3, i % 3].subgridspec(2, 1, height_ratios=(1, .07))
            coefficient_axes.append(coefficient_figure.add_subplot(panels[0]))
            type_axes.append(coefficient_figure.add_subplot(panels[1]))
        magnitude, phase, argand, difference, efficiency, delta_phase = coefficient_axes
        m = self.display_orders
        kinds = self.data.solutions[0].output_kinds[self._display_mask]
        for axis in type_axes:
            # A separate categorical row cannot be read as coefficient data.
            axis.set(xlim=(m[0]-.75, m[-1]+.75), ylim=(0, 1), xticks=[], yticks=[])
            axis.set_ylabel('Type', rotation=0, ha='right', va='center', fontsize=8, color=INK)
            for spine in axis.spines.values():
                spine.set_visible(False)
            for order, kind in zip(m, kinds):
                axis.axvspan(order-.5, order+.5, color={
                    'propagating': '#dcece7', 'evanescent': '#f0f0f0', 'grazing': '#e8e0ef',
                }[kind], ec='white', linewidth=.5)
                axis.text(order, .5, kind[0].upper(), ha='center', va='center',
                          fontsize=8 if len(m) <= 15 else 6, color=INK)
        a['selected_markers'] = []
        coefficient_labels = {
            'hbvp': 'Absorber: direct HBVP',
            'thin': 'Ideal mask: from z = 0',
            'fft': 'FFT of same HBVP field\n'
                   r'$U_{\mathrm{HBVP}}(x,z_{\mathrm{ref}})$',
        }
        for name, axis, title, label, limits in (
            ('magnitude', magnitude, 'Coefficient magnitude', '|Cₘ|', (0, self.coefficient_limit)),
            ('phase', phase, 'Wrapped coefficient phase', 'arg Cₘ (rad)', (-3.4, 3.4)),
            ('efficiency', efficiency, 'Transmitted scalar flux', 'ηₘ = Re(βₘ) |Cₘ|² / k₀', (0, self.efficiency_limit))):
            for key, color, marker in [('hbvp', BLUE, 'o'), ('thin', ORANGE, 'x')]:
                a[name+'_'+key], = axis.plot(m, np.zeros(m.size), marker, color=color, ms=4,
                                            label=coefficient_labels[key])
            axis.set(title=title, ylabel=label, ylim=limits)
        for name, axis in [('magnitude', magnitude), ('phase', phase)]:
            a[name+'_fft'], = axis.plot(m, np.zeros(m.size), 's', mfc='none', mec=FFT_COLOR, ms=7,
                                        label=coefficient_labels['fft'])
        # Keep the explanatory legend above even the largest coefficient.
        magnitude.set_ylim(0, 1.35*self.coefficient_limit)
        magnitude.legend(fontsize=8, loc='upper left')
        a['difference'], = difference.plot(m, np.zeros(m.size), 'o', color=INK, ms=4)
        a['delta_phase'], = delta_phase.plot(m, np.zeros(m.size), 'o', color=INK, ms=4)
        difference.set(title='Absolute complex difference', ylabel='|C HBVP − C ideal|', ylim=(0, self.difference_limit))
        delta_phase.set(title='Relative phase: both signals significant', ylabel='Δφₘ (rad)', ylim=(-3.4, 3.4))
        for axis in (magnitude, phase, difference, efficiency, delta_phase):
            axis.set(xlabel='Order m', xlim=(m[0]-.75, m[-1]+.75))
            if len(m) <= 15:
                axis.set_xticks(m)
            a['selected_markers'].append(axis.axvline(0, color=CYAN, alpha=.5, lw=1))
            _style(axis)
        for key, color, marker in [('hbvp', BLUE, '-o'), ('thin', ORANGE, '--x')]:
            a['argand_'+key], = argand.plot([], [], marker, color=color, ms=6, label=coefficient_labels[key])
        a['argand_fft'], = argand.plot([], [], 's', mfc='none', mec=FFT_COLOR, ms=9,
                                      label=coefficient_labels['fft'])
        argand.legend(fontsize=7, loc='upper right')
        argand.axhline(0, color=INK, alpha=.2)
        argand.axvline(0, color=INK, alpha=.2)
        argand.set(xlabel='Re Cₘ', ylabel='Im Cₘ', aspect='equal',
                   xlim=(-self.coefficient_limit, self.coefficient_limit), ylim=(-self.coefficient_limit, self.coefficient_limit))
        a['arrow'] = argand.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops={'arrowstyle': '->', 'color': CYAN, 'lw': 2})
        _style(argand)

        mag, phase, power, error = self.figures[2].subplots(2, 2).flat
        for name, axis in [('sweep_magnitude', mag), ('sweep_phase', phase)]:
            for key, color, line, legend in [('hbvp', BLUE, '-o', 'Physical absorber'), ('thin', ORANGE, '--x', 'Ideal amplitude mask')]:
                a[name+'_'+key], = axis.plot([], [], line, color=color, ms=4, label=legend)
                if name == 'sweep_phase':
                    a[name+'_'+key].set_linestyle('none')
        mag.set(title='Selected coefficient magnitude', ylabel='|Cₘ|', ylim=(0, self.coefficient_limit))
        mag.legend(fontsize=8)
        phase.set(title='Absolute phase at z = −h − distance', ylabel='arg Cₘ (rad)', ylim=(-3.4, 3.4))
        for name, color in [('R', INK), ('T', BLUE), ('A', ORANGE)]:
            a['power_'+name], = power.plot([], [], '-o', color=color, ms=4, label=name)
        power.set(title='Physical scalar power fractions', ylabel='Fraction of incident flux', ylim=(0, 1))
        power.legend(fontsize=8)
        a['sweep_error'], = error.plot([], [], '-o', color=INK, ms=4)
        error.set(title='Difference from ideal: selected order', ylabel='|ΔCₘ|', ylim=(0, self.difference_limit))
        a['thickness_markers'] = []
        for axis in (mag, phase, power, error):
            margin = max(.01, .03*np.ptp(self.data.thicknesses))
            axis.set(xlabel='Absorber thickness h (µm)',
                     xlim=(min(self.data.thicknesses)-margin, max(self.data.thicknesses)+margin))
            a['thickness_markers'].append(axis.axvline(c.thickness, color=CYAN, alpha=.5))
            _style(axis)

    def _update(self, change=None):
        if self._closed or self._resetting:
            return
        owner = None if change is None else change.get('owner')
        plane_changed = self.planes is None or owner is self.distance or change is None
        if plane_changed:
            self.planes = self.data.planes(self.distance.value)
        index = self.thickness.value
        self.plane = p = self.planes[index]
        solution = self.data.solutions[index]
        h = solution.configuration.thickness
        a = self._artists
        if change is None or owner is self.compare_ideal:
            for name in ('profile_thin', 'field_phase_thin', 'field_argand_thin'):
                a[name].set_visible(self.compare_ideal.value)
            handles = [a['profile_hbvp']]
            if self.compare_ideal.value:
                handles.append(a['profile_thin'])
            a['profile_hbvp'].axes.legend(handles=handles, fontsize=8, loc='upper right')
            a['field_argand_legend'].set_visible(self.compare_ideal.value)
        if plane_changed or owner is self.thickness:
            self.fields = {key: getattr(p, key) @ self.basis for key in ('thin', 'hbvp')}
            self.fft_coefficients = field_fourier_coefficients(self.fields['hbvp'], self.x, self.config.period, p.orders)
            self.fft_error = float(np.max(abs(self.fft_coefficients-p.hbvp)))
            a['field_argand'].set_offsets(np.column_stack((self.fields['hbvp'].real, self.fields['hbvp'].imag)))
            a['field_argand_thin'].set_data(self.fields['thin'].real, self.fields['thin'].imag)
            a['material'].set_data(self.materials[index])
            a['physical_field'].set_data(self.maps[index])
            for line in a['exit_lines']:
                line.set_ydata([-h]*2)
            for line in a['plane_lines']:
                line.set_ydata([p.z_ref]*2)
        q = self.quantity.value
        self.point_index = point = int(np.searchsorted(self.x, self.x_position.value))
        for key, u in self.fields.items():
            values = {'real': u.real, 'imag': u.imag, 'amplitude': abs(u), 'intensity': abs(u)**2,
                      'phase': visible_phase(u, break_wraps=True)}[q]
            a['profile_'+key].set_data(self.x, values)
            a['field_phase_'+key].set_data(self.x, visible_phase(u, break_wraps=True))
            a['magnitude_'+key].set_ydata(abs(getattr(p, key)[self._display_mask]))
            a['phase_'+key].set_ydata(getattr(p, key+'_phase')[self._display_mask])
            a['efficiency_'+key].set_ydata(getattr(p, key+'_efficiency')[self._display_mask])
            if key == 'hbvp':
                local_phase = visible_phase(u[point:point+1])[0]
                a['profile_point'].set_data([self.x[point]], [local_phase if q == 'phase' else values[point]])
                a['field_phase_point'].set_data([self.x[point]], [local_phase])
                a['field_argand_point'].set_data([u[point].real], [u[point].imag])
        for name in ('profile', 'field_phase'):
            a[name+'_position'].set_xdata([self.x[point]]*2)
        label, limits = {'amplitude': ('|U|', (0, self.amplitude_limit)),
            'intensity': ('|U|²', (0, self.amplitude_limit**2)), 'real': ('Re U', (-self.amplitude_limit, self.amplitude_limit)),
            'imag': ('Im U', (-self.amplitude_limit, self.amplitude_limit)), 'phase': ('arg U (rad)', (-3.4, 3.4))}[q]
        a['profile_hbvp'].axes.set(title=f'Common-plane profile: z = {p.z_ref:.2f} µm', ylabel=label, ylim=limits)
        a['magnitude_fft'].set_ydata(abs(self.fft_coefficients[self._display_mask]))
        a['phase_fft'].set_ydata(visible_phase(self.fft_coefficients[self._display_mask]))
        a['difference'].set_ydata(abs(p.difference[self._display_mask]))
        a['delta_phase'].set_ydata(p.phase_difference[self._display_mask])
        j = self.order.value+self.config.M
        for key in ('hbvp', 'thin'):
            value = getattr(p, key)[j]
            a['argand_'+key].set_data([0, value.real], [0, value.imag])
        a['argand_fft'].set_data([self.fft_coefficients[j].real], [self.fft_coefficients[j].imag])
        a['argand_hbvp'].axes.set_title(f'm = {self.order.value:+d}: {p.kinds[j]}\nCyan: ideal → physical', fontsize=10)
        a['arrow'].xy = (p.hbvp[j].real, p.hbvp[j].imag)
        a['arrow'].set_position((p.thin[j].real, p.thin[j].imag))
        for line in a['selected_markers']:
            line.set_xdata([self.order.value]*2)
        for line in a['thickness_markers']:
            line.set_xdata([h]*2)
        for key in ('hbvp', 'thin'):
            values = np.array([getattr(plane, key)[j] for plane in self.planes])
            a['sweep_magnitude_'+key].set_data(self.data.thicknesses, abs(values))
            a['sweep_phase_'+key].set_data(self.data.thicknesses, visible_phase(values))
        a['sweep_error'].set_data(self.data.thicknesses, [abs(plane.difference[j]) for plane in self.planes])
        for name, attr in [('R', 'reflectance'), ('T', 'transmittance'), ('A', 'absorption')]:
            a['power_'+name].set_data(self.data.thicknesses, [getattr(s, attr) for s in self.data.solutions])
        local = self.fields['hbvp'][point]
        self.readout.value = (f'<b>h = {h:.2f} µm; z_ref = {p.z_ref:.2f} µm.</b> '
            f'Opening width W<sub>open</sub> = {self.config.clear_width:g} µm, period d = {self.config.period:g} µm; '
            f'absorber n = {self.config.n_absorber:g} + {self.config.kappa:g}i; λ₀ = {self.config.wavelength:g} µm. '
            f'Illustrative material; {2*self.config.M+1} retained orders. '
            'Thin mask at z = 0, propagated to z_ref. Both models: Uᵢ(0) = 1; no fitted phase. Phase gaps hide wraps or amplitudes below 10⁻⁶.'
            f'<br><b>Physical flux:</b> R = {solution.reflectance:.5f}, T = {solution.transmittance:.5f}, '
            f'A = {solution.absorption:.5f}; R + T + A = {solution.energy_balance:.10f}. '
            'A is integrated material loss; the ideal mask does not determine R or A.'
            f'<br><b>HBVP at x = {self.x[point]:+.4f} µm:</b> U = {_complex_text(local)}, '
            f'|U| = {abs(local):.5f}, arg U = {_phase_text(visible_phase([local])[0])} rad. '
            'The highlighted spatial and Argand points represent this same local field.')
        if not np.all(self._display_mask):
            self.readout.value += (f'<br>Displayed orders: {self.display_orders[0]:+d}…{self.display_orders[-1]:+d}; '
                f'fields, FFT checks and flux totals use all {2*self.config.M+1} solved harmonics.')
        status = 'within' if self.fft_error <= FIELD_FFT_ATOL else 'EXCEEDS'
        self.fft_readout.value = (
            '<b>Complex-field Fourier check:</b> the transmitted field '
            'U<sub>HBVP</sub>(x, z<sub>ref</sub>) shown in tab A; '
            f'{self.x.size} samples over one period; '
            f'max |C FFT − C direct| = {self.fft_error:.2e}, {status} the absolute {FIELD_FFT_ATOL:.0e} tolerance. '
            'This checks representation consistency, not an independent scattering calculation.'
            f'<br>m = {self.order.value:+d}: C direct = {_complex_text(p.hbvp[j])}; '
            f'C FFT = {_complex_text(self.fft_coefficients[j])}; '
            f'|difference| = {abs(self.fft_coefficients[j]-p.hbvp[j]):.2e}. '
            'The FFT uses complex U, not |U| or |U|².')
        self._table()
        self.quantity.disabled = self.tabs.selected_index != 0
        self.x_position.disabled = self.tabs.selected_index != 0
        self.compare_ideal.disabled = self.tabs.selected_index != 0
        self.order.disabled = self.tabs.selected_index == 0
        self.pictures[self.tabs.selected_index].value = figure_png(self.figures[self.tabs.selected_index])
        if self.tabs.selected_index == 1 and self.sweep.selected_index == 0:
            self.pictures[2].value = figure_png(self.figures[2])

    def _table(self):
        p, rows = self.plane, []
        for i, m in enumerate(p.orders):
            if abs(m) > 4 and m != self.order.value:
                continue
            values = (f'{m:+d}', p.kinds[i], f'{abs(p.thin[i]):.5f}', f'{abs(p.hbvp[i]):.5f}',
                _phase_text(p.thin_phase[i]), _phase_text(p.hbvp_phase[i]), _phase_text(p.phase_difference[i]),
                _complex_text(p.difference[i]), f'{p.thin_efficiency[i]:.5f}', f'{p.hbvp_efficiency[i]:.5f}')
            rows.append('<tr>'+''.join(f'<td style="padding:3px 8px;white-space:nowrap">{v}</td>' for v in values)+'</tr>')
        headers = ('m', 'Type', '|C ideal|', '|C HBVP|', 'arg ideal', 'arg HBVP', 'Δφ', 'ΔC', 'η ideal', 'η HBVP')
        self.table.value = ('<p>The separate Type rows identify propagating (P), evanescent (E) and grazing (G) orders. '
            f'Selected m = {self.order.value:+d}. Phases in radians; — means undefined or negligible amplitude. '
            'Table shows |m| ≤ 4 plus the selected order.</p><div style="overflow-x:auto"><table style="font-size:12px"><tr>'+
            ''.join(f'<th style="padding:3px 8px">{h}</th>' for h in headers)+'</tr>'+''.join(rows)+'</table></div>')

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
        self.sweep.unobserve(self._update, names='selected_index')
        self.reset_button.on_click(self._reset, remove=True)
        for picture in self.pictures:
            picture.value = b''
        for figure in self.figures:
            figure.clear()
        self.figures.clear()
        self._artists.clear()
        self.data = self.planes = self.plane = self.fields = self.basis = self.x = None
        self.fft_coefficients = self.fft_error = None
        self.maps = self.materials = self.map_x = self.map_z = None
        _close_widget_tree(self.ui)
