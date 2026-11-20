"""Views of one solved scalar grating, without re-solving during observation.

Lengths displayed in micrometres. ScalarGratingData owns fixed sampled data;
explorers own independent controls and disposable figures. Only the reference
explorer's three displays share an observation plane. No figures are saved.
"""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .grating_intro import _close_widget_tree
from .plotting import figure_png
from .scalar_grating import ScalarGratingConfiguration, solve_scalar_grating


BLUE, ORANGE, INK, CYAN = '#2866a3', '#b85c17', '#243444', '#07858f'
KINDS = {'propagating': BLUE, 'grazing': '#79539e', 'evanescent': ORANGE}
QUANTITIES = [('Real field Re U', 'real'), ('Amplitude |U|', 'amplitude'),
              ('Intensity |U|²', 'intensity'), ('Phase arg U', 'phase')]
STAGES = ('geometry', 'field', 'harmonics', 'reconstruction', 'reference')
PHASE_FLOOR = 1e-6  # Absolute field units; incident amplitude is one.


def _order_display_mask(orders, limit):
    """Select plotted orders without truncating the solution or field sums."""
    if limit is None:
        return np.ones(len(orders), dtype=bool)
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or limit < 0:
        raise ValueError('display_order_limit must be a nonnegative integer or None.')
    return abs(orders) <= limit


def visible_phase(values, *, relative_to=None, break_wraps=False):
    """Wrapped radians, NaN for amplitudes <1e-6 of the unit incident field.

    Optionally reference phases to a complex scalar. Break line segments at
    the +/-pi branch cut instead of drawing an artificial connecting ramp.
    """
    values = np.asarray(values, dtype=complex)
    phase = np.angle(values if relative_to is None else values*np.conj(relative_to))
    phase = np.where(abs(values) >= PHASE_FLOOR, phase, np.nan)
    if relative_to is not None and abs(relative_to) < PHASE_FLOOR:
        phase = np.full(values.shape, np.nan)
    if break_wraps and phase.ndim == 1:
        phase[1:][abs(np.diff(phase)) > np.pi] = np.nan
    return phase


class ScalarGratingData:
    """Fixed geometry and cached samples of one solution (no widgets).

    Accepts either an existing absorbing or lossless solution. The no-argument
    default retains the legacy lossless M=20 example; Notebook 0_5 explicitly
    supplies its absorbing M=24 solution. Uses 512 x samples across two
    periods and 601 z samples. Slice projection uses 256 points in one period.
    Sampling is increased for larger M; this is not a convergence estimator.
    """

    def __init__(self, solution=None):
        self.solution = (solve_scalar_grating(ScalarGratingConfiguration(M=20))
                         if solution is None else solution)
        self.config = c = self.solution.configuration
        self.z_min, self.z_max = -c.thickness-1.5, .6
        nx = max(512, 8*(2*c.M+1))
        self.x = -c.period+(np.arange(nx)+.5)*(2*c.period/nx)
        self.z = np.linspace(self.z_min, self.z_max, 601)
        self.harmonics = self.solution.harmonics(self.z)
        self.field = self.harmonics @ np.exp(1j*self.solution.kx[:, None]*self.x)
        self.index = c.refractive_index(self.x, self.z)
        self.is_absorbing = hasattr(self.solution, 'absorption')
        ns = max(256, 4*(2*c.M+1))
        self.x_slice = -c.period/2+np.arange(ns)*c.period/ns
        self.basis = np.exp(1j*self.solution.kx[:, None]*self.x_slice)
        self.amplitude_limit = 1.08*float(np.max(abs(self.field)))
        self.coefficient_limit = 1.08*float(np.max(abs(self.harmonics)))
        self.depth = np.linspace(0., 1.5, 241)
        self.output_harmonics = self.solution.harmonics(-c.thickness-self.depth)

    def material_quantity(self, name):
        """Display real index or extinction without discarding complex data."""
        if name == 'n':
            values, cmap, label = self.index.real, 'Blues', 'Real index n'
        elif name == 'kappa':
            values, cmap, label = self.index.imag, 'Greys', 'Extinction coefficient κ'
        else:
            raise ValueError('Material quantity must be n or kappa.')
        limits = (float(values.min()), max(float(values.max()), float(values.min())+.01))
        return values, cmap, limits, label

    def quantity(self, name):
        if name == 'real':
            return self.field.real, 'RdBu_r', (-self.amplitude_limit, self.amplitude_limit), 'Re U'
        if name == 'amplitude':
            return abs(self.field), 'viridis', (0., self.amplitude_limit), '|U|'
        if name == 'intensity':
            return abs(self.field)**2, 'inferno', (0., self.amplitude_limit**2), '|U|²'
        if name == 'phase':
            return visible_phase(self.field), 'twilight', (-np.pi, np.pi), 'arg U (rad)'
        raise ValueError('Unknown field quantity.')

    def plane(self, z, *, output_only=False):
        if np.ndim(z) or np.iscomplexobj(z) or not np.isfinite(z):
            raise ValueError('Plane position must be a finite real scalar.')
        upper = -self.config.thickness if output_only else self.z_max
        if not self.z_min <= z <= upper:
            raise ValueError('Plane lies outside the displayed region or transmitted half-space.')
        coefficients = self.solution.at_reference_plane(z)
        sampled = self.solution.reconstruct(self.x_slice, z)[0]
        reconstructed = coefficients @ self.basis
        projected = sampled @ self.basis.conj().T/self.x_slice.size
        return dict(z=float(z), coefficients=coefficients, field=sampled,
                    reconstructed=reconstructed, projected=projected,
                    amplitude=abs(sampled), phase=visible_phase(sampled, break_wraps=True),
                    coefficient_phase=visible_phase(coefficients),
                    relative_phase=visible_phase(coefficients, relative_to=coefficients[self.config.M]),
                    error=float(np.max(abs(reconstructed-sampled))))

    def evolution(self, order):
        if order not in self.solution.orders:
            raise ValueError('Order is not retained in this solution.')
        i = int(order)+self.config.M
        values, initial = self.output_harmonics[:, i], self.solution.transmission[i]
        if abs(initial) < PHASE_FLOOR:
            ratio = phase_advance = np.full(self.depth.shape, np.nan)
        else:
            ratio = abs(values)/abs(initial)
            phase_advance = np.unwrap(np.angle(values*np.conj(initial)))
            phase_advance[abs(values) < PHASE_FLOOR] = np.nan
        return dict(order=order, kind=self.solution.output_kinds[i], coefficients=values,
                    ratio=ratio, phase_advance=phase_advance, beta=self.solution.beta_output[i])


def _style(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(alpha=.15)
    axis.tick_params(labelsize=8)
    axis.xaxis.label.set_size(9)
    axis.yaxis.label.set_size(9)
    axis.title.set_size(11)


class ScalarGratingExplorer:
    """One lesson section using shared data but independently owned controls.

    For stage='reference', evolution_ui and efficiency_ui are lazy linked
    displays for sections 6/7. They use the same z_ref control and solution.
    Call close() before replacement; rerun later display cells after replacing
    the reference explorer. Data may be shared safely between explorers.
    display_order_limit restricts coefficient plots and order selectors only;
    field reconstruction, Fourier checks and flux totals use every solved order.
    """

    def __init__(self, stage='reference', data=None, *, display_order_limit=None):
        if stage not in STAGES:
            raise ValueError(f'stage must be one of {STAGES}.')
        import ipywidgets as widgets
        self.stage, self.model = stage, (ScalarGratingData() if data is None else data)
        self._display_mask = _order_display_mask(self.model.solution.orders, display_order_limit)
        self.display_orders = self.model.solution.orders[self._display_mask]
        self._closed = self._resetting = False
        self._artists, self._controls = {}, []
        self.data = None
        self._evolution_ui = self._efficiency_ui = None
        self.evolution_figure = self.evolution_picture = None
        self.efficiency_figure = self.efficiency_picture = None
        self.quantity = self.orders = self.z_slice = self.z_ref = self.order = None
        self.material = None
        c = self.model.config
        style = {'description_width': 'initial'}

        def slider(label, value, minimum, maximum):
            result = widgets.FloatSlider(description=label, value=value, min=minimum, max=maximum,
                step=.005, readout_format='.3f', continuous_update=False, style=style,
                layout=widgets.Layout(width='470px', max_width='100%'))
            self._controls.append(result)
            return result

        if stage in ('geometry', 'field') and self.model.is_absorbing:
            self.material = widgets.Dropdown(options=[('Extinction κ', 'kappa'), ('Real index n', 'n')],
                value='kappa', description='Material:', style=style)
            self._controls.append(self.material)
        if stage in ('field', 'reference'):
            self.quantity = widgets.Dropdown(options=QUANTITIES, value='intensity', description='Show:', style=style)
            self._controls.append(self.quantity)
        if stage == 'harmonics':
            choices = tuple(int(m) for m in self.display_orders)
            self.orders = widgets.SelectMultiple(options=choices, value=tuple(m for m in (0, 1, 3) if m in choices),
                rows=4, description='Orders m:', style=style, layout=widgets.Layout(width='200px'))
            self._controls.append(self.orders)
        if stage == 'reconstruction':
            self.z_slice = slider('Slice z (µm)', -c.thickness/2, self.model.z_min, self.model.z_max)
        if stage == 'reference':
            self.z_ref = slider('Output plane z_ref (µm)', -c.thickness-.05, self.model.z_min, -c.thickness)
            self.order = widgets.Dropdown(options=[int(m) for m in self.display_orders],
                value=min(3, int(self.display_orders[-1])), description='Follow order m:', style=style)
            self._controls.append(self.order)  # Displayed only in the evolution panel.
        self._defaults = [(control, control.value) for control in self._controls]
        self.reset_button = widgets.Button(description='Reset this view')
        self.reset_button.on_click(self._reset)
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png',
            layout=widgets.Layout(width='100%', max_width='1250px', height='auto'))
        displayed_controls = [control for control in self._controls if control is not self.order]
        self.reset_button.layout.display = 'none' if stage == 'geometry' and self.material is None else ''
        self.ui = widgets.VBox([widgets.Box([*displayed_controls, self.reset_button],
            layout=widgets.Layout(display='flex', flex_flow='row wrap', align_items='center')),
            self.readout, self.picture])
        # This undisplayed ownership root also releases lazy/shared widget trees once.
        self._owned = widgets.VBox([self.ui, *([self.order] if self.order is not None else [])])
        sizes = {'geometry': (6.5, 4.5), 'field': (11, 4.5), 'harmonics': (9, 4.5),
                 'reconstruction': (11, 6.4), 'reference': (12.5, 7.2)}
        self.figure = Figure(figsize=sizes[stage], dpi=110, facecolor='white', constrained_layout=True)
        FigureCanvasAgg(self.figure)
        self._build()
        for control in self._controls:
            control.observe(self._update, names='value')
        self._update()

    def _interfaces(self, axis):
        axis.axhline(0, color=INK, lw=.8, ls='--')
        axis.axhline(-self.model.config.thickness, color=INK, lw=.8, ls='--')
        axis.set_ylim(self.model.z_min, self.model.z_max)

    def _map(self, axis, *, material=False):
        model, c = self.model, self.model.config
        if material:
            values, cmap, limits, label = model.material_quantity(self.material.value if self.material else 'n')
        else:
            values, cmap, limits, label = model.quantity(self.quantity.value if self.quantity else 'amplitude')
        dz = model.z[1]-model.z[0]
        image = axis.imshow(values, origin='lower', extent=(-c.period, c.period,
            model.z[0]-dz/2, model.z[-1]+dz/2), cmap=cmap, vmin=limits[0], vmax=limits[1],
            aspect='equal', interpolation='nearest')
        self._interfaces(axis)
        axis.set(xlabel='x (µm)', ylabel='z (µm)', title='Prescribed material' if material else f'Total field: {label}')
        colorbar = self.figure.colorbar(image, ax=axis, shrink=.8, pad=.025)
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        _style(axis)
        axis.grid(False)
        if material:
            self._artists.update(material_map=image, material_axis=axis, material_colorbar=colorbar)
        else:
            self._artists.update(map=image, map_axis=axis, colorbar=colorbar)
        return image

    def _build(self):
        stage, c = self.stage, self.model.config
        if stage == 'geometry':
            axis = self.figure.subplots()
            self._map(axis, material=True)
            labels = ('Air: incident region', 'Absorbing ridge', 'Air: transmitted region') if self.model.is_absorbing else (
                'Incident region', 'Grating', 'Transmitted region')
            for height, label in zip((.4, -c.thickness/2, -c.thickness-.6), labels):
                axis.text(.93*c.period, height, label, fontsize=10, ha='right', va='center',
                          bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=2))
            if self.model.is_absorbing:
                axis.text(c.opening_shift, -c.thickness/2, 'Air', fontsize=9, ha='center', va='center')
            axis.annotate('', xy=(-.8*c.period, .04), xytext=(-.8*c.period, .49),
                          arrowprops=dict(arrowstyle='->', color=INK))
            axis.text(-.8*c.period+.08, .49, 'Incident wave', fontsize=9, ha='left')
            axis.text(-.35*c.period, .01, 'z = 0', fontsize=8, va='bottom')
            axis.text(-.35*c.period, -c.thickness-.03, 'z = −h', fontsize=8, va='top')
        elif stage == 'field':
            material, field = self.figure.subplots(1, 2)
            self._map(material, material=True)
            self._map(field)
        elif stage == 'harmonics':
            self._artists['harmonics_axis'] = self.figure.subplots()
        else:
            axes = self.figure.subplots(2, 3 if stage == 'reference' else 2)
            self._map(axes[0, 0])
            self._artists['plane'] = axes[0, 0].axhline(0, color=CYAN, lw=2)
            complex_axis = axes[0, 1]
            a, = complex_axis.plot([], [], color=BLUE, label='Re U')
            b, = complex_axis.plot([], [], color=ORANGE, label='Im U')
            self._artists.update(real=a, imag=b)
            complex_axis.set(xlabel='x (µm)', ylabel='Signed field', title='Complex field on the slice',
                             ylim=(-self.model.amplitude_limit, self.model.amplitude_limit))
            if stage == 'reconstruction':
                ar, = complex_axis.plot([], [], 'o', ms=3, markevery=14, mfc='none', color=BLUE, label='Sum: Re U')
                br, = complex_axis.plot([], [], 's', ms=3, markevery=14, mfc='none', color=ORANGE, label='Sum: Im U')
                self._artists.update(sum_real=ar, sum_imag=br)
            complex_axis.legend(fontsize=8, loc='upper right')
            amplitude_axis, phase_axis = axes[1, 0], axes[1, 1]
            m = self.display_orders
            colors = [KINDS[k] for k in self.model.solution.output_kinds[self._display_mask]] if stage == 'reference' else BLUE
            bars = amplitude_axis.bar(m, np.zeros(m.size), color=colors, width=.75)
            phase, = phase_axis.plot(m, np.full(m.size, np.nan), 'o', ms=4, color=INK)
            self._artists.update(bars=bars, coefficient_phase=phase)
            amplitude_axis.set(xlabel='Order m', ylabel='Coefficient magnitude',
                               title='Fourier coefficients on the slice', ylim=(0, self.model.coefficient_limit))
            phase_axis.set(xlabel='Order m', ylabel='Wrapped phase (rad)', title='Coefficient phase',
                           ylim=(-np.pi-.2, np.pi+.2), yticks=[-np.pi, 0, np.pi], yticklabels=['−π', '0', 'π'])
            for axis in (amplitude_axis, phase_axis):
                axis.set_xlim(m[0]-.75, m[-1]+.75)
                if len(m) <= 15:
                    axis.set_xticks(m)
            if stage == 'reference':
                amplitude, = axes[0, 2].plot([], [], color=CYAN)
                field_phase, = axes[1, 2].plot([], [], color=CYAN)
                self._artists.update(amplitude=amplitude, field_phase=field_phase)
                axes[0, 2].set(xlabel='x (µm)', ylabel='|U|', title='Amplitude on the output plane',
                               ylim=(0, self.model.amplitude_limit))
                axes[1, 2].set(xlabel='x (µm)', ylabel='arg U (rad)', title='Phase on the output plane',
                               ylim=(-np.pi-.2, np.pi+.2), yticks=[-np.pi, 0, np.pi], yticklabels=['−π', '0', 'π'])
            for axis in axes.flat:
                _style(axis)
            axes[0, 0].grid(False)
            for axis in [complex_axis, *([axes[0, 2], axes[1, 2]] if stage == 'reference' else [])]:
                axis.set_xlim(-c.period/2, c.period/2)

    def _update(self, change=None):
        if self._closed or self._resetting:
            return
        c, model = self.model.config, self.model
        if change and change.get('owner') is self.order:
            self._update_evolution()
            return
        if self.material is not None and (change is None or change.get('owner') is self.material):
            values, cmap, limits, label = model.material_quantity(self.material.value)
            self._artists['material_map'].set_data(values)
            self._artists['material_map'].set_cmap(cmap)
            self._artists['material_map'].set_clim(*limits)
            self._artists['material_colorbar'].set_label(label, fontsize=9)
            self._artists['material_axis'].set_title(f'Prescribed material: {label}', fontsize=11)
        if self.stage == 'geometry':
            if model.is_absorbing:
                self.readout.value = (f'λ₀ = {c.wavelength:g} µm; d = {c.period:g} µm; h = {c.thickness:g} µm; '
                    f'clear width CD = {c.clear_width:g} µm; open fraction D_open = {c.open_fraction:g}. '
                    f'Absorbing ridges: n + iκ = {c.n_absorber:g} + {c.kappa:g}i. '
                    'Centred clear openings and both exterior regions: air (n=1, κ=0). '
                    'Unit incident amplitude; normal incidence toward −z. Illustrative absorber, no substrate.')
            else:
                self.readout.value = (f'λ₀ = {c.wavelength:g} µm; d = {c.period:g} µm; h = {c.thickness:g} µm; '
                    f'ridge fraction D_ridge = {c.duty_cycle:g}. n<sub>ridge</sub> = {c.n_ridge:g}, '
                    f'n<sub>groove</sub> = {c.n_groove:g}; n<sub>in</sub> = {c.n_incident:g}, '
                    f'n<sub>out</sub> = {c.n_output:g}. Unit incident amplitude; normal incidence toward −z.')
        elif self.stage == 'harmonics':
            axis = self._artists['harmonics_axis']
            axis.clear()
            axis.axhspan(-c.thickness, 0, color=INK, alpha=.08)
            for m in self.orders.value:
                axis.plot(abs(model.harmonics[:, m+c.M]), model.z, label=f'm = {m:+d}')
            self._interfaces(axis)
            axis.set(xlabel='Harmonic magnitude |Fₘ(z)|', ylabel='z (µm)',
                     xlim=(0, model.coefficient_limit), title='Harmonics of the same total field')
            _style(axis)
            if self.orders.value:
                axis.legend(fontsize=9)
            self.readout.value = ('Select a few orders (Ctrl/Cmd-click). The shaded layer couples their amplitudes. '
                                  'Above z=0, A₀ includes incident plus reflected fields.')
            if model.is_absorbing:
                self.readout.value += ' Material loss occurs inside the absorbing ridges; the exterior is air.'
        else:
            if self.quantity is not None and (change is None or change.get('owner') is self.quantity):
                values, cmap, limits, label = model.quantity(self.quantity.value)
                self._artists['map'].set_data(values)
                self._artists['map'].set_cmap(cmap)
                self._artists['map'].set_clim(*limits)
                self._artists['colorbar'].set_label(label, fontsize=9)
                self._artists['map_axis'].set_title(f'Total field: {label}', fontsize=11)
            if self.stage == 'field':
                self.readout.value = (f'Orders −{c.M}…+{c.M}; one truncated scattering solution. '
                    'Above the grating: incident + reflected field. Re U is a t=0 snapshot; '
                    '|U|² is relative scalar intensity. Phase is masked below |U| = 10⁻⁶.')
            else:
                z = self.z_ref.value if self.stage == 'reference' else self.z_slice.value
                self.data = plane = model.plane(z, output_only=self.stage == 'reference')
                a = self._artists
                a['plane'].set_ydata([z, z])
                a['real'].set_data(model.x_slice, plane['field'].real)
                a['imag'].set_data(model.x_slice, plane['field'].imag)
                for bar, height in zip(a['bars'], abs(plane['coefficients'][self._display_mask])):
                    bar.set_height(height)
                a['coefficient_phase'].set_ydata(plane['coefficient_phase'][self._display_mask])
                if self.stage == 'reference':
                    a['amplitude'].set_data(model.x_slice, plane['amplitude'])
                    a['field_phase'].set_data(model.x_slice, plane['phase'])
                    self.readout.value = (f'<b>z_ref = {z:.3f} µm; distance below grating = {-c.thickness-z:.3f} µm.</b> '
                        'Cₘ = Fₘ(z_ref). Blue bars: propagating; orange: evanescent; purple: grazing. '
                        'Phase uses the fixed incident-wave reference; gaps hide wraps or |U| &lt; 10⁻⁶. '
                        'All controls sample the cached solution; colour and axis scales stay fixed.')
                    self._update_evolution()
                    self._update_efficiency()
                else:
                    a['sum_real'].set_data(model.x_slice, plane['reconstructed'].real)
                    a['sum_imag'].set_data(model.x_slice, plane['reconstructed'].imag)
                    self.readout.value = (f'<b>z = {z:.3f} µm.</b> Curves: sampled U; markers: sum of Fₘ exp(ikₓ,ₘx). '
                        f'Maximum difference {plane["error"]:.1e}. All {2*c.M+1} harmonics are retained. '
                        'This checks two representations of the same truncated solution, not an independent solver.')
        if self.stage in ('harmonics', 'reconstruction', 'reference') and not np.all(self._display_mask):
            self.readout.value += (f' Displayed orders: {self.display_orders[0]:+d}…{self.display_orders[-1]:+d}; '
                                   f'fields and totals use all {2*c.M+1} solved harmonics.')
        self.picture.value = figure_png(self.figure)

    @property
    def evolution_ui(self):
        if self.stage != 'reference' or self._closed:
            raise ValueError('The evolution view requires an open reference explorer.')
        if self._evolution_ui is None:
            import ipywidgets as widgets
            self.evolution_readout = widgets.HTML()
            self.evolution_picture = widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1100px'))
            self._evolution_ui = widgets.VBox([widgets.Box([self.z_ref, self.order, self.reset_button],
                layout=widgets.Layout(display='flex', flex_flow='row wrap', align_items='center')),
                                              self.evolution_readout, self.evolution_picture])
            self._owned.children += (self._evolution_ui,)
            self.evolution_figure = Figure(figsize=(10.5, 3.7), dpi=110, facecolor='white', constrained_layout=True)
            FigureCanvasAgg(self.evolution_figure)
            magnitude, phase = self.evolution_figure.subplots(1, 2)
            line, = magnitude.plot([], [], color=ORANGE)
            phase_line, = phase.plot([], [], color=BLUE)
            markers = [axis.axvline(0, color=CYAN, lw=1.5) for axis in (magnitude, phase)]
            self._artists.update(evolution_magnitude=line, evolution_phase=phase_line, evolution_markers=markers)
            magnitude.set(ylabel='|Cₘ| / |tₘ|', title='Amplitude relative to the bottom interface', ylim=(0, 1.05))
            max_phase = float(self.model.solution.beta_output.real.max()*self.model.depth[-1])
            phase.set(ylabel='Phase advance from z = −h (rad)', title='Unwrapped phase advance', ylim=(-.2, max(.5, 1.05*max_phase)))
            for axis in (magnitude, phase):
                axis.set(xlabel='Distance Δ below the grating (µm)', xlim=(0, self.model.depth[-1]))
                _style(axis)
            self._update_evolution()
        return self._evolution_ui

    def _update_evolution(self):
        if self._evolution_ui is None:
            return
        model = self.model
        self.evolution_data = data = model.evolution(self.order.value)
        self._artists['evolution_magnitude'].set_data(model.depth, data['ratio'])
        self._artists['evolution_phase'].set_data(model.depth, data['phase_advance'])
        for marker in self._artists['evolution_markers']:
            marker.set_xdata([-model.config.thickness-self.z_ref.value]*2)
        counts = {kind: np.count_nonzero(model.solution.output_kinds == kind) for kind in KINDS}
        self.evolution_readout.value = (f'<b>m = {self.order.value:+d}: {data["kind"]}.</b> '
            f'Output orders: {counts["propagating"]} propagating, {counts["grazing"]} grazing, '
            f'{counts["evanescent"]} evanescent. Cyan marks the plane shared with section 5. '
            'Phase is hidden once |Cₘ| &lt; 10⁻⁶; ratios are undefined if |tₘ| &lt; 10⁻⁶.')
        self.evolution_picture.value = figure_png(self.evolution_figure)

    @property
    def efficiency_ui(self):
        if self.stage != 'reference' or self._closed:
            raise ValueError('The efficiency view requires an open reference explorer.')
        if self._efficiency_ui is None:
            import ipywidgets as widgets
            self.efficiency_readout = widgets.HTML()
            self.efficiency_picture = widgets.Image(format='png',
                layout=widgets.Layout(width='100%', max_width='1250px', height='auto'))
            self._efficiency_ui = widgets.VBox([widgets.Box([self.z_ref, self.reset_button],
                layout=widgets.Layout(display='flex', flex_flow='row wrap', align_items='center')),
                self.efficiency_readout, self.efficiency_picture])
            self._owned.children += (self._efficiency_ui,)
            self.efficiency_figure = Figure(figsize=(12, 7), dpi=110,
                facecolor='white', constrained_layout=True)
            FigureCanvasAgg(self.efficiency_figure)
            magnitude, phase, flux, balance = self.efficiency_figure.subplots(2, 2).flat
            s, mask, m = self.model.solution, self._display_mask, self.display_orders
            kinds = s.output_kinds[mask]
            colors = [KINDS[kind] for kind in kinds]
            amplitude_bars = magnitude.bar(m, np.zeros(m.size), color=colors, width=.7)
            phase_points = phase.scatter(m, np.full(m.size, np.nan), c=colors, s=28)
            efficiencies = s.transmission_efficiency[mask]
            flux_bars = flux.bar(m, efficiencies, color=colors, width=.7)
            zero_power = efficiencies == 0
            flux.scatter(m[zero_power], np.zeros(np.count_nonzero(zero_power)),
                facecolors='white', edgecolors=np.asarray(colors)[zero_power],
                s=28, zorder=3, clip_on=False, label='Zero normal power')
            flux.legend(loc='upper right', fontsize=8)
            magnitude.set(ylabel=r'$|C_m|$', title='Transmitted coefficient magnitude',
                ylim=(0, max(.05, 1.12*float(np.max(abs(s.transmission[mask]))))))
            phase.set(ylabel=r'$\arg(C_m C_0^*)$ (rad)', title='Phase relative to order 0',
                ylim=(-np.pi-.25, np.pi+.25), yticks=[-np.pi, 0, np.pi],
                yticklabels=[r'$-\pi$', '0', r'$\pi$'])
            flux.set(ylabel=r'Transmitted efficiency $\eta_m^{(T)}$',
                title='Normal power carried by each order',
                ylim=(0, max(.01, 1.18*float(np.max(efficiencies)))))
            magnitude.legend(handles=[Patch(facecolor=KINDS[kind], label=kind.capitalize())
                for kind in KINDS if kind in kinds], fontsize=8, loc='upper right')
            for axis in (magnitude, phase, flux):
                axis.set(xlabel='Order m', xlim=(m[0]-.75, m[-1]+.75))
                if len(m) <= 15:
                    axis.set_xticks(m)
            totals = [float(sum(s.reflection_efficiency)), float(sum(s.transmission_efficiency))]
            labels = ['Reflection R', 'Transmission T']
            if self.model.is_absorbing:
                totals.append(s.absorption)
                labels.append('Absorption A')
            balance_bars = balance.bar(labels, totals,
                color=[INK, BLUE, '#6b7d4b'][:len(totals)], width=.55)
            balance.bar_label(balance_bars, labels=[f'{value:.5f}' for value in totals],
                              padding=4, fontsize=9)
            balance.set(ylabel='Fraction of incident power', ylim=(0, 1.08),
                title='Power accounting: all solved orders')
            self._artists.update(efficiency_magnitude=amplitude_bars,
                efficiency_phase=phase_points, efficiency_flux=flux_bars,
                efficiency_balance=balance_bars)
            for axis in (magnitude, phase, flux, balance):
                _style(axis)
            self._update_efficiency()
        return self._efficiency_ui

    def _update_efficiency(self):
        if self._efficiency_ui is None:
            return
        s, p = self.model.solution, self.data
        for bar, height in zip(self._artists['efficiency_magnitude'],
                               abs(p['coefficients'][self._display_mask])):
            bar.set_height(height)
        self._artists['efficiency_phase'].set_offsets(np.column_stack((
            self.display_orders, p['relative_phase'][self._display_mask])))
        accounting = (f'R = {sum(s.reflection_efficiency):.6f}, '
                      f'T = {sum(s.transmission_efficiency):.6f}, ')
        if self.model.is_absorbing:
            accounting += (f'A = {s.absorption:.6f}, R + T + A = {s.energy_balance:.12f}. '
                           'A is independently integrated material absorption. ')
        else:
            accounting += f'R + T = {s.power_balance:.12f}. '
        self.efficiency_readout.value = (
            f'<b>z_ref = {p["z"]:.3f} µm.</b> '
            f'Showing orders {self.display_orders[0]:+d}…{self.display_orders[-1]:+d}; '
            'power totals include all retained orders. '
            'Phase is relative to m=0; missing points mean a coefficient is too small to report. '
            'Hollow circles mark zero outgoing normal power, including evanescent and grazing orders.'
            '<br>Scalar flux: '+accounting+
            'Efficiencies and totals stay fixed as the observation plane moves in the output medium.')
        self.efficiency_picture.value = figure_png(self.efficiency_figure)

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
        self.reset_button.on_click(self._reset, remove=True)
        self.picture.value = b''
        self.figure.clear()
        if self.evolution_figure is not None:
            self.evolution_picture.value = b''
            self.evolution_figure.clear()
        if self._efficiency_ui is not None:
            self.efficiency_readout.value = ''
            self.efficiency_picture.value = b''
            self.efficiency_figure.clear()
        self.figure = self.evolution_figure = self.model = self.data = None
        self.efficiency_figure = None
        self.evolution_data = None
        self._artists.clear()
        _close_widget_tree(self._owned)
