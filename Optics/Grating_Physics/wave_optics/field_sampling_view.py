"""Owned, reusable spatial-sampling explorer for Notebook 9.2."""
import io

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .field_sampling import DEFAULT_ORDERS, DEFAULT_COEFFICIENTS, periodic_field, sampling_experiment


def sampling_comparison_png(cases, *, period_nm=1000., orders=DEFAULT_ORDERS,
                            coefficients=DEFAULT_COEFFICIENTS,
                            highlight_order=None, show_alias=False):
    """Compare two grids using the same physical field and fixed plot scales.

    Each case is (samples_per_period, periods). Only Re U is shown spatially;
    the Fourier coefficients always come from the full complex field. For two
    one-period grids, the horizontal spectral coordinate is the order index.
    """
    if len(cases) != 2:
        raise ValueError('Use two cases for a side-by-side teaching comparison.')
    data = [sampling_experiment(count, periods, period_nm=period_nm,
            orders=orders, coefficients=coefficients) for count, periods in cases]
    figure = Figure(figsize=(11, 6), dpi=110)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    figure.subplots_adjust(left=.075, right=.985, bottom=.105, top=.94,
                           hspace=.27, wspace=.22)
    left = min(d['x_start_nm'] for d in data)
    right = max(d['x_start_nm']+d['L_nm'] for d in data)
    full_x = np.linspace(left, right, 2401)
    full_field = periodic_field(full_x, period_nm, orders, coefficients)
    frequency_limit = (max(1, max(abs(np.asarray(orders))))+1.5)/period_nm
    order_axis = all(d['periods'] == 1 for d in data)
    axis_scale = period_nm if order_axis else 1.
    horizontal_limit = frequency_limit*axis_scale
    bound = max(1., float(np.sum(abs(np.asarray(coefficients)))))*1.08
    coefficient_bound = max(1., max(float(np.max(abs(d['coefficients']))) for d in data)*1.4)
    for row, d in enumerate(data):
        spatial, spectral = axes[row]
        spatial.axvspan(d['x_start_nm'], d['x_start_nm']+d['L_nm'],
                       color='#fff2df', zorder=0)
        spatial.plot(full_x, full_field.real, color='#c6c6c6', lw=1.4)
        spatial.plot(d['fine_x_nm'], d['exact_fine'].real, color='#222222',
                     lw=1.8, label='Exact field')
        spatial.plot(d['fine_x_nm'], d['reconstructed_fine'].real, '--',
                     color='#2674af', lw=1.6, label='Field from FFT')
        spatial.plot(d['x_nm'], d['field'].real, 'o', color='#d46b19',
                     ms=3.8, label='Stored samples')
        spatial.set(xlim=(left, right), ylim=(-bound, bound), ylabel='Re U',
                    title=f"{d['N']} points over {d['L_nm']:.0f} nm; dx = {d['dx_nm']:g} nm")
        spatial.tick_params(labelbottom=row == 1)
        spatial.grid(alpha=.15)
        fx = np.fft.fftshift(d['frequencies'])
        magnitudes = abs(np.fft.fftshift(d['coefficients']))
        true_positions = np.asarray(orders)/period_nm*axis_scale
        bin_positions = fx*axis_scale
        spectral.vlines(true_positions, 0, abs(np.asarray(coefficients)),
                        color='#777777', lw=4, alpha=.5)
        spectral.plot(true_positions, abs(np.asarray(coefficients)), 'o',
                      color='#222222', ms=6, label=r'Known $|C_m|$')
        spectral.vlines(bin_positions, 0, magnitudes, color='#d46b19', lw=1)
        spectral.plot(bin_positions, magnitudes, 'x', color='#d46b19', ms=6,
                      label=r'FFT $|\widehat C_m|$' if order_axis else 'FFT coefficients / bins')
        title = (f"{d['N']}-point FFT: recovered order coefficients" if order_axis else
                 f"FFT bins {1/d['L_nm']:.4f} cycles/nm apart")
        spectral.set(xlim=(-horizontal_limit, horizontal_limit), ylim=(-.035, coefficient_bound),
                     ylabel='Coefficient magnitude', title=title)
        if order_axis:
            spectral.set_xticks(np.arange(np.ceil(-horizontal_limit), np.floor(horizontal_limit)+1))
        spectral.grid(alpha=.15)
        spectral.tick_params(labelbottom=row == 1)
        if highlight_order is not None:
            index = np.flatnonzero(np.asarray(orders) == highlight_order)
            if not index.size:
                raise ValueError('highlight_order must be a known order.')
            height = abs(np.asarray(coefficients)[index[0]])
            spectral.annotate(f"m = {highlight_order}: bin q = {highlight_order*d['periods']}",
                xy=(highlight_order/period_nm*axis_scale, height),
                xytext=(.96, .65), textcoords='axes fraction', ha='right',
                color='#2674af', fontsize=10,
                arrowprops={'arrowstyle':'->', 'color':'#2674af'})
        if show_alias and not d['resolved']:
            known_modes = np.asarray(orders)
            mode = known_modes[np.argmax(abs(known_modes))]
            aliased_index = (mode*d['periods']) % d['N']
            aliased_frequency = d['frequencies'][aliased_index]
            aliased_order = aliased_frequency*period_nm
            value = abs(d['coefficients'][aliased_index])
            nyquist = .5/d['dx_nm']*axis_scale
            if nyquist < horizontal_limit:
                spectral.axvspan(-horizontal_limit, -nyquist, color='#e6e6e6', zorder=0)
                spectral.axvspan(nyquist, horizontal_limit, color='#e6e6e6', zorder=0)
                for boundary in (-nyquist, nyquist):
                    spectral.axvline(boundary, color='#999999', ls=':', lw=1)
            minimum_bin = int(round(np.min(d['frequencies'])*period_nm))
            maximum_bin = int(round(np.max(d['frequencies'])*period_nm))
            if order_axis:
                spectral.set_title(f"{d['N']}-point FFT: bins m = {minimum_bin}, ..., {maximum_bin}")
            source_x = mode/period_nm*axis_scale
            target_x = aliased_frequency*axis_scale
            arrow_y = coefficient_bound*.88
            spectral.annotate('', xy=(target_x, arrow_y), xytext=(source_x, arrow_y),
                arrowprops={'arrowstyle':'->', 'color':'#a43834', 'lw':1.5})
            spectral.text((source_x+target_x)/2, arrow_y+coefficient_bound*.025,
                f'alias: {mode} {aliased_order-mode:+g} = {aliased_order:g}',
                ha='center', va='bottom', color='#a43834', fontsize=9)
            spectral.plot(target_x, value, 'x', color='#a43834', ms=9, mew=2)
            spectral.annotate('Combined FFT coefficient\n' +
                rf'$|\widehat C_{{{aliased_order:g}}}| = {value:.2f}$',
                xy=(target_x, value), xytext=(.025, .53), textcoords='axes fraction',
                ha='left', color='#a43834', fontsize=8,
                arrowprops={'arrowstyle':'->', 'color':'#a43834'})
            spectral.text(source_x, coefficient_bound*.54, 'No separate\nFFT bin here',
                          ha='center', va='center', color='#666666', fontsize=8)
    axes[0, 0].legend(loc='upper right', fontsize=7)
    axes[0, 1].legend(loc='upper right', fontsize=7)
    axes[1, 0].set_xlabel('Position x (nm); shaded region is the sampled window')
    axes[1, 1].set_xlabel('Order m (one period; fx = m/p)' if order_axis else
                         'Spatial frequency fx (cycles/nm)')
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', facecolor='white')
    figure.clear()
    return buffer.getvalue()


def _phase(values, *, break_wraps=False):
    phase = np.where(abs(values) > 1e-8, np.angle(values), np.nan)
    if break_wraps:
        jumps = np.flatnonzero(abs(np.diff(phase)) > np.pi)+1
        phase[jumps] = np.nan
    return phase


class FieldSamplingExplorer:
    """Vary numerical grids while the prescribed physical field stays fixed."""

    def __init__(self, *, period_nm=1000., orders=DEFAULT_ORDERS,
                 coefficients=DEFAULT_COEFFICIENTS):
        import ipywidgets as widgets
        self.period_nm = period_nm
        self.orders = np.array(orders, copy=True)
        self.coefficients = np.array(coefficients, dtype=complex, copy=True)
        self._closed = self._resetting = False
        style = {'description_width': 'initial'}
        self.samples = widgets.SelectionSlider(options=(6, 8, 10, 12, 16, 24, 32, 48, 64),
            value=32, description='Samples per period', continuous_update=False, style=style,
            layout=widgets.Layout(width='400px'))
        self.periods = widgets.ToggleButtons(options=(1, 2, 4), value=1,
            description='Periods in window', style=style)
        self.start = widgets.Dropdown(options=[('Centred window', -.5), ('Start at x=0', 0.),
            ('Start at x=0.137 L', .137)], value=-.5, description='Grid origin', style=style)
        self.reset_button = widgets.Button(description='Reset sampling')
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1250px'))
        controls = widgets.Box([self.samples, self.periods, self.start, self.reset_button],
            layout=widgets.Layout(display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([controls, self.readout, self.picture])
        self.figure = Figure(figsize=(11.6, 6.8), dpi=105, constrained_layout=True)
        FigureCanvasAgg(self.figure)
        self.axes = self.figure.subplots(2, 3)
        self._build_artists()
        self._controls = (self.samples, self.periods, self.start)
        for control in self._controls:
            control.observe(self._update, names='value')
        self.reset_button.on_click(self._reset)
        self._update()

    def _build_artists(self):
        self.field_artists = []
        field_axes = [*self.axes[0], self.axes[1, 0]]
        bound = max(1., float(np.sum(abs(self.coefficients))))*1.08
        for i, (ax, title) in enumerate(zip(field_axes, ('Real field', 'Imaginary field', 'Amplitude', 'Wrapped phase'))):
            exact, = ax.plot([], [], color='#222222', lw=1.8, label='Prescribed field')
            reconstructed, = ax.plot([], [], '--', color='#2674af', lw=1.6, label='FFT interpolant')
            samples, = ax.plot([], [], 'o', color='#d46b19', ms=4, label='Stored samples')
            ax.set(title=title, xlabel='Position x (nm)',
                   ylabel=('Re U', 'Im U', '|U|', 'Phase (rad)')[i])
            ax.set_ylim((-bound, bound) if i < 2 else ((0, bound) if i == 2 else (-3.4, 3.4)))
            ax.grid(alpha=.16)
            self.field_artists.append((exact, reconstructed, samples))
        self.axes[0, 0].legend(fontsize=7, loc='upper right')
        magnitude, phase = self.axes[1, 1:]
        self.known_stems = magnitude.vlines([], 0, [], color='#222222', lw=2, label='Known coefficients')
        self.fft_stems = magnitude.vlines([], 0, [], color='#d46b19', lw=1)
        self.known_magnitudes, = magnitude.plot([], [], 'o', color='#222222', ms=6)
        self.fft_magnitudes, = magnitude.plot([], [], 'x', color='#d46b19', ms=6, label='FFT coefficients')
        self.known_phases, = phase.plot([], [], 'o', color='#222222', ms=6, label='Known coefficients')
        self.fft_phases, = phase.plot([], [], 'x', color='#d46b19', ms=6, label='FFT coefficients')
        self.nyquist_lines = [magnitude.axvline(0, ls=':', color='#b04d8a', lw=1.2) for _ in range(2)]
        max_mode = max(1, int(np.max(abs(self.orders))))
        self.frequency_limit = (max_mode+1.5)/self.period_nm
        for ax, label in ((magnitude, '|coefficient|'), (phase, 'Coefficient phase (rad)')):
            ax.set(xlabel='Spatial frequency fx (cycles/nm)', ylabel=label,
                   xlim=(-self.frequency_limit, self.frequency_limit))
            ax.grid(alpha=.16)
        magnitude.set(title='Known spectrum and FFT bins', ylim=(-.035, max(1., max(abs(self.coefficients))*1.3)))
        phase.set(title='Phase where amplitude is appreciable', ylim=(-3.4, 3.4))
        magnitude.legend(fontsize=7, loc='upper right')

    def _update(self, change=None):
        if self._closed or self._resetting:
            return
        data = self.data = sampling_experiment(self.samples.value, self.periods.value,
            period_nm=self.period_nm, start_fraction=self.start.value,
            orders=self.orders, coefficients=self.coefficients)
        transforms = (np.real, np.imag, np.abs, lambda u: _phase(u, break_wraps=True))
        for ax, artists, quantity in zip([*self.axes[0], self.axes[1, 0]], self.field_artists, transforms):
            exact, interpolant, samples = artists
            exact.set_data(data['fine_x_nm'], quantity(data['exact_fine']))
            interpolant.set_data(data['fine_x_nm'], quantity(data['reconstructed_fine']))
            samples.set_data(data['x_nm'], _phase(data['field']) if ax is self.axes[1, 0] else quantity(data['field']))
            ax.set_xlim(data['x_start_nm'], data['x_start_nm']+data['L_nm'])
        true_fx = self.orders/self.period_nm
        fx, coefficients = data['frequencies'], data['coefficients']
        self.known_stems.set_segments([[(f, 0), (f, a)] for f, a in zip(true_fx, abs(self.coefficients))])
        self.fft_stems.set_segments([[(f, 0), (f, a)] for f, a in zip(fx, abs(coefficients))])
        self.known_magnitudes.set_data(true_fx, abs(self.coefficients))
        self.fft_magnitudes.set_data(fx, abs(coefficients))
        self.known_phases.set_data(true_fx, _phase(self.coefficients))
        self.fft_phases.set_data(fx, _phase(coefficients))
        nyquist = .5/data['dx_nm']
        for line, value in zip(self.nyquist_lines, (-nyquist, nyquist)):
            line.set_xdata([value, value])
        status = ('All specified harmonics are strictly below Nyquist.' if data['resolved'] else
                  'Some specified harmonics are at or above Nyquist; the field is not uniquely resolved.')
        coefficient_check = (f"Maximum coefficient difference: {data['coefficient_error']:.2e}. "
                             if data['resolved'] else 'The FFT bins now represent an aliased field. ')
        self.readout.value = (
            f"<b>{status}</b><br>N={data['N']}; L={data['L_nm']:.0f} nm; "
            f"dx={data['dx_nm']:.3f} nm; df={1/data['L_nm']:.5f} cycles/nm; "
            f"Nyquist={nyquist:.5f} cycles/nm. Dotted lines mark Nyquist when visible. "
            f"<br>Sample FFT/inverse error: {data['roundtrip_error']:.2e}; "
            f"maximum field difference on the fine comparison grid: {data['field_error']:.2e}. "
            f"{coefficient_check}The prescribed field stays fixed while these controls change its representation.")
        buffer = io.BytesIO()
        self.figure.savefig(buffer, format='png', facecolor='white')
        self.picture.value = buffer.getvalue()

    def _reset(self, _):
        if self._closed:
            return
        self._resetting = True
        self.samples.value, self.periods.value, self.start.value = 32, 1, -.5
        self._resetting = False
        self._update()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.reset_button.on_click(self._reset, remove=True)
        from .grating_intro import _close_widget_tree
        _close_widget_tree(self.ui)
        self.figure.clear()
        self.data = None
