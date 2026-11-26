"""Notebook comparison of a 2D absorbing screen and an ideal amplitude mask."""
import io
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .amplitude_grating_2d import AbsorbingComparison2DData, field_fourier_coefficients_2d


def _phase(values):
    return np.where(abs(values) > 1e-6, np.angle(values), np.nan)


class AbsorbingGrating2DExplorer:
    """Solve once; changing the view, order or distance never reruns FMM.

    Display with display(explorer.ui). Configuration is supplied through data;
    call close() before replacing a displayed explorer.
    """

    def __init__(self, data=None):
        import ipywidgets as widgets
        self.data = AbsorbingComparison2DData() if data is None else data
        self.config = c = self.data.configuration
        self._closed = False
        size = max(64, 2**int(np.ceil(np.log2(2*max(c.Mx, c.My)+2))))
        self.x = ((np.arange(size)+.5)/size-.5)*c.period_x
        self.y = ((np.arange(size)+.5)/size-.5)*c.period_y
        self._basis_x = np.exp(1j*self.data.solution.kx[:, None]*self.x)
        self._basis_y = np.exp(1j*self.data.solution.ky[:, None]*self.y)
        self._cache = {}
        self.distance = widgets.FloatSlider(value=0., min=0., max=5*c.wavelength, step=c.wavelength/40,
            description='Distance below exit', continuous_update=False, readout_format='.3f',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='420px'))
        self.order = widgets.Dropdown(options=[(f'({m:+d}, {n:+d})', i)
            for i, (m,n) in enumerate(self.data.solution.orders)], value=self.data.solution.incident_index,
            description='Order (m,n)', style={'description_width': 'initial'})
        self.readout, self.order_readout = widgets.HTML(), widgets.HTML()
        self.pictures = [widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1200px'))
                         for _ in range(3)]
        self.tabs = widgets.Tab(children=[widgets.VBox([picture]) for picture in self.pictures])
        for i, title in enumerate(('1–2 · Structure and exit field', '3 · Complex order lattice', '4 · Propagation')):
            self.tabs.set_title(i, title)
        self.reset_button = widgets.Button(description='Reset view')
        note = widgets.HTML('Plane-wave illumination; one periodic unit cell. Lengths use the configuration’s '
            'common unit. Propagation uses full ASM order factors. Phase is hidden where |field/coefficient| '
            '&lt; 10⁻⁶ of the unit incident field. Indices are illustrative, not fitted material data.')
        self.ui = widgets.VBox([widgets.HBox([self.distance, self.order, self.reset_button]),
                               self.readout, self.order_readout, self.tabs, note])
        self.figures = [Figure(figsize=(11, 6.7), dpi=100, constrained_layout=True) for _ in range(3)]
        for figure in self.figures:
            FigureCanvasAgg(figure)
        self._images, self._markers = [], []
        self._build_figures()
        for control in (self.distance, self.order, self.tabs):
            control.observe(self._update, names='selected_index' if control is self.tabs else 'value')
        self.reset_button.on_click(self._reset)
        self._update()

    def _field(self, coefficients):
        return np.einsum('a,ay,ax->yx', coefficients, self._basis_y, self._basis_x, optimize=True)

    def _build_figures(self):
        c = self.config
        self._spatial_extent = [-c.period_x/2, c.period_x/2, -c.period_y/2, c.period_y/2]
        order_extent = [-c.Mx-.5, c.Mx+.5, -c.My-.5, c.My+.5]
        titles = [
            ('Material Re n', 'Material Im n', 'Exact ideal |t(x,y)|',
             'FMM exit |U|', 'FMM exit phase', 'Ideal exit |U| · retained Fourier sum'),
            ('FMM |C / incident exit reference|', 'Ideal |C / incident exit reference|', '|C FMM − C ideal|',
             'FMM coefficient phase', 'Ideal coefficient phase', 'Relative phase arg(C FMM C ideal*)'),
            ('FMM propagated |U|', 'Ideal propagated |U|', 'FMM propagated phase', 'Ideal propagated phase'),
        ]
        for tab, figure in enumerate(self.figures):
            axes = figure.subplots(2, 2 if tab == 2 else 3)
            images, markers = [], []
            for i, axis in enumerate(axes.flat):
                phase = (tab == 0 and i == 4) or (tab == 1 and i >= 3) or (tab == 2 and i >= 2)
                im = axis.imshow(np.zeros((2,2)), origin='lower', aspect='equal',
                    extent=order_extent if tab == 1 else self._spatial_extent,
                    cmap='twilight' if phase else 'viridis',
                    vmin=-np.pi if phase else 0, vmax=np.pi if phase else 1,
                    interpolation='nearest')
                images.append(im)
                figure.colorbar(im, ax=axis, shrink=.8)
                axis.set(title=titles[tab][i], xlabel='Order m' if tab == 1 else 'x',
                         ylabel='Order n' if tab == 1 else 'y')
                if tab == 1:
                    marker, = axis.plot([0],[0], 's', mfc='none', mec='red', mew=1.5, ms=12)
                    markers.append(marker)
            self._images.append(images)
            self._markers.append(markers)
        self._amplitude_max = max(1., np.sum(abs(self.data.solution.transmission)),
                                  np.sum(abs(self.data.ideal.transmission)))
        s, ideal = self.data.solution, self.data.ideal
        material = c.refractive_index(self.x, self.y, [-c.thickness/2])[0]
        physical, prescribed = self._field(s.transmission), self._field(ideal.transmission)
        mask = abs(ideal.exact_object_field(self.x, self.y))
        for im, values in zip(self._images[0],
                              (material.real, material.imag, mask, abs(physical), _phase(physical), abs(prescribed))):
            im.set_data(values)
        self._images[0][0].set_clim(0, max(1., float(material.real.max())))
        self._images[0][1].set_clim(0, max(.01, float(material.imag.max())))
        for i in (3,5):
            self._images[0][i].set_clim(0, self._amplitude_max)

    def _plane(self, distance):
        # Retain only the current plane, avoiding an unbounded slider cache.
        if distance not in self._cache:
            actual, ideal = self.data.coefficients(distance)
            physical_field, ideal_field = self._field(actual), self._field(ideal)
            extracted = field_fourier_coefficients_2d(physical_field, self.x, self.y, self.config)
            self._cache = {distance: (actual, ideal, physical_field, ideal_field,
                                     float(np.max(abs(extracted-actual))))}
        return self._cache[distance]

    def _update(self, change=None):
        if self._closed:
            return
        tab = self.tabs.selected_index
        c, s = self.config, self.data.solution
        distance = 0. if tab == 0 else float(self.distance.value)
        actual, ideal, physical_field, ideal_field, fft_error = self._plane(distance)
        reference = self.data.ideal.incident_reference
        self.distance.disabled = tab == 0
        self.order.disabled = tab != 1
        self.readout.value = (
            f'<b>Scalar FMM:</b> R={s.reflectance:.6f}, T={s.transmittance:.6f}, '
            f'A={s.absorption:.6f}; R+T+A={s.energy_balance:.10f}. '
            f'Cutoffs ({c.Mx}, {c.My}); {len(s.orders)} harmonics. '
            f'<br>Clear fractions ({c.fill_x:g}, {c.fill_y:g}); h={c.thickness:g}; '
            f'absorber index={c.n_background}; opening index={c.n_inclusion}. '
            f'<br>Observation z={-c.thickness-distance:.4f}. '
            f'Complex-field FFT2 check: max |C FFT − C direct|={fft_error:.2e}. '
            'This checks representation consistency; refine FMM cutoffs separately.')
        j = self.order.value
        m, n = s.orders[j]
        if tab == 1:
            self.order_readout.value = (
                f'<b>Order ({m:+d},{n:+d}) · {s.output_kinds[j]}:</b> '
                f'C FMM / reference={actual[j]/reference:.5g}; '
                f'C ideal / reference={ideal[j]/reference:.5g}. '
                'Both use the same incident phase, with no phase fitting. '
                'These are field coefficients, not efficiencies.')
            shape = (2*c.Mx+1, 2*c.My+1)
            relative = np.where((abs(actual)>1e-6)&(abs(ideal)>1e-6), np.angle(actual*ideal.conj()), np.nan)
            values = (abs(actual), abs(ideal), abs(actual-ideal),
                      _phase(actual/reference), _phase(ideal/reference), relative)
            for i, (im, value) in enumerate(zip(self._images[1], values)):
                im.set_data(value.reshape(shape).T)
                if i < 3:
                    maximum = max(.01, float(np.max(abs(actual))), float(np.max(abs(ideal)))) if i < 2 else max(.01,float(np.max(value)))
                    im.set_clim(0, maximum)
            for marker in self._markers[1]:
                marker.set_data([m], [n])
        else:
            self.order_readout.value = ''
        if tab == 2:
            for i, (im, value) in enumerate(zip(self._images[2],
                    (abs(physical_field), abs(ideal_field), _phase(physical_field), _phase(ideal_field)))):
                im.set_data(value)
                if i < 2:
                    im.set_clim(0, self._amplitude_max)
        if tab is not None:
            buffer = io.BytesIO()
            self.figures[tab].savefig(buffer, format='png', facecolor='white')
            self.pictures[tab].value = buffer.getvalue()

    def _reset(self, _):
        self.distance.value = 0.
        self.order.value = self.data.solution.incident_index
        self.tabs.selected_index = 0

    def close(self):
        if self._closed:
            return
        self._closed = True
        for control in (self.distance, self.order, self.tabs):
            control.unobserve(self._update, names='selected_index' if control is self.tabs else 'value')
        self.reset_button.on_click(self._reset, remove=True)
        from .grating_intro import _close_widget_tree
        _close_widget_tree(self.ui)
        for figure in self.figures:
            figure.clear()
