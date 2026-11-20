"""Small, independent amplitude/intensity lessons for notebook six.

Field values use a fixed arbitrary unit. Relative intensity refers to a
unit-amplitude propagating wave in the same medium and direction. All figures
use fixed scales and are rendered in memory; importing creates no widgets.
"""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .diffraction import diffraction_orders
from .grating_intro import _close_widget_tree, _uniform_field_image
from .plotting import figure_png
from .waves import field_from_phasor, plane_wave_phasor, wavenumber
from ._view_cache import ViewCache


BLUE, ORANGE, INK, GREY = '#2866a3', '#b85c17', '#243444', '#727c86'
STAGES = ('amplitude', 'phase', 'phasor', 'average', 'intensity', 'order')


def cycle_samples(amplitude=1.0, phase_deg=0.0, sample_count=400):
    """Sample one complete cycle once, with no duplicated endpoint.

    The local phasor is A exp(i phi), and the real field uses exp(-i omega t).
    ``mean_square`` is an average of the sampled real field, whereas
    ``relative_intensity`` is the phasor magnitude squared in reference units.
    """
    amplitude, phase_deg = float(amplitude), float(phase_deg)
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError('Amplitude must be finite and nonnegative.')
    if not np.isfinite(phase_deg):
        raise ValueError('Phase must be finite.')
    if not isinstance(sample_count, (int, np.integer)) or sample_count < 8:
        raise ValueError('Use at least eight equally spaced samples per cycle.')
    tau = np.linspace(0, 1, sample_count, endpoint=False)
    phasor = plane_wave_phasor([0.0], [1.0], amplitude=amplitude,
                              phase=np.deg2rad(phase_deg))
    field = field_from_phasor(phasor, time=tau, omega=2*np.pi)
    return dict(tau=tau, field=field, phasor=complex(phasor),
                mean_field=float(np.mean(field)),
                mean_square=float(np.mean(field**2)),
                relative_intensity=float(abs(phasor)**2))


def example_order_vector():
    """Notebook-six m=+1: lambda=2 um, d=4 um, normal incidence, toward -z."""
    data = diffraction_orders(2.0, 4.0, 0.0, 0.0, orders=[1])
    direction = data['directions'][0].copy()
    direction[2] *= -1  # Adapt the shared helper's +z convention to the series.
    return wavenumber(2.0)*direction


def _style(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(alpha=.14)
    axis.tick_params(labelsize=10)


def _cycle_axis(axis, data):
    # Close the plotted curve only; the average never counts t/T=1 twice.
    tau = np.append(data['tau'], 1.0)
    field = np.append(data['field'], data['field'][0])
    axis.plot(tau, field, color=BLUE, lw=2.2, label='Selected wave')
    axis.axhline(0, color=GREY, lw=.8)
    axis.set(xlim=(0, 1), ylim=(-2.25, 2.25), xticks=[0, .25, .5, .75, 1],
             xlabel='Time t/T at one fixed position', ylabel='Signed field u (field units)')
    _style(axis)


class AmplitudeIntensityExplorer:
    """One teaching view with its own parameters, allocated only when requested."""

    def __init__(self, stage='amplitude'):
        import ipywidgets as widgets

        if stage not in STAGES:
            raise ValueError(f'Stage must be one of {STAGES}.')
        self.stage, self._closed = stage, False
        self._cache = ViewCache()
        self._order_mesh = self._order_phasor = None
        style = {'description_width': 'initial'}
        self.amplitude = widgets.FloatSlider(value=1, min=0, max=2, step=.1,
            description='Amplitude A (field units)', readout_format='.1f',
            continuous_update=False, style=style,
            layout=widgets.Layout(width='340px', max_width='100%'))
        controls = [self.amplitude]
        self.phase = None
        if stage in ('phase', 'phasor', 'average', 'order'):
            self.phase = widgets.FloatSlider(value=0, min=-180, max=180, step=15,
                description='Phase offset (degrees)', readout_format='.0f',
                continuous_update=False, style=style,
                layout=widgets.Layout(width='310px', max_width='100%'))
            controls.append(self.phase)
        self.time = None
        if stage in ('phasor', 'order'):
            self.time = widgets.FloatSlider(value=0, min=0, max=1, step=.025,
                description='Model time t/T', readout_format='.3f', continuous_update=False,
                style=style, layout=widgets.Layout(width='280px', max_width='100%'))
            controls.append(self.time)
        self._controls = tuple(controls)
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1080px', height='auto'))
        row = widgets.Box(controls, layout=widgets.Layout(
            display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([row, self.readout, self.picture])
        height = 6.0 if stage == 'average' else 4.4
        self.figure = Figure(figsize=(10.0, height), dpi=110, facecolor='white')
        FigureCanvasAgg(self.figure)
        for control in controls:
            control.observe(self._update, names='value')
        self._update()

    def _update(self, change=None):
        if self._closed:
            return
        a = self.amplitude.value
        phase_deg = self.phase.value if self.phase is not None else 0.0
        tau = self.time.value if self.time is not None else 0.0
        self.data = self._cache.get('cycle', (a, phase_deg), lambda: cycle_samples(a, phase_deg))
        if (self.stage == 'order' and self._order_mesh is not None
                and change is not None and change.get('owner') is self.time):
            self._order_mesh.set_data(field_from_phasor(self._order_phasor, time=tau, omega=2*np.pi))
            self.picture.value = figure_png(self.figure)
            return
        self._order_mesh = None
        self.figure.clear()
        self.figure.subplots_adjust(left=.10, right=.96, bottom=.17, top=.85,
                                    wspace=.42, hspace=.55)
        getattr(self, f'_draw_{self.stage}')(a, phase_deg, tau)
        self.picture.value = figure_png(self.figure)

    def _draw_amplitude(self, a, phase_deg, tau):
        axis = self.figure.add_subplot(111)
        _cycle_axis(axis, self.data)
        axis.plot(self.data['tau'], np.cos(2*np.pi*self.data['tau']),
                  color=GREY, ls='--', lw=1.2, label='Reference: A = 1')
        axis.axhline(a, color=ORANGE, ls=':', lw=1.4)
        axis.axhline(-a, color=ORANGE, ls=':', lw=1.4)
        axis.set_title('Change the height; keep one cycle in the same time')
        axis.legend(loc='upper right', fontsize=9)
        self.readout.value = (f'Field limits: <b>{-a:.1f} to +{a:.1f}</b> field units. '
            'The vertical scale stays fixed. Dashed and solid curves are alternative settings of one wave.')

    def _draw_phase(self, a, phase_deg, tau):
        axis = self.figure.add_subplot(111)
        _cycle_axis(axis, self.data)
        axis.plot(self.data['tau'], a*np.cos(2*np.pi*self.data['tau']),
                  color=GREY, ls='--', lw=1.2, label='Same A, phase offset = 0')
        axis.axhline(a, color=ORANGE, ls=':', lw=1)
        axis.axhline(-a, color=ORANGE, ls=':', lw=1)
        axis.set_title('Change phase: move the cycle while keeping its height')
        axis.legend(loc='upper right', fontsize=9)
        self.readout.value = (f'<b>A = {a:.1f}</b>; phase offset = <b>{phase_deg:+.0f}°</b>; '
            f'field at t = 0: <b>{self.data["phasor"].real:+.3f}</b>. '
            + ('At A = 0 the field vanishes; its phase is undefined.' if a == 0 else
               'Both curves have the same amplitude. They are compared, not added.'))

    def _draw_phasor(self, a, phase_deg, tau):
        complex_axis, time_axis = self.figure.subplots(1, 2)
        phasor = self.data['phasor']
        rotating = phasor*np.exp(-2j*np.pi*tau)
        complex_axis.axhline(0, color=GREY, lw=.8)
        complex_axis.axvline(0, color=GREY, lw=.8)
        if a > 0:
            circle = np.linspace(0, 2*np.pi, 240)
            complex_axis.plot(a*np.cos(circle), a*np.sin(circle), color=GREY, lw=.8, alpha=.5)
            complex_axis.annotate('', xy=(phasor.real, phasor.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.4))
            complex_axis.annotate('', xy=(rotating.real, rotating.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.6, linestyle='--'))
            complex_axis.plot([rotating.real, rotating.real], [0, rotating.imag],
                              color=ORANGE, lw=1, ls=':')
        complex_axis.plot(rotating.real, 0, 'o', color=ORANGE, ms=6)
        complex_axis.set(xlim=(-2.35, 2.35), ylim=(-2.35, 2.35), aspect='equal',
            xlabel='Real part (field units)', ylabel='Imaginary part (field units)',
            title='Complex-amplitude plane')
        complex_axis.plot([], [], color=BLUE, lw=2, label='Fixed phasor U')
        complex_axis.plot([], [], color=ORANGE, ls='--', label='With time factor')
        complex_axis.legend(loc='upper center', bbox_to_anchor=(.5, -.22),
                            ncol=1, fontsize=8, frameon=False)
        _style(complex_axis)
        _cycle_axis(time_axis, self.data)
        time_axis.plot(tau, rotating.real, 'o', color=ORANGE, ms=7)
        time_axis.set_title('Real part gives the field reading')
        self.figure.subplots_adjust(left=.08, bottom=.24, top=.87, wspace=.38)
        phase_note = 'Phase undefined at zero amplitude.' if a == 0 else f'Phase of U: {phase_deg:+.0f}°.'
        self.readout.value = (f'<b>U = {phasor.real:+.3f} {phasor.imag:+.3f}i</b>; '
            f'|U| = <b>{abs(phasor):.3f}</b>; u at t/T = {tau:.3f}: '
            f'<b>{rotating.real:+.3f}</b>. {phase_note}')

    def _draw_average(self, a, phase_deg, tau):
        field_axis, square_axis = self.figure.subplots(2, 1)
        _cycle_axis(field_axis, self.data)
        field_axis.set_title('Model field at one observation point')
        field_axis.axhline(self.data['mean_field'], color=ORANGE, ls='--',
                           label='Cycle average = 0')
        field_axis.legend(loc='upper right', fontsize=9)
        tt = np.append(self.data['tau'], 1.0)
        squared = np.append(self.data['field']**2, self.data['field'][0]**2)
        square_axis.plot(tt, squared, color=BLUE, lw=2)
        square_axis.axhline(self.data['mean_square'], color=ORANGE, lw=1.6,
                            ls='--', label=f'Cycle average = {self.data["mean_square"]:.3f}')
        square_axis.set(xlim=(0, 1), ylim=(-.15, 4.4), xticks=[0, .25, .5, .75, 1],
            xlabel='Time t/T at the same position', ylabel='Squared field u² (field units²)',
            title='Squared field and its cycle average')
        _style(square_axis)
        square_axis.legend(loc='upper right', fontsize=9)
        self.figure.subplots_adjust(left=.10, bottom=.12, top=.91, hspace=.68)
        self.readout.value = (f'<b>Mean of u: {self.data["mean_field"]:.3f}</b>; '
            f'<b>mean of u²: {self.data["mean_square"]:.3f}</b> = A²/2. '
            'Detector signal is proportional to the mean of u². '
            'The panels have different vertical quantities and units.')

    def _draw_intensity(self, a, phase_deg, tau):
        axis = self.figure.add_subplot(111)
        amplitudes = np.linspace(0, 2, 241)
        axis.plot(amplitudes, amplitudes**2, color=BLUE, lw=2.3)
        axis.plot([a, a], [0, a*a], color=ORANGE, ls=':', lw=1.2)
        axis.plot([0, a], [a*a, a*a], color=ORANGE, ls=':', lw=1.2)
        axis.plot(a, a*a, 'o', color=ORANGE, ms=8)
        axis.set(xlim=(-.04, 2.1), ylim=(-.12, 4.35), xticks=[0, .5, 1, 1.5, 2],
            yticks=[0, 1, 2, 3, 4], xlabel='Amplitude A (field units)',
            ylabel='Relative intensity I / I_ref', title='Intensity grows as the square of amplitude')
        _style(axis)
        self.readout.value = (f'<b>A = {a:.1f}</b> field units → '
            f'<b>I / I_ref = {a*a:.2f}</b>. Reference: A_ref = 1 field unit, '
            'with the same medium and propagation direction.')

    def _draw_order(self, a, phase_deg, tau):
        field_axis, detector_axis = self.figure.subplots(1, 2)
        def order_grid():
            kv = example_order_vector()
            x, z = np.linspace(-3, 3, 241), np.linspace(-3, 0, 181)
            xx, zz = np.meshgrid(x, z)
            points = np.stack((xx, np.zeros_like(xx), zz), axis=-1)
            return kv, x, z, plane_wave_phasor(points, kv)

        kv, x, z, unit_field = self._cache.get('order_grid', (), order_grid)
        phasor = a*np.exp(1j*np.deg2rad(phase_deg))*unit_field
        self._order_phasor = phasor
        field = field_from_phasor(phasor, time=tau, omega=2*np.pi)
        picture = _uniform_field_image(field_axis, x, z, field, vmin=-2, vmax=2, cmap='RdBu_r')
        self._order_mesh = picture
        field_axis.axhline(0, color=INK, lw=2, clip_on=False)
        detector_z = -1.5
        detector_colour = '#12bbd5'
        field_axis.axhline(detector_z, color=detector_colour, lw=2, ls='--')
        field_axis.text(-2.85, detector_z + .12, 'Detector plane', color=INK,
                        fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
        field_axis.set(xlim=(-3, 3), ylim=(-3, 0), aspect='equal', xlabel='x (µm)',
                       ylabel='z (µm)', title='Instantaneous field: xz slice')
        self.figure.colorbar(picture, ax=field_axis, orientation='horizontal', pad=.25,
                            fraction=.06, ticks=[-2, 0, 2], label='Signed field u (field units)')
        direction = kv/wavenumber(2.0)
        start = np.array([-2.2, -.15])
        field_axis.annotate('', xy=start + direction[[0, 2]], xytext=start,
            arrowprops=dict(arrowstyle='->', color=INK if a > 0 else GREY, lw=2,
                            linestyle='-' if a > 0 else '--'))
        y = np.linspace(-1.5, 1.5, 121)
        # A uniform plane wave has |U|² = A² on the whole detector.
        # The reference has the same incidence direction, so its projection cancels.
        relative_intensity = np.full((len(y), len(x)), a*a)
        detector = _uniform_field_image(detector_axis, x, y, relative_intensity,
                                        vmin=0, vmax=4, cmap='gray')
        detector_axis.set(xlim=(-3, 3), ylim=(-1.5, 1.5), aspect='equal',
            xlabel='x (µm)', ylabel='y (µm)',
            title='Averaged detector image: xy plane')
        detector_axis.text(.5, .5, f'I / I_ref = {a*a:.2f}',
                           ha='center', va='center', transform=detector_axis.transAxes,
                           color=INK if a*a > 2 else 'white', fontsize=11)
        for spine in detector_axis.spines.values():
            spine.set_color(detector_colour)
            spine.set_linewidth(1.5)
        self.figure.colorbar(detector, ax=detector_axis, orientation='horizontal', pad=.25,
                            fraction=.06, ticks=[0, 1, 2, 3, 4], label='Relative intensity I / I_ref')
        self.figure.subplots_adjust(left=.075, right=.97, top=.90, bottom=.17, wspace=.35)
        self.readout.value = (f'λ = 2 µm; d = 4 µm; normal incidence; m = +1. '
            f'<b>Assigned A_m = {a:.1f}</b>; <b>I_m / I_ref = {a*a:.2f}</b>. '
            + ('At A_m = 0 there is no field; phase is undefined.' if a == 0 else
               'Phase and model time move the field bands; detector brightness stays fixed.'))

    def close(self):
        """Release all owned figures, images, widgets and callbacks; safe twice."""
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        self._order_mesh = self._order_phasor = None
        self._cache.clear()
        _close_widget_tree(self.ui)
