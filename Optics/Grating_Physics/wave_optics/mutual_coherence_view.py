"""Experimental layouts and compact supplemental views for Notebook 7 V2."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from . import mutual_coherence as model
from .grating_intro import _close_widget_tree
from .plotting import figure_png
from .superposition import two_wave_cycle


STAGES = ('fields', 'coherence', 'visibility', 'phase', 'exposure', 'orders')


class MutualCoherenceExplorer:
    """One figure, fixed axes, retained artists and a single callback per control."""

    def __init__(self, stage):
        import ipywidgets as widgets
        if stage not in STAGES:
            raise ValueError('Unknown mutual-coherence lesson stage.')
        self.stage, self._closed, self._suspend_updates = stage, False, False
        self.phase = self.ratio = self.magnitude = self.duty = self.state = None
        self._controls = []

        def slider(label, default, low, high, step):
            control = widgets.FloatSlider(description=label, value=default, min=low,
                max=high, step=step, continuous_update=False,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='350px', max_width='100%'))
            self._controls.append(control)
            return control

        if stage in ('fields', 'phase', 'orders'):
            self.phase = slider('Relative phase / EO β (°)', 0, 0, 360, 5)
        if stage in ('fields', 'orders'):
            self.ratio = slider('Amplitude ratio A₂/A₁' if stage == 'fields' else '|C₁|/|C₀|', 1, 0, 2, .05)
        if stage == 'coherence':
            self.magnitude = widgets.ToggleButtons(options=[('1', 1.), ('0.5', .5), ('0', 0.)],
                value=1., description='Prescribed |γ|')
            self._controls.append(self.magnitude)
        if stage == 'visibility':
            self.ratio = slider('Intensity ratio I₂/I₁', 1, 0, 4, .05)
        if stage in ('visibility', 'orders'):
            self.magnitude = slider('Prescribed |γ|', 1, 0, 1, .05)
        if stage == 'exposure':
            self.phase = slider('Switched phase β (°)', 180, 0, 180, 5)
            self.duty = slider('Fraction of exposure at 0°', .5, 0, 1, .05)
            self.state = widgets.ToggleButtons(options=[('0° setting', 0), ('β setting', 1)],
                                               value=0, description='Resolved state')
            self._controls.append(self.state)
        self._defaults = [(control, control.value) for control in self._controls]
        self.reset_button = widgets.Button(description='Reset', icon='undo')
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1000px'))
        self.ui = widgets.VBox([widgets.HBox(self._controls, layout=widgets.Layout(flex_flow='row wrap')),
                               self.reset_button, self.picture, self.readout])
        self.figure = Figure(figsize=(10, 3.3), dpi=110, facecolor='white')
        FigureCanvasAgg(self.figure)
        self.axes = self.figure.subplots(1, 2)
        self.figure.subplots_adjust(left=.065, right=.97, bottom=.20, top=.85, wspace=.30)
        self.x = np.linspace(-2, 2, 801) if stage != 'orders' else np.linspace(-8, 8, 801)
        self.basis = model.two_order_basis(self.x) if stage == 'orders' else None
        self.spatial_phase = 2*np.pi*self.x if self.basis is None else self.basis['spatial_phase']
        self.vmax = 9. if stage in ('fields', 'visibility') else 1.5 if stage == 'orders' else 4.
        self.artists = []
        if stage == 'fields':
            for label, color in [('u₁', '#2866a3'), ('u₂', '#b85c17'), ('u₁+u₂', '#29313c')]:
                self.artists.append(self.axes[0].plot([], [], label=label, color=color,
                    linestyle='--' if label == 'u₂' else '-', linewidth=1.6 if label == 'u₂' else 2.2)[0])
            self.axes[0].set(xlim=(0, 1), ylim=(-3.2, 3.2), xlabel='Model time t/T', ylabel='Signed field')
            self.axes[0].set_title('Two fields and their coherent sum')
            self.axes[1].set_title('Add fields, then calculate intensity')
            self.axes[0].legend(loc='upper right', fontsize=8)
            self.bars = self.axes[1].bar(['I₁', 'I₂', 'I₁+I₂', 'I(sum)'], [1, 1, 2, 4],
                                       color=['#2866a3', '#b85c17', '#999999', '#29313c'])
            self.axes[1].set(ylim=(0, 9), ylabel='Relative scalar intensity')
        else:
            self.image = self.axes[0].imshow(np.zeros((24, 801)), origin='lower',
                extent=(self.x[0], self.x[-1], 0, 1), cmap='gray', vmin=0, vmax=self.vmax,
                interpolation='nearest', aspect='auto')
            self.axes[0].set(yticks=[], xlabel='x (μm)' if stage == 'orders' else 'x/Λ')
            self.axes[0].set_title('Resolved detector pattern')
            self.line = self.axes[1].plot(self.x, np.zeros_like(self.x), color='#29313c', label='Recorded')[0]
            self.reference = self.axes[1].axhline(2., color='#999999', linestyle='--', label='I₁+I₂')
            self.axes[1].set(xlim=(self.x[0], self.x[-1]), ylim=(0, self.vmax),
                xlabel='x (μm)' if stage == 'orders' else 'x/Λ', ylabel='Relative scalar intensity')
            self.axes[1].legend(loc='upper right', fontsize=8)
        for control in self._controls:
            control.observe(self._update, names='value')
        self.reset_button.on_click(self._reset)
        self._update()

    def _update(self, change=None):
        if self._closed or self._suspend_updates:
            return
        beta = self.phase.value if self.phase is not None else 0.
        ratio = self.ratio.value if self.ratio is not None else 1.
        magnitude = self.magnitude.value if self.magnitude is not None else 1.
        if self.stage == 'fields':
            self.data = two_wave_cycle(1., ratio, beta)
            for artist, y in zip(self.artists, (*self.data['fields'], self.data['field_sum'])):
                artist.set_data(self.data['tau'], y)
            for bar, value in zip(self.bars, [1, ratio**2, 1+ratio**2, self.data['relative_intensity']]):
                bar.set_height(value)
            note = (f'I₁+I₂ = {1+ratio**2:.3f}; cross term = {self.data["interference_term"]:+.3f}; '
                    f'<b>I(sum) = {self.data["relative_intensity"]:.3f}</b>. Fields add before squaring.')
        else:
            i1, i2 = (.16, (.4*ratio)**2) if self.stage == 'orders' else (1., ratio if self.stage == 'visibility' else 1.)
            gamma = complex(magnitude)*np.exp(1j*np.deg2rad(beta))
            if self.stage == 'exposure':
                gamma = model.phase_average([0, beta], weights=[self.duty.value, 1-self.duty.value])
                recorded = model.averaged_intensity(i1, i2, gamma, self.spatial_phase)
                current = model.averaged_intensity(i1, i2, 1., self.spatial_phase + np.deg2rad(beta*self.state.value))
                image = current
                self.axes[0].set_title('Time-resolved setting (fully coherent)')
                self.axes[1].set_title('Completed exposure average')
                prefix = (f'Exposure weights: ({self.duty.value:.2f}, {1-self.duty.value:.2f}). '
                          'Changing the resolved setting leaves the completed average fixed. ')
            else:
                recorded = (model.two_order_intensity(self.basis, ratio, beta, magnitude)
                            if self.stage == 'orders' else model.averaged_intensity(i1, i2, gamma, self.spatial_phase))
                image, prefix = recorded, ''
                self.axes[1].set_title('Resolved intensity profile')
            visibility = model.fringe_visibility(i1, i2, min(1., abs(gamma)))
            gamma_phase = None if abs(gamma) < 1e-12 else float(np.rad2deg(np.angle(gamma)))
            defined = i1 > 0 and i2 > 0
            self.data = dict(intensity=recorded, gamma=gamma if defined else None,
                             coherence_phase_deg=gamma_phase if defined else None,
                             visibility=visibility, intensity1=i1, intensity2=i2, phase_deg=beta)
            self.image.set_data(np.broadcast_to(image, (24, image.size)))
            self.line.set_ydata(recorded)
            self.reference.set_ydata([i1+i2, i1+i2])
            correlation = (f'|γ| = {abs(gamma):.3f}; phase of γ at the reference = '
                           + (f'{gamma_phase:.1f}°' if gamma_phase is not None else 'undefined (γ = 0)')) if defined else 'γ is undefined: one arm is dark'
            if self.stage == 'orders':
                # Reference correlation is defined before the fixed propagation phase.
                prefix += 'm = (0,+1), λ = 2 μm, d = 4 μm; fringe period = 4 μm. '
            note = prefix + f'<b>{correlation}</b>; V = {visibility:.3f}. '
            note += ('Prescribed correlation; no stochastic source is being simulated.'
                     if self.stage in ('coherence', 'visibility', 'orders') else
                     'Fixed phase changes correlation phase, not its magnitude.' if self.stage == 'phase' else
                     'Reduced exposure contrast comes from unresolved phase variation, not intrinsic incoherence.')
        self.readout.value = note
        self.picture.value = figure_png(self.figure)

    def _reset(self, button=None):
        if self._closed:
            return
        self._suspend_updates = True
        try:
            for control, default in self._defaults:
                control.value = default
        finally:
            self._suspend_updates = False
        self._update()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.reset_button.on_click(self._reset, remove=True)
        self.picture.value = b''
        self.data = self.basis = self.spatial_phase = None
        self.figure.clear()
        self.figure = None
        self.axes = ()
        self.image = self.line = self.reference = self.bars = None
        self.x = None
        self.artists.clear()
        _close_widget_tree(self.ui)


class ApparatusBackground:
    """Lazy, independently configured reuse of Notebook 7's tested apparatus."""

    def __init__(self):
        import ipywidgets as widgets
        self._closed, self.apparatus = False, None
        self.panel = widgets.VBox()
        self.ui = widgets.Accordion(children=[self.panel], selected_index=None)
        self.ui.set_title(0, 'Open optional experimental apparatus')
        self.ui.observe(self._open, names='selected_index')

    def _open(self, change=None):
        if self._closed or self.ui.selected_index is None or self.apparatus is not None:
            return
        import ipywidgets as widgets
        from .interference_setup import MirrorTiltExplorer
        self.apparatus = MirrorTiltExplorer(wavelength_nm=632.8, power_mw=1.,
            diameter_mm=.54, crossing_diameter_mm=.54, mm_per_unit=1.8, optics_phase_deg=0.)
        a = self.apparatus
        self.panel.children = (widgets.HTML('Schematic geometry; Gaussian footprints and resolved detector views reuse Notebook 7. '
            'Tilt changes crossing angle and overlap; EO changes phase. This monochromatic model assumes full mutual coherence.'),
            a.modulated_ui, a.detector_modulated_ui)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.ui.unobserve(self._open, names='selected_index')
        if self.apparatus is not None:
            self.apparatus.close()
        _close_widget_tree(self.ui)


def _close_owned_widgets(widgets):
    """Close only adapter-owned models; shared apparatus children have one owner."""
    for widget in reversed(widgets):
        for attribute in ('layout', 'style'):
            value = getattr(widget, attribute, None)
            if value is not None:
                value.close()
        widget.close()


class ExperimentalInterferometer:
    """V2 layout adapter around Notebook 7's original linked experimental figures."""

    def __init__(self):
        import ipywidgets as widgets
        from .interference_setup import MirrorTiltExplorer
        self._closed = False
        self.apparatus = MirrorTiltExplorer(wavelength_nm=632.8, power_mw=1.,
            diameter_mm=.54, crossing_diameter_mm=.54, mm_per_unit=1.8, optics_phase_deg=0.)
        a = self.apparatus
        self.reset_button = widgets.Button(description='Reset apparatus', icon='undo')
        self.reset_button.on_click(self._reset)
        self.ui = widgets.VBox([a.tilt, a.detector_z, self.reset_button, a.ui])
        self._owned = [self.reset_button, self.ui]
        self._detector_ui = None

    @property
    def detector_ui(self):
        import ipywidgets as widgets
        if self._closed:
            raise RuntimeError('Rerun the setup cell before opening the detector.')
        if self._detector_ui is None:
            # Original renderer, mode selector, overview and readout; geometry
            # controls are displayed once, beside the opening apparatus.
            self._detector_ui = widgets.VBox(self.apparatus.detector_ui.children[2:])
            self._owned.append(self._detector_ui)
        return self._detector_ui

    def _reset(self, button=None):
        if self._closed:
            return
        a = self.apparatus
        changed = a.tilt.value != 0 or a.detector_z.value != -6.9
        # A reset is one geometry update, rather than one redraw per slider.
        for control in a._controls:
            control.unobserve(a._update, names='value')
        try:
            a.tilt.value, a.detector_z.value = 0., -6.9
        finally:
            for control in a._controls:
                control.observe(a._update, names='value')
        if changed:
            a._update()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.reset_button.on_click(self._reset, remove=True)
        self.apparatus.close()
        _close_owned_widgets(self._owned)


class ExperimentalEOView:
    """Disposable EO controls sharing the opening interferometer's physical state."""

    def __init__(self, experiment):
        import ipywidgets as widgets
        self._closed = False
        self.experiment = experiment
        a = experiment.apparatus
        # Force the original lazy views once; their numeric caches and artists
        # remain owned by the shared apparatus, not by this disposable layout.
        setup = a.modulated_ui
        detector = a.detector_modulated_ui
        self.phase = widgets.FloatSlider(value=a.eo_phase.value, min=0, max=180,
            step=5, description='Fixed EO phase β (°)', continuous_update=False,
            style={'description_width': 'initial'}, layout=widgets.Layout(width='550px', max_width='100%'))
        self.reset_button = widgets.Button(description='Reset EO phase', icon='undo')
        self.note = widgets.HTML('Fixed phase: |γ₁₂| = 1. Geometry and local arm intensities stay fixed. '
            'The optional exposure panel below uses its own explicitly averaged phase sequence.')
        self.ui = widgets.VBox([self.phase, self.reset_button, self.note, setup,
                               *detector.children[3:]])
        self._owned = [self.phase, self.reset_button, self.note, self.ui]
        self.phase.observe(self._update, names='value')
        a.eo_phase.observe(self._sync, names='value')
        self.reset_button.on_click(self._reset)

    def _update(self, change=None):
        if not self._closed and not self.experiment._closed:
            self.experiment.apparatus.eo_phase.value = self.phase.value

    def _sync(self, change):
        if not self._closed:
            self.phase.value = change['new']

    def _reset(self, button=None):
        if not self._closed:
            self.phase.value = 0.

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.phase.unobserve(self._update, names='value')
        self.experiment.apparatus.eo_phase.unobserve(self._sync, names='value')
        self.reset_button.on_click(self._reset, remove=True)
        _close_owned_widgets(self._owned)


class YoungBackground:
    """Keep the original extended-source experiment off the main reading path."""

    def __init__(self):
        import ipywidgets as widgets
        self._closed, self.explorer = False, None
        self.panel = widgets.VBox()
        self.ui = widgets.Accordion(children=[self.panel], selected_index=None)
        self.ui.set_title(0, 'Optional: extended-source Young experiment')
        self.ui.observe(self._open, names='selected_index')

    def _open(self, change=None):
        if self._closed or self.ui.selected_index is None or self.explorer is not None:
            return
        from .spatial_coherence import SpatialCoherenceExplorer
        self.explorer = SpatialCoherenceExplorer(wavelength_nm=632.8,
            source_distance_mm=500., detector_distance_mm=1000.)
        self.panel.children = (self.explorer.ui,)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.ui.unobserve(self._open, names='selected_index')
        if self.explorer is not None:
            self.explorer.close()
        _close_widget_tree(self.ui)
