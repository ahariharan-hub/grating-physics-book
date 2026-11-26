"""Independent illustrations for the scalar-optics foundations notebook."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.colors import PowerNorm

from .grating_intro import _close_widget_tree
from .plotting import figure_png
from .scalar_theory import (
    boundary_family, helmholtz_trial, scalar_solution, slit_far_field, slit_propagation,
)
from ._view_cache import ViewCache


BLUE, ORANGE, INK, GREY, PURPLE = '#2866a3', '#b85c17', '#243444', '#727c86', '#79539e'
STAGES = ('helmholtz', 'solutions', 'superposition', 'boundary', 'diffraction')


def _style(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(alpha=.14)
    axis.tick_params(labelsize=9)
    axis.title.set_fontsize(11)


class ScalarTheoryExplorer:
    """One section's controls, figure, data and disposable widget tree."""

    def __init__(self, stage='helmholtz'):
        import ipywidgets as widgets

        # Older open notebook cells used this name before the explorers merged.
        if stage == 'more_solutions':
            stage = 'solutions'
        if stage not in STAGES:
            raise ValueError(f'Stage must be one of {STAGES}.')
        self.stage, self._closed, self._resetting = stage, False, False
        self._cache = ViewCache()
        self._time_image = self._time_cursor = None
        self._superposition_lines = None
        self._last_time = 0.
        self.amplitude = self.phase = self.time = self.ratio = None
        self.solution = self.fraction = self.width = None
        self.transverse = self.evanescent = self.degree = None
        self.waist = self.axial = None
        self.view = self.beam_axial = None
        controls = []

        def slider(description, value, lower, upper, step, fmt):
            control = widgets.FloatSlider(description=description, value=value,
                min=lower, max=upper, step=step, readout_format=fmt,
                continuous_update=False, style={'description_width': 'initial'},
                layout=widgets.Layout(width='340px', max_width='100%'))
            controls.append(control)
            return control

        self.wavelength = (slider('Wavelength λ (µm)', .6, .4, .8, .025, '.3f')
            if stage == 'diffraction' else slider('Wavelength λ (µm)', 2, 1, 3, .25, '.2f'))
        if stage in ('helmholtz', 'solutions'):
            self.amplitude = slider('Amplitude A', 1, 0, 1, .1, '.1f')
        if stage == 'superposition':
            self.amplitude = slider('Backward-wave amplitude', .5, 0, 1, .1, '.1f')
        if stage != 'diffraction':
            self.phase = slider('Relative phase φ₂ (°)' if stage == 'superposition' else 'Phase offset φ (°)',
                                60 if stage == 'superposition' else 0, 0, 360, 15, '.0f')
        if stage == 'helmholtz':
            self.ratio = slider('Trial q/k', 1, .5, 1.5, .05, '.2f')
        if stage == 'solutions':
            self.solution = widgets.Dropdown(description='Solution',
                options=[('Plane wave', 'plane'), ('Standing wave', 'standing'),
                         ('Spherical wave', 'spherical'), ('Cylindrical wave', 'cylindrical'),
                         ('Bessel beam', 'bessel'), ('Spherical multipole', 'multipole'),
                         ('Evanescent wave', 'evanescent'),
                         ('Gaussian beam (paraxial)', 'gaussian')], value='plane',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='340px', max_width='100%'))
            controls.append(self.solution)
            self.view = widgets.ToggleButtons(description='View',
                options=[('Field & amplitude', 'field'), ('Beam intensity', 'beam')], value='field',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='440px', max_width='100%'))
            controls.append(self.view)
            self.transverse = slider('Transverse k⊥/k', .6, .15, .95, .05, '.2f')
            self.evanescent = slider('Evanescent q/k', 1.25, 1.05, 2, .05, '.2f')
            self.degree = widgets.IntSlider(description='Angular degree ℓ', value=1,
                min=0, max=2, step=1, continuous_update=False,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='340px', max_width='100%'))
            controls.append(self.degree)
            self.waist = slider('Waist w₀/λ', 2, 2, 6, .25, '.2f')
            self.axial = slider('Axial plane z/z_R', 0, -2, 2, .1, '.1f')
            self.beam_axial = slider('Observation z/λ', 0, -10, 10, .5, '.1f')
        if stage in ('solutions', 'boundary', 'superposition'):
            self.time = slider('Model time t/T', 0, 0, 1, .025, '.3f')
        if stage == 'boundary':
            self.fraction = slider('Backward fraction b', 0, 0, 1, .05, '.2f')
        if stage == 'diffraction':
            self.width = slider('Slit width W_open (µm)', 20, 10, 40, 1, '.0f')
        self._controls = tuple(controls)
        self._defaults = tuple((control, control.value) for control in controls)
        self.reset_button = widgets.Button(description='Reset this example',
                                          layout=widgets.Layout(width='180px'))
        self.reset_button.on_click(self._reset)
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png',
            layout=widgets.Layout(width='100%', max_width='1100px', height='auto'))
        self.ui = widgets.VBox([
            widgets.Box([*controls, self.reset_button], layout=widgets.Layout(
                display='flex', flex_flow='row wrap', align_items='center')),
            self.readout, self.picture])
        self.figure = Figure(figsize=(10, 5.9 if stage in ('solutions', 'boundary') else 5.3),
                             dpi=110, facecolor='white')
        FigureCanvasAgg(self.figure)
        self.data = None
        for control in controls:
            control.observe(self._update, names='value')
        self._update()

    def _update(self, change=None):
        if self._closed or self._resetting:
            return
        if change is not None and change.get('owner') is self.time:
            tau = self.time.value
            if self._superposition_lines is not None:
                signed = np.real(self.data['components']*np.exp(-2j*np.pi*tau))
                self.data['real_fields'] = signed
                for line, values in zip(self._superposition_lines, (*signed, signed.sum(axis=0))):
                    line.set_ydata(values)
                self.picture.value = figure_png(self.figure)
                return
            if self._time_cursor is not None:
                self.data['real_field'] = np.real(self.data['field']*np.exp(-2j*np.pi*tau))
                self._time_cursor.set_ydata([tau, tau])
                self.picture.value = figure_png(self.figure)
                self._last_time = tau
                return
            if self._time_image is not None:
                signed = np.real(self.data['field']*np.exp(-2j*np.pi*tau))
                self.data['signed_field'] = signed
                self._time_image.set_data(np.ma.array(signed, mask=self.data['mask']))
                self.readout.value = self.readout.value.replace(
                    f't/T = {self._last_time:.3f}', f't/T = {tau:.3f}')
                self.picture.value = figure_png(self.figure)
                self._last_time = tau
                return
            if self.stage == 'solutions' and self.view.value == 'beam' and self.solution.value in ('plane', 'bessel', 'gaussian'):
                return  # Hidden time control cannot alter a monochromatic intensity view.
        self._time_image = self._time_cursor = None
        self._superposition_lines = None
        self.figure.clear()
        self.figure.set_size_inches(10, 5.9 if self.stage in ('solutions', 'boundary') else 5.3)
        getattr(self, f'_draw_{self.stage}')()
        self.picture.value = figure_png(self.figure)
        self._last_time = self.time.value if self.time is not None else 0.

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

    def _draw_helmholtz(self):
        field_axis, residual_axis = self.figure.subplots(1, 2)
        x = np.linspace(-4, 4, 641)
        self.data = helmholtz_trial(x, self.wavelength.value, self.ratio.value,
                                   self.amplitude.value, self.phase.value)
        k, q = self.data['k'], self.data['q']
        field_axis.plot(x, self.data['field'].real, color=BLUE, lw=2.6, label='Re U')
        field_axis.plot(x, -self.data['second'].real/k**2, color=ORANGE,
                        ls='--', lw=1.8, label="Re(−U″ / k²)")
        field_axis.set(xlim=(-4, 4), ylim=(-2.4, 2.4), xlabel='x (µm)',
                       ylabel='Field units', title='Helmholtz requires these curves to agree')
        field_axis.legend(loc='upper center', bbox_to_anchor=(.5, -.22),
                          ncol=2, frameon=False, fontsize=9)
        residual_axis.axhline(0, color=GREY, lw=1)
        residual_axis.plot(x, self.data['residual'].real/k**2, color=PURPLE, lw=2.1)
        residual_axis.set(xlim=(-4, 4), ylim=(-1.4, 1.4), xlabel='x (µm)',
                          ylabel='Re[(U″ + k²U) / k²] (field units)',
                          title='The residual must vanish everywhere')
        for axis in (field_axis, residual_axis):
            _style(axis)
        self.figure.subplots_adjust(left=.08, right=.97, top=.88, bottom=.25, wspace=.43)
        residual_norm = np.max(np.abs(self.data['residual']))/k**2
        verdict = ('The zero field solves the equation for any trial q.' if self.amplitude.value == 0
                   else 'This trial is a Helmholtz solution.' if np.isclose(abs(q), k)
                   else 'This trial is not a Helmholtz solution at the selected k.')
        self.readout.value = (f'k = {k:.3f} rad/µm; q = {q:.3f} rad/µm; q/k = {q/k:.2f}. '
            f'<b>max |U″ + k²U| / k² = {residual_norm:.3f}</b> field units. {verdict} '
            'Derivatives here are evaluated analytically.')

    def _draw_superposition(self):
        spatial, intensity_axis = self.figure.subplots(1, 2)
        z = self._cache.get('addition_grid', (), lambda: np.linspace(0, 2, 801))
        key = (self.wavelength.value, self.amplitude.value, self.phase.value)

        def waves():
            k = 2*np.pi/self.wavelength.value
            return np.array([np.exp(1j*k*z), self.amplitude.value*np.exp(
                1j*(-k*z+np.deg2rad(self.phase.value)))])

        components = self._cache.get('addition_components', key, waves)
        field = components.sum(axis=0)
        intensity = self._cache.get('addition_intensity', key, lambda: np.abs(field)**2)
        signed = np.real(components*np.exp(-2j*np.pi*self.time.value))
        lines = []
        for values, color, label, style in zip((*signed, signed.sum(axis=0)),
                (BLUE, ORANGE, INK), ('u₁: forward', 'u₂: backward', 'u₁ + u₂'), ('--', ':', '-')):
            lines.append(spatial.plot(z, values, color=color, label=label, ls=style, lw=2)[0])
        self._superposition_lines = lines
        spatial.axhline(0, color=GREY, lw=.8)
        spatial.set(xlim=(0, 2), ylim=(-2.1, 2.1), xlabel='z (µm)', ylabel='Signed field (field units)',
                    title='At one instant: add signed fields')
        spatial.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2, frameon=False, fontsize=9)
        intensity_axis.plot(z, intensity, color=INK, lw=2.3, label='|U₁ + U₂|²')
        separate = 1+self.amplitude.value**2
        intensity_axis.axhline(separate, color=PURPLE, ls='--', label='|U₁|² + |U₂|²')
        intensity_axis.set(xlim=(0, 2), ylim=(0, 4.15), xlabel='z (µm)', ylabel='Relative scalar intensity',
                           title='After optical-cycle averaging')
        intensity_axis.legend(loc='upper center', bbox_to_anchor=(.5, -.2), frameon=False, fontsize=9)
        for axis in (spatial, intensity_axis):
            _style(axis)
        self.figure.subplots_adjust(left=.08, right=.97, top=.88, bottom=.29, wspace=.4)
        self.data = dict(z=z, components=components, field=field, real_fields=signed,
                         relative_intensity=intensity)
        self.readout.value = (
            '<b>Fields add at each point.</b> Move model time to watch the signed fields change; '
            'the cycle-averaged intensity stays fixed. The dashed reference adds the two separate intensities. '
            f'At these amplitudes, intensity ranges from {(1-self.amplitude.value)**2:.2f} '
            f'to {(1+self.amplitude.value)**2:.2f}.')

    def _draw_boundary(self):
        spatial, history = self.figure.subplots(1, 2)
        z, times = self._cache.get('boundary_grid', (),
            lambda: (np.linspace(0, 6, 481), np.linspace(0, 1, 161)))
        key = (self.wavelength.value, self.fraction.value, self.phase.value)
        base = self._cache.get('boundary_field', key,
            lambda: boundary_family(z, *key))
        self.data = dict(base, real_field=np.real(base['field']*np.exp(-2j*np.pi*self.time.value)))
        field = self.data['field']
        spatial.plot(z, field.real, color=BLUE, lw=2.4, label='Re U(z)')
        spatial.plot(z, field.imag, color=ORANGE, ls='--', lw=1.8, label='Im U(z)')
        spatial.plot(0, field[0].real, 'o', color=BLUE, ms=7, clip_on=False)
        spatial.plot(0, field[0].imag, 's', color=ORANGE, ms=5, clip_on=False)
        spatial.set(xlim=(0, 6), ylim=(-1.15, 1.15), xlabel='z (µm)',
                    ylabel='Field units', title='The same boundary value allows different fields')
        spatial.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=2,
                       frameon=False, fontsize=9)
        real_field = self._cache.get('boundary_history', key,
            lambda: np.real(field[None, :]*np.exp(-2j*np.pi*times[:, None])))
        picture = history.imshow(real_field, extent=(0, 6, 0, 1), origin='lower',
                                 aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        self._time_cursor = history.axhline(self.time.value, color=PURPLE, ls=':', lw=1.7)
        history.set(xlim=(0, 6), ylim=(0, 1), yticks=[0, .25, .5, .75, 1],
                    xlabel='z (µm)', ylabel='Time t/T', title='Crest motion reveals the propagation direction')
        if np.isclose(self.fraction.value, .5):
            nodes = np.arange(self.wavelength.value/4, 6, self.wavelength.value/2)
            for node in nodes:
                history.axvline(node, color='white', ls=':', lw=.8, alpha=.85)
            history.set_title('Standing wave: nodes stay at fixed z')
        elif self.fraction.value in (0, 1):
            history.set_title('Forward: crests move toward +z' if self.fraction.value == 0
                              else 'Backward: crests move toward −z')
        for axis in (spatial, history):
            _style(axis)
        self.figure.subplots_adjust(left=.08, right=.97, top=.89, bottom=.28, wspace=.40)
        bar = self.figure.add_axes([.595, .125, .34, .025])
        self.figure.colorbar(picture, cax=bar, orientation='horizontal', ticks=[-1, 0, 1],
                            label='Signed real field u(z,t)')
        self.data.update(times=times, history=real_field)
        u0, derivative = field[0], self.data['derivative'][0]
        self.readout.value = (f'b = {self.fraction.value:.2f}; forward amplitude 1−b = {1-self.fraction.value:.2f}; '
            f'backward amplitude b = {self.fraction.value:.2f}. '
            f'<b>U(0) = {u0.real:+.3f} {u0.imag:+.3f}i</b> for every b.<br>'
            f'U′(0) = {derivative.real:+.3f} {derivative.imag:+.3f}i field units/µm. '
            'The purple line selects the current model time. Outgoing propagation into z > 0 selects b = 0.')

    def _draw_solutions(self):
        from .helmholtz_families import helmholtz_family

        kind = self.solution.value
        wavelength = self.wavelength.value
        k = 2*np.pi/wavelength
        has_beam_view = kind in ('plane', 'bessel', 'gaussian')
        beam_view = has_beam_view and self.view.value == 'beam'
        self.view.layout.display = '' if has_beam_view else 'none'
        self.beam_axial.layout.display = '' if beam_view and kind != 'gaussian' else 'none'
        self.phase.layout.display = self.time.layout.display = 'none' if beam_view else ''
        self.picture.layout.max_width = '1300px' if beam_view else '1100px'
        for control, active in ((self.transverse, 'bessel'), (self.evanescent, 'evanescent'),
                                (self.degree, 'multipole'), (self.waist, 'gaussian'),
                                (self.axial, 'gaussian')):
            control.layout.display = '' if kind == active else 'none'
        if beam_view:
            self._draw_beam_propagation()
            return
        if kind == 'gaussian':
            self._draw_gaussian_solution()
            return
        options = dict(wavelength=wavelength, kind=kind, amplitude=1., phase_deg=0.)
        family_parameter = (self.transverse.value if kind == 'bessel' else
                            self.evanescent.value if kind == 'evanescent' else
                            self.degree.value if kind == 'multipole' else None)
        family_key = (kind, wavelength, family_parameter)
        factor = self.amplitude.value*np.exp(1j*np.deg2rad(self.phase.value))

        def evaluate(points):
            if kind in ('plane', 'standing', 'spherical'):
                return scalar_solution(points, **options)
            return helmholtz_family(points, **options, transverse_ratio=self.transverse.value,
                                    evanescent_ratio=self.evanescent.value, degree=self.degree.value)

        scene, amplitude_axis = self.figure.subplots(1, 2)
        def grid():
            x = np.linspace(-4, 4, 281)
            z = np.linspace(0 if kind == 'evanescent' else -4, 4, 281)
            xx, zz = np.meshgrid(x, z)
            return x, z, xx, zz, np.stack((xx, np.zeros_like(xx), zz), axis=-1)

        x, z, xx, zz, points = self._cache.get('solution_grid', (kind == 'evanescent',), grid)
        field = factor*self._cache.get('solution_field', family_key, lambda: evaluate(points))
        signed = np.real(field*np.exp(-2j*np.pi*self.time.value))
        mask = (np.hypot(xx, zz) < wavelength/2 if kind in ('spherical', 'cylindrical')
                else np.zeros_like(xx, dtype=bool))
        picture = scene.imshow(np.ma.array(signed, mask=mask), extent=(-4, 4, z[0], 4),
            origin='lower', cmap='RdBu_r', vmin=-2, vmax=2, aspect='equal', interpolation='nearest')
        self._time_image = picture
        titles = {'plane': 'Plane wave: xz slice',
                  'standing': 'Standing wave: xz slice',
                  'spherical': 'Outgoing spherical wave: xz slice',
                  'cylindrical': 'Outgoing cylindrical wave: xz slice',
                  'bessel': 'Bessel beam: xz slice',
                  'multipole': f'Regular spherical multipole\nℓ = {self.degree.value}, m = 0 (xz slice)',
                  'evanescent': 'Evanescent wave: xz slice, z ≥ 0'}
        scene.set(xlim=(-4, 4), ylim=(z[0], 4), xticks=[-4, -2, 0, 2, 4],
                  yticks=[0, 1, 2, 3, 4] if kind == 'evanescent' else [-4, -2, 0, 2, 4],
                  xlabel='x (µm)', ylabel='z (µm)', title=titles[kind])
        if kind in ('spherical', 'cylindrical'):
            scene.add_patch(Circle((0, 0), wavelength/2, facecolor='white', edgecolor=GREY, lw=.8))
            scene.plot(0, 0, 'o', color=INK, ms=4)
            scene.text(0, -.2, 'source' if kind == 'spherical' else 'line axis y',
                       ha='center', va='top', fontsize=8)
        if kind in ('plane', 'bessel'):
            scene.annotate('', xy=(3, 2.5), xytext=(3, .5),
                           arrowprops=dict(arrowstyle='->', color=INK, lw=2))
            scene.text(3, .1, '+z', ha='center', fontsize=9)
        def profile_grid():
            distance = np.linspace(0, 6, 601)
            ray = np.zeros((len(distance), 3))
            ray[:, 0 if kind == 'bessel' else 2] = distance
            return distance, ray

        distance, ray = self._cache.get('solution_profile_grid', (kind == 'bessel',), profile_grid)
        profile = factor*self._cache.get('solution_profile', family_key, lambda: evaluate(ray))
        if kind in ('spherical', 'cylindrical'):
            profile = np.where(distance >= wavelength/2, profile, np.nan+1j*np.nan)
        labels = {'plane': 'Distance along +z at x = y = 0 (µm)',
                  'standing': 'Distance along +z at x = y = 0 (µm)',
                  'spherical': 'Radius r along the +z axis (µm)',
                  'cylindrical': 'Distance ρ from the line axis (µm)',
                  'bessel': 'Transverse radius ρ at z = 0 (µm)',
                  'multipole': 'Radius r along the +z axis (µm)',
                  'evanescent': 'Depth z at x = y = 0 (µm)'}
        amplitude_axis.plot(distance, np.abs(profile), color=BLUE, lw=2.3)
        amplitude_axis.set(xlim=(0, 6), ylim=(0, 2.15), xlabel=labels[kind],
                           ylabel='Amplitude |U| (field units)', title='Spatial amplitude stays fixed in time')
        for axis in (scene, amplitude_axis):
            _style(axis)
        self.figure.subplots_adjust(left=.075, right=.97, top=.88, bottom=.26, wspace=.38)
        bar = self.figure.add_axes([.095, .12, .36, .025])
        self.figure.colorbar(picture, cax=bar, orientation='horizontal', ticks=[-2, 0, 2],
                            label='Signed real field (fixed scale)')
        self.data = dict(points=points, field=field, signed_field=signed, mask=mask,
                         profile_points=ray, profile=profile)
        if kind == 'plane':
            detail = 'Parallel wavefronts; uniform amplitude A; propagation toward +z.'
        elif kind == 'standing':
            detail = 'U = A cos(kz) exp(iφ): two opposite plane waves, each of amplitude A/2. Fixed nodes.'
        elif kind == 'spherical':
            detail = ('U = Aλ exp[i(kr + φ)] / r: A is the amplitude at r = λ. '
                      'The disk r < λ/2 is hidden only in the drawing. This is a slice of a 3D spherical wave.')
        elif kind == 'cylindrical':
            detail = ('The source is a line parallel to y. A is the amplitude at ρ = λ; '
                      'ρ < λ/2 is hidden only in the drawing. The exact radial function is Hankel H₀⁽¹⁾.')
        elif kind == 'bessel':
            ratio = self.transverse.value
            detail = (f'k⊥/k = {ratio:.2f}; k_z/k = {np.sqrt(1-ratio**2):.3f}; '
                      f'cone angle = {np.rad2deg(np.arcsin(ratio)):.1f}°. '
                      'J₀ gives transverse rings. The ideal field extends to infinite radius; '
                      'the display window is not a physical aperture.')
        elif kind == 'multipole':
            detail = (f'U = A j_ℓ(kr) P_ℓ(cos θ) exp(iφ), ℓ = {self.degree.value}, m = 0. '
                      'This regular solution is finite at the origin and includes incoming and outgoing radial parts. '
                      'A is a coefficient, not the peak amplitude.')
        else:
            ratio = self.evanescent.value
            alpha = k*np.sqrt(ratio**2-1)
            detail = (f'q/k = {ratio:.2f}; α = {alpha:.3f} µm⁻¹; amplitude decay length 1/α = {1/alpha:.3f} µm. '
                      'The phase travels along x while the amplitude decays into z > 0.')
        self.readout.value = (f'λ = {wavelength:.2f} µm; model t/T = {self.time.value:.3f}. '+detail)

    def _draw_gaussian_solution(self):
        from .helmholtz_families import gaussian_beam

        wavelength, amplitude = self.wavelength.value, self.amplitude.value
        waist = self.waist.value*wavelength
        rayleigh = np.pi*waist**2/wavelength
        z = self.axial.value*rayleigh
        width = waist*np.sqrt(1+self.axial.value**2)
        options = dict(wavelength=wavelength, amplitude=1., phase_deg=0., waist_ratio=self.waist.value)
        key = (wavelength, self.waist.value, self.axial.value)
        factor = amplitude*np.exp(1j*np.deg2rad(self.phase.value))

        def gaussian_grid():
            coordinate = np.linspace(-3, 3, 281)
            xx, yy = np.meshgrid(coordinate*waist, coordinate*waist)
            return np.stack((xx, yy, np.full_like(xx, z)), axis=-1)

        points = self._cache.get('gaussian_grid', key, gaussian_grid)
        field = factor*self._cache.get('gaussian_field', key, lambda: gaussian_beam(points, **options))
        signed = np.real(field*np.exp(-2j*np.pi*self.time.value))
        scene, amplitude_axis = self.figure.subplots(1, 2)
        picture = scene.imshow(signed, extent=(-3, 3, -3, 3), origin='lower',
            cmap='RdBu_r', vmin=-2, vmax=2, aspect='equal', interpolation='nearest')
        self._time_image = picture
        scene.set(xlim=(-3, 3), ylim=(-3, 3), xticks=[-3, -1.5, 0, 1.5, 3],
                  yticks=[-3, -1.5, 0, 1.5, 3], xlabel='x / w₀', ylabel='y / w₀',
                  title=f'Gaussian beam (paraxial)\nTransverse xy slice, z/z_R = {self.axial.value:.1f}')
        distance = self._cache.get('gaussian_distance', (), lambda: np.linspace(0, 3, 601))
        ray = self._cache.get('gaussian_ray', key,
            lambda: np.column_stack((distance*waist, np.zeros_like(distance), np.full_like(distance, z))))
        profile = factor*self._cache.get('gaussian_profile', key, lambda: gaussian_beam(ray, **options))
        amplitude_axis.plot(distance, np.abs(profile), color=BLUE, lw=2.3, label='Selected plane')
        amplitude_axis.plot(distance, amplitude*np.exp(-distance**2), color=GREY,
                            lw=1.5, ls=':', label='At the waist (z = 0)')
        amplitude_axis.set(xlim=(0, 3), ylim=(0, 2.15), xlabel='Transverse radius ρ / w₀ at the selected z',
                           ylabel='Amplitude |U| (field units)', title='Beam width grows away from the waist')
        amplitude_axis.legend(loc='upper right', fontsize=9, frameon=False)
        for axis in (scene, amplitude_axis):
            _style(axis)
        self.figure.subplots_adjust(left=.075, right=.97, top=.88, bottom=.26, wspace=.38)
        bar = self.figure.add_axes([.095, .12, .36, .025])
        self.figure.colorbar(picture, cax=bar, orientation='horizontal', ticks=[-2, 0, 2],
                            label='Signed real field (fixed scale)')
        self.data = dict(points=points, field=field, signed_field=signed,
                         mask=np.zeros_like(signed, dtype=bool), profile_points=ray, profile=profile,
                         profile_distance=distance, waist=waist, rayleigh=rayleigh, z=z, width=width)
        self.readout.value = (
            '<b>Paraxial Gaussian beam:</b> an approximate solution of full Helmholtz. '
            f'λ = {wavelength:.2f} µm; w₀ = {waist:.2f} µm; z_R = {rayleigh:.2f} µm; '
            f'z = {z:.2f} µm; w(z)/w₀ = {width/waist:.3f}; peak |U| = {amplitude*waist/width:.3f}. '
            f'Model t/T = {self.time.value:.3f}. '
            'This is a transverse xy slice, with distances measured in waist radii. '
            'A is the on-axis amplitude at the waist. Move the axial plane to see spreading.')

    def _draw_beam_propagation(self):
        from .helmholtz_families import gaussian_beam, helmholtz_family

        kind, wavelength = self.solution.value, self.wavelength.value
        amplitude, phase = self.amplitude.value, self.phase.value
        beam_data = dict(beam_kind=kind)
        if kind == 'gaussian':
            waist = self.waist.value*wavelength
            rayleigh = np.pi*waist**2/wavelength
            transverse_limit, axial_limit = 3*waist, 2*rayleigh
            selected_z = self.axial.value*rayleigh
            width = waist*np.sqrt(1+self.axial.value**2)
            beam_data.update(waist=waist, rayleigh=rayleigh, width=width)

            def evaluate(points):
                return gaussian_beam(points, wavelength, 1., 0., self.waist.value)

            title = 'Gaussian beam (paraxial)'
            detail = (f'w₀ = {waist:.2f} µm; z_R = {rayleigh:.2f} µm; '
                      f'w(z) = {width:.2f} µm. Dashed curves mark ±w(z), '
                      'where intensity is 1/e² of the local on-axis value; they are not hard edges. '
                      'The beam narrows toward the waist at z = 0 and spreads beyond it.')
        elif kind == 'bessel':
            from scipy.special import jn_zeros

            first_zero = self._cache.get('bessel_first_zero', (), lambda: jn_zeros(0, 1)[0])
            central_radius = first_zero/(2*np.pi/wavelength*self.transverse.value)
            transverse_limit, axial_limit = 3*central_radius, 10*wavelength
            selected_z = self.beam_axial.value*wavelength
            beam_data['central_radius'] = central_radius

            def evaluate(points):
                return helmholtz_family(points, wavelength, kind, 1., 0.,
                                        transverse_ratio=self.transverse.value)

            title = 'Ideal Bessel beam'
            detail = (f'First-zero radius = {central_radius:.2f} µm. '
                      'Dashed lines mark that zero, enclosing the central core. '
                      'The rings and core retain their profile along z. The ideal beam extends to '
                      'infinite radius; this plotting window is not an aperture.')
        else:
            transverse_limit, axial_limit = 3*wavelength, 10*wavelength
            selected_z = self.beam_axial.value*wavelength

            def evaluate(points):
                return scalar_solution(points, wavelength, 'plane', 1., 0.)

            title = 'Plane-wave reference'
            detail = ('Uniform intensity everywhere. The plane wave has no finite beam width; '
                      'the borders of these plots are only viewing limits.')

        parameter = (self.waist.value if kind == 'gaussian' else
                     self.transverse.value if kind == 'bessel' else None)
        key = (kind, wavelength, parameter)
        factor = amplitude*np.exp(1j*np.deg2rad(phase))

        def beam_grid():
            x = np.linspace(-transverse_limit, transverse_limit, 281)
            z = np.linspace(-axial_limit, axial_limit, 321)
            xx, zz = np.meshgrid(x, z)
            points = np.stack((xx, np.zeros_like(xx), zz), axis=-1)
            xx, yy = np.meshgrid(x, x)
            return x, z, points, xx, yy, np.linspace(-transverse_limit, transverse_limit, 601)

        x, z, longitudinal_points, xx, yy, profile_x = self._cache.get('beam_grid', key, beam_grid)
        longitudinal_unit = self._cache.get('beam_longitudinal_field', key,
            lambda: evaluate(longitudinal_points))
        longitudinal_field = factor*longitudinal_unit
        longitudinal_intensity = self._cache.get('beam_longitudinal_intensity', (*key, amplitude),
            lambda: amplitude**2*np.abs(longitudinal_unit)**2)
        selected_key = (*key, selected_z)
        transverse_points, ray = self._cache.get('beam_observation_points', selected_key,
            lambda: (np.stack((xx, yy, np.full_like(xx, selected_z)), axis=-1),
                     np.column_stack((profile_x, np.zeros_like(profile_x), np.full_like(profile_x, selected_z)))))

        # Plane and ideal Bessel profiles change only by a common phase along z.
        # Evaluate their zero-z profiles once; moving the observation plane keeps
        # both intensity arrays intact while updating the exposed complex fields.
        profile_key = selected_key if kind == 'gaussian' else key

        def profile_at_reference(points):
            if kind != 'gaussian':
                points = points.copy()
                points[..., 2] = 0
            return evaluate(points)

        transverse_unit = self._cache.get('beam_transverse_field', profile_key,
            lambda: profile_at_reference(transverse_points))
        profile_unit = self._cache.get('beam_profile', profile_key,
            lambda: profile_at_reference(ray))
        kz = 2*np.pi/wavelength*(np.sqrt(1-self.transverse.value**2) if kind == 'bessel' else 1.)
        axial_factor = np.exp(1j*kz*selected_z) if kind != 'gaussian' else 1.
        transverse_field = factor*axial_factor*transverse_unit
        profile = factor*axial_factor*profile_unit
        transverse_intensity = self._cache.get('beam_transverse_intensity', (*profile_key, amplitude),
            lambda: amplitude**2*np.abs(transverse_unit)**2)
        profile_intensity = self._cache.get('beam_profile_intensity', (*profile_key, amplitude),
            lambda: amplitude**2*np.abs(profile_unit)**2)

        self.figure.set_size_inches(12.4, 5.9)
        longitudinal, transverse, profile_axis = self.figure.subplots(1, 3)
        self.figure.suptitle(title, fontsize=13, color=INK, y=.97)
        picture = longitudinal.imshow(longitudinal_intensity,
            extent=(-transverse_limit, transverse_limit, -axial_limit, axial_limit),
            origin='lower', cmap='inferno', vmin=0, vmax=1, aspect='auto')
        longitudinal.axhline(selected_z, color='#58e5ee', lw=1.7, label='Selected plane')
        if kind == 'gaussian':
            envelope = waist*np.sqrt(1+(z/rayleigh)**2)
            for sign in (-1, 1):
                longitudinal.plot(sign*envelope, z, '--', color='white', lw=1.1)
        elif kind == 'bessel':
            for sign in (-1, 1):
                longitudinal.plot([sign*central_radius]*2, [-axial_limit, axial_limit],
                                  '--', color='white', lw=1.1)
        longitudinal.set(xlabel='x (µm)', ylabel='z (µm)',
                         title='Propagation: xz section at y = 0\nAxes use different scales')
        longitudinal.legend(loc='upper right', fontsize=8, facecolor='white', framealpha=.85)
        transverse.imshow(transverse_intensity,
            extent=(-transverse_limit, transverse_limit, -transverse_limit, transverse_limit),
            origin='lower', cmap='inferno', vmin=0, vmax=1, aspect='equal')
        transverse.set(xlabel='x (µm)', ylabel='y (µm)',
                       title=f'Transverse xy plane\nz = {selected_z:.2f} µm')
        profile_axis.plot(profile_x, profile_intensity, color=BLUE, lw=2.2,
                          label='Selected plane')
        if kind == 'gaussian':
            profile_axis.plot(profile_x, amplitude**2*np.exp(-2*(profile_x/waist)**2),
                              ':', color=GREY, lw=1.5, label='At the waist')
            profile_axis.legend(loc='upper right', fontsize=8, frameon=False)
        profile_axis.set(xlim=(-transverse_limit, transverse_limit), ylim=(0, 1.05),
                         xlabel='x (µm), y = 0', ylabel='Relative intensity |U|²',
                         title='Intensity across the selected plane')
        for axis in (longitudinal, transverse, profile_axis):
            _style(axis)
        # Grid lines would resemble extra wave structure on the intensity maps.
        longitudinal.grid(False)
        transverse.grid(False)
        self.figure.subplots_adjust(left=.07, right=.98, top=.82, bottom=.26, wspace=.40)
        bar = self.figure.add_axes([.08, .12, .54, .025])
        self.figure.colorbar(picture, cax=bar, orientation='horizontal', ticks=[0, .5, 1],
                            label='Relative intensity |U|² — shared fixed scale')
        self.data = dict(**beam_data, longitudinal_points=longitudinal_points,
                         longitudinal_field=longitudinal_field, longitudinal_intensity=longitudinal_intensity,
                         transverse_points=transverse_points, transverse_field=transverse_field,
                         transverse_intensity=transverse_intensity, profile_points=ray,
                         profile=profile, profile_intensity=profile_intensity, z_selected=selected_z)
        self.readout.value = (f'<b>{title}:</b> λ = {wavelength:.2f} µm; selected z = {selected_z:.2f} µm. '
            + detail + '<br>The cyan line selects the transverse plane. All panels use |U|² with '
            'one fixed reference amplitude of 1, so brightness changes are preserved. '
            'A global phase shift and optical-cycle time do not change this monochromatic intensity.')

    def _draw_diffraction(self):
        self.figure.set_size_inches(12.8, 5.8)
        self.picture.layout.max_width = '1400px'
        aperture, propagation, screen = self.figure.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1.15, 1.1]})
        width, wavelength = self.width.value, self.wavelength.value
        xx = np.array([-30., -width/2, -width/2, width/2, width/2, 30.])
        aperture.plot(xx, [0, 0, 1, 1, 0, 0], color=BLUE, lw=2.4)
        aperture.fill_between([-width/2, width/2], 0, 1, color=BLUE, alpha=.12)
        aperture.annotate('', xy=(width/2, 1.15), xytext=(-width/2, 1.15),
                          arrowprops=dict(arrowstyle='<->', color=PURPLE, lw=1.5))
        aperture.text(0, 1.21, rf'$W_{{\mathrm{{open}}}}$ = {width:.0f} µm', ha='center', fontsize=10, color=PURPLE)
        aperture.set(xlim=(-30, 30), ylim=(-.08, 1.42), xlabel='Aperture x (µm)',
                     ylabel='U(x,0) = t(x), unit incident amplitude', title='1. Prescribe the slit field\nz = 0')
        near = self._cache.get('slit_propagation', (width, wavelength),
                               lambda: slit_propagation(width, wavelength))
        picture = propagation.imshow(near['intensity'], extent=(-80.25, 80.25, -1/480, 1+1/480),
            origin='lower', aspect='auto', cmap='inferno', norm=PowerNorm(.5, vmin=0, vmax=2))
        propagation.set(xlim=(-80, 80), ylim=(0, 1), xticks=[-80, -40, 0, 40, 80],
                        xlabel='x (µm)', ylabel='Forward distance z (mm)',
                        title='2. Propagate the prescribed field\nFirst millimetre (angular spectrum)')
        propagation.plot([-width/2, width/2], [0, 0], color='#58e5ee', lw=4, clip_on=False)
        screen_x = np.linspace(-100, 100, 1201)
        self.data = dict(slit_far_field(screen_x, width, wavelength, 1000), propagation=near)
        screen.plot(screen_x, self.data['relative_intensity'], color=INK, lw=2.3)
        zero = self.data['first_zero_mm']
        for sign in (-1, 1):
            if zero <= 100:
                screen.axvline(sign*zero, color=PURPLE, ls=':', lw=1)
                screen.plot(sign*zero, 0, 'o', color=PURPLE, ms=5)
        screen.set(xlim=(-100, 100), ylim=(-.04, 1.08), xlabel='Detector X (mm)',
                   ylabel='I(X) / I(0)', title='3. Observe the far-field pattern\nDetector at L = 1000 mm')
        for axis in (aperture, propagation, screen):
            _style(axis)
        propagation.grid(False)
        self.figure.subplots_adjust(left=.065, right=.98, top=.82, bottom=.28, wspace=.52)
        bar = self.figure.add_axes([.395, .12, .23, .025])
        self.figure.colorbar(picture, cax=bar, orientation='horizontal', ticks=[0, .25, 1, 2],
                            label='|U|² / incident intensity')
        self.readout.value = (f'λ = {wavelength:.3f} µm; W<sub>open</sub> = {width:.0f} µm. '
            f'<b>Detector first zeros: X = ±{zero:.2f} mm.</b><br>'
            'The middle map stops at 1 mm; the detector is much farther away, at 1000 mm. '
            'Map brightness uses a fixed incident-intensity reference; the detector curve uses its own central peak. '
            'Axes use different scales. Narrow the slit and compare the spreading and detector width.')

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.reset_button.on_click(self._reset, remove=True)
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        self._time_image = self._time_cursor = None
        self._superposition_lines = None
        self._cache.clear()
        _close_widget_tree(self.ui)
