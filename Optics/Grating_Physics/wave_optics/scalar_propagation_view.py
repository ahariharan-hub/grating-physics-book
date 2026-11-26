"""Disposable notebook illustrations of spectral geometry and propagation."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.colors import PowerNorm

from .grating_intro import _close_widget_tree
from .plotting import figure_png
from .scalar_propagation import (
    fresnel_phase_error, spectrum_geometry, transfer_function,
)
from .scalar_theory import slit_propagation
from ._view_cache import ViewCache


BLUE, ORANGE, INK, GREY, PURPLE = '#2866a3', '#b85c17', '#243444', '#727c86', '#79539e'
STAGES = ('geometry', 'transfer', 'spectrum', 'component', 'slit')


def _style(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(alpha=.15)
    axis.tick_params(labelsize=9)


def _physical_geometry(normalized, wavelength):
    """Restore inverse-length units without changing normalized directions."""
    return dict(normalized, **{name: normalized[name]/wavelength
                              for name in ('fx', 'fy', 'k', 'kx', 'ky', 'kz')})


class ScalarPropagationExplorer:
    """One interactive section; constructing it creates its own widget tree."""

    def __init__(self, stage='geometry'):
        if stage not in STAGES:
            raise ValueError(f'Stage must be one of {STAGES}.')
        import ipywidgets as widgets

        self.stage, self._closed, self._resetting = stage, False, False
        self._cache = ViewCache()
        self.alpha = self.beta = self.distance = None
        self.width = self.keep_phase = self.wavelength = None
        self._slit_artists = None
        controls = []

        def slider(description, value, lower, upper, step, fmt):
            control = widgets.FloatSlider(description=description, value=value,
                min=lower, max=upper, step=step, readout_format=fmt,
                continuous_update=False, style={'description_width': 'initial'},
                layout=widgets.Layout(width='335px', max_width='100%'))
            controls.append(control)
            return control

        if stage in ('geometry', 'transfer', 'slit'):
            self.wavelength = slider('Wavelength λ (µm)', .6, .4,
                                     .8 if stage == 'slit' else 1, .025, '.3f')
        if stage in ('spectrum', 'slit'):
            self.width = slider('Slit width W_open (µm)', 20, 10, 40, 1, '.0f')
        if stage == 'spectrum':
            self.keep_phase = widgets.Checkbox(value=True,
                description='Keep spectrum phase', indent=False,
                layout=widgets.Layout(width='220px'))
            controls.append(self.keep_phase)
        if stage == 'component':
            self.alpha = slider('Transverse frequency λ fₓ', .6, 0, 1.5, .05, '.2f')
        if stage == 'slit':
            self.distance = slider('Observation z (µm)', 500, 0, 1000, 5, '.0f')
        if stage == 'geometry':
            self.alpha = slider('α = λ fₓ', .3, -1.25, 1.25, .05, '.2f')
            self.beta = slider('β = λ fᵧ', .4, -1.25, 1.25, .05, '.2f')
        elif stage == 'transfer':
            self.distance = slider('Distance z/λ', 2, 0, 20, .25, '.2f')
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
        self.figure = Figure(figsize=(10.5, 5.5 if stage == 'geometry' else 4.6),
                             dpi=110, facecolor='white')
        FigureCanvasAgg(self.figure)
        self.data = None
        for control in controls:
            control.observe(self._update, names='value')
        self._update()

    def _update(self, change=None):
        if self._closed or self._resetting:
            return
        if (self.stage in ('geometry', 'transfer') and change is not None
                and change.get('owner') is self.wavelength and self.data is not None):
            self.data['geometry'] = _physical_geometry(self._normalized_geometry, self.wavelength.value)
            if self.stage == 'geometry':
                self._geometry_readout(self.data['geometry'])
            else:
                self.data['distance'] = self.distance.value*self.wavelength.value
                self._transfer_readout(self.data['distance'])
            return  # All plotted coordinates/curves use wavelength-normalized units.
        if (self.stage == 'slit' and change is not None
                and change.get('owner') is self.distance and self._slit_artists is not None):
            self._select_slit_plane()
            self.picture.value = figure_png(self.figure)
            return
        self.figure.clear()
        getattr(self, f'_draw_{self.stage}')()
        self.picture.value = figure_png(self.figure)

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

    def _draw_geometry(self):
        disk, scene = self.figure.subplots(1, 2)
        wavelength = self.wavelength.value
        alpha, beta = self.alpha.value, self.beta.value
        self._normalized_geometry = self._cache.get('geometry', (alpha, beta),
            lambda: spectrum_geometry(alpha, beta, 1.))
        geometry = _physical_geometry(self._normalized_geometry, wavelength)
        gamma = complex(geometry['gamma'])
        q = float(geometry['q'])
        propagating = bool(geometry['propagating'])
        grazing = q == 1
        color = BLUE if propagating else ORANGE

        disk.add_patch(Circle((0, 0), 1, facecolor=BLUE, edgecolor=BLUE,
                              alpha=.12, lw=1.5))
        disk.add_patch(Circle((0, 0), 1, fill=False, edgecolor=BLUE, lw=1.4))
        disk.axhline(0, color=GREY, lw=.8)
        disk.axvline(0, color=GREY, lw=.8)
        disk.plot([0, alpha], [0, beta], color=color, lw=1.4)
        disk.plot(alpha, beta, 'o', color=color, ms=8)
        disk.text(-1.31, 1.24, 'Evanescent outside the circle', fontsize=9, color=ORANGE)
        disk.text(0, -.78, 'Propagating disk', ha='center', fontsize=10, color=BLUE)
        disk.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), xlabel=r'$\alpha=\lambda f_x=k_x/k$',
                 ylabel=r'$\beta=\lambda f_y=k_y/k$', title='One selected spectral component')
        disk.set_aspect('equal', adjustable='box')

        def grid():
            x, z = np.linspace(-2, 2, 321), np.linspace(0, 4, 321)
            xx, zz = np.meshgrid(x, z)
            return x, z, xx, zz

        x, z, xx, zz = self._cache.get('geometry_grid', (), grid)
        field = self._cache.get('geometry_field', (alpha, beta),
            lambda: np.exp(2j*np.pi*(alpha*xx+gamma*zz)))
        shown = scene.imshow(field.real, origin='lower', extent=(-2, 2, 0, 4),
            cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal', interpolation='nearest')
        projected = np.array([alpha, gamma.real])
        if propagating and np.linalg.norm(projected) > 1e-12:
            start = np.array([0., .45])
            end = start+1.2*projected
            scene.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=INK, lw=2.1))
            scene.text(end[0]+.07, end[1]+.10, r'projected $\mathbf{k}$', color=INK,
                fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2))
        elif propagating:
            scene.text(0, 3.55, 'Wavevector is along y\n(out of this slice)', ha='center',
                       fontsize=9, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        scene.set(xlabel=r'$x/\lambda$', ylabel=r'$z/\lambda$',
                  title=r'Signed field $\mathrm{Re}\,U$ in the $y=0$ slice')
        self.figure.subplots_adjust(left=.075, right=.96, top=.88, bottom=.22, wspace=.33)
        colorbar_axis = self.figure.add_axes([.615, .105, .28, .025])
        colorbar = self.figure.colorbar(shown, cax=colorbar_axis, orientation='horizontal', ticks=[-1, 0, 1])
        colorbar.set_label('Field units (unit boundary amplitude)', fontsize=9)
        for axis in (disk, scene):
            _style(axis)
        scene.grid(False)

        self.data = dict(geometry=geometry, x_over_wavelength=x, z_over_wavelength=z,
                         field=field, signed_field=field.real)
        self._geometry_readout(geometry)

    def _geometry_readout(self, geometry):
        wavelength = self.wavelength.value
        alpha, beta = self.alpha.value, self.beta.value
        gamma, q = complex(geometry['gamma']), float(geometry['q'])
        propagating, grazing = bool(geometry['propagating']), q == 1
        if propagating:
            kind = 'Grazing propagation' if grazing else 'Propagating plane wave'
            azimuth = float(geometry['azimuth_deg'])
            azimuth_text = f'{azimuth:.1f}°' if np.isfinite(azimuth) else 'undefined on axis'
            description = (f'θ = {float(geometry["theta_deg"]):.1f}° from +z; '
                           f'azimuth = {azimuth_text}.')
            gamma_text = f'{gamma.real:.3f}'
        else:
            kind = 'Evanescent component'
            decay_length = wavelength/(2*np.pi*gamma.imag)
            description = (f'1/e amplitude decay length = {decay_length:.3f} µm '
                f'({decay_length/wavelength:.3f} λ). No real polar propagation angle.')
            gamma_text = f'{gamma.imag:.3f} i'
        self.readout.value = (
            f'<b>{kind}.</b> α² + β² = {q:.3f}; γ = k<sub>z</sub>/k = {gamma_text}. '
            f'f<sub>x</sub> = {alpha/wavelength:.3f} µm⁻¹; '
            f'f<sub>y</sub> = {beta/wavelength:.3f} µm⁻¹. {description} '
            'The x–z view is a slice; β still contributes to γ. '
            'Normalized axes keep the shape fixed when λ alone changes.')

    def _draw_transfer(self):
        longitudinal, magnitude, phase = self.figure.subplots(1, 3)
        wavelength = self.wavelength.value
        distance = self.distance.value*wavelength
        s = self._cache.get('transfer_grid', (), lambda: np.linspace(0, 1.4, 701))
        self._normalized_geometry = self._cache.get('transfer_geometry', (),
            lambda: spectrum_geometry(s, 0, 1.))
        geometry = _physical_geometry(self._normalized_geometry, wavelength)
        normalized_distance = self.distance.value
        asm, fresnel, error = self._cache.get('transfer_curves', (normalized_distance,),
            lambda: (transfer_function(s, 0, 1., normalized_distance, 'asm'),
                     transfer_function(s, 0, 1., normalized_distance, 'fresnel'),
                     fresnel_phase_error(s, 0, 1., normalized_distance)))
        gamma = geometry['gamma']

        longitudinal.plot(s, gamma.real, color=BLUE, lw=2, label='ASM: Re γ')
        longitudinal.plot(s, gamma.imag, color=ORANGE, lw=2, label='ASM: Im γ')
        longitudinal.plot(s, 1-s*s/2, color=PURPLE, lw=1.8, ls='--', label='Fresnel: 1 − s²/2')
        longitudinal.set(ylim=(-.07, 1.13), ylabel=r'$k_z/k$', title='Longitudinal wavevector')
        magnitude.plot(s, abs(asm), color=BLUE, lw=2.2, label='ASM')
        magnitude.plot(s, abs(fresnel), color=PURPLE, lw=1.8, ls='--', label='Fresnel')
        magnitude.set(ylim=(-.06, 1.12), ylabel='Transfer magnitude |H|', title='Phase or decay?')
        phase.plot(s, error['exact'], color=BLUE, lw=2.2, label='Exact difference')
        phase.plot(s, error['leading'], color=PURPLE, lw=1.8, ls='--', label=r'Leading $-(k z)s^4/8$')
        minimum = -np.pi*self.distance.value
        phase.set(ylim=(min(-1., 1.08*minimum), max(.08, -.05*minimum)),
                  ylabel='ASM phase − Fresnel phase (rad)', title='Accumulated phase error')
        phase.text(1.2, .5, 'Not a\npropagating\nphase error', ha='center', va='center',
                   fontsize=8, color=GREY, transform=phase.get_xaxis_transform())
        for axis in (longitudinal, magnitude, phase):
            axis.axvspan(1, 1.4, color=GREY, alpha=.08)
            axis.axvline(1, color=GREY, ls=':', lw=1)
            axis.set(xlim=(0, 1.4), xlabel=r'$s=\lambda\sqrt{f_x^2+f_y^2}$')
            axis.legend(loc='upper center', bbox_to_anchor=(.5, -.20),
                        frameon=False, fontsize=8)
            _style(axis)
        self.figure.subplots_adjust(left=.065, right=.98, top=.88, bottom=.29, wspace=.46)
        self.data = dict(s=s, geometry=geometry, asm=asm, fresnel=fresnel,
                         error=error, distance=distance)
        self._transfer_readout(distance)

    def _transfer_readout(self, distance):
        self.readout.value = (
            f'<b>z = {distance:.3f} µm = {self.distance.value:.2f} λ.</b> '
            'The circle edge is s = 1. ASM preserves propagating amplitudes and decays '
            'evanescent amplitudes; Fresnel always has unit magnitude. '
            '<b>Signed phase error means exact ASM phase minus Fresnel phase</b>, '
            'without wrapping; its propagating-wave definition stops at s = 1. '
            'Fresnel is intended for s ≪ 1. Normalized curves depend on z/λ.')

    def _draw_spectrum(self):
        boundary, spectral, rebuilt = self.figure.subplots(1, 3)
        width = self.width.value

        def decompose():
            # Centred spatial and frequency grids. The factor dx approximates
            # the continuous 1D Fourier integral; it is removed on inversion.
            dx, count = .0625, 8192
            x = (np.arange(count)-count//2)*dx
            field = np.clip((width/2-np.abs(x))/dx+.5, 0, 1)
            fx = np.fft.fftshift(np.fft.fftfreq(count, dx))
            spectrum = dx*np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field)))
            return x, field, fx, spectrum, dx

        x, field, fx, spectrum, dx = self._cache.get('slit_spectrum', (width,), decompose)
        retained = spectrum if self.keep_phase.value else abs(spectrum)
        reconstructed = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(retained)))/dx
        selected, frequencies = abs(x) <= 40, abs(fx) <= .3
        boundary.plot(x[selected], field[selected], color=BLUE, lw=2)
        boundary.set(title='1. Known field at z = 0', xlabel='x (µm)',
                     ylabel='Real field (unit incident amplitude)', xlim=(-40, 40), ylim=(-.15, 1.2))
        spectral.plot(fx[frequencies], abs(spectrum[frequencies]), color=PURPLE, lw=2)
        spectral.axvline(-1/width, color=GREY, ls=':', lw=1)
        spectral.axvline(1/width, color=GREY, ls=':', lw=1)
        spectral.set(title='2. Angular-spectrum magnitude', xlabel=r'$f_x$ (cycles/µm)',
                     ylabel=r'$|\widetilde U_{\mathrm{exit}}(f_x)|$ (field units × µm)', xlim=(-.3, .3), ylim=(0, 42))
        rebuilt.plot(x[selected], field[selected], color=BLUE, lw=3, alpha=.35, label='Input')
        rebuilt.plot(x[selected], reconstructed[selected].real, color=ORANGE, lw=1.8,
                     ls='--', label='Reconstructed')
        rebuilt.set(title='3. Add the components again', xlabel='x (µm)',
                    ylabel='Real field', xlim=(-40, 40))
        rebuilt.set_ylim((-.15, 1.2) if self.keep_phase.value else (-1.5, 4.5))
        rebuilt.legend(frameon=False, fontsize=9)
        for axis in (boundary, spectral, rebuilt):
            _style(axis)
            axis.title.set_fontsize(11)
        self.figure.subplots_adjust(left=.065, right=.985, bottom=.20, top=.85, wspace=.42)
        error = float(np.max(abs(reconstructed-field)))
        self.data = dict(x_um=x, boundary_field=field, fx=fx, spectrum=spectrum,
                         reconstructed=reconstructed, dx_um=dx, reconstruction_error=error)
        message = ('The complex spectrum reconstructs the sampled input.' if self.keep_phase.value
                   else 'Phase removed: the magnitude is unchanged, but the reconstructed field is wrong (right axis expanded).')
        self.readout.value = (f'<b>{message}</b> First spectral zeros: ±{1/width:.3f} cycles/µm. '
            f'Maximum field mismatch: {error:.2g}. Both fields here are real; a negative value means a π phase reversal.')

    def _draw_component(self):
        axis = self.figure.subplots()
        s = self.alpha.value
        z = np.linspace(0, 3, 601)  # distances in wavelengths
        # Evaluate the shared transfer function at each distance.
        multiplier = np.array([transfer_function(s, 0., 1., depth) for depth in z])
        reference = np.array([transfer_function(.6, 0., 1., depth) for depth in z])
        geometry = spectrum_geometry(s, 0., 1.)
        axis.plot(z, reference.real, color=GREY, lw=1.2, alpha=.55,
                  label='Reference: Re H at λfₓ = 0.60')
        axis.plot(z, multiplier.real, color=BLUE, lw=2, label='Selected: Re H')
        axis.plot(z, abs(multiplier), color=ORANGE, lw=2, ls='--', label='Selected: |H|')
        axis.axhline(0, color=GREY, lw=.7)
        axis.set(xlim=(0, 3), ylim=(-1.15, 1.15), xlabel=r'Distance $z/\lambda$',
                 ylabel='Unit boundary amplitude', title='One component: phase change or amplitude decay?')
        _style(axis)
        axis.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=3, frameon=False, fontsize=9)
        self.figure.subplots_adjust(left=.09, right=.97, top=.86, bottom=.28)
        gamma = complex(geometry['gamma'])
        if s < 1:
            kind = f'Propagating: k<sub>z</sub>/k = {gamma.real:.3f}. The amplitude stays 1; the phase advances.'
        elif s == 1:
            kind = 'Grazing: k<sub>z</sub> = 0. H = 1, so this component has no z variation.'
        else:
            kind = (f'Evanescent: k<sub>z</sub>/k = {gamma.imag:.3f} i. '
                    f'Amplitude falls to 1/e after {1/(2*np.pi*gamma.imag):.3f} λ.')
        self.readout.value = f'<b>{kind}</b> Re H is a signed field factor at a fixed instant, not intensity.'
        self.data = dict(s=s, geometry=geometry, z_over_wavelength=z, transfer=multiplier)

    def _draw_slit(self):
        data = self._cache.get('slit_map', (self.width.value, self.wavelength.value),
            lambda: slit_propagation(self.width.value, self.wavelength.value,
                                     distances_um=np.arange(0., 1001., 5.)))
        self.data = dict(data)
        boundary, scene, profile = self.figure.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1.2, 1]})
        x = data['x_um']
        boundary.plot(x, data['boundary_field'], color=BLUE, lw=2)
        boundary.set(xlabel='x (µm)', ylabel=r'$|U_{\mathrm{exit}}|$', title='Known aperture field', ylim=(-.06, 1.15))
        shown = scene.imshow(data['intensity'], origin='lower', extent=(-80.25, 80.25, -2.5, 1002.5),
            aspect='auto', cmap='inferno', norm=PowerNorm(.5, vmin=0, vmax=2), interpolation='nearest')
        scene.set(xlabel='x (µm)', ylabel='z (µm)', title='Propagation into +z', ylim=(0, 1000))
        cursor = scene.axhline(self.distance.value, color='#64dce0', lw=1.6)
        curve, = profile.plot(x, data['intensity'][0], color=INK, lw=1.8)
        profile.set(xlabel='x (µm)', ylabel=r'$I_{\rm rel}=|U|^2$', ylim=(0, 2))
        for axis in (boundary, scene, profile):
            axis.set_xlim(-80, 80)
            _style(axis)
            axis.title.set_fontsize(11)
        scene.grid(False)
        self.figure.subplots_adjust(left=.065, right=.98, bottom=.29, top=.86, wspace=.45)
        color_axis = self.figure.add_axes([.397, .12, .235, .025])
        self.figure.colorbar(shown, cax=color_axis, orientation='horizontal', ticks=[0, .5, 1, 1.5, 2])
        color_axis.set_xlabel('Relative intensity (fixed scale)', fontsize=9)
        self._slit_artists = (cursor, curve, profile)
        self._select_slit_plane()

    def _select_slit_plane(self):
        distance = self.distance.value
        # Slider planes are cached. Direct programmatic intermediate values
        # still compute the requested plane, without rebuilding the whole map.
        index = int(np.argmin(abs(self.data['z_um']-distance)))
        if abs(self.data['z_um'][index]-distance) < 1e-10:
            field = self.data['field'][index]
        else:
            field = self._cache.get('slit_plane',
                (self.width.value, self.wavelength.value, distance),
                lambda: slit_propagation(self.width.value, self.wavelength.value,
                                         distances_um=[distance]))['field'][0]
        intensity = abs(field)**2
        self.data.update(selected_field=field, selected_intensity=intensity, distance_um=distance)
        cursor, curve, profile = self._slit_artists
        cursor.set_ydata([distance, distance])
        curve.set_ydata(intensity)
        profile.set_title(f'Observation at z = {distance:g} µm', fontsize=11)
        if distance:
            number = self.width.value**2/(self.wavelength.value*distance)
            description = f'a²/(λz) = {number:.3f}; smaller values approach the far-field shape.'
        else:
            description = 'At z = 0 the field is exactly the prescribed sampled aperture.'
        self.readout.value = (f'<b>{description}</b> The cyan line selects the profile. '
            'Intensity uses the same incident reference throughout; no curve is normalized to its own peak.')

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
        self._normalized_geometry = None
        self._slit_artists = None
        self._cache.clear()
        _close_widget_tree(self.ui)


def slit_setup_png():
    """A single physical scene, computed from the same prescribed slit model."""
    data = slit_propagation(20, .6, distances_um=np.linspace(0, 1000, 151))
    figure = Figure(figsize=(10.5, 3.8), dpi=110, facecolor='white')
    FigureCanvasAgg(figure)
    scene, detector = figure.subplots(1, 2, gridspec_kw={'width_ratios': [2.5, 1]})
    dz = data['z_um'][1]-data['z_um'][0]
    scene.imshow(data['intensity'].T, origin='lower', extent=(-dz/2, 1000+dz/2, -80.25, 80.25),
                 cmap='inferno', norm=PowerNorm(.5, vmin=0, vmax=2),
                 aspect='auto', interpolation='nearest')
    scene.axvspan(-280, 0, color=BLUE, alpha=.12)
    for crest in (-240, -180, -120, -60):
        scene.plot([crest, crest], [-80, 80], color=BLUE, lw=1.2)
    scene.plot([0, 0], [-80, -10], color=GREY, lw=5, solid_capstyle='butt')
    scene.plot([0, 0], [10, 80], color=GREY, lw=5, solid_capstyle='butt')
    scene.axvline(1000, color='#64dce0', lw=2)
    scene.set(xlim=(-280, 1000), ylim=(-80, 80), xlabel='z (µm) →', ylabel='x (µm)',
              title='Incident wave → slit → known field → homogeneous propagation')
    scene.text(-145, -65, 'Crests\n(schematic)', ha='center', fontsize=9, color=BLUE)
    scene.annotate(r'$W_{\mathrm{open}}$ = 20 µm', xy=(0, 0), xytext=(150, 55), color='white', fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='white'))
    scene.text(650, -67, 'Calculated intensity', color='white', fontsize=9)
    detector.plot(data['x_um'], data['intensity'][-1], color=INK, lw=2)
    detector.axvspan(-10, 10, color=BLUE, alpha=.1)
    detector.set(xlim=(-80, 80), ylim=(0, 2), xlabel='x (µm)', ylabel=r'$I_{\rm rel}$',
                 title='Observation at z = 1000 µm')
    detector.text(0, 1.7, 'Geometrical\nopening', ha='center', color=BLUE, fontsize=9)
    for axis in (scene, detector):
        _style(axis)
        axis.title.set_fontsize(10)
    scene.grid(False)
    figure.subplots_adjust(left=.065, right=.98, bottom=.20, top=.85, wspace=.34)
    result = figure_png(figure)
    figure.clear()
    return result
