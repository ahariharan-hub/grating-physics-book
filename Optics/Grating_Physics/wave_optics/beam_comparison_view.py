"""Chapter 1 comparison; existing paraxial physics, retained Matplotlib artists."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .helmholtz_families import gaussian_beam
from .plotting import figure_png


class BeamComparison:
    """Finite-power Gaussian versus an ideal plane wave, all lengths in micrometres."""

    def __init__(self):
        import ipywidgets as widgets
        self._closed = self._resetting = False
        self.updates = 0
        style = {'description_width': 'initial'}
        def slider(label, value, lo, hi, step):
            return widgets.FloatSlider(description=label, value=value, min=lo,
                max=hi, step=step, continuous_update=False, style=style,
                layout=widgets.Layout(width='330px', max_width='100%'))
        self.wavelength = slider('Wavelength (µm)', .6, .4, .8, .05)
        self.waist = slider('Waist radius w₀ (µm)', 3, 2, 5, .25)
        self.screen = slider('Screen z (µm)', 0, 0, 40, 2)
        self.controls = (self.wavelength, self.waist, self.screen)
        self.reset_button = widgets.Button(description='Reset')
        self.picture = widgets.Image(format='png', layout=widgets.Layout(width='100%'))
        self.caption = widgets.HTML()
        self.figure = Figure(figsize=(11, 7.1), dpi=110, layout='constrained')
        FigureCanvasAgg(self.figure)
        axes = self.figure.subplots(2, 2)
        self.axes = axes
        self.x = np.linspace(-12, 12, 241)
        self.z = np.linspace(0, 40, 1001)
        zz, xx = np.meshgrid(self.z, self.x)
        self.points = np.stack((xx, np.zeros_like(xx), zz), axis=-1)
        self.images, self.cursors, self.profiles = [], [], []
        for col, title in enumerate(('Ideal plane wave', 'Finite Gaussian beam (paraxial)')):
            axis = axes[0, col]
            im = axis.imshow(np.zeros(xx.shape), extent=(0, 40, -12, 12),
                origin='lower', aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                interpolation='nearest')
            self.images.append(im)
            self.cursors.append(axis.axvline(0, color='#299746', lw=2))
            axis.annotate('', xy=(12, 0), xytext=(4, 0),
                arrowprops={'arrowstyle': '->', 'color': '#111111', 'lw': 2.5})
            axis.text(4, 1.3, 'central +z direction', fontsize=8,
                bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
            axis.set(title=title+'\nFinite viewing window; signed field at t = 0',
                xlabel='z (µm)', ylabel='x (µm)')
            profile = axes[1, col]
            self.profiles.append(profile.plot(self.x, np.ones_like(self.x), color='#2866a3')[0])
            profile.set(xlim=(-12, 12), ylim=(0, 1.05), xlabel='x at screen (µm)',
                ylabel='Relative intensity |U|² / |A₀|²', title='Screen intensity (not signed field)')
            profile.grid(alpha=.2)
        self.width_lines = [axes[0, 1].plot(self.z, np.zeros_like(self.z),
            '--', color='#333333', lw=1)[0] for _ in range(2)]
        self.figure.colorbar(self.images[0], ax=list(axes[0]), shrink=.75,
            label='Signed field Re(U) / A₀')
        self.ui = widgets.VBox([widgets.HTML(
            '<b>Laser → collimated beam → movable observation screen</b><br>'
            'Green line: screen. Dashed curves: 1/e² intensity radius, not hard edges. '
            'Both arrows mark a direction, not a beam width.'),
            widgets.HBox([self.wavelength, self.waist], layout=widgets.Layout(flex_flow='row wrap')),
            widgets.HBox([self.screen, self.reset_button]), self.picture, self.caption])
        for control in self.controls:
            control.observe(self._changed, names='value')
        self.reset_button.on_click(self.reset)
        self._update()

    def _changed(self, change):
        if not self._closed and not self._resetting:
            self._update()

    def _update(self):
        wavelength, waist, screen = (c.value for c in self.controls)
        rayleigh = np.pi*waist**2/wavelength
        width = waist*np.sqrt(1+(self.z/rayleigh)**2)
        beam = gaussian_beam(self.points, wavelength=wavelength, waist_ratio=waist/wavelength)
        plane = np.cos(2*np.pi*self.points[..., 2]/wavelength)
        sample = np.column_stack((self.x, np.zeros_like(self.x), np.full_like(self.x, screen)))
        intensity = abs(gaussian_beam(sample, wavelength=wavelength, waist_ratio=waist/wavelength))**2
        self.images[0].set_data(plane)
        self.images[1].set_data(beam.real)
        self.profiles[1].set_ydata(intensity)
        for cursor in self.cursors:
            cursor.set_xdata([screen, screen])
        for line, sign in zip(self.width_lines, (1, -1)):
            line.set_ydata(sign*width)
        screen_width = waist*np.sqrt(1+(screen/rayleigh)**2)
        self.caption.value = (f'Screen z = {screen:g} µm; beam radius there = {screen_width:.2f} µm. '
            'Fixed colour/intensity scales; A₀ = 1 at the waist. Changing waist keeps peak '
            'waist intensity fixed, not total power. Microscopic dimensions exaggerate spreading; '
            'a broad laboratory beam spreads much more slowly.')
        self.data = dict(beam=beam, plane=plane, profile=intensity,
            waist=waist, screen_width=screen_width, rayleigh=rayleigh)
        self.picture.value = figure_png(self.figure)
        self.updates += 1

    def reset(self, button=None):
        if self._closed:
            return
        self._resetting = True
        try:
            for control, default in zip(self.controls, (.6, 3, 0)):
                control.value = default
        finally:
            self._resetting = False
        self._update()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for control in self.controls:
            control.unobserve(self._changed, names='value')
        self.reset_button.on_click(self.reset, remove=True)
        def close_tree(widget):
            for child in getattr(widget, 'children', ()):
                close_tree(child)
            widget.close()
        close_tree(self.ui)
        self.figure.clear()
