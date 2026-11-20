"""One owned analytical Gaussian-beam experiment, with retained artists."""
from itertools import combinations
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from .gaussian_order_beams import GaussianOrderConfiguration, GaussianOrderExperiment
from .grating_intro import _close_widget_tree
from .plotting import figure_png
from .gaussian_order_interactive_view import GaussianOrderInteractiveView, fitted_view_bounds
from .gaussian_fringe_diagnostics import fringe_diagnostics
from .gaussian_lesson_figures import GaussianLessonAids


DETECTOR_CMAP = LinearSegmentedColormap.from_list('detector_signal', [
    (0., '#020305'), (.02, '#351008'), (.15, '#c64b0a'),
    (.45, '#ffad32'), (1., '#fff3c4')])


def beam_shell(model, order, z_limits=(-80., 0.)):
    """Display mesh of rho=w(ell), clipped to the illustrated z half-space.

    This is a 1/e² contour relative to each local on-axis intensity, not an
    aperture or physical beam boundary. The field itself is not truncated.
    """
    c = model.configuration
    transverse, direction = model.basis(order)
    ell = np.linspace(-40, 112, 39)
    azimuth = np.linspace(0, 2*np.pi, 25)
    width = c.waist*np.sqrt(1+(ell/c.rayleigh)**2)
    rings = (ell[:, None, None]*direction
             + width[:, None, None]*(np.cos(azimuth)[None, :, None]*transverse
             + np.sin(azimuth)[None, :, None]*np.array([0., 1., 0.])))
    faces = []
    for i in range(len(ell)-1):
        for j in range(len(azimuth)-1):
            polygon = list(rings[[i, i+1, i+1, i], [j, j, j+1, j+1]])
            for boundary, sign in ((z_limits[1], -1), (z_limits[0], 1)):
                clipped = []
                for a, b in zip(polygon, polygon[1:]+polygon[:1]):
                    inside_a, inside_b = sign*(a[2]-boundary) >= 0, sign*(b[2]-boundary) >= 0
                    if inside_a:
                        clipped.append(a)
                    if inside_a != inside_b:
                        clipped.append(a+(b-a)*(boundary-a[2])/(b[2]-a[2]))
                polygon = clipped
                if not polygon:
                    break
            if len(polygon) >= 3:
                faces.append(np.array(polygon))
    return faces


class GaussianOrderBeamsExplorer:
    def __init__(self):
        import ipywidgets as w
        self._closed = self._busy = False
        self.updates = self.model_updates = 0
        self._key = None
        def slider(label, value, lo, hi, step):
            return w.FloatSlider(description=label, value=value, min=lo, max=hi,
                step=step, continuous_update=False, readout_format='.2f' if label.startswith('λ') else '.0f',
                style={'description_width': '90px'}, layout=w.Layout(width='100%'))
        self.wavelength = slider('λ (µm)', 1., .6, 1.2, .1)
        self.pitch = slider('Pitch d (µm)', 4., 4., 8., .5)
        self.pitch.readout_format = '.1f'
        self.waist = slider('Waist w₀ (µm)', 6., 4., 12., 1.)
        self.detector = slider('Depth s (µm)', 40., 0., 80., 1.)
        self.phase = slider('Phase β (°)', 0., -180., 180., 15.)
        self.pair = w.Dropdown(description='Orders (m,p)', options=[],
            style={'description_width': '90px'}, layout=w.Layout(width='100%'))
        self.reset_button = w.Button(description='Reset', icon='undo', layout=w.Layout(width='90px'))
        self.readout = w.HTML()
        self.interactive = GaussianOrderInteractiveView()
        self.picture = self.interactive.figure
        self.section_picture = w.Image(format='png', layout=w.Layout(width='100%'))
        self._inputs = (self.wavelength, self.pitch, self.waist, self.detector, self.phase, self.pair)
        self.x = np.linspace(-80., 80., 1601)
        self.depths = np.arange(81.)
        self.figure = Figure(figsize=(12, 7.5), dpi=105, layout='constrained')
        FigureCanvasAgg(self.figure)
        grid = self.figure.add_gridspec(2, 2, height_ratios=(1.6, .8), wspace=.12)
        self.geometry_axis = self.figure.add_subplot(grid[0, 0], projection='3d')
        self.detector_axis = self.figure.add_subplot(grid[0, 1])
        profile = self.figure.add_subplot(grid[1, 0])
        overlap = self.figure.add_subplot(grid[1, 1])
        self.section_figure = Figure(figsize=(12, 7), dpi=105, layout='constrained')
        FigureCanvasAgg(self.section_figure)
        self.axes = self.section_figure.subplots(2, 2)
        geometry = self.geometry_axis
        geometry.set(title='Incident beam → grating → two finite beams',
            xlabel='x (µm)', ylabel='y (µm)', zlabel='z (µm)',
            xlim=(-64, 64), ylim=(-64, 64), zlim=(-80, 48))
        geometry.set_box_aspect((1, 1, 1))
        geometry.zaxis.labelpad = 0
        geometry.zaxis.label.set_fontsize(8)
        geometry.zaxis.set_rotate_label(False)
        geometry.zaxis.label.set_rotation(90)
        geometry.view_init(elev=24, azim=-64)
        geometry.set_proj_type('ortho')
        grating = Poly3DCollection([[(-64, -64, 0), (64, -64, 0),
                                     (64, 64, 0), (-64, 64, 0)]],
                                  facecolors='#bac5d0', alpha=.13, edgecolors='#667585')
        geometry.add_collection3d(grating)
        geometry.text(-54, 20, 2, 'Grating, z=0', fontsize=7)
        self.grating_ticks = Line3DCollection([], colors='#667585', linewidths=.5, alpha=.5)
        geometry.add_collection3d(self.grating_ticks, autolim=False)
        self.incident_shell = Poly3DCollection([], facecolors='#17806b', edgecolors='none', alpha=.25)
        geometry.add_collection3d(self.incident_shell, autolim=False)
        self.incident_arrow = Line3DCollection([[(0, 0, 30), (0, 0, 8)],
            [(0, 0, 8), (-2, 0, 13)], [(0, 0, 8), (2, 0, 13)]], colors='#17806b', linewidths=2.5)
        geometry.add_collection3d(self.incident_arrow, autolim=False)
        geometry.text(8, 0, 24, 'Incident', color='#126550', fontsize=8)
        self.shells = []
        self.direction_arrows = []
        self.axes_3d = []
        for color in ('#2866a3', '#b85c17'):
            shell = Poly3DCollection([], facecolors=color, edgecolors='none', alpha=.13)
            geometry.add_collection3d(shell, autolim=False)
            self.shells.append(shell)
            arrow = Line3DCollection([], colors=color, linewidths=3)
            geometry.add_collection3d(arrow, autolim=False)
            self.direction_arrows.append(arrow)
            self.axes_3d.append(geometry.plot([], [], [], '--', color=color, lw=1)[0])
        self.detector_surface = Poly3DCollection([], edgecolors='none', linewidths=.1,
                                                antialiaseds=False, alpha=1.)
        geometry.add_collection3d(self.detector_surface, autolim=False)
        self.detector_outline, = geometry.plot([], [], [], color='#7b858f', lw=2)
        self.detector_x = np.linspace(-64, 64, 801)
        self.detector_y = np.linspace(-64, 64, 321)
        self._display_stride = (8, 16)
        xx, yy = np.meshgrid(self.detector_x, self.detector_y)
        self._detector_points = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
        self._detector_key = None
        self.detector_axis.set(title='XY detector: signal (false colour)',
            xlabel='x (µm)', ylabel='y (µm)', xlim=(-64, 64), ylim=(-64, 64))
        self.detector_image = self.detector_axis.imshow(np.zeros_like(xx),
            extent=(-64.08, 64.08, -64.2, 64.2), origin='lower', aspect='equal',
            cmap=DETECTOR_CMAP, vmin=0, vmax=4, interpolation='nearest')
        self.detector_axis.set_facecolor('#020305')
        for spine in self.detector_axis.spines.values():
            spine.set_color('#4b5563')
            spine.set_linewidth(2.)
        self.detector_axis.axhline(0, color='#a1aab5', ls='--', lw=.6, alpha=.5)
        self.figure.colorbar(self.detector_image, ax=self.detector_axis, shrink=.7,
                             label='Relative intensity · false colour')
        self.images = []
        titles = ('Assigned beam m: intensity', 'Assigned beam n: intensity',
                  'Coherent combined intensity', 'Signed interference cross term')
        for axis, title, cmap, limit in zip(self.axes.flat, titles,
                ('Blues', 'Oranges', 'gray', 'RdBu_r'), (1., 1., 4., 2.)):
            axis.set(title=title, xlabel='x (µm)', ylabel='z (µm)', xlim=(-64, 64), ylim=(-80, 0))
            image = axis.imshow(np.zeros((81, 1601)), extent=(-80.05, 80.05, -80.5, .5),
                origin='upper', aspect='equal', cmap=cmap,
                vmin=-limit if cmap == 'RdBu_r' else 0, vmax=limit)
            self.images.append(image)
            axis.axhline(0, color='#243444', lw=2)
        self.section_figure.colorbar(self.images[0], ax=list(self.axes[0]), shrink=.7,
                             label='Individual intensity (waist peak = 1)')
        self.section_figure.colorbar(self.images[2], ax=self.axes[1, 0], shrink=.7, label='Combined intensity')
        self.section_figure.colorbar(self.images[3], ax=self.axes[1, 1], shrink=.7, label='2 Re(Uₘ* Uₚ)')
        self.axes[0, 0].text(.02, .05, 'Dashed lines: axes, not beam boundaries',
                            transform=self.axes[0, 0].transAxes, fontsize=7)
        self.axis_lines = [a.plot([], [], '--', color=c, lw=1)[0]
            for a, c in zip(self.axes[0], ('#2866a3', '#b85c17'))]
        self.detector_lines = [a.axhline(-40, color='#299746', lw=1) for a in list(self.axes.flat)[:4]]
        profile.set(title='XY detector plane: y = 0 profile', xlabel='x (µm)',
                    ylabel='Relative intensity', xlim=(-64, 64), ylim=(-2.1, 4.1))
        self.profiles = [profile.plot([], [], color=c, lw=1, label=l)[0]
            for c, l in (('#2866a3', 'Iₘ'), ('#b85c17', 'Iₚ'), ('#243444', 'Combined'), ('#79539e', 'Cross term'))]
        profile.legend(fontsize=7)
        overlap.set(title='Detector-line spatial overlap (not coherence)',
                    xlabel='Depth s = −z (µm)', ylabel='Oₗᵢₚₑ', xlim=(0, 80), ylim=(-.02, 1.02))
        self.overlap_line, = overlap.plot([], [], color='#17806b')
        self.overlap_marker = overlap.axvline(40, color='#299746', lw=1)
        for axis in (*self.axes.flat, profile, overlap):
            axis.tick_params(labelsize=8)
            axis.title.set_fontsize(10)
        self.sections = w.Accordion(children=[self.section_picture], selected_index=None)
        self.sections.set_title(0, 'Optional xz cross-sections: individual beams, sum and cross term')
        self.sections.observe(self._sections_changed, names='selected_index')
        self._sections_dirty = True
        self.lesson_aids = GaussianLessonAids(self)
        self.fringe_panel = w.Accordion(children=[self.interactive.spacing_figure], selected_index=None)
        self.fringe_panel.set_title(0, 'Fringe spacing: nominal carrier versus local Gaussian phase')
        def control_group(title, controls):
            return w.VBox([w.HTML(f'<b>{title}</b>'), *controls], layout=w.Layout(
                flex='1 1 310px', min_width='280px', padding='6px 10px',
                border='1px solid #d9dfe5', margin='0 6px 4px 0'))
        self.control_panel = w.HBox([
            control_group('Beam & grating', [self.wavelength, self.pitch, self.waist]),
            control_group('Orders & detector', [self.pair, self.detector, self.phase])],
            layout=w.Layout(flex_flow='row wrap', width='100%', max_width='900px'))
        self.ui = w.VBox([
            self.control_panel,
            w.HBox([self.readout, self.reset_button], layout=w.Layout(
                align_items='center', justify_content='space-between', max_width='900px', width='100%')),
            self.picture, self.fringe_panel, self.sections])
        for control in self._inputs:
            control.observe(self._changed, names='value')
        self.reset_button.on_click(self.reset)
        self._update()

    def _changed(self, change):
        if not self._busy and not self._closed:
            self._update()

    def _sections_changed(self, change):
        if not self._closed and self.sections.selected_index is not None and self._sections_dirty:
            self.section_picture.value = figure_png(self.section_figure)
            self._sections_dirty = False

    def _update_geometry_3d(self):
        # Prescribed normal-incidence input, same waist scale; no scattering
        # relation or power/efficiency balance is inferred from these contours.
        self.incident_shell.set_verts(beam_shell(self.model, 0, z_limits=(0., 32.)))
        for shell, line, arrow, order in zip(self.shells, self.axes_3d,
                                            self.direction_arrows, self.pair.value):
            transverse, direction = self.model.basis(order)
            shell.set_verts(beam_shell(self.model, order))
            end = 80*direction/(-direction[2])
            line.set_data_3d([0, end[0]], [0, end[1]], [0, end[2]])
            start, tip = 12*direction, 30*direction
            arrow.set_segments([[start, tip], [tip, tip-5*direction+2*transverse],
                                 [tip, tip-5*direction-2*transverse]])
        positions = np.arange(-64, 64.01, self.pitch.value)
        self.grating_ticks.set_segments([[(x, -64, 0), (x, 64, 0)] for x in positions])

    def _update_detector_3d(self, key):
        detector_key = (key, self.detector.value)
        if detector_key != self._detector_key:
            self.view_bounds=fitted_view_bounds(self.model,self.pair.value,self.detector.value)
            bounds=self.view_bounds
            side=bounds['side']
            self.detector_x=np.linspace(*bounds['x'],int(round(side/.1))+1)
            self.detector_y=np.linspace(*bounds['y'],int(round(side/.4))+1)
            xx,yy=np.meshgrid(self.detector_x,self.detector_y)
            self._detector_points=np.stack((xx,yy,np.full_like(xx,-self.detector.value)),axis=-1)
            self.detector_fields = np.stack([
                self.model.field(self._detector_points, order) for order in self.pair.value])
            self._detector_key = detector_key
            # Coarsen ONLY the display mesh; cell colors are area-averaged below.
            sy, sx = self._display_stride
            x, y = np.meshgrid(self.detector_x[::sx], self.detector_y[::sy])
            z = np.full_like(x, -self.detector.value)
            p = np.stack((x, y, z), axis=-1)
            faces = np.stack((p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]), axis=2)
            self.detector_surface.set_verts(faces.reshape(-1, 4, 3))
            lo,hi=bounds['x'];bottom,top=bounds['y']
            self.detector_outline.set_data_3d([lo,hi,hi,lo,lo],
                [bottom,bottom,top,top,bottom], [-self.detector.value]*5)
            self.detector_image.set_extent((lo-.05,hi+.05,bottom-.2,top+.2))
            self.detector_axis.set(xlim=bounds['x'],ylim=bounds['y'])
            self.geometry_axis.set(xlim=bounds['scene_x'],ylim=bounds['scene_y'],zlim=bounds['scene_z'])
        a, b = self.detector_fields
        self.detector_intensity = abs(a+b*np.exp(1j*np.deg2rad(self.phase.value)))**2
        self.detector_image.set_data(self.detector_intensity)
        # Average sample corners to cell centres, then display-mesh cells.
        intensity = self.detector_intensity
        cells = (intensity[:-1, :-1]+intensity[1:, :-1]
                 +intensity[:-1, 1:]+intensity[1:, 1:])/4
        sy, sx = self._display_stride
        ny, nx = cells.shape
        self.detector_display_intensity = cells.reshape(ny//sy, sy, nx//sx, sx).mean(axis=(1, 3))
        colors = DETECTOR_CMAP(Normalize(0, 4)(self.detector_display_intensity.ravel()))
        self.detector_surface.set_facecolors(colors)
        self.detector_surface.set_edgecolors(colors)
        self.detector_axis.set_title(f'Detector at z={-self.detector.value:g} µm · false colour')

    def _update(self):
        if self._closed:
            return
        self._busy = True
        self.updates += 1
        try:
            geometry = (self.wavelength.value, self.pitch.value, self.waist.value)
            if geometry != getattr(self, '_geometry', None):
                self.model = GaussianOrderExperiment(GaussianOrderConfiguration(*geometry))
                self._geometry = geometry
                old = (0, 1) if getattr(self, '_reset_pair', False) else self.pair.value
                legal = list(combinations(self.model.orders, 2))
                self.pair.options = [(f'({m:+d}, {n:+d})', (m, n)) for m, n in legal]
                self.pair.value = old if old in legal else ((0, 1) if (0, 1) in legal else legal[0])
            key = (geometry, self.pair.value)
            if key != self._key:
                self.data = self.model.sample(self.x, self.depths, self.pair.value)
                self._key = key
                self.model_updates += 1
                for line, order in zip(self.axis_lines, self.pair.value):
                    direction = self.model.basis(order)[1]
                    line.set_data(-self.depths*direction[0]/direction[2], -self.depths)
                for image, array in zip(self.images[:2], self.data['intensities']):
                    image.set_data(array)
                self.overlap_line.set_data(self.depths, self.data['overlap'])
                self._phase_value = None
                self.axes[0, 0].set_title(f'Assigned beam m={self.pair.value[0]:+d}: intensity')
                self.axes[0, 1].set_title(f'Assigned beam p={self.pair.value[1]:+d}: intensity')
                self._update_geometry_3d()
            if self.phase.value != self._phase_value:
                a, b = self.data['fields']
                b = b*np.exp(1j*np.deg2rad(self.phase.value))
                self.cross = 2*np.real(np.conj(a)*b)
                self.combined = abs(a+b)**2
                self.images[2].set_data(self.combined)
                self.images[3].set_data(self.cross)
                self._phase_value = self.phase.value
            row = int(round(self.detector.value))
            for line, array in zip(self.profiles,
                    (*self.data['intensities'], self.combined, self.cross)):
                line.set_data(self.x, array[row])
            for line in self.detector_lines:
                line.set_ydata([-self.detector.value]*2)
            self.overlap_marker.set_xdata([self.detector.value]*2)
            self._update_detector_3d(key)
            fringe_key = (key, self.detector.value)
            if fringe_key != getattr(self, '_fringe_key', None):
                self.fringes = fringe_diagnostics(*self.data['fields'][:, row], self.x,
                    self.pitch.value/abs(self.pair.value[1]-self.pair.value[0]))
                self._fringe_key = fringe_key
            c = self.model.configuration
            angles = [np.rad2deg(np.arcsin(m*c.wavelength/c.period)) for m in self.pair.value]
            self.readout.value = (
                f'θₘ={angles[0]:+.1f}°, θₚ={angles[1]:+.1f}° &nbsp;·&nbsp; '
                f'zᴿ={c.rayleigh:.1f} µm &nbsp;·&nbsp; '
                f'<b>O<sub>line</sub>={self.data["overlap"][row]:.3f}</b>')
            self.interactive.update(self, beam_shell, DETECTOR_CMAP)
            self.lesson_aids.refresh()
            self._sections_dirty = True
            self._sections_changed(None)
        finally:
            self._busy = False

    def reset(self, button=None):
        if self._closed:
            return
        self._busy = True
        try:
            for control, value in ((self.wavelength, 1.), (self.pitch, 4.),
                    (self.waist, 6.), (self.detector, 40.), (self.phase, 0.)):
                control.value = value
            # Rebuild the legal pair list before choosing the default.
            self._geometry = None
            self._reset_pair = True
        finally:
            self._busy = False
        self._update()
        self.interactive.reset_camera()
        self._reset_pair = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        for control in self._inputs:
            control.unobserve(self._changed, names='value')
        self.reset_button.on_click(self.reset, remove=True)
        self.sections.unobserve(self._sections_changed, names='selected_index')
        self.lesson_aids.close()
        _close_widget_tree(self.ui)
        self.section_picture.value = b''
        self.figure.clear()
        self.section_figure.clear()
        self.data = self.cross = self.combined = None
        self.detector_fields = self.detector_intensity = self.detector_display_intensity = None
        self.fringes = None
