"""Matplotlib and Plotly views, independent of Jupyter widget controls."""

import io
import html
from itertools import product, combinations
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyArrowPatch, Circle, Patch
from matplotlib.lines import Line2D
from matplotlib.collections import EllipseCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .waves import (
    direction_cosines, field_from_phasor, plane_wave_phasor,
    scalar_plane_wave, wavenumber, wavevector,
)
from .diffraction import sphere_residuals


def figure_png(figure):
    """Render a Matplotlib figure without requiring an interactive backend."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=figure.dpi,
                   facecolor='white', pil_kwargs={'compress_level': 2})
    return buffer.getvalue()


class PlaneWavePlot:
    """Render a 1D or 2D plane wave and its wavevector to PNG bytes."""

    def __init__(self, dimensions, minimum_wavelength=1.0):
        if dimensions not in (1, 2):
            raise ValueError('Use dimensions=1 or 2.')
        self.dimensions = dimensions
        self.minimum_wavelength = minimum_wavelength
        self._wave_signature = None
        self.figure = Figure(figsize=(9.4, 3.5 if dimensions == 1 else 4.8), dpi=105)
        FigureCanvasAgg(self.figure)
        if dimensions == 1:
            self._build_1d()
        else:
            self._build_2d()

    def _build_1d(self):
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.85)
        self.x = np.linspace(-5, 5, 1000)
        self.line, = self.ax.plot(self.x, np.zeros_like(self.x), color='#2266a5', lw=2.4)
        self.point, = self.ax.plot([0], [1], 'o', color='#db6c24', ms=8, zorder=5)
        self.ax.axvline(0, color='0.7', ls=':', lw=1)
        self.ax.axhline(0, color='0.75', lw=0.8)
        self.ax.set(xlim=(-5, 5), ylim=(-1.35, 1.6),
                    xlabel='Position x (length units)', ylabel='Scalar field u / A')
        self.ax.set_yticks([-1, 0, 1])
        self.ax.spines[['top', 'right']].set_visible(False)
        self.period_arrow = FancyArrowPatch(
            (-4, 1.3), (-2, 1.3), arrowstyle='<->', mutation_scale=12,
            lw=1.5, color='#6c4295')
        self.ax.add_patch(self.period_arrow)
        self.period_text = self.ax.text(-3, 1.39, r'$\lambda$', color='#6c4295', ha='center')
        self.title = self.ax.set_title('Travelling wave', pad=14)
        self.ax.legend([self.point], ['Field at fixed position x = 0'],
                       loc='lower left', fontsize=9, framealpha=0.95)

    def _build_2d(self):
        self.field_ax, self.k_ax = self.figure.subplots(1, 2)
        self.figure.subplots_adjust(left=0.07, right=0.98, bottom=0.19, top=0.88, wspace=0.35)
        self.x = np.linspace(-4, 4, 160)
        self.z = self.x.copy()
        self.X, self.Z = np.meshgrid(self.x, self.z)
        self.field_image = self.field_ax.imshow(
            np.zeros_like(self.X), origin='lower', extent=(-4, 4, -4, 4),
            cmap='RdBu_r', vmin=-1, vmax=1, interpolation='bilinear')
        self.field_ax.set(xlabel='x (length units)', ylabel='z (length units)',
                          title='Real space: scalar field u / A', aspect='equal')
        self.field_ax.set_xticks([-4, -2, 0, 2, 4])
        self.field_ax.set_yticks([-4, -2, 0, 2, 4])
        self.direction_arrow = FancyArrowPatch(
            (0, 0), (0, 1), arrowstyle='-|>', mutation_scale=19,
            lw=2.4, color='#161b22')
        self.field_ax.add_patch(self.direction_arrow)
        self.field_ax.text(0.02, 0.02, 'Arrow: phase direction (unit vector)',
            transform=self.field_ax.transAxes, fontsize=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
        self.field_ax.text(0.02, 0.98, 'Red: +1    White: 0    Blue: -1',
            transform=self.field_ax.transAxes, va='top', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
        limit = 2 * np.pi / self.minimum_wavelength * 1.30
        self.k_ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal',
                      xlabel=r'$k_x$ (radians / length unit)',
                      ylabel=r'$k_z$ (radians / length unit)', title='Wavevector space')
        self.k_ax.set_xticks([-6, -3, 0, 3, 6])
        self.k_ax.set_yticks([-6, -3, 0, 3, 6])
        self.k_ax.axhline(0, color='0.8', lw=0.8)
        self.k_ax.axvline(0, color='0.8', lw=0.8)
        self.circle = Circle((0, 0), radius=np.pi, fill=False, ec='0.65', ls='--', lw=1.1)
        self.k_ax.add_patch(self.circle)
        self.kx_line, = self.k_ax.plot([0, 0], [0, 0], color='#d67a17', lw=2)
        self.kz_line, = self.k_ax.plot([0, 0], [0, 0], color='#238c69', lw=2)
        self.k_arrow = FancyArrowPatch(
            (0, 0), (0, np.pi), arrowstyle='-|>', mutation_scale=17,
            lw=2.5, color='#75429a', zorder=4)
        self.k_ax.add_patch(self.k_arrow)
        self.k_label = self.k_ax.text(0, 0, r'$\mathbf{k}$', color='#75429a', fontsize=12)
        self.k_ax.spines[['top', 'right']].set_visible(False)
        self.k_ax.legend(handles=[
            Line2D([], [], color='#75429a', lw=2, label=r'$\mathbf{k}$'),
            Line2D([], [], color='#d67a17', lw=2, label=r'$k_x$'),
            Line2D([], [], color='#238c69', lw=2, label=r'$k_z$')],
            loc='lower center', bbox_to_anchor=(0.775, 0.005),
            bbox_transform=self.figure.transFigure,
            frameon=False, ncol=3, fontsize=10)

    def render(self, wavelength, direction, fraction, speed=1.0):
        self.c = speed
        self.k = wavenumber(wavelength)
        self.omega = self.c * self.k
        self.period = wavelength / self.c
        self.fraction = fraction
        phase = 2 * np.pi * self.fraction
        signature = (wavelength, direction)
        geometry_changed = signature != self._wave_signature
        self.time_label = f'<span>t / T = {self.fraction:.2f}</span>'
        if self.dimensions == 1:
            self.kx = direction * self.k
            if geometry_changed:
                self._spatial_phase = self.kx * self.x
            self.field = np.cos(self._spatial_phase - phase)
            self.line.set_ydata(self.field)
            self.point.set_ydata([np.cos(-phase)])
            self.period_arrow.set_positions((-4, 1.3), (-4 + wavelength, 1.3))
            self.period_text.set_position((-4 + wavelength / 2, 1.39))
            direction = '+x (right)' if self.kx > 0 else '-x (left)'
            self.title.set_text(f'Phase propagation toward {direction}')
            components = f'k<sub>x</sub> = {self.kx:+.3f}'
        else:
            theta = np.deg2rad(direction)
            self.kx, self.kz = wavevector(wavelength, direction)[[0, 2]]
            if geometry_changed:
                self._spatial_phase = self.kx*self.X + self.kz*self.Z
            self.field = np.cos(self._spatial_phase - phase)
            self.field_image.set_data(self.field)
            self.direction_arrow.set_positions((0, 0), (np.sin(theta), np.cos(theta)))
            self.circle.set_radius(self.k)
            self.k_arrow.set_positions((0, 0), (self.kx, self.kz))
            self.kx_line.set_data([0, self.kx], [0, 0])
            self.kz_line.set_data([self.kx, self.kx], [0, self.kz])
            self.k_label.set_position((1.10 * self.kx, 1.10 * self.kz))
            self.k_label.set_ha('left' if self.kx >= 0 else 'right')
            self.k_label.set_va('bottom' if self.kz >= 0 else 'top')
            components = f'k<sub>x</sub> = {self.kx:+.3f}; k<sub>z</sub> = {self.kz:+.3f}'
        self._wave_signature = signature
        self.summary_html = (
            f'<div style="padding:6px 0">|k| = {self.k:.3f}; {components}'
            f' &nbsp; (radians / length unit)<br>'
            f'T = {self.period:.2f}; t = {self.fraction * self.period:.2f}'
            f' &nbsp; (time units); &omega; = {self.omega:.3f} rad / time unit</div>')
        return figure_png(self.figure)


class ScalarPlaneWave3DPlot:
    """Render constant-phase planes in a cube and the corresponding k sphere."""

    def __init__(self, minimum_wavelength=1.0):
        self.minimum_wavelength = minimum_wavelength
        self._geometry_signature = self._front_signature = self._camera_signature = None
        self.half_width = 2.0
        self.corners = np.array(list(product((-self.half_width, self.half_width), repeat=3)))
        self.edges = [(a, b) for a, b in combinations(self.corners, 2)
                      if np.count_nonzero(a != b) == 1]
        self.figure = Figure(figsize=(11.4, 5.4), dpi=100)
        FigureCanvasAgg(self.figure)
        self.space_ax = self.figure.add_subplot(121, projection='3d')
        self.k_ax = self.figure.add_subplot(122, projection='3d')
        self.figure.subplots_adjust(left=0.01, right=0.96, bottom=0.14, top=0.85, wspace=0.08)
        for ax in (self.space_ax, self.k_ax):
            ax.set_box_aspect((1, 1, 1))
            ax.set_proj_type('ortho')
            ax.grid(False)
            ax.tick_params(labelsize=8, pad=0)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False
                axis.pane.set_edgecolor('0.9')
        self.space_ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2),
            xlabel='x (length units)', ylabel='y (length units)', zlabel='z (length units)')
        for setter in (self.space_ax.set_xticks, self.space_ax.set_yticks, self.space_ax.set_zticks):
            setter([-2, 0, 2])
        self.space_ax.set_title('Real space: moving wavefront planes', pad=18)
        for a, b in self.edges:
            self.space_ax.plot(*np.array([a, b]).T, color='0.65', linewidth=0.7, alpha=0.6)
        self.front_collection = Poly3DCollection([], linewidths=0.6)
        self.space_ax.add_collection3d(self.front_collection)
        limit = 1.30 * 2 * np.pi / self.minimum_wavelength
        self.k_ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
                      xlabel=r'$k_x$', ylabel=r'$k_y$', zlabel=r'$k_z$')
        for setter in (self.k_ax.set_xticks, self.k_ax.set_yticks, self.k_ax.set_zticks):
            setter([-6, 0, 6])
        self.k_ax.set_title('Wavevector space: sphere radius k', pad=18)
        self.figure.text(0.75, 0.095, 'Wavevector axes: radians / length unit',
                         ha='center', fontsize=9, color='0.35')
        self.component_lines = [self.k_ax.plot([], [], [], color=color, linewidth=2)[0]
                                for color in ('#d67a17', '#2266a5', '#238c69')]
        self.k_text = self.k_ax.text(0, 0, 0, r'$\mathbf{k}$', color='#75429a', fontsize=12)
        self.direction_text = self.space_ax.text(0, 0, 0, r'$\hat{\mathbf{k}}$', fontsize=12)
        longitude, colatitude = np.meshgrid(np.linspace(0, 2*np.pi, 25), np.linspace(0, np.pi, 13))
        self.unit_sphere = (np.sin(colatitude)*np.cos(longitude),
                            np.sin(colatitude)*np.sin(longitude), np.cos(colatitude))
        self.sphere = self.k_arrow = self.direction_arrow = None
        self.figure.legend(handles=[
            Patch(facecolor='#df792a', alpha=0.65, label='Crest: u/A = +1'),
            Patch(facecolor='#2776b4', alpha=0.65, label='Trough: u/A = -1')],
            loc='lower center', bbox_to_anchor=(0.25, 0.015), frameon=False, ncol=2, fontsize=9)
        self.figure.legend(handles=[Line2D([], [], color=color, lw=2, label=label)
            for color, label in [('#75429a', r'$\mathbf{k}$'), ('#d67a17', r'$k_x$'),
                                 ('#2266a5', r'$k_y$'), ('#238c69', r'$k_z$')]],
            loc='lower center', bbox_to_anchor=(0.75, 0.015), frameon=False, ncol=4, fontsize=10)

    def _plane_polygon(self, normal, distance):
        # Intersect an infinite constant-phase plane with the display cube.
        points = []
        for a, b in self.edges:
            da, db = a @ normal - distance, b @ normal - distance
            if abs(da) < 1e-10:
                points.append(a)
            if abs(db) < 1e-10:
                points.append(b)
            if da * db < 0:
                points.append(a + da / (da - db) * (b - a))
        if not points:
            return None
        vertices = np.unique(np.round(points, 10), axis=0)
        if len(vertices) < 3:
            return None
        reference = np.eye(3)[np.argmin(np.abs(normal))]
        basis_a = np.cross(normal, reference)
        basis_a /= np.linalg.norm(basis_a)
        basis_b = np.cross(normal, basis_a)
        centred = vertices - vertices.mean(axis=0)
        angles = np.arctan2(centred @ basis_b, centred @ basis_a)
        return vertices[np.argsort(angles)]

    def field_at(self, points):
        # A scalar at every spatial point; useful for checking phase geometry.
        return scalar_plane_wave(points, self.k_vector, phase=-2*np.pi*self.fraction)

    def render(self, wavelength, theta_deg, phi_deg, fraction,
               elevation=25, azimuth=-55, speed=1.0):
        self.c = speed
        self.k = wavenumber(wavelength)
        self.period = wavelength / self.c
        self.fraction = fraction
        self.direction = direction_cosines(theta_deg, phi_deg)
        self.k_vector = self.k * self.direction
        geometry_signature = (wavelength, theta_deg, phi_deg)
        front_signature = (*geometry_signature, fraction)
        if front_signature != self._front_signature:
            self._update_fronts(wavelength)
            self._front_signature = front_signature
        if geometry_signature != self._geometry_signature:
            self._update_geometry()
            self._geometry_signature = geometry_signature
        camera_signature = (elevation, azimuth)
        if camera_signature != self._camera_signature:
            for ax in (self.space_ax, self.k_ax):
                ax.view_init(elev=elevation, azim=azimuth)
            self._camera_signature = camera_signature
        self.time_label = f't / T = {self.fraction:.2f}'
        kx, ky, kz = self.k_vector
        self.summary_html = (
            f'<div style="padding:6px 0">|<b>k</b>| = k = {self.k:.3f}; '
            f'(k<sub>x</sub>, k<sub>y</sub>, k<sub>z</sub>) = '
            f'({kx:+.3f}, {ky:+.3f}, {kz:+.3f}) radians / length unit<br>'
            f'T = {self.period:.2f}; t = {self.fraction*self.period:.2f} time units</div>')
        return figure_png(self.figure)

    def _update_fronts(self, wavelength):
        """Only changing phase or wave geometry moves the physical wavefronts."""
        extent = self.half_width * np.abs(self.direction).sum()
        self.fronts = []
        faces, colours, edges = [], [], []
        for shift, value, rgb in [(0.0, 1.0, (0.87, 0.47, 0.16)),
                                  (0.5, -1.0, (0.15, 0.46, 0.71))]:
            start = int(np.floor(-extent/wavelength - self.fraction - shift))
            stop = int(np.ceil(extent/wavelength - self.fraction - shift))
            for order in range(start, stop + 1):
                distance = wavelength * (order + shift + self.fraction)
                polygon = self._plane_polygon(self.direction, distance)
                if polygon is None:
                    continue
                self.fronts.append((value, distance, polygon))
                faces.append(polygon)
                colours.append((*rgb, 0.18))
                edges.append((*rgb, 0.55))
        self.front_collection.set_verts(faces)
        self.front_collection.set_facecolor(colours)
        self.front_collection.set_edgecolor(edges)

    def _update_geometry(self):
        """Retain the k sphere and arrows throughout playback and camera moves."""
        for artist in (self.sphere, self.k_arrow, self.direction_arrow):
            if artist is not None:
                artist.remove()
        self.sphere = self.k_ax.plot_wireframe(
            *(self.k * coordinates for coordinates in self.unit_sphere),
            color='0.6', alpha=0.35, linewidth=0.55, rstride=2, cstride=2)
        self.k_arrow = self.k_ax.quiver(0, 0, 0, *self.k_vector,
            color='#75429a', linewidth=2.8, arrow_length_ratio=0.16)
        self.direction_arrow = self.space_ax.quiver(0, 0, 0, *(1.5*self.direction),
            color='#151515', linewidth=2.6, arrow_length_ratio=0.18)
        vertices = np.array([[0., 0., 0.], [self.k_vector[0], 0., 0.],
            [self.k_vector[0], self.k_vector[1], 0.], self.k_vector])
        for line, a, b in zip(self.component_lines, vertices[:-1], vertices[1:]):
            line.set_data_3d(*np.array([a, b]).T)
        self.k_text.set_position_3d(tuple(1.12*self.k_vector))
        self.direction_text.set_position_3d(tuple(1.7*self.direction))


class PlaneWavePhasorPlot:
    """A local phasor and its real time trace, for a unit-amplitude scalar wave.

    The phasor is time independent. The second arrow includes exp(-i omega t)
    and rotates clockwise as t/T increases. Neither arrow is a wavevector.
    """

    def __init__(self):
        self.figure = Figure(figsize=(9.4, 3.8), dpi=105)
        FigureCanvasAgg(self.figure)
        self.complex_ax, self.time_ax = self.figure.subplots(
            1, 2, gridspec_kw={'width_ratios': [1, 1.65]})
        self.figure.subplots_adjust(left=0.07, right=0.98, bottom=0.24,
                                    top=0.84, wspace=0.37)
        self.complex_ax.set(
            xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect='equal',
            xlabel='Real amplitude / A', ylabel='Imaginary amplitude / A',
            title='Complex plane at position P')
        self.complex_ax.set_xticks([-1, 0, 1])
        self.complex_ax.set_yticks([-1, 0, 1])
        self.complex_ax.axhline(0, color='0.75', lw=0.8)
        self.complex_ax.axvline(0, color='0.75', lw=0.8)
        self.complex_ax.add_patch(Circle((0, 0), 1, fill=False, ec='0.7', lw=1))
        self.reference_arrow = FancyArrowPatch(
            (0, 0), (1, 0), arrowstyle='-|>', mutation_scale=15,
            color='#2266a5', linestyle='--', linewidth=1.8, zorder=3)
        self.rotating_arrow = FancyArrowPatch(
            (0, 0), (1, 0), arrowstyle='-|>', mutation_scale=18,
            color='#75429a', linewidth=2.6, zorder=4)
        self.complex_ax.add_patch(self.reference_arrow)
        self.complex_ax.add_patch(self.rotating_arrow)
        self.reference_tip, = self.complex_ax.plot(
            [], [], 'o', color='#2266a5', ms=7, mfc='white', mew=1.5, zorder=5)
        self.projection_line, = self.complex_ax.plot(
            [], [], ':', color='#db6c24', lw=1.6)
        self.real_point, = self.complex_ax.plot(
            [], [], 'o', color='#db6c24', ms=7, zorder=6)
        self.time_samples = np.linspace(0, 1, 401)
        self.time_trace, = self.time_ax.plot([], [], color='#2266a5', lw=2.2)
        self.time_cursor = self.time_ax.axvline(0, color='0.65', ls=':', lw=1)
        self.time_point, = self.time_ax.plot([], [], 'o', color='#db6c24', ms=8)
        self.time_ax.axhline(0, color='0.8', lw=0.8)
        self.time_ax.set(
            xlim=(0, 1), ylim=(-1.25, 1.25), xlabel='Time t / T',
            ylabel='Real field u(P, t) / A', title='Oscillation at the same fixed position')
        self.time_ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        self.time_ax.set_yticks([-1, 0, 1])
        for axis in [self.complex_ax, self.time_ax]:
            axis.spines[['top', 'right']].set_visible(False)
        self.figure.legend(handles=[
            Line2D([], [], color='#2266a5', ls='--', lw=1.8,
                   label=r'Phasor $\tilde u(P)$: fixed in time'),
            Line2D([], [], color='#75429a', lw=2.6,
                   label=r'$\tilde u(P)e^{-i\omega t}$: rotating'),
            Line2D([], [], color='#db6c24', marker='o', ls='none',
                   label='Real part: physical field')],
            loc='lower center', bbox_to_anchor=(0.5, 0.005),
            ncol=3, frameon=False, fontsize=9)

    def render(self, k_vector, position, fraction):
        """Render a point in 1D, 2D, or 3D; fraction is t/T."""
        self.k_vector = np.asarray(k_vector, dtype=float)
        self.position = np.asarray(position, dtype=float)
        self.fraction = float(fraction)
        self.phasor = complex(plane_wave_phasor(self.position, self.k_vector))
        self.instantaneous = self.phasor * np.exp(-2j * np.pi * self.fraction)
        self.real_value = float(field_from_phasor(self.phasor, self.fraction, 2*np.pi))
        fixed_tip = (self.phasor.real, self.phasor.imag)
        moving_tip = (self.instantaneous.real, self.instantaneous.imag)
        self.reference_arrow.set_positions((0, 0), fixed_tip)
        self.rotating_arrow.set_positions((0, 0), moving_tip)
        self.reference_tip.set_data([fixed_tip[0]], [fixed_tip[1]])
        self.projection_line.set_data([moving_tip[0], moving_tip[0]], [moving_tip[1], 0])
        self.real_point.set_data([self.real_value], [0])
        self.time_trace.set_data(
            self.time_samples, field_from_phasor(self.phasor, self.time_samples, 2*np.pi))
        self.time_cursor.set_xdata([self.fraction, self.fraction])
        self.time_point.set_data([self.fraction], [self.real_value])
        return figure_png(self.figure)


class SpatialPhasorGridPlot:
    """Local complex-plane glyphs at sampled positions of a scalar plane wave.

    Glyph centres are spatial coordinates. Within each glyph, right means
    positive real amplitude and up means positive imaginary amplitude.
    The 3D field is displayed as three x-z sections at different y values.
    """

    def __init__(self, dimensions):
        if dimensions not in (1, 2, 3):
            raise ValueError('Use dimensions=1, 2, or 3.')
        self.dimensions = dimensions
        size = {1: (9.4, 4.8), 2: (6.8, 5.6), 3: (11.4, 4.8)}[dimensions]
        self.figure = Figure(figsize=size, dpi=105)
        FigureCanvasAgg(self.figure)
        self.sample_axes = []
        self.positions = []
        self.glyphs = []
        self.arrows = []
        self.real_markers = []
        if dimensions == 1:
            self.wave_ax, grid_ax = self.figure.subplots(
                2, 1, sharex=True, gridspec_kw={'height_ratios': [1.4, 1]})
            self.figure.subplots_adjust(left=0.08, right=0.97, bottom=0.20,
                                        top=0.83, hspace=0.27)
            samples = np.linspace(-1, 1, 13)
            positions = samples[:, None]
            centers = np.column_stack([samples, np.zeros_like(samples)])
            self._add_grid(grid_ax, positions, centers, radius=0.065)
            grid_ax.set(xlim=(-1.16, 1.16), ylim=(-0.13, 0.13),
                        xlabel='Position x (length units)', aspect='equal', adjustable='box')
            grid_ax.set_yticks([])
            grid_ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            grid_ax.set_title('One local complex plane at each x', fontsize=11, pad=12)
            grid_ax.spines[['top', 'right', 'left']].set_visible(False)
            self.wave_x = np.linspace(-1, 1, 401)
            self.wave_line, = self.wave_ax.plot([], [], color='#2266a5', lw=2.2)
            self.sample_values, = self.wave_ax.plot([], [], 'o', color='#db6c24', ms=5)
            self.wave_ax.axhline(0, color='0.8', lw=0.8)
            self.wave_ax.set(ylim=(-1.25, 1.25), ylabel='Real part / A')
            self.wave_ax.set_yticks([-1, 0, 1])
            self.wave_ax.spines[['top', 'right']].set_visible(False)
        else:
            axes = np.atleast_1d(self.figure.subplots(1, 3 if dimensions == 3 else 1))
            self.figure.subplots_adjust(left=0.065, right=0.98, bottom=0.20,
                                        top=0.81, wspace=0.28)
            samples = np.linspace(-1, 1, 7)
            X, Z = np.meshgrid(samples, samples)
            centers = np.column_stack([X.ravel(), Z.ravel()])
            self.slice_y = np.array([-0.5, 0.0, 0.5]) if dimensions == 3 else None
            for index, axis in enumerate(axes):
                if dimensions == 2:
                    positions = centers.copy()
                    title = 'Positions in the x-z plane'
                else:
                    y = self.slice_y[index]
                    positions = np.column_stack([X.ravel(), np.full(X.size, y), Z.ravel()])
                    title = f'x-z slice at y = {y:+.1f}'
                self._add_grid(axis, positions, centers, radius=0.115)
                axis.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect='equal',
                         xlabel='x (length units)', ylabel='z (length units)')
                axis.set_xticks([-1, 0, 1])
                axis.set_yticks([-1, 0, 1])
                axis.set_title(title, fontsize=11, pad=12)
                axis.spines[['top', 'right']].set_visible(False)
        self.heading = self.figure.suptitle('', fontsize=13, y=0.98)
        self.figure.text(0.5, 0.085,
            'Each small circle is a local complex plane: right = +Re, up = +Im.',
            ha='center', fontsize=10, color='0.25')
        self.figure.text(0.5, 0.04,
            'Purple arrow: complex value. Orange dot: its real part. All amplitudes A = 1.',
            ha='center', fontsize=10, color='0.25')

    def _add_grid(self, axis, positions, centers, radius):
        count = len(positions)
        circles = EllipseCollection(
            np.full(count, 2*radius), np.full(count, 2*radius), np.zeros(count),
            units='xy', offsets=centers, offset_transform=axis.transData,
            facecolors='none', edgecolors='0.75', linewidths=0.65)
        axis.add_collection(circles)
        arrows = axis.quiver(centers[:, 0], centers[:, 1], np.ones(count), np.zeros(count),
            angles='xy', scale_units='xy', scale=1/radius, color='#75429a',
            width=0.005 if self.dimensions == 3 else 0.0035,
            headwidth=3.4, headlength=4.5, headaxislength=4, pivot='tail', zorder=3)
        marker, = axis.plot([], [], 'o', color='#db6c24', ms=3.5, zorder=4)
        self.sample_axes.append(axis)
        self.positions.append(positions)
        self.glyphs.append((centers, radius))
        self.arrows.append(arrows)
        self.real_markers.append(marker)

    def render(self, k_vector, fraction=0.0, include_time=True):
        """Return PNG bytes. Time is applied uniformly to all spatial samples."""
        self.k_vector = np.asarray(k_vector, dtype=float)
        if self.k_vector.shape != (self.dimensions,):
            raise ValueError('k_vector must match the grid spatial dimension.')
        self.fraction = float(fraction)
        self.include_time = bool(include_time)
        self.phasors = [plane_wave_phasor(points, self.k_vector) for points in self.positions]
        factor = np.exp(-2j*np.pi*self.fraction) if include_time else 1
        self.values = [phasors * factor for phasors in self.phasors]
        for values, arrows, marker, (centers, radius) in zip(
                self.values, self.arrows, self.real_markers, self.glyphs):
            arrows.set_UVC(values.real, values.imag)
            marker.set_data(centers[:, 0] + radius*values.real, centers[:, 1])
        quantity = (r'Complex field $\tilde{u}(\mathbf{r})e^{-i\omega t}$'
                    if include_time else r'Spatial phasors $\tilde{u}(\mathbf{r})$: time suppressed')
        self.heading.set_text(quantity)
        if self.dimensions == 1:
            values = plane_wave_phasor(self.wave_x[:, None], self.k_vector) * factor
            self.wave_line.set_data(self.wave_x, values.real)
            self.sample_values.set_data(self.positions[0][:, 0], self.values[0].real)
            self.wave_ax.set_title('Real field at the sampled positions' if include_time
                                   else 'Real part of the spatial phasor (the field at t = 0)',
                                   fontsize=11, pad=10)
        return figure_png(self.figure)


GEOMETRY_COLOURS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#7851A9",
]


INK, GRID = "#263442", "#bec8d2"


def geometry_order_colour(order):
    """Use the same colour for +m and -m; reserve the first colour for zero."""
    magnitude = abs(int(order))
    if magnitude == 0:
        return GEOMETRY_COLOURS[0]
    return GEOMETRY_COLOURS[1 + (magnitude - 1) % (len(GEOMETRY_COLOURS) - 1)]


def geometry_segments(segments):
    coordinates = [[], [], []]
    for segment in segments:
        for point in segment:
            for axis in range(3):
                coordinates[axis].append(float(point[axis]))
        for axis in coordinates:
            axis.append(None)
    return coordinates


def geometry_wire(center, upper=False):
    curves = []
    azimuth = np.linspace(0, 2*np.pi, 73)
    polar_max = np.pi/2 if upper else np.pi
    for polar in np.linspace(0, polar_max, 7):
        curves.append(np.column_stack([
            np.sin(polar)*np.cos(azimuth),
            np.sin(polar)*np.sin(azimuth), np.full_like(azimuth, np.cos(polar))
        ]) + center)
    polar = np.linspace(0, polar_max, 49)
    for az in np.linspace(0, 2*np.pi, 12, endpoint=False):
        curves.append(np.column_stack([
            np.sin(polar)*np.cos(az), np.sin(polar)*np.sin(az), np.cos(polar)
        ]) + center)
    return geometry_segments(curves)


def geometry_line3d(xyz, colour=GRID, width=2, dash="solid", name=""):
    return go.Scatter3d(x=xyz[0], y=xyz[1], z=xyz[2], mode="lines",
                        line=dict(color=colour, width=width, dash=dash),
                        name=name, showlegend=False, hoverinfo="skip")


def geometry_line2d(x, y, colour=GRID, width=2, dash="solid"):
    return go.Scatter(x=x, y=y, mode="lines",
                      line=dict(color=colour, width=width, dash=dash),
                      showlegend=False, hoverinfo="skip")


def geometry_base(left_title, right_title):
    fig = make_subplots(rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.49, 0.51], horizontal_spacing=0.09,
        subplot_titles=[left_title, right_title])
    axis = dict(showbackground=False, gridcolor="#e5eaf0", zerolinecolor=GRID,
                showspikes=False, nticks=5)
    fig.update_layout(
        template="plotly_white", height=550, autosize=True,
        margin=dict(l=35, r=25, t=70, b=65),
        font=dict(family="Arial, sans-serif", size=12, color=INK),
        uirevision="diffraction-geometry",
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data",
                   dragmode="orbit", uirevision="keep-camera",
                   camera=dict(eye=dict(x=1.4, y=-2.0, z=1.2))),
        legend=dict(orientation="h", y=-0.13, x=0.5, xanchor="center"),
        hovermode="closest")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _coordinate_projection(data, projection):
    k = data['k']
    na_ratio = data['numerical_aperture'] / data['refractive_index']
    choices = {
        'cartesian': (k, 'kₓ (rad/μm)', 'kᵧ (rad/μm)', 'Cartesian transverse components'),
        'direction': (1.0, 'α = kₓ/k', 'β = kᵧ/k', 'Direction-cosine disk'),
        'frequency': (k / (2*np.pi), 'fₓ (cycles/μm)', 'fᵧ (cycles/μm)', 'Spatial-frequency plane'),
        'pupil': (1 / na_ratio, 'σₓ = kₓ/(k₀ NA)', 'σᵧ = kᵧ/(k₀ NA)', 'Normalized pupil coordinates'),
    }
    if projection not in choices:
        raise ValueError(f"Unknown projection: {projection}")
    return choices[projection]


def coordinate_figure(data, projection="direction"):
    """One wavevector and its transverse projection, with lengths in micrometres."""
    k = data['k']
    vector = data['cartesian']
    alpha, beta, gamma = data['direction_cosines']
    na_ratio = data['numerical_aperture'] / data['refractive_index']
    scale, xlabel, ylabel, title = _coordinate_projection(data, projection)
    fig = geometry_base(f'Wavevector sphere: k = {k:.3f} rad/μm', title)
    fig.layout.uirevision = 'wavevector-coordinates'
    wire = [[None if value is None else k * value for value in axis]
            for axis in geometry_wire(np.zeros(3))]
    fig.add_trace(geometry_line3d(wire, '#ccd4dc', 1), row=1, col=1)
    fig.add_trace(geometry_line3d(np.column_stack([np.zeros(3), vector]),
                                 '#7055b5', 7), row=1, col=1)
    fig.add_trace(go.Cone(
        x=[vector[0]], y=[vector[1]], z=[vector[2]],
        u=[alpha], v=[beta], w=[gamma], anchor='tip',
        sizemode='absolute', sizeref=0.15*k, showscale=False,
        colorscale=[[0, '#7055b5'], [1, '#7055b5']], hoverinfo='skip',
        showlegend=False), row=1, col=1)
    # Dashed component steps terminate at the actual vector tip.
    components = np.array([[0, 0, 0], [vector[0], 0, 0],
                           [vector[0], vector[1], 0], vector])
    fig.add_trace(geometry_line3d(components.T, '#bc7d33', 3, 'dash'), row=1, col=1)
    angle = np.linspace(0, 2*np.pi, 181)
    rim = k * np.array([na_ratio*np.cos(angle), na_ratio*np.sin(angle),
                       np.full_like(angle, np.sqrt(max(0, 1-na_ratio**2)))])
    fig.add_trace(geometry_line3d(rim, '#15806b', 4), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[vector[0]], y=[vector[1]], z=[vector[2]], mode='markers',
        marker=dict(size=3, color='#7055b5'), showlegend=False,
        hovertemplate='kₓ=%{x:.4f}<br>kᵧ=%{y:.4f}<br>k_z=%{z:.4f}<extra>rad/μm</extra>'
    ), row=1, col=1)
    for radius, color, dash, name in [
        (scale, '#8996a3', 'dot', 'Propagating limit'),
        (scale * na_ratio, '#15806b', 'solid', 'Collection rim (+z)')
    ]:
        fig.add_trace(go.Scatter(
            x=radius*np.cos(angle), y=radius*np.sin(angle), mode='lines',
            line=dict(color=color, width=2, dash=dash), name=name,
            hoverinfo='skip'), row=1, col=2)
    point = scale * np.array([alpha, beta])
    fig.add_trace(go.Scatter(
        x=[0, point[0]], y=[0, point[1]], mode='lines+markers',
        line=dict(color='#7055b5', width=3),
        marker=dict(size=[3, 11], symbol=['circle', 'circle' if gamma >= 0 else 'x']),
        name=f'Selected wave ({data["hemisphere"]})',
        hovertemplate='horizontal=%{x:.4f}<br>vertical=%{y:.4f}<extra></extra>'
    ), row=1, col=2)
    for axis, label in zip(('xaxis', 'yaxis', 'zaxis'),
                           ('kₓ (rad/μm)', 'kᵧ (rad/μm)', 'k_z (rad/μm)')):
        fig.layout.scene[axis].update(title=label, range=[-1.13*k, 1.13*k])
    limit = 1.12 * scale
    fig.update_xaxes(title=xlabel, range=[-limit, limit], row=1, col=2)
    fig.update_yaxes(title=ylabel, range=[-limit, limit], row=1, col=2)
    fig.update_layout(height=540, margin=dict(l=35, r=25, t=60, b=90))
    return fig


def update_coordinate_figure(figure, data, projection='direction', previous=None,
                             previous_projection=None):
    """Update persistent traces and only the layout properties that can change."""
    k = data['k']
    vector = data['cartesian']
    alpha, beta, gamma = data['direction_cosines']
    na_ratio = data['numerical_aperture']/data['refractive_index']
    scale, xlabel, ylabel, title = _coordinate_projection(data, projection)
    scale_changed = (previous is None or previous_projection is None
                     or previous_projection != projection
                     or _coordinate_projection(previous, previous_projection)[0] != scale)
    angle = np.linspace(0, 2*np.pi, 181)
    with figure.batch_update():
        if previous is None or previous['k'] != k:
            wire = [[None if value is None else k*value for value in axis]
                    for axis in geometry_wire(np.zeros(3))]
            figure.data[0].update(x=wire[0], y=wire[1], z=wire[2])
            for axis in ('xaxis', 'yaxis', 'zaxis'):
                figure.layout.scene[axis].range = [-1.13*k, 1.13*k]
            figure.layout.annotations[0].text = f'Wavevector sphere: k = {k:.3f} rad/μm'
        figure.data[1].update(x=[0, vector[0]], y=[0, vector[1]], z=[0, vector[2]])
        figure.data[2].update(x=[vector[0]], y=[vector[1]], z=[vector[2]],
                              u=[alpha], v=[beta], w=[gamma], sizeref=.15*k)
        components = np.array([[0, 0, 0], [vector[0], 0, 0],
                               [vector[0], vector[1], 0], vector]).T
        figure.data[3].update(x=components[0], y=components[1], z=components[2])
        figure.data[4].update(x=k*na_ratio*np.cos(angle), y=k*na_ratio*np.sin(angle),
                              z=k*np.full_like(angle, np.sqrt(max(0, 1-na_ratio**2))))
        figure.data[5].update(x=[vector[0]], y=[vector[1]], z=[vector[2]])
        for trace, radius in zip(figure.data[6:8], (scale, scale*na_ratio)):
            trace.update(x=radius*np.cos(angle), y=radius*np.sin(angle))
        point = scale*np.array([alpha, beta])
        figure.data[8].update(x=[0, point[0]], y=[0, point[1]],
            marker=dict(symbol=['circle', 'circle' if gamma >= 0 else 'x']),
            name=f'Selected wave ({data["hemisphere"]})')
        figure.layout.annotations[1].text = title
        if scale_changed:
            figure.layout.xaxis.update(title=xlabel, range=[-1.12*scale, 1.12*scale])
            figure.layout.yaxis.update(title=ylabel, range=[-1.12*scale, 1.12*scale])


def coordinate_readout(data):
    """Compact HTML readout for the coordinate explorer; no files are written."""
    def vector(values):
        return '(' + ', '.join(f'{v:+.4f}' for v in values) + ')'

    azimuth = data['spherical'][2]
    azimuth_text = 'undefined at pole' if azimuth is None else f'{azimuth:.1f}°'
    rows = [
        ('Cartesian (kₓ, kᵧ, k_z)', vector(data['cartesian']) + ' rad/μm'),
        ('Spherical (k, θ, φ)', f'({data["k"]:.4f} rad/μm, {data["spherical"][1]:.1f}°, {azimuth_text})'),
        ('Cylindrical (k⊥, φ, k_z)', f'({data["cylindrical"][0]:.4f} rad/μm, {azimuth_text}, {data["cartesian"][2]:+.4f} rad/μm)'),
        ('Direction cosines (α, β, γ)', vector(data['direction_cosines'])),
        ('Spatial frequencies (fₓ, fᵧ, f_z)', vector(data['spatial_frequency']) + ' cycles/μm'),
        ('Normalized pupil (σₓ, σᵧ)', vector(data['pupil'])),
    ]
    table = '<table style="font-size:13px">' + ''.join(
        f'<tr><td style="padding:2px 20px 2px 0">{label}</td><td>{value}</td></tr>'
        for label, value in rows) + '</table>'
    if data['hemisphere'] == '-z':
        status = 'Backward wave: outside the +z collection cone, even if its projected point is inside the pupil.'
    elif data['collected']:
        status = 'On or inside the ideal +z collection cone.'
    else:
        status = 'Propagating wave outside the collection cone.'
    return (f'<p>λ in medium = {data["medium_wavelength"]:.4f} μm; '
            f'hemisphere: <b>{data["hemisphere"]}</b>; '
            f'pupil radius = {data["pupil_radius"]:.4f}.<br>{status}</p>' + table)


def geometry_figures(data, views=('direction', 'ewald')):
    """Build only the requested views, returned in the requested order."""
    views = tuple(views)
    if len(set(views)) != len(views) or any(v not in ('direction', 'ewald') for v in views):
        raise ValueError("views must contain unique 'direction'/'ewald' names.")
    direction = ewald = None
    if not len(data["allowed"]):
        raise ValueError("At least one real order is needed to draw the geometry.")
    d = dict(data, colours=[geometry_order_colour(m) for m in data["allowed"]])
    si, rho = d["incident"], d["rho"]
    angles = np.linspace(0, 2*np.pi, 241)
    arc = np.linspace(0, np.pi, 161)
    s, q = d["directions"], d["changes"]
    labels, allowed = d["labels"], d["allowed"]
    texts = [f"m={m:+d}" for m in allowed] if len(allowed) <= 11 else [""]*len(allowed)
    symbols = ["diamond" if d["grazing"][i] else "circle"
               for i in np.flatnonzero(d["valid"])]
    if 'direction' in views:
        direction = geometry_base("Transmitted direction hemisphere",
                                  "Harvey's direction-cosine disk")
        direction.add_trace(geometry_line3d(geometry_wire(np.zeros(3), upper=True)), row=1, col=1)
        latitude = [rho*np.cos(arc), np.full_like(arc, si[1]), rho*np.sin(arc)]
        direction.add_trace(geometry_line3d(latitude, INK, 3), row=1, col=1)
        rays = geometry_segments([[np.zeros(3), point] for point in s])
        direction.add_trace(geometry_line3d(rays, "#8293a4", 2), row=1, col=1)
        projections = geometry_segments([[point, [point[0], point[1], 0]] for point in s])
        direction.add_trace(geometry_line3d(projections, "#879aab", 2, "dot"), row=1, col=1)
        direction.add_trace(go.Scatter3d(
            x=s[:, 0], y=s[:, 1], z=s[:, 2], mode="markers+text",
            text=texts, textposition="top center", textfont=dict(size=11),
            marker=dict(size=5, color=d["colours"], symbol=symbols),
            customdata=[[int(m)] for m in allowed], showlegend=False,
            hovertemplate="m=%{customdata[0]}<br>α=%{x:.4f}<br>β=%{y:.4f}<br>γ=%{z:.4f}<extra></extra>"
        ), row=1, col=1)
        direction.add_trace(go.Scatter3d(
            x=[si[0]], y=[si[1]], z=[si[2]], mode="markers",
            marker=dict(size=8, color=INK, symbol="circle-open", line=dict(width=2)),
            showlegend=False, hovertemplate="Incident direction = zero order<extra></extra>"
        ), row=1, col=1)
        direction.add_trace(go.Scatter(
            x=np.cos(angles), y=np.sin(angles), mode="lines",
            line=dict(color=GRID), fill="toself", fillcolor="#f3f6fa",
            hoverinfo="skip", showlegend=False
        ), row=1, col=2)
        xlim = [min(-1.2, d["a"].min()-0.16), max(1.2, d["a"].max()+0.16)]
        direction.add_trace(geometry_line2d(xlim, [si[1]]*2, INK, 1, "dash"), row=1, col=2)
        states = ["evanescent" if not valid else ("grazing" if grazing else "propagating")
                  for valid, grazing in zip(d["valid"], d["grazing"])]
        candidate_colours = [geometry_order_colour(m) for m in labels]
        candidate_symbols = ["x" if not valid else ("diamond" if grazing else "circle")
                             for valid, grazing in zip(d["valid"], d["grazing"])]
        direction.add_trace(go.Scatter(
            x=d["a"], y=d["b"], mode="markers+text",
            text=[f"m={m:+d}" for m in labels] if len(labels) <= 13 else [""]*len(labels),
            textposition="bottom center", textfont=dict(size=10),
            marker=dict(size=10, color=candidate_colours, symbol=candidate_symbols),
            customdata=[[int(m), state] for m, state in zip(labels, states)],
            showlegend=False,
            hovertemplate="m=%{customdata[0]} · %{customdata[1]}<br>α=%{x:.4f}<br>β=%{y:.4f}<extra></extra>"
        ), row=1, col=2)
        direction.add_trace(go.Scatter(
            x=[si[0]], y=[si[1]], mode="markers",
            marker=dict(size=16, color=INK, symbol="circle-open", line=dict(width=2)),
            showlegend=False, hovertemplate="Incident direction = zero order<extra></extra>"
        ), row=1, col=2)
        direction.update_layout(scene=dict(
            xaxis=dict(title="α = kx/k", range=[-1.1, 1.1]),
            yaxis=dict(title="β = ky/k", range=[-1.1, 1.1]),
            zaxis=dict(title="γ = kz/k", range=[-0.05, 1.12])))
        direction.update_xaxes(title_text="α = kx/k", range=xlim)
        direction.update_yaxes(title_text="β = ky/k", range=[-1.18, 1.18])

    if 'ewald' in views:
        center = -si
        qx = labels*d["spacing"]
        qxlim = [min(center[0]-1.13, qx.min()-0.16), max(center[0]+1.13, qx.max()+0.16)]
        qzlim = [center[2]-1.12, max(center[2]+1.18, 0.2)]
        circle_x, circle_z = center[0]+rho*np.cos(angles), center[2]+rho*np.sin(angles)
        upper_x, upper_z = center[0]+rho*np.cos(arc), center[2]+rho*np.sin(arc)
        ewald = geometry_base("Ewald sphere: centre C, radius 1",
                              f"Rod plane Δk_y=0 · section radius ρ={rho:.3f}")
        ewald.add_trace(geometry_line3d(geometry_wire(center)), row=1, col=1)
        ewald.add_trace(geometry_line3d([circle_x, np.zeros_like(angles), circle_z],
                                       GRID, 2, "dash"), row=1, col=1)
        ewald.add_trace(geometry_line3d([upper_x, np.zeros_like(arc), upper_z], INK, 4), row=1, col=1)
        rods = {}
        for valid, colour in [(True, "#889baa"), (False, "#889baa")]:
            rods[valid] = geometry_segments([[[x, 0, qzlim[0]], [x, 0, qzlim[1]]]
                                           for x, is_valid in zip(qx, d["valid"]) if is_valid == valid])
            ewald.add_trace(geometry_line3d(rods[valid], colour, 2, "dot"), row=1, col=1)
        ewald.add_trace(go.Scatter3d(
            x=q[:, 0], y=q[:, 1], z=q[:, 2], mode="markers+text", text=texts,
            textposition="top center", textfont=dict(size=11),
            marker=dict(size=5, color=d["colours"], symbol=symbols),
            customdata=[[int(m)] for m in allowed], showlegend=False,
            hovertemplate="m=%{customdata[0]}<br>Δk_x/k=%{x:.4f}<br>Δk_y/k=%{y:.4f}<br>Δk_z/k=%{z:.4f}<extra></extra>"
        ), row=1, col=1)
        ewald.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]], mode="markers+text",
            marker=dict(size=5, color=INK, symbol="square"), text=["C = −ki/k"],
            textposition="bottom center", showlegend=False,
            hovertemplate="Sphere centre C<extra></extra>"
        ), row=1, col=1)
        ewald.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode="markers", marker=dict(size=8, color=INK, symbol="circle-open"),
            showlegend=False, hovertemplate="O: q=0, zero order<extra></extra>"
        ), row=1, col=1)
        chosen_index = int(np.flatnonzero(allowed == d["chosen"])[0])
        for role, start, vector, colour, name, dash in [
            ("incident", center, si, INK, "Incident ki/k", "solid"),
            ("outgoing", center, s[chosen_index], d["colours"][chosen_index],
             f"Outgoing km/k (m={d['chosen']:+d})", "solid"),
            ("change", np.zeros(3), q[chosen_index], "#914e97", "Change Δk/k", "dash"),
        ]:
            end = start+vector
            line = geometry_line3d(geometry_segments([[start, end]]), colour, 5, dash, name)
            line.update(showlegend=True, hoverinfo="name", meta={"role": role + "-vector"})
            ewald.add_trace(line, row=1, col=1)
            nonzero = np.linalg.norm(vector) > 1e-12
            ewald.add_trace(go.Cone(
                x=[end[0]] if nonzero else [], y=[end[1]] if nonzero else [],
                z=[end[2]] if nonzero else [], u=[vector[0]] if nonzero else [],
                v=[vector[1]] if nonzero else [], w=[vector[2]] if nonzero else [],
                anchor="tip", sizemode="absolute", sizeref=0.07,
                colorscale=[[0, colour], [1, colour]], showscale=False,
                hoverinfo="skip", showlegend=False, meta={"role": role + "-head"}
            ), row=1, col=1)
        ewald.add_trace(geometry_line2d(circle_x, circle_z, GRID, 2, "dash"), row=1, col=2)
        ewald.add_trace(geometry_line2d(upper_x, upper_z, INK, 3), row=1, col=2)
        for valid, colour in [(True, "#889baa"), (False, "#889baa")]:
            ewald.add_trace(geometry_line2d(rods[valid][0], rods[valid][2],
                                           colour, 1, "dot"), row=1, col=2)
        ewald.add_trace(go.Scatter(
            x=q[:, 0], y=q[:, 2], mode="markers+text", text=texts,
            textposition="top center", textfont=dict(size=10),
            marker=dict(size=10, color=d["colours"], symbol=symbols),
            customdata=[[int(m)] for m in allowed], showlegend=False,
            hovertemplate="m=%{customdata[0]}<br>Δk_x/k=%{x:.4f}<br>Δk_z/k=%{y:.4f}<extra></extra>"
        ), row=1, col=2)
        ewald.add_trace(go.Scatter(
            x=[center[0]], y=[center[2]], mode="markers",
            marker=dict(size=12, color=INK, symbol="cross"), showlegend=False,
            hovertemplate="Centre of the section, not generally the full sphere centre<extra></extra>"
        ), row=1, col=2)
        ewald.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(size=16, color=INK, symbol="circle-open", line=dict(width=2)),
            showlegend=False, hovertemplate="O: q=0, zero order<extra></extra>"
        ), row=1, col=2)
        ewald.update_layout(scene=dict(
            xaxis=dict(title="Δk_x/k", range=qxlim),
            yaxis=dict(title="Δk_y/k", range=[center[1]-1.12, center[1]+1.12]),
            zaxis=dict(title="Δk_z/k", range=qzlim)))
        ewald.update_xaxes(title_text="Δk_x/k", range=qxlim)
        ewald.update_yaxes(title_text="Δk_z/k", range=qzlim)
    # A persistent selection mark in every panel makes the dropdown visible in either tab.
    if direction is not None:
        selected_ray = geometry_line3d([[], [], []], INK, 6)
        selected_ray.meta = {"role": "selected-ray"}
        direction.add_trace(selected_ray, row=1, col=1)
    for figure, role3d, role2d, axes3d, axes2d in [
        (direction, "selected-direction", "selected-disk", ("α", "β", "γ"), ("α", "β")),
        (ewald, "selected-ewald", "selected-section", ("Δk_x/k", "Δk_y/k", "Δk_z/k"), ("Δk_x/k", "Δk_z/k")),
    ]:
        if figure is None:
            continue
        figure.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode="markers",
            marker=dict(size=10, color="#111827", symbol="square-open",
                        line=dict(width=3)),
            meta={"role": role3d}, customdata=[[0]], showlegend=False,
            hovertemplate=("Highlighted m=%{customdata[0]}<br>"
                           + axes3d[0] + "=%{x:.4f}<br>"
                           + axes3d[1] + "=%{y:.4f}<br>"
                           + axes3d[2] + "=%{z:.4f}<extra></extra>")
        ), row=1, col=1)
        figure.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(size=21, color="#111827", symbol="square-open",
                        line=dict(width=2.5)),
            meta={"role": role2d}, customdata=[[0]], showlegend=False,
            hovertemplate=("Highlighted m=%{customdata[0]}<br>"
                           + axes2d[0] + "=%{x:.4f}<br>"
                           + axes2d[1] + "=%{y:.4f}<extra></extra>")
        ), row=1, col=2)
    update_geometry_highlight(direction, ewald, data)
    figures = dict(direction=direction, ewald=ewald)
    return tuple(figures[view] for view in views)


def update_geometry_highlight(direction, ewald, data):
    """Move the selection and its triangle without rebuilding the physical geometry."""
    chosen = int(data["chosen"])
    index = int(np.flatnonzero(data["allowed"] == chosen)[0])
    s = data["directions"][index]
    q = data["changes"][index]
    colour = geometry_order_colour(chosen)
    direction_roles = {
        trace.meta["role"]: trace for trace in (direction.data if direction is not None else ())
        if isinstance(trace.meta, dict) and "role" in trace.meta}
    ewald_roles = {
        trace.meta["role"]: trace for trace in (ewald.data if ewald is not None else ())
        if isinstance(trace.meta, dict) and "role" in trace.meta}

    for roles, role, point in [
        (direction_roles, "selected-direction", s),
        (direction_roles, "selected-disk", s[:2]),
        (ewald_roles, "selected-ewald", q),
        (ewald_roles, "selected-section", q[[0, 2]]),
    ]:
        if not roles:
            continue
        properties = dict(x=[point[0]], y=[point[1]], customdata=[[chosen]])
        if len(point) == 3:
            properties["z"] = [point[2]]
        roles[role].update(properties)
    if direction_roles:
        ray = geometry_segments([[np.zeros(3), s]])
        direction_roles["selected-ray"].update(
            x=ray[0], y=ray[1], z=ray[2], line=dict(color=colour))
    if not ewald_roles:
        return

    for role, start, vector, line_colour, label in [
        ("outgoing", -data["incident"], s, colour, f"Outgoing km/k (m={chosen:+d})"),
        ("change", np.zeros(3), q, "#914e97", "Change Δk/k"),
    ]:
        end = start + vector
        xyz = geometry_segments([[start, end]])
        ewald_roles[role + "-vector"].update(
            x=xyz[0], y=xyz[1], z=xyz[2], name=label,
            line=dict(color=line_colour))
        nonzero = np.linalg.norm(vector) > 1e-12
        ewald_roles[role + "-head"].update(
            x=[end[0]] if nonzero else [], y=[end[1]] if nonzero else [],
            z=[end[2]] if nonzero else [], u=[vector[0]] if nonzero else [],
            v=[vector[1]] if nonzero else [], w=[vector[2]] if nonzero else [],
            colorscale=[[0, line_colour], [1, line_colour]])


def export_geometry_html(directions_figure, ewald_figure, wavelength, period,
                         theta_deg, phi_deg, selected_order,
                         target=None):
    """Save standalone HTML, defaulting to the supplemental figures directory.

    An explicit relative target remains relative to the current working directory.
    """
    # This browser export keeps rotation, zoom, hover, and the current parameter values.
    # Parameter sliders in the notebook use the live Python kernel.
    target = (Path(__file__).resolve().parents[1] / "supplement" / "figures"
              / "diffraction_geometry_interactive.html"
              if target is None else Path(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    config = dict(responsive=True, scrollZoom=True, displaylogo=False)
    directions = go.Figure(directions_figure).to_html(
        full_html=False, include_plotlyjs=True, config=config, div_id="direction-geometry")
    ewald = go.Figure(ewald_figure).to_html(
        full_html=False, include_plotlyjs=False, config=config, div_id="ewald-geometry")
    params = (f"λ = {wavelength:g} nm · d = {period:g} nm · "
              f"θi = {theta_deg:g}° · φi = {phi_deg:g}° · "
              f"highlighted m = {selected_order:+d}")
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Interactive diffraction geometry</title>"
        "<style>body{font:16px system-ui,sans-serif;color:#263442;max-width:1250px;"
        "margin:25px auto;padding:0 20px}h1{font-size:25px}p{line-height:1.5}</style>"
        "</head><body><h1>Harvey and Ewald: diffraction geometry</h1><p>"
        + html.escape(params) + "</p><p>Drag to rotate the spheres; scroll to zoom; "
        "hover over an order to read its coordinates. Large squares mark the selected order. "
        "Opposite orders (+m and -m) share a colour; zero has its own colour. "
        "Use the notebook sliders to change the physical parameters.</p>"
        + directions
        + "<p>The Ewald sphere has radius 1 and centre −ki/k. "
        "The rod-plane section has radius ρ = √(1−βi²). "
        "The filled upper-arc intersections are transmitted orders; endpoints are grazing.</p>"
        + ewald + "</body></html>")
    target.write_text(page, encoding="utf-8")
    return target


def vector_text(vector):
    """Format coordinates at the precision used in the teaching notebook."""
    return "(" + ", ".join(f"{0.0 if abs(value) < 0.00005 else value:+.4f}"
                             for value in vector) + ")"


def diffraction_order_table(data):
    """Return a Markdown table including real and evanescent candidate orders."""
    rows = [
        "| Order m | Disk position (alpha, beta) | Direction (alpha, beta, gamma) | q / k | State |",
        "| ---: | --- | --- | --- | --- |",
    ]
    real_index = 0
    for index, order in enumerate(data["labels"]):
        if not data["valid"][index]:
            state, direction, change = "evanescent", "no real direction", "—"
        else:
            state = "grazing" if data["grazing"][index] else "propagating"
            direction = vector_text(data["directions"][real_index])
            change = vector_text(data["changes"][real_index])
            real_index += 1
        disk = vector_text([data["a"][index], data["b"][index]])
        rows.append(f"| {int(order)} | {disk} | {direction} | {change} | {state} |")
    return (f"Incident unit vector: **{vector_text(data['incident'])}**; "
            f"order spacing in the disk: **{data['spacing']:.4f}**.\n\n" + "\n".join(rows))


def diffraction_residual_report(data):
    """Format the numerical checks of the two sphere equations."""
    residuals = sphere_residuals(data)
    if residuals is None:
        return "No real directions among the requested order labels."
    harvey, ewald = residuals
    return (f"Maximum direction-sphere residual: {harvey:.2e}\n"
            f"Maximum normalized Ewald-sphere residual: {ewald:.2e}")
