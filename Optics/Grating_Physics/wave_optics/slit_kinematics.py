"""Candidate propagating directions for one isolated slit, without strengths.

The slit is uniform along y and illuminated with ky=0. Angles are measured
from -z toward +x in a homogeneous medium; all images stay in memory.
"""

import numpy as np


def slit_direction(wavelength=2., angle_deg=30.):
    """One member of the continuous outgoing semicircle (including grazing)."""
    if (not np.all(np.isfinite([wavelength, angle_deg])) or wavelength <= 0
            or abs(angle_deg) > 90):
        raise ValueError('Use a positive wavelength and an angle from -90 to 90 degrees.')
    angle = np.deg2rad(angle_deg)
    k = 2*np.pi/wavelength
    return dict(wavelength=float(wavelength), angle_deg=float(angle_deg), k=k,
                kx=k*np.sin(angle), kz=-k*np.cos(angle))


def draw_slit_kinematics(figure, data):
    """Real-space direction sketch beside the continuous set of wavevectors."""
    blue, orange, green, grey, ink = '#2866a3', '#b85c17', '#23734d', '#c8ced5', '#243444'
    figure.clear()
    space, vectors = figure.subplots(1, 2)
    figure.subplots_adjust(left=.06, right=.95, top=.81, bottom=.22, wspace=.33)
    figure.suptitle('One isolated slit: a continuous choice of candidate directions', fontsize=14)
    figure.text(.5, .89, 'Move the candidate angle. No grating period or order number is needed.',
                ha='center', fontsize=10, color=ink)

    def arrow(axis, end, color, start=(0, 0), width=1.7):
        axis.annotate('', xy=end, xytext=start,
                      arrowprops=dict(arrowstyle='->', color=color, lw=width))

    # These are schematic directions, not the emitted field or power rays.
    for height in (.32, .56, .80):
        space.plot([-.7, .7], [height, height], color=blue, lw=1.4, alpha=.65)
    arrow(space, (0, .07), blue, (0, .96))
    space.text(.78, .61, 'Incident\nplane wave\n' + r'$\mathbf{k}_i$',
               fontsize=9, color=blue, va='center')
    for edges in ((-1.35, -.16), (.16, 1.35)):
        space.plot(edges, [0, 0], color=ink, lw=6, solid_capstyle='butt')
    space.text(-1.3, .10, 'Opaque screen', fontsize=9, color=ink)
    space.text(.23, -.13, 'One opening', fontsize=9, color=ink)
    sample_angles = np.deg2rad(np.arange(-75, 76, 15))
    for angle in sample_angles:
        arrow(space, (1.15*np.sin(angle), -1.15*np.cos(angle)), grey, width=1.)
    sx, sz = data['kx']/data['k'], data['kz']/data['k']
    arrow(space, (1.15*sx, 1.15*sz), orange, width=2.8)
    space.text(0, -1.39, f'Selected candidate θ = {data["angle_deg"]:+.1f}°',
               ha='center', fontsize=10, color=orange)
    space.set(xlim=(-1.45, 1.45), ylim=(-1.52, 1.02), aspect='equal',
              title='Direction sketch in physical space')
    space.title.set_fontsize(11)
    space.axis('off')

    k, kx, kz = data['k'], data['kx'], data['kz']
    angles = np.linspace(-np.pi/2, np.pi/2, 501)
    vectors.axhline(0, color=grey, lw=1)
    vectors.axvline(0, color=grey, lw=1)
    vectors.plot(k*np.sin(angles), -k*np.cos(angles), color=blue, lw=2)
    vectors.scatter(k*np.sin(sample_angles), -k*np.cos(sample_angles),
                    s=13, color=grey, zorder=3)
    vectors.plot([kx, kx], [0, kz], ls='--', color=green, lw=1.2)
    arrow(vectors, (kx, 0), green, width=2.3)
    arrow(vectors, (kx, kz), orange, width=2.8)
    vectors.scatter([kx], [kz], s=35, color=orange, zorder=4)
    vectors.text(kx/2, .09*k, 'kₓ' if abs(kx) > 1e-12 else 'kₓ = 0',
                 color=green, ha='center', fontsize=11)
    vectors.text(.52*kx+.06*k, .52*kz-.06*k, r'$\mathbf{k}(\theta)$',
                 color=orange, fontsize=12)
    vectors.set(xlim=(-1.23*k, 1.23*k), ylim=(-1.18*k, .38*k), aspect='equal',
                xlabel='kₓ (rad/µm)', ylabel='k_z (rad/µm)',
                title='Wavevector tips fill the entire blue arc')
    vectors.title.set_fontsize(11)
    vectors.spines[['top', 'right']].set_visible(False)
    figure.text(.735, .15, r'Every candidate has $|\mathbf{k}(\theta)|=k=2\pi/\lambda$'
                + f' = {k:.3f} rad/µm',
                ha='center', fontsize=10, color=ink)
    figure.text(.5, .09, 'Grey arrows and dots sample a continuum; their number and lengths do not represent power.',
                ha='center', fontsize=9, color=ink)
    figure.text(.5, .035, 'The slit field combines plane-wave components. This view selects one direction without assigning its amplitude.',
                ha='center', fontsize=9, color=ink)


class SlitKinematicsExplorer:
    """Independent direction selector; it has no period, order or intensity."""

    def __init__(self):
        import ipywidgets as widgets
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        self._closed = False
        style = {'description_width': 'initial'}
        self.wavelength = widgets.FloatSlider(value=2, min=1, max=4, step=.1,
            description='λ in medium (µm)', continuous_update=False, style=style,
            layout=widgets.Layout(width='380px'))
        self.angle = widgets.FloatSlider(value=30, min=-90, max=90, step=.1,
            description='Candidate θ from −z (°)', continuous_update=False, style=style,
            layout=widgets.Layout(width='460px'))
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1250px', height='auto'))
        controls = widgets.Box([self.wavelength, self.angle], layout=widgets.Layout(
            display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([controls, self.readout, self.picture])
        self.figure = Figure(figsize=(12, 5.7), dpi=105, facecolor='white')
        FigureCanvasAgg(self.figure)
        for control in (self.wavelength, self.angle):
            control.observe(self._update, names='value')
        self._update()

    def _update(self, change=None):
        if self._closed:
            return
        from .plotting import figure_png
        self.data = slit_direction(self.wavelength.value, self.angle.value)
        data = self.data
        draw_slit_kinematics(self.figure, data)
        self.picture.value = figure_png(self.figure)
        grazing = (' This endpoint is grazing, parallel to the screen.'
                   if abs(data['angle_deg']) == 90 else '')
        self.readout.value = (
            '<div style="padding:8px 0;line-height:1.7">'
            f'<b>Candidate θ = {data["angle_deg"]:+.1f}°.</b> '
            f'k = {data["k"]:.3f}; kₓ = {data["kx"]:+.3f}; k_z = {data["kz"]:+.3f} rad/µm.<br>'
            'kₓ = k sin θ; k_z = −k cos θ. The same wavelength permits every angle along the arc.'
            f'{grazing} These controls are independent of the grating examples.</div>')

    def close(self):
        if self._closed:
            return
        from .grating_intro import _close_widget_tree
        self._closed = True
        for control in (self.wavelength, self.angle):
            control.unobserve(self._update, names='value')
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        _close_widget_tree(self.ui)
