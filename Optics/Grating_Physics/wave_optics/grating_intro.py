"""Grating teaching views; no field propagation or efficiency model.

Lengths are in micrometres, with wavelength measured in the medium. The
grating repeats along x, is uniform along y, and lies at z=0. Signed incidence
angles go from -z toward +x: incidence is from z>0, transmission toward z<0.
All field views are snapshots at t=0, A=1.
Figures and SVG/PNG representations are created in memory only.
"""

import io

import ipywidgets as widgets
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .diffraction import diffraction_orders
from .plotting import figure_png
from .waves import wavevector


def _uniform_field_image(axis, x, z, values, **kwargs):
    """Draw an evenly sampled field with the same cell edges as pcolormesh.

    Coordinates denote sample centres and may run in either direction.
    Callers retain their existing physical limits and colour normalization.
    """
    dx, dz = x[1] - x[0], z[1] - z[0]
    kwargs.setdefault('aspect', axis.get_aspect())
    return axis.imshow(values, origin='lower', interpolation='nearest',
        extent=(x[0]-dx/2, x[-1]+dx/2, z[0]-dz/2, z[-1]+dz/2),
        zorder=1, **kwargs)


def amplitude_grating(x, period, opening_width):
    """Ideal amplitude transmission, with openings centred on integer periods.

    Values at an opening's exact edge are set to zero. Zero and full opening
    widths give a uniformly closed and uniformly open mask, respectively.
    This is a prescribed thin mask, not a solved electromagnetic boundary.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite([period, opening_width])) or not np.all(np.isfinite(x)):
        raise ValueError('Coordinates and grating dimensions must be finite.')
    if period <= 0 or not 0 <= opening_width <= period:
        raise ValueError('Require period > 0 and 0 <= opening_width <= period.')
    if opening_width == period:
        return np.ones_like(x)
    local_x = (x + period / 2) % period - period / 2
    return (np.abs(local_x) < opening_width / 2).astype(float)


def _figure(*, figsize):
    figure = Figure(figsize=figsize, dpi=115, facecolor='white')
    FigureCanvasAgg(figure)
    return figure


def _svg_and_release(figure):
    try:
        buffer = io.StringIO()
        figure.savefig(buffer, format='svg', facecolor='white')
        return buffer.getvalue()
    finally:
        figure.clear()


def grating_geometry_figure():
    """Original plan and side views; return an unmanaged Matplotlib figure."""
    figure = _figure(figsize=(10, 4))
    plan, side = figure.subplots(1, 2)
    figure.subplots_adjust(left=.035, right=.98, bottom=.10, top=.86, wspace=.24)
    plan.set_title('Looking at the surface: x–y plane', fontsize=12, pad=15)
    for centre in (-2, 0, 2):
        plan.add_patch(Rectangle((centre - 1, -.9), 2, 1.8,
                                 facecolor='#b7bdc5', edgecolor='none'))
        plan.add_patch(Rectangle((centre - .5, -.9), 1, 1.8,
                                 facecolor='#dff4fc', edgecolor='#3286a7'))
    # Measure the period between corresponding left edges of adjacent openings.
    for start, end, height, label in ((-.5, 1.5, 1.15, 'period d'),
                                       (-.5, .5, -1.2, r'opening width $W_{\mathrm{open}}$')):
        if label == 'period d':
            plan.vlines([start, end], .9, height + .07,
                        color='#72808f', linewidth=.8)
        plan.annotate('', (end, height), (start, height),
                      arrowprops=dict(arrowstyle='<->', color='#172b4d',
                                      shrinkA=0, shrinkB=0))
        plan.text((start + end) / 2, height + (.10 if height > 0 else -.25),
                  label, ha='center', fontsize=10)
    plan.annotate('+y: along the slits / grooves', xy=(-2, .8), xytext=(-2, -1.85),
                  ha='center', fontsize=10,
                  arrowprops=dict(arrowstyle='->', color='#172b4d'))
    plan.annotate('+x: repetition', xy=(3.3, -.05), xytext=(1.4, -.05),
                  ha='center', fontsize=10,
                  arrowprops=dict(arrowstyle='->', color='#172b4d'))
    plan.text(-2.9, 1.4, '⊙ +z out of page', fontsize=10)
    plan.text(.9, -1.85, 'Blue = opening\nGrey = blocked', ha='center', fontsize=10)
    plan.set(xlim=(-3.5, 3.6), ylim=(-2.15, 1.85))
    plan.axis('off')

    side.set_title('Side view: x–z plane', fontsize=12, pad=15)
    for centre in (-2, 0, 2):
        side.plot([centre - 1, centre - .5], [0, 0], color='#56616d', lw=7,
                  solid_capstyle='butt')
        side.plot([centre + .5, centre + 1], [0, 0], color='#56616d', lw=7,
                  solid_capstyle='butt')
        side.plot([centre - .5, centre + .5], [0, 0], color='#3286a7', lw=2)
    side.annotate('', (3.4, 0), (-3.2, 0), arrowprops=dict(arrowstyle='->'))
    side.annotate('', (0, 1.65), (0, -.7), arrowprops=dict(arrowstyle='->'))
    side.text(3.15, -.35, '+x', fontsize=11)
    side.text(.18, 1.5, '+z: surface normal', fontsize=10)
    side.text(-3.05, -.4, 'Grating at z = 0', fontsize=10)
    side.text(.4, .7, 'Incident side\nz > 0', fontsize=10)
    side.text(.4, -1.15, 'Transmission side\nz < 0', fontsize=10)
    side.annotate('', (-1.5, .08), (-1.5, 1.25),
                  arrowprops=dict(arrowstyle='->', lw=2, color='#14758c'))
    side.text(-1.65, .65, r'incident $\mathbf{k}_i$', ha='right', fontsize=10)
    side.text(-3.05, -1.8, '⊗ +y into page', fontsize=10)
    side.set(xlim=(-3.5, 3.6), ylim=(-2.15, 1.85))
    side.axis('off')
    return figure


def grating_geometry_svg():
    """Render the coordinate diagram to an in-memory SVG and release it."""
    return _svg_and_release(grating_geometry_figure())


def order_preview_figure():
    """Equal-length direction markers from the existing order calculation.

    No order amplitudes are computed or implied. At the chosen wavelength and
    period the five indicated directions are all the geometrically allowed
    transmitted directions, not necessarily five nonzero outgoing waves.
    """
    data = diffraction_orders(wavelength=1.0, period=2.5, theta_deg=0, phi_deg=0)
    # Mirror the shared +z solution across z=0 for this notebook's -z
    # transmission. Tangential components and order labels stay the same.
    directions = data['directions'].copy()
    directions[:, 2] *= -1
    figure = _figure(figsize=(7.5, 3.9))
    axis = figure.subplots()
    figure.subplots_adjust(left=.06, right=.94, top=.87, bottom=.13)
    axis.axhline(0, color='#56616d', lw=4)
    axis.plot([0, 0], [0, -1.26], '--', color='#b7bdc5', lw=1)
    for order, direction in zip(data['allowed'], directions):
        dx, _, dz = direction
        axis.annotate('', (dx, dz), (0, 0),
                      arrowprops=dict(arrowstyle='->', lw=2, color='#14758c',
                                      shrinkA=0, shrinkB=0))
        label = 'm = 0' if order == 0 else f'm = {order:+d}'
        axis.text(1.13 * dx, 1.13 * dz, label, ha='center', va='center', fontsize=11)
    axis.annotate('', (0, .05), (0, .48),
                  arrowprops=dict(arrowstyle='->', lw=2, color='#333333'))
    axis.text(.06, .32, r'incident $\mathbf{k}_i$', fontsize=10, va='center')
    axis.text(-1.08, .08, 'grating', fontsize=10)
    axis.text(1.02, .08, '+x', fontsize=10)
    axis.text(.05, -1.24, '−z', fontsize=10)
    axis.set_title('Allowed directions only — arrow lengths do not show power', fontsize=11)
    axis.set(xlim=(-1.25, 1.25), ylim=(-1.36, .52), aspect='equal')
    axis.axis('off')
    return figure


def order_preview_svg():
    """Render the order preview in memory, without retaining its figure."""
    return _svg_and_release(order_preview_figure())


def order_component_figure():
    """Return a standalone rendering of the default m=+1 component example."""
    wavelength, period, radius = 1.0, 2.5, 2.0
    alpha = wavelength / period  # normal incidence, m=+1
    direction = np.array([alpha, 0.0, -np.sqrt(1 - alpha**2)])
    figure = _figure(figsize=(14, 6.2))
    _draw_order_components(figure, direction, wavelength, period, 0, radius, 1,
                           radius_limit=4, k_limit=2 * np.pi)
    return figure


def _draw_order_components(figure, direction, wavelength, period, angle, radius,
                           order, *, radius_limit, k_limit):
    """Draw one real direction from the explorer, including zero and grazing.

    Fixed limits within each panel make changes to R, k and 1/lambda visible. The
    panels use different physical units and need not use the same scale.
    """
    k = 2 * np.pi / wavelength
    theta = np.arctan2(direction[0], -direction[2])
    side = -1 if direction[0] < 0 else 1
    grazing = direction[2] == 0
    figure.clear()
    spatial, vector, frequency = figure.subplots(1, 3)
    figure.subplots_adjust(left=.025, right=.985, bottom=.28, top=.79, wspace=.24)
    label = '0' if order == 0 else f'{order:+d}'
    figure.suptitle(f'One direction, three coordinate scales: m = {label}'
                   + (' (grazing)' if grazing else ''), fontsize=13)
    figure.text(.5, .89, f'λ = {wavelength:g} µm; d = {period:g} µm; '
                f'θᵢ = {angle:g}° from −z; R = {radius:g} µm',
                ha='center', fontsize=10)
    for axis, scale, limit, title, horizontal, vertical, diagonal in (
            (spatial, radius, radius_limit, 'Position on the reference sphere\n(µm)',
             r'$x_{\rm sphere}$', r'$z_{\rm sphere}$', r'$R$'),
            (vector, k, k_limit, r'Order wavevector $\mathbf{k}_m$' + '\n(rad/µm)',
             r'$k_{mx}$', r'$k_{mz}$', r'$\mathbf{k}_m$'),
            (frequency, 1 / wavelength, k_limit / (2 * np.pi),
             'Spatial-frequency components\n(cycles/µm)',
             r'$f_{x,m}$', r'$f_{z,m}$', r'$1/\lambda$')):
        xx, zz = scale * direction[[0, 2]]
        arc = np.linspace(-np.pi / 2, np.pi / 2, 241)
        axis.plot(scale * np.sin(arc), -scale * np.cos(arc),
                  color='#b8c7d8', lw=1.3)
        axis.plot([-1.07 * limit, 1.07 * limit], [0, 0], color='#aab2bb', lw=.8)
        axis.plot([0, 0], [0, -1.08 * limit], '--', color='#aab2bb', lw=.8)
        axis.plot([xx, xx, 0], [0, zz, zz], '--', color='#7297b4', lw=1)
        for endpoint, color, width in (((xx, 0), '#1496bd', 2),
                                       ((0, zz), '#1496bd', 2),
                                       ((xx, zz), '#14627d', 2.8)):
            if np.linalg.norm(endpoint) > 1e-12 * limit:
                axis.annotate('', endpoint, (0, 0),
                              arrowprops=dict(arrowstyle='->', color=color, lw=width,
                                              shrinkA=0, shrinkB=0))
        axis.scatter([xx], [zz], s=42, color='#2866a3', zorder=4)
        if abs(theta) > 1e-12:
            angle_arc = np.linspace(0, theta, 41)
            axis.plot(.34 * scale * np.sin(angle_arc),
                      -.34 * scale * np.cos(angle_arc), color='#b85c17', lw=1.6)
            axis.annotate(r'$\theta_m$',
                (.42 * scale * np.sin(theta / 2), -.42 * scale * np.cos(theta / 2)),
                xytext=(side * 3, -9), textcoords='offset points',
                ha='center', color='#b85c17', fontsize=11)
        zero_x = abs(direction[0]) < 1e-12
        axis.annotate(horizontal + (' = 0' if zero_x else ''), (xx / 2, 0),
                      xytext=(10 if zero_x else 0, 10), textcoords='offset points',
                      ha='left' if zero_x else 'center', fontsize=11)
        axis.annotate(vertical + (' = 0' if grazing else ' < 0'), (0, zz / 2),
                      xytext=(-side * 10, -22 if grazing else 0),
                      textcoords='offset points', ha='right' if side > 0 else 'left',
                      va='center', fontsize=11)
        axis.annotate(diagonal, (.68 * xx, .68 * zz),
                      xytext=(side * 10, -20 if grazing else -2),
                      textcoords='offset points', ha='left' if side > 0 else 'right',
                      color='#14627d', fontsize=12)
        axis.text(.025 * limit, -1.08 * limit, '−z', va='top', fontsize=10)
        axis.text(1.07 * limit, .025 * limit, '+x', ha='center', fontsize=10)
        axis.set(xlim=(-1.18 * limit, 1.18 * limit),
                 ylim=(-1.2 * limit, .22 * limit), aspect='equal')
        axis.set_title(title, fontsize=11, pad=15)
        axis.axis('off')
    sphere = radius * direction
    k_vector = k * direction
    frequency_vector = direction / wavelength
    for axis, formulas, values in (
            (spatial, r'$x_{\rm sphere}=R\sin\theta_m$' + '\n'
             + r'$z_{\rm sphere}=-R\cos\theta_m$',
             f'(x, z) = ({sphere[0]:.3f}, {sphere[2]:.3f}) µm'),
            (vector, r'$k_{mx}=k\sin\theta_m$' + '\n'
             + r'$k_{mz}=-k\cos\theta_m$',
             r'$(k_{mx},\,k_{mz})$' + f' = ({k_vector[0]:.3f}, {k_vector[2]:.3f}) rad/µm'),
            (frequency, r'$f_{x,m}=\sin\theta_m/\lambda$' + '\n'
             + r'$f_{z,m}=-\cos\theta_m/\lambda$',
             r'$(f_{x,m},\,f_{z,m})$' + f' = ({frequency_vector[0]:.3f}, '
             f'{frequency_vector[2]:.3f}) cycles/µm')):
        position = axis.get_position()
        centre = (position.x0 + position.x1) / 2
        figure.text(centre, .18, formulas, ha='center', fontsize=11)
        figure.text(centre, .125, values, ha='center', fontsize=10)
    figure.text(.5, .035, f'Same angle: θₘ = {np.rad2deg(theta):.2f}°; '
                f'same normalized horizontal component: αₘ = {direction[0]:.3f}.\n'
                f'Diagonal lengths: R = {radius:g} µm; '
                + r'$|\mathbf{k}_m|=k$' + f' = {k:.3f} rad/µm; '
                f'1/λ = {1 / wavelength:.3f} cycles/µm.',
                ha='center', fontsize=10)


def order_component_svg():
    """Render the component triangles in memory and release the figure."""
    return _svg_and_release(order_component_figure())


def order_surface_intersections(k_vectors, radius, distance):
    """Intersect representative rays from the origin with two surfaces.

    Rows of k_vectors are real, nonzero wavevectors directed toward z<=0.
    The sphere has radius R=radius; the plane is at z=-distance. All lengths
    share one unit. Grazing rays meet the sphere but have no plane point,
    represented by a NaN row and a false reaches_plane entry. No field,
    beam footprint, or power is calculated.
    """
    if np.iscomplexobj(k_vectors):
        raise ValueError('Use real wavevectors for propagating or grazing orders.')
    vectors = np.asarray(k_vectors, dtype=float)
    lengths = np.asarray([radius, distance], dtype=float)
    if (vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) == 0
            or not np.all(np.isfinite(vectors))):
        raise ValueError('Wavevectors must be a nonempty finite array of shape (N, 3).')
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError('Sphere radius and plane distance must be positive and finite.')
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ValueError('Each wavevector must have a finite, nonzero magnitude.')
    directions = vectors / norms[:, None]
    if np.any(directions[:, 2] > 0):
        raise ValueError('Use outgoing wavevectors directed toward z <= 0.')
    sphere = float(radius) * directions
    reaches_plane = directions[:, 2] < 0
    plane = np.full_like(directions, np.nan)
    ray_lengths = -float(distance) / directions[reaches_plane, 2]
    plane[reaches_plane] = ray_lengths[:, None] * directions[reaches_plane]
    return dict(directions=directions, sphere=sphere, plane=plane,
                reaches_plane=reaches_plane)


def _close_widget_tree(root):
    """Release one owned widget tree, including its layouts and styles."""
    closed = set()

    def close_tree(widget):
        if not isinstance(widget, widgets.Widget) or id(widget) in closed:
            return
        closed.add(id(widget))
        for child in getattr(widget, 'children', ()):
            close_tree(child)
        close_tree(getattr(widget, 'layout', None))
        close_tree(getattr(widget, 'style', None))
        widget.close()

    close_tree(root)


class OrderSurfaceExplorer:
    """In-plane diffraction directions on a sphere and a reference plane.

    Wavelength, period and signed incidence determine the allowed orders.
    Radius and plane distance change their geometrical intersections. The
    x-z view and its horizontal projection ruler use fixed, matching
    x scales. The view reports off-screen plane points.
    """

    def __init__(self):
        self._closed = False
        self._suspend_updates = False
        self._reset_component_selection = False
        self._surface_signature = None
        self._component_signature = None

        def slider(value, minimum, maximum, step, label):
            return widgets.FloatSlider(value=value, min=minimum, max=maximum,
                step=step, description=label, readout_format='.2f',
                continuous_update=False, style={'description_width': 'initial'},
                layout=widgets.Layout(width='310px'))

        self.wavelength = slider(1, 1, 4, .1, 'λ in medium (µm)')
        self.period = slider(2.5, .5, 4, .1, 'Period d (µm)')
        self.angle = slider(0, -60, 60, 5, 'θᵢ from −z (°)')
        self.angle.readout_format = '.0f'
        self.radius = slider(2, 1, 4, .25, 'Sphere radius R (µm)')
        self.distance = slider(3, 1, 6, .25, 'Plane distance L (µm)')
        self._controls = (self.wavelength, self.period, self.angle, self.radius, self.distance)
        self._defaults = tuple(control.value for control in self._controls)
        self.reset = widgets.Button(description='Reset example',
                                    layout=widgets.Layout(width='150px'))
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1100px', height='auto'))
        physics = widgets.Box(list(self._controls[:3]), layout=widgets.Layout(
            display='flex', flex_flow='row wrap', align_items='center'))
        surfaces = widgets.Box([self.radius, self.distance, self.reset], layout=widgets.Layout(
            display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([physics, surfaces, self.readout, self.picture])
        self.figure = _figure(figsize=(10, 8.4))
        # A second display location shares this explorer's physical controls.
        # The explorer owns both widget trees and both unmanaged figures.
        self.component_order = widgets.Dropdown(options=[('m = +1', 1)], value=1,
            description='Inspect order', style={'description_width': 'initial'},
            layout=widgets.Layout(width='270px'))
        self.component_readout = widgets.HTML()
        self.component_picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1400px', height='auto'))
        self.components_ui = widgets.VBox([
            self.component_order, self.component_readout, self.component_picture])
        self.component_figure = _figure(figsize=(14, 6.2))
        self.component_order.observe(self._update, names='value')
        for control in self._controls:
            control.observe(self._update, names='value')
        self.reset.on_click(self._reset_example)
        self._update()

    def _reset_example(self, button):
        if self._closed:
            return
        self._suspend_updates = True
        try:
            for control, value in zip(self._controls, self._defaults):
                control.value = value
        finally:
            self._suspend_updates = False
        self._reset_component_selection = True
        self._update()

    def _update(self, change=None):
        if self._closed or self._suspend_updates:
            return
        wavelength, period, angle, radius, distance = (c.value for c in self._controls)
        # The shared helper uses a +z polar angle. Azimuth 180 gives a negative
        # x tilt; reflecting z then gives this notebook's incident convention.
        orders = diffraction_orders(wavelength, period, abs(angle),
                                    180.0 if angle < 0 else 0.0)
        self.orders = orders['allowed'].copy()
        directions = orders['directions'].copy()
        directions[:, 1] = 0.0  # exactly in-plane, including sin(pi) roundoff
        directions[:, 2] *= -1
        grazing = orders['grazing'][orders['valid']]
        directions[grazing, 2] = 0.0  # respect the helper's grazing tolerance
        self.data = order_surface_intersections(directions, radius, distance)
        self.directions = self.data['directions']
        sphere, plane = self.data['sphere'], self.data['plane']
        reaches_plane = self.data['reaches_plane']
        previous = self.component_order.value
        preferred = 1 if self._reset_component_selection else previous
        self._reset_component_selection = False
        selected = int(preferred) if preferred in self.orders else 0
        selection_note = ''
        if preferred not in self.orders:
            selection_note = (f'Order m = {preferred:+d} no longer has a real direction; '
                              'showing m = 0. ')
        options = tuple(((f'm = {int(order):+d}' if order else 'm = 0')
                         + (' (grazing)' if at_grazing else ''),
                         int(order)) for order, at_grazing in zip(self.orders, grazing))
        # Changing options can briefly select their first entry. Ignore those
        # intermediate notifications and render the final selection once.
        self._suspend_updates = True
        try:
            self.component_order.options = options
            self.component_order.value = selected
        finally:
            self._suspend_updates = False
        x_limit, z_bottom, z_top = 9.0, -7.1, 1.8
        visible = reaches_plane & (np.abs(plane[:, 0]) <= x_limit)
        self.data['visible_plane'] = visible
        sphere_color, plane_color, ray_color = '#2866a3', '#ac560c', '#65727e'
        surface_signature = (wavelength, period, angle, radius, distance)
        surface_changed = surface_signature != self._surface_signature
        if surface_changed:
            self.figure.clear()
            # Reserve a separate strip for a coordinate ruler, not a third surface.
            # Its x scale must line up with the equal-aspect spatial drawing.
            axis = self.figure.add_axes([.085, .40, .885, .52])
            ruler = self.figure.add_axes([.085, .135, .885, .08], sharex=axis)

            arc = np.linspace(-np.pi / 2, np.pi / 2, 241)
            axis.plot(radius * np.sin(arc), -radius * np.cos(arc),
                      color=sphere_color, linewidth=1.5)
            axis.vlines(sphere[:, 0], z_bottom, sphere[:, 2],
                        color='#bdcddd', linewidth=.8, linestyles='--', zorder=0)
            axis.axhline(-distance, color=plane_color, linewidth=1.5)
            axis.axhline(0, color='#d9dee3', linewidth=.8, zorder=0)
            axis.plot([-.65, .65], [0, 0], color='#303943', linewidth=4)
            axis.text(-x_limit + .2, .15, 'grating: z = 0', fontsize=10)
            incident = wavevector(wavelength, angle)
            incident[2] *= -1
            tail = -1.2 * incident[[0, 2]] / np.linalg.norm(incident)
            axis.annotate('', (0, 0), tail,
                          arrowprops=dict(arrowstyle='->', color='#303943', lw=1.5))
            axis.text(tail[0] + .12, tail[1] + .16, r'incident $\mathbf{k}_i$', fontsize=9)

            for index, (label, direction, on_sphere) in enumerate(zip(
                    self.orders, self.directions, sphere)):
                plane_length = -distance / direction[2] if reaches_plane[index] else 0
                # When the plane is inside the sphere, extend each ray to both
                # surfaces. Clip very long near-grazing rays to the fixed view.
                length = max(radius * (1.1 if grazing[index] else 1), plane_length)
                if direction[0] != 0:
                    length = min(length, x_limit / abs(direction[0]))
                if direction[2] < 0:
                    length = min(length, z_bottom / direction[2])
                end = length * direction
                axis.plot([0, end[0]], [0, end[2]], color=ray_color, linewidth=1.1)
                order_text = '0' if label == 0 else f'{label:+d}'
                axis.annotate(order_text, (on_sphere[0], on_sphere[2]),
                              xytext=(0, 10 + 13 * (index % 2)), textcoords='offset points',
                              ha='center', fontsize=9, color=sphere_color)
                if visible[index]:
                    point = plane[index]
                    align = 'right' if point[0] > x_limit - .5 else (
                        'left' if point[0] < -x_limit + .5 else 'center')
                    axis.annotate(order_text, (point[0], point[2]),
                                  xytext=(0, -18 - 13 * (index % 2)),
                                  textcoords='offset points', ha=align, fontsize=9,
                                  color=plane_color)
                elif reaches_plane[index]:
                    axis.annotate('', (end[0], end[2]), (.93 * end[0], .93 * end[2]),
                                  arrowprops=dict(arrowstyle='->', color=ray_color, lw=1.1))

            axis.scatter(sphere[:, 0], sphere[:, 2], s=36, marker='o', color=sphere_color,
                         zorder=4, label='Sphere marker')
            axis.scatter(plane[visible, 0], plane[visible, 2], s=36, marker='s',
                         facecolors='none', edgecolors=plane_color, linewidths=1.4,
                         zorder=5, label='Plane marker')
            axis.set(xlim=(-x_limit, x_limit), ylim=(z_bottom, z_top), aspect='equal',
                     xlabel='x (µm)', ylabel='z (µm)',
                     title='The same orders on two surfaces — x–z view (y = 0)')
            axis.grid(alpha=.12)
            axis.legend(loc='upper center', bbox_to_anchor=(.5, -.16),
                        ncol=2, frameon=False, fontsize=10)
            # Applying the aspect constraint first keeps the ruler aligned even
            # if the figure dimensions are changed by a caller.
            axis.apply_aspect()
            position = axis.get_position()
            ruler.set_position([position.x0, .135, position.width, .08])
            ruler.axhline(0, color='#bdcddd', linewidth=1)
            ruler.scatter(sphere[:, 0], np.zeros(len(sphere)), s=36,
                          color=sphere_color, zorder=3)
            for index, (label, point) in enumerate(zip(self.orders, sphere)):
                ruler.annotate('0' if label == 0 else f'{label:+d}', (point[0], 0),
                               xytext=(0, 8 + 10 * (index % 2)), textcoords='offset points',
                               ha='center', fontsize=9, color=sphere_color)
            ruler.set(ylim=(-.5, 1.2), yticks=[],
                      xlabel=r'Sphere horizontal coordinate $x_{\rm sphere}$ (µm)')
            ruler.set_title('Projection ruler: the same blue points, horizontal positions only',
                            fontsize=10, pad=14)
            for spine in ruler.spines.values():
                spine.set_visible(False)
            self.figure.text(.5, .025,
                'Labels give order m. Circles and squares identify surfaces; no power is encoded.',
                ha='center', fontsize=10)

        def labels(values):
            return ', '.join('0' if value == 0 else f'{value:+d}' for value in values)

        messages = [
            f'<b>{len(self.orders)} orders have real directions</b>: '
            f'{np.count_nonzero(reaches_plane)} travel into z&lt;0; '
            f'{np.count_nonzero(grazing)} are grazing.',
            f'Order labels (including grazing): {labels(self.orders)}.',
            f'λ/d = {wavelength / period:.3f}; θᵢ = {angle:g}°; '
            f'R = {radius:.2f} µm; plane z = −L = −{distance:.2f} µm.',
        ]
        if len(self.orders) > 1:
            messages.append('Projected sphere points: horizontal step '
                            f'R λ/d = {radius * wavelength / period:.3f} µm.')
        else:
            messages.append('One allowed order: no neighbouring pair to compare on the ruler.')
        if np.any(grazing):
            messages.append(f'Grazing orders {labels(self.orders[grazing])}: '
                            'sphere markers only; these rays do not meet the plane.')
        offscreen = reaches_plane & ~visible
        if np.any(offscreen):
            messages.append(f'Plane intersections outside the shown x range (±{x_limit:g} µm): '
                            f'm = {labels(self.orders[offscreen])}. Arrows at the edges continue these rays.')
        index = int(np.flatnonzero(self.orders == selected)[0])
        if reaches_plane[index]:
            theta_m = np.rad2deg(np.arctan2(self.directions[index, 0], -self.directions[index, 2]))
            messages.append(f'Selected order m = {labels([selected])}: θₘ = {theta_m:.2f}°, '
                            f'x on plane = {plane[index, 0]:.3f} µm = L tan θₘ.')
        self.readout.value = '<div style="padding:8px 0">' + '<br>'.join(messages) + '</div>'
        if surface_changed:
            self.picture.value = figure_png(self.figure)
            self._surface_signature = surface_signature
        direction = self.directions[index]
        k = 2 * np.pi / wavelength
        self.component_data = dict(order=selected, direction=direction.copy(),
            k_vector=k * direction, sphere=sphere[index].copy(),
            spatial_frequency=direction / wavelength,
            alpha=float(direction[0]),
            theta_deg=float(np.rad2deg(np.arctan2(direction[0], -direction[2]))),
            grazing=bool(grazing[index]))
        component_signature = (wavelength, period, angle, radius, selected)
        component_changed = component_signature != self._component_signature
        if component_changed:
            _draw_order_components(self.component_figure, direction, wavelength, period,
                angle, radius, selected, radius_limit=self.radius.max,
                k_limit=2 * np.pi / self.wavelength.min)
        if grazing[index]:
            geometry_note = 'Grazing: all three vertical components are zero; the triangles are flat.'
        elif abs(direction[0]) < 1e-12:
            geometry_note = 'Normal outgoing direction (θₘ = 0): the horizontal component is zero.'
        else:
            geometry_note = ('The horizontal component points '
                             + ('right' if direction[0] > 0 else 'left')
                             + '; the vertical component points toward −z.')
        self.component_readout.value = (
            '<div style="padding:8px 0">' + selection_note + geometry_note
            + '<br>Uses the controls above. Each panel keeps its own fixed scale; '
              'L changes only the observation-plane view.</div>')
        if component_changed:
            self.component_picture.value = figure_png(self.component_figure)
            self._component_signature = component_signature

    def close(self):
        """Disconnect controls and release only this explorer's resources."""
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.component_order.unobserve(self._update, names='value')
        self.reset.on_click(self._reset_example, remove=True)
        self.picture.value = b''
        self.component_picture.value = b''
        self.figure.clear()
        self.component_figure.clear()
        self.figure = None
        self.component_figure = None
        self.data = None
        self.component_data = None
        _close_widget_tree(self.ui)
        _close_widget_tree(self.components_ui)


class GratingSurfaceExplorer:
    """One incident-field snapshot and its unwrapped surface phase.

    Lesson 2.2 defaults to an 8 µm view and equal open/closed widths.
    adjustable_duty=True exposes period (nm) and open-fraction controls.
    There is no playback or clock link. Controls rerender one owned figure;
    close() disconnects callbacks and releases widgets, layouts and styles.
    """

    def __init__(self, *, field_width_um=8., adjustable_duty=False):
        if not np.isfinite(field_width_um) or field_width_um <= 0:
            raise ValueError('The displayed field width must be positive and finite.')
        self._closed = False
        self._incident_cache = None
        self.field_width_um = float(field_width_um)
        self.adjustable_duty = adjustable_duty
        half_field = self.field_width_um/2
        samples = int(np.ceil(self.field_width_um/(.005 if adjustable_duty else .02)))+1
        self._x = np.linspace(-half_field, half_field, samples)
        self._z = np.linspace(8, 0, 241)
        self.figure = _figure(figsize=(10, 4.6))
        if adjustable_duty:
            self.period = widgets.IntSlider(value=2000, min=200, max=4000, step=10,
                description='Period d (nm)', continuous_update=False,
                style={'description_width': 'initial'}, layout=widgets.Layout(width='310px'))
            self.duty_cycle = widgets.FloatSlider(value=.5, min=.05, max=.95, step=.01,
                description='Duty D (open fraction)', readout_format='.2f', continuous_update=False,
                style={'description_width': 'initial'}, layout=widgets.Layout(width='310px'))
        else:
            self.period = widgets.FloatSlider(value=2, min=1, max=4, step=.1,
                description='Period d (µm)', readout_format='.1f', continuous_update=False,
                style={'description_width': 'initial'}, layout=widgets.Layout(width='310px'))
        self.wavelength = widgets.FloatSlider(value=2, min=1, max=4, step=.1,
            description='λ in medium (µm)', readout_format='.1f', continuous_update=False,
            style={'description_width': 'initial'}, layout=widgets.Layout(width='310px'))
        self.angle = widgets.IntSlider(value=0, min=-60, max=60, step=5,
            description='θᵢ from −z (°)', continuous_update=False,
            style={'description_width': 'initial'}, layout=widgets.Layout(width='280px'))
        self.normal = widgets.Button(description='Normal incidence',
                                     layout=widgets.Layout(width='160px'))
        geometry_controls = ((self.period, self.duty_cycle) if adjustable_duty
                             else (self.period,))
        self._controls = (*geometry_controls, self.wavelength, self.angle)
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1100px', height='auto'))
        shown_controls = [*self._controls, self.normal]
        row = widgets.Box(shown_controls, layout=widgets.Layout(
            display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([row, self.readout, self.picture])
        for control in self._controls:
            control.observe(self._update, names='value')
        self.normal.on_click(self._normal_incidence)
        self._update()

    def _normal_incidence(self, button):
        if not self._closed:
            self.angle.value = 0

    def _update(self, change=None):
        if self._closed:
            return
        if self.adjustable_duty:
            period = self.period.value/1000
            width = period*self.duty_cycle.value
            closed_width = period-width
        else:
            period = self.period.value
            width = closed_width = period/2
        wavelength, angle = self.wavelength.value, self.angle.value
        half_field = self.field_width_um/2
        x, z = self._x, self._z
        key = (wavelength, angle)
        if self._incident_cache is None or self._incident_cache[0] != key:
            k_vector = wavevector(wavelength, angle)
            # The helper uses +z; this lesson measures incidence from -z.
            k_vector[2] *= -1
            surface_phase = k_vector[0] * x
            # y=0 and t=0: evaluate the same scalar plane wave directly on
            # broadcast axes, without constructing an (Nz, Nx, 3) point grid.
            incident = np.cos(surface_phase[None, :] + k_vector[2] * z[:, None])
            self._incident_cache = (key, k_vector, incident, surface_phase)
        _, k_vector, incident, surface_phase = self._incident_cache
        self.data = dict(x=x, z=z, incident=incident, surface_phase=surface_phase,
                         k_vector=k_vector, phase_step=k_vector[0] * period,
                         period_um=period, opening_width_um=width,
                         closed_width_um=closed_width, field_width_um=self.field_width_um,
                         spatial_frequency_per_um=1/period, G_rad_per_um=2*np.pi/period,
                         periods_in_field=self.field_width_um/period, duty_cycle=width/period,
                         transmission=amplitude_grating(x, period, width))

        self.figure.clear()
        field, phase = self.figure.subplots(1, 2)
        self.figure.subplots_adjust(left=.075, right=.97, bottom=.16, top=.84, wspace=.35)
        picture = _uniform_field_image(field, x, z, incident,
                                       cmap='RdBu_r', vmin=-1, vmax=1)
        self.figure.colorbar(picture, ax=field, fraction=.04, pad=.025,
                            label='Signed incident field uᵢ/A')
        centres = period * np.arange(-int(np.ceil(half_field / period)) - 1,
                                      int(np.ceil(half_field / period)) + 2)
        for centre in centres:
            # Dark segments block; the breaks are the openings.
            for start, end in ((centre - period / 2, centre - width / 2),
                               (centre + width / 2, centre + period / 2)):
                field.plot([start, end], [0, 0], color='#222222', lw=7,
                           solid_capstyle='butt')
            phase.axvspan(centre - width / 2, centre + width / 2,
                          color='#dff4fc', zorder=0)
        direction = k_vector / np.linalg.norm(k_vector)
        tail = -2.7 * direction[[0, 2]]
        field.plot([0, 0], [3.1, 0], '--', color='#333333', lw=1)
        field.annotate('', xy=(0, 0), xytext=tail,
                       arrowprops=dict(arrowstyle='->', lw=2, color='#111111'))
        field.text(tail[0] + .15, tail[1] + .28, r'incident $\mathbf{k}_i$', fontsize=10,
                   bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2))
        field.set(xlim=(-half_field, half_field), ylim=(-.25, 8), aspect='equal',
                  xlabel='x (µm)', ylabel='z (µm)',
                  title='Incident wave above the grating\nSnapshot at t = 0; travelling toward the grating')
        phase.plot(x, surface_phase, color='#14758c', lw=2)
        visible_centres = centres[np.abs(centres) <= half_field]
        phase.scatter(visible_centres, k_vector[0] * visible_centres,
                      color='#14758c', s=25, zorder=3)
        phase.axhline(0, color='#888888', lw=.7, zorder=0)
        limit = max(np.pi, np.max(np.abs(surface_phase)) * 1.12)
        phase.set(xlim=(-half_field, half_field), ylim=(-limit, limit), xlabel='x (µm)',
                  ylabel='Incident phase (rad)',
                  title='Incident phase at the surface\nBlue bands = openings; dots = their centres')
        phase.grid(axis='y', alpha=.2)
        step = self.data['phase_step']
        status = ('Normal incidence: every opening receives the same phase.' if angle == 0
                  else 'Tilted incidence: compare the phase at neighbouring opening centres.')
        if self.adjustable_duty:
            geometry = (
                f'<b>Fixed field: {self.field_width_um:g} µm</b> '
                f'(x = −{half_field:g} to +{half_field:g} µm).<br>'
                f'<b>d = {period*1000:g} nm ({period:g} µm)</b>; '
                f'<b>D = {width/period:.2f} ({100*width/period:g}% open)</b>; '
                f'W<sub>open</sub> = Dd = {width*1000:g} nm; '
                f'W<sub>close</sub> = (1−D)d = {closed_width*1000:g} nm.<br>'
                f'<b>grating spatial frequency 1/d = {1/period:.4g} cycles/µm</b>; '
                f'<b>G = 2π/d = {2*np.pi/period:.4f} rad/µm</b>.<br>'
                f'L/d = {self.field_width_um/period:.4g} periods across the field.<br>')
        else:
            geometry = f'W<sub>open</sub> = d/2 = {width:.2f} µm; '
        self.readout.value = (
            f'<div style="padding:8px 0">{geometry}<b>{status}</b><br>A = 1; '
            f'k<sub>ix</sub> = {k_vector[0]:+.3f} rad/µm; '
            f'k<sub>iz</sub> = {k_vector[2]:+.3f} rad/µm; '
            f'phase step k<sub>ix</sub>d = {step:+.3f} rad '
            f'({step / np.pi:+.3f}π). '
            'Phase is unwrapped; differences of 2π describe the same phase in a cycle.</div>')
        self.picture.value = figure_png(self.figure)

    def close(self):
        """Dispose only this explorer; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.normal.on_click(self._normal_incidence, remove=True)
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        self._incident_cache = None
        self._x = self._z = None
        _close_widget_tree(self.ui)
