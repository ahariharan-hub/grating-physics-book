"""Incident/selected-order plane waves using notebook two's x-z geometry.

Both sides use the same wavelength in a homogeneous medium. Incidence is
from z>0; outgoing directions have kz<=0. Individual components are drawn
with unit amplitude and zero phase at the origin, not as a boundary solution.
All figure/image buffers remain in memory.
"""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.patheffects as path_effects

from .diffraction import diffraction_orders
from .plotting import figure_png
from .grating_intro import _close_widget_tree, _uniform_field_image
from .order_formation import wavefront_fields
from .order_paths import path_difference_data, draw_path_geometry
from .order_array import MAX_OPENINGS, opening_array_data, draw_opening_array


BLUE = '#2866a3'
ORANGE = '#b85c17'
GREY = '#89939e'
INK = '#243444'
GREEN = '#23734d'


def _outgoing_grid():
    """Use the same fixed physical window in the order and comparison views."""
    return np.linspace(-4.5, 4.5, 361), np.linspace(-4.5, 0, 181)


def _period_ruler(axis, period, height, label_height):
    ruler = axis.annotate('', (period, height), (0, height),
                          arrowprops=dict(arrowstyle='<->', color=INK, lw=1.3,
                                          shrinkA=0, shrinkB=0), zorder=7)
    ruler.arrow_patch.set_path_effects([
        path_effects.Stroke(linewidth=3.5, foreground='white'), path_effects.Normal()])
    axis.text(period/2, label_height, f'd = {period:g} µm', ha='center', va='center',
              fontsize=9, color=INK, zorder=7,
              bbox=dict(facecolor='white', edgecolor='none', alpha=.94, pad=2))


def order_wave_data(wavelength=2., period=2., angle_deg=30., selected_order=-1,
                    *, _field_cache=None):
    """Real directions, including grazing, and the selected field snapshot.

    Explorers may retain their own bounded field cache; ordinary calls return
    freshly calculated arrays. The cache holds only the current two fields.
    """
    if not np.isfinite(angle_deg) or not -90 < angle_deg < 90:
        raise ValueError('Use a finite incidence angle between -90 and 90 degrees.')
    result = diffraction_orders(wavelength, period, abs(angle_deg),
                                180. if angle_deg < 0 else 0.)
    orders = result['allowed'].copy()
    directions = result['directions'].copy()
    directions[:, 1] = 0.
    directions[:, 2] *= -1
    grazing = result['grazing'][result['valid']].copy()
    directions[grazing, 0] = np.sign(directions[grazing, 0])
    directions[grazing, 2] = 0.
    chosen = int(selected_order) if selected_order in orders else 0
    index = int(np.flatnonzero(orders == chosen)[0])
    k = 2*np.pi/wavelength
    theta = np.deg2rad(angle_deg)
    incident = k*np.array([np.sin(theta), 0., -np.cos(theta)])
    vectors = k*directions
    angles = np.rad2deg(np.arctan2(directions[:, 0], -directions[:, 2]))
    cache = {} if _field_cache is None else _field_cache
    if 'grid' not in cache:
        x, z_below = _outgoing_grid()
        cache['grid'] = (x, z_below, -z_below[::-1])
    x, z_below, z_above = cache['grid']

    def field(name, z, vector):
        key = tuple(vector)
        if name not in cache or cache[name][0] != key:
            # y=0, t=0: broadcasting axes avoids constructing 3D point arrays.
            cache[name] = (key, np.cos(vector[0]*x[None, :] + vector[2]*z[:, None]))
        return cache[name][1]

    return dict(wavelength=float(wavelength), period=float(period), angle_deg=float(angle_deg),
                k=k, incident=incident, orders=orders, directions=directions,
                vectors=vectors, angles=angles, grazing=grazing,
                selected=chosen, index=index, x=x, z_above=z_above, z_below=z_below,
                incident_field=field('incident', z_above, incident),
                outgoing_field=field('outgoing', z_below, vectors[index]),
                incident_step=incident[0]*period,
                outgoing_step=vectors[index, 0]*period)


def cell_comparison_limits(wavelength, period, angle_deg):
    """Continuous comparison values that have a real outgoing direction."""
    if (not np.all(np.isfinite([wavelength, period, angle_deg]))
            or wavelength <= 0 or period <= 0 or not -90 < angle_deg < 90):
        raise ValueError('Use positive finite lengths and -90 < incidence < 90 degrees.')
    incident_sine = np.sin(np.deg2rad(angle_deg))
    return ((-1-incident_sine)*period/wavelength,
            (1-incident_sine)*period/wavelength)


def cell_comparison_data(wavelength=2., period=2., angle_deg=30., comparison_order=-1.):
    """Continuous wave versus cell copies carrying the incident phase offset.

    m_test=(kx-kix)*d/(2*pi) is continuous here. It is an order number only
    at integers. The copied pattern is a compatibility construction, not an
    actual discontinuous scattered field. Its right-minus-left phase jump
    at a seam is -2*pi*m_test modulo 2*pi.
    """
    low, high = cell_comparison_limits(wavelength, period, angle_deg)
    if (not np.isfinite(comparison_order)
            or comparison_order < low-1e-12 or comparison_order > high+1e-12):
        raise ValueError('The comparison value must have a real outgoing direction.')
    alpha = np.sin(np.deg2rad(angle_deg))+comparison_order*wavelength/period
    outgoing_angle = float(np.rad2deg(np.arcsin(np.clip(alpha, -1, 1))))
    x, z = _outgoing_grid()
    boundaries = np.arange(int(np.ceil(x[0]/period)),
                            int(np.floor(x[-1]/period))+1)*period
    continuous, copied = wavefront_fields(x[None, :], -z[:, None],
                                          wavelength, period, angle_deg, outgoing_angle)
    difference = float(2*np.pi*comparison_order)
    seam_factor = np.exp(-1j*difference)
    matched = abs(seam_factor-1) < 1e-9
    k = 2*np.pi/wavelength
    direction = np.array([np.clip(alpha, -1, 1), -np.sqrt(max(0., 1-alpha*alpha))])
    phase_x = np.linspace(0, period, 101)
    return dict(wavelength=wavelength, period=period, angle_deg=angle_deg,
                comparison_order=float(comparison_order), outgoing_angle=outgoing_angle,
                x=x, z=z, cell_boundaries=boundaries, continuous=continuous, copied=copied,
                k=k, incident_kx=k*np.sin(np.deg2rad(angle_deg)),
                outgoing_kx=k*direction[0], outgoing_kz=k*direction[1], direction=direction,
                phase_x=phase_x, phase_from_a=k*direction[0]*phase_x,
                incident_step=k*np.sin(np.deg2rad(angle_deg))*period,
                outgoing_step=k*np.clip(alpha, -1, 1)*period,
                difference=difference, seam_factor=seam_factor,
                seam_jump=float(np.angle(seam_factor)), matched=matched,
                order=int(round(comparison_order)) if matched else None)


def _comparison_vectors(axis, data, *, walk_height=-3.65):
    """Scaled trial outgoing vector and projection inside the base cell."""
    period, direction = data['period'], data['direction']
    length = min(1.7, .8*period/max(abs(direction[0]), 1e-12))
    displacement = length*direction
    start = np.array([period/2-displacement[0]/2, -1.1])
    end = start+displacement
    horizontal = np.array([end[0], start[1]])
    box = dict(facecolor='white', alpha=.94, edgecolor='none', pad=1.5)

    def arrow(point, color, width):
        mark = axis.annotate('', point, start,
            arrowprops=dict(arrowstyle='->', color=color, lw=width, shrinkA=0, shrinkB=0),
            zorder=8)
        mark.arrow_patch.set_path_effects([
            path_effects.Stroke(linewidth=width+2, foreground='white'), path_effects.Normal()])

    arrow(end, BLUE, 2.4)
    if abs(direction[1]) < 1e-9:
        label = (r'$\mathbf{k}_{\mathrm{out}},\quad k_{\mathrm{out},x}=+k$'
                 if direction[0] > 0 else
                 r'$\mathbf{k}_{\mathrm{out}},\quad k_{\mathrm{out},x}=-k$')
        axis.text(period/2, start[1]-.32, label, fontsize=11, ha='center',
                  color=BLUE, bbox=box, zorder=9)
    else:
        if abs(direction[0]) > 1e-9:
            arrow(horizontal, ORANGE, 2)
            projection, = axis.plot([end[0], horizontal[0]], [end[1], horizontal[1]],
                                     '--', color=INK, lw=1, zorder=7)
            projection.set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground='white'), path_effects.Normal()])
            axis.text(period/2, start[1]+.18, r'$k_{\mathrm{out},x}$', fontsize=12, ha='center',
                      color=ORANGE, bbox=box, zorder=9)
        else:
            axis.text(start[0]+.18, start[1], r'$k_{\mathrm{out},x}=0$', fontsize=10,
                      color=ORANGE, bbox=box, zorder=9)
        axis.annotate(r'$\mathbf{k}_{\mathrm{out}}$', (start+end)/2, xytext=(-22, -3),
                      textcoords='offset points', fontsize=12, color=BLUE, bbox=box, zorder=9)

    # The horizontal walk changes x only: the common kz*z and time terms cancel.
    height = walk_height
    line, = axis.plot([0, period], [height, height], '--', color=INK, lw=1.3, zorder=7)
    line.set_path_effects([
        path_effects.Stroke(linewidth=3.5, foreground='white'), path_effects.Normal()])
    for xx, label in ((0, 'A'), (period, 'B')):
        axis.plot(xx, height, 'o', markersize=6, markerfacecolor='white',
                  markeredgecolor=INK, markeredgewidth=1.2, zorder=8)
        axis.annotate(label, (xx, height), xytext=(0, -17), textcoords='offset points',
                      ha='center', fontsize=10, bbox=box, zorder=9)


def draw_cell_comparison(figure, data):
    """Connect the wavevector projection, an A-to-B phase walk, and the seam."""
    figure.clear()
    grid = figure.add_gridspec(2, 2, height_ratios=[1.25, 1])
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    phase_axis = figure.add_subplot(grid[1, 0])
    explanation = figure.add_subplot(grid[1, 1])
    figure.subplots_adjust(left=.07, right=.91, bottom=.13, top=.90, wspace=.24, hspace=.40)
    period, value = data['period'], data['comparison_order']
    figure.suptitle(fr'From $k_{{\mathrm{{out}},x}}$ to the seam: comparison $m_{{\mathrm{{test}}}} = {value:+.3f}$', fontsize=14)
    for axis, wave, title in zip(axes, (data['continuous'], data['copied']),
                                ('Continuous plane wave at this direction',
                                 'One cell copied with the incident phase offset')):
        picture = _uniform_field_image(axis, data['x'], data['z'], wave.real,
                                        cmap='RdBu_r', vmin=-1, vmax=1)
        for boundary in data['cell_boundaries']:
            axis.axvline(boundary, color=INK, lw=.7, ls=':', alpha=.65)
        axis.set(xlim=(data['x'][0], data['x'][-1]), ylim=(data['z'][0], data['z'][-1]),
                 aspect='equal', xticks=np.arange(-4, 5, 2), yticks=np.arange(-4, 1),
                 xlabel='x (µm)', ylabel='z (µm)',
                 title=title)
        axis.title.set_fontsize(10)
        _period_ruler(axis, period, -.27, -.57)
        _comparison_vectors(axis, data)
    color_axis = figure.add_axes([.935, .575, .015, .265])
    figure.colorbar(picture, cax=color_axis, ticks=[-1, 0, 1], label='Field value cos φ')

    # These are phase changes relative to A at the marked height, not the
    # absolute phase of the colour at A. Compare one-sided limits at B.
    end, next_start = data['outgoing_step'], data['incident_step']
    phase_axis.plot(data['phase_x'], data['phase_from_a'], color=ORANGE, lw=2.2)
    phase_axis.plot(0, 0, 'o', color=INK, ms=5)
    phase_axis.plot(period, end, 'o', color=ORANGE, ms=7, label='End of this cell')
    phase_axis.plot(period, next_start, 'o', color=GREEN, ms=11, markerfacecolor='none',
                    markeredgewidth=2, label='Start of next copy')
    phase_axis.axhline(0, color=GREY, lw=.8)
    phase_axis.axvline(period, color=GREY, lw=.8, ls=':')
    if abs(end-next_start) > 1e-9:
        phase_axis.annotate('', (period*1.09, next_start), (period*1.09, end),
            arrowprops=dict(arrowstyle='<->', color=INK, lw=1.2, shrinkA=0, shrinkB=0))
    low, high = min(0, end, next_start), max(0, end, next_start)
    span = max(np.pi, high-low)
    step = max(.5, np.ceil(span/np.pi/4*2)/2)
    ticks = np.arange(np.floor(low/np.pi/step), np.ceil(high/np.pi/step)+1)*step
    phase_axis.set_yticks(ticks*np.pi, ['0' if tick == 0 else f'{tick:g}π' for tick in ticks])
    phase_axis.set(xlim=(-.06*period, 1.22*period), ylim=(low-.22*span, high+.24*span),
                   xticks=[0, period/2, period], xticklabels=['A: 0', f'{period/2:g}', f'B: d = {period:g}'],
                   xlabel='x along the marked A–B line (µm)',
                   ylabel='Phase change from A (rad)',
                   title='Same height, same instant: follow the phase from A to B')
    phase_axis.title.set_fontsize(10)
    phase_axis.spines[['top', 'right']].set_visible(False)
    phase_axis.grid(axis='y', alpha=.18)

    explanation.axis('off')
    lines = [
        (.99, '1. Direction → horizontal phase change', INK, 'bold'),
        (.86, fr'$k = {data["k"]:.3f}$; $k_{{\mathrm{{out}},x}} = {data["outgoing_kx"]:+.3f}$ rad/µm', BLUE, 'normal'),
        (.69, '2. Orange dot: end of this cell at B', ORANGE, 'bold'),
        (.56, fr'$\Delta\phi_{{\mathrm{{out}}}} = k_{{\mathrm{{out}},x}}d = {end/np.pi:+.3f}\pi$ rad', ORANGE, 'normal'),
        (.39, '3. Green circle: start of the next copy at B', GREEN, 'bold'),
        (.26, f'Δφᵢ = kᵢₓd = {next_start/np.pi:+.3f}π rad', GREEN, 'normal'),
    ]
    for yy, label, color, weight in lines:
        explanation.text(0, yy, label, transform=explanation.transAxes, va='top',
                          fontsize=10, color=color, fontweight=weight)
    if data['matched']:
        status = (f'End − next start = {data["difference"]/np.pi:+.3f}π: whole cycles.\n'
                  'Same phase in the cycle: the seam joins.')
    else:
        status = (f'End − next start = {data["difference"]/np.pi:+.3f}π: not whole cycles.\n'
                  'Different phase in the cycle: broken seam.')
    explanation.text(0, .055, status, transform=explanation.transAxes, va='top',
                      fontsize=10, color=GREEN if data['matched'] else ORANGE)
    figure.text(.5, .025, 'Colours show the field; the orange graph shows phase change from A. Full 2π cycles count as the same phase.',
                ha='center', fontsize=9, color=INK)


def draw_path_connection(figure, wave, paths):
    """Keep the familiar plane wave beside the corresponding ray geometry."""
    figure.clear()
    grid = figure.add_gridspec(2, 2, height_ratios=[1.2, 1])
    field = figure.add_subplot(grid[0, 0])
    explanation = figure.add_subplot(grid[1, 0])
    geometry = figure.add_subplot(grid[:, 1])
    figure.subplots_adjust(left=.07, right=.96, top=.86, bottom=.13, wspace=.27, hspace=.12)
    figure.suptitle('The same phase cycles, measured as extra path length', fontsize=14)
    figure.text(.5, .925,
                f'd = {wave["period"]:g} µm; λ = {wave["wavelength"]:g} µm; '
                fr'$\theta_i = {wave["angle_deg"]:+.1f}°$; $\theta_{{\mathrm{{out}}}} = {wave["outgoing_angle"]:+.2f}°$',
                ha='center', fontsize=10, color=INK)
    _uniform_field_image(field, wave['x'], wave['z'], wave['continuous'].real,
                          cmap='RdBu_r', vmin=-1, vmax=1)
    field.axhline(0, color=GREY, lw=2)
    _comparison_vectors(field, wave, walk_height=0.)
    _period_ruler(field, wave['period'], -3.65, -3.95)
    field.set(xlim=(wave['x'][0], wave['x'][-1]), ylim=(wave['z'][0], .2), aspect='equal',
              xlabel='x (µm)', ylabel='z (µm)', title=r'Outgoing wave: phase from A to B is $k_{\mathrm{out},x}d$')
    field.title.set_fontsize(11)
    draw_path_geometry(geometry, paths)
    explanation.axis('off')
    if abs(wave['incident_kx']) < 1e-10:
        opening = 'Normal incidence: equal incoming paths.'
    else:
        opening = 'Tilted incidence: include the incoming contribution.'
    rows = [
        (.96, opening, INK, 'bold'),
        (.78, fr'Outgoing: $\ell_{{\mathrm{{out}}}} = d\sin\theta_{{\mathrm{{out}}}} = {paths["outgoing_projection"]:+.3f}$ µm', ORANGE, 'normal'),
        (.62, f'Incoming: ℓᵢ = d sin θᵢ = {paths["incident_projection"]:+.3f} µm', GREEN, 'normal'),
        (.43, f'ΔL = route A − route B = ℓ(out) − ℓᵢ = {paths["delta_length"]:+.3f} µm', INK, 'normal'),
        (.25, fr'$k\Delta L = (k_{{\mathrm{{out}},x}}-k_{{ix}})d = {wave["difference"]/np.pi:+.3f}\pi$ rad', INK, 'bold'),
        (.06, f'ΔL / λ = {paths["delta_length"]/wave["wavelength"]:+.3f} phase cycles', INK, 'bold'),
    ]
    for yy, label, color, weight in rows:
        explanation.text(0, yy, label, transform=explanation.transAxes, va='top',
                          fontsize=9.5, color=color, fontweight=weight)
    status = ('Whole cycles: the paths agree in phase and the cell seams join.' if wave['matched']
              else 'Fractional cycle difference: the paths disagree in phase and the cell seams break.')
    figure.text(.5, .06, status, ha='center', fontsize=10, color=GREEN if wave['matched'] else ORANGE)
    figure.text(.5, .025,
                'Dashed lines are perpendicular to the rays. Coloured segments show extra lengths; the readout gives their signs.',
                ha='center', fontsize=9, color=INK)


def draw_order_waves(figure, data):
    """Reuse the section-three field colours; put the order overview at right."""
    figure.clear()
    grid = figure.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1.1, 1])
    field = figure.add_subplot(grid[:, 0])
    fan = figure.add_subplot(grid[0, 1])
    listing = figure.add_subplot(grid[1, 1])
    figure.subplots_adjust(left=.065, right=.98, bottom=.10, top=.86,
                           wspace=.30, hspace=.17)
    selected, index = data['selected'], data['index']
    figure.suptitle('Incident wave and one selected diffraction order', fontsize=14)
    _uniform_field_image(field, data['x'], data['z_above'], data['incident_field'],
                          cmap='RdBu_r', vmin=-1, vmax=1)
    picture = _uniform_field_image(field, data['x'], data['z_below'], data['outgoing_field'],
                                    cmap='RdBu_r', vmin=-1, vmax=1)
    figure.colorbar(picture, ax=field, fraction=.038, pad=.025,
                    label='Normalized signed field', ticks=[-1, 0, 1])
    period = data['period']
    centres = np.arange(-int(np.ceil(4.5/period))-1,
                        int(np.ceil(4.5/period))+2)*period
    field.axhline(0, color=GREY, lw=.8)
    for centre in centres:
        for start, end in ((centre-period/2, centre-period/4),
                           (centre+period/4, centre+period/2)):
            field.plot([start, end], [0, 0], color='#222222', lw=6,
                        solid_capstyle='butt', zorder=5)
    label_box = dict(facecolor='white', alpha=.90, edgecolor='none', pad=3)
    field.text(-4.22, 4.05, r'Incident wave $\mathbf{k}_i$', fontsize=10, bbox=label_box)
    field.text(-4.22, -4.10, fr'Outgoing $\mathbf{{k}}_m$: order $m = {selected:+d}$',
                fontsize=10, bbox=label_box)
    field.text(4.25, .25, 'grating: z = 0', ha='right', fontsize=9, bbox=label_box)
    tail = -2.6*data['incident'][[0, 2]]/data['k']
    field.annotate('', (0, .08), tail,
                   arrowprops=dict(arrowstyle='->', color=INK, lw=2.3))
    direction = data['directions'][index, [0, 2]]
    start = np.array([0., -1.0 if data['grazing'][index] else -.12])
    field.annotate('', start+2.8*direction, start,
                   arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.8))
    _period_ruler(field, period, .50, .82)
    field.set(xlim=(data['x'][0], data['x'][-1]),
              ylim=(data['z_below'][0], data['z_above'][-1]), aspect='equal',
              xlabel='x (µm)', ylabel='z (µm)',
              title='2D wave snapshot at t = 0')
    field.title.set_fontsize(12)

    fan.axhline(0, color=GREY, lw=2)
    fan.plot([0, 0], [0, -1.12], '--', color=GREY, lw=.8)
    fan.text(.04, -1.23, '−z', fontsize=9, color=GREY)
    fan.text(1.18, .08, '+x', fontsize=9, color=GREY)
    fan.text(-1.28, .12, 'grating', fontsize=9, color=GREY)
    for order, vector in zip(data['orders'], data['directions']):
        chosen = order == selected
        color = ORANGE if chosen else BLUE
        end = vector[[0, 2]]
        fan.annotate('', end, (0, 0),
                     arrowprops=dict(arrowstyle='->', color=color,
                                     lw=2.8 if chosen else 1.1, shrinkA=0, shrinkB=0),
                     zorder=4 if chosen else 3)
        fan.text(1.14*end[0], 1.14*end[1], f'{order:+d}', ha='center', va='center',
                  fontsize=10, color=color, fontweight='bold' if chosen else 'normal',
                  bbox=dict(facecolor='white', edgecolor='none', pad=.8))
    fan.set(xlim=(-1.45, 1.45), ylim=(-1.4, .35), aspect='equal',
            title='Possible order directions; selected order in orange')
    fan.title.set_fontsize(11)
    fan.axis('off')

    rows = [[f'{order:+d}', f'{angle:+.2f}' + (' *' if grazing else ''), f'{vector[0]:+.3f}']
            for order, angle, vector, grazing in zip(data['orders'], data['angles'],
                                                     data['vectors'], data['grazing'])]
    table = listing.table(cellText=rows,
                          colLabels=['Order m', 'Angle θₘ (°)', 'kₘₓ (rad/µm)'],
                          cellLoc='center', colLoc='center', loc='center',
                          colWidths=[.25, .35, .40])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.28)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#dce2e8')
        if row == 0:
            cell.set_facecolor('#eef2f6')
            cell.set_text_props(weight='bold', color=INK)
        elif row == index+1:
            cell.set_facecolor('#fff0df')
            cell.set_text_props(weight='bold', color=ORANGE)
    listing.axis('off')
    figure.text(.76, .045, '* Grazing: parallel to the surface.' if np.any(data['grazing'])
                else 'Angles are measured from −z toward +x.',
                ha='center', fontsize=9, color=INK)
    figure.text(.28, .025, 'Each field is normalized; order strengths are not shown.',
                ha='center', fontsize=9, color=INK)


class OrderWaveExplorer:
    """Local wavelength/period/incidence controls plus a real-order selector."""

    def __init__(self, wavelength=2., period=2., angle_deg=30., selected_order=-1):
        import ipywidgets as widgets

        self._closed = False
        self._updating = False
        self._preferred = selected_order
        self._comparison_follows_order = True
        self._defaults = (2., 2., 30.)
        self._paths_ui = None
        self.path_figure = None
        self.path_data = None
        self._array_ui = None
        self.array_figure = None
        self.array_data = None
        self._array_signature = None
        self._field_cache = {}
        self._array_cache = {}
        self._comparison_signature = None
        self._path_signature = None
        self.figure = Figure(figsize=(12.4, 7.1), dpi=105, facecolor='white')
        FigureCanvasAgg(self.figure)

        def slider(value, low, high, step, description):
            return widgets.FloatSlider(value=value, min=low, max=high, step=step,
                description=description, readout_format='.1f', continuous_update=False,
                style={'description_width': 'initial'}, layout=widgets.Layout(width='300px'))

        self.period = slider(period, 1, 4, .1, 'Period d (µm)')
        self.wavelength = slider(wavelength, 1, 4, .1, 'λ in medium (µm)')
        self.angle = slider(angle_deg, -60, 60, 5, 'θᵢ from −z (°)')
        self._controls = (self.period, self.wavelength, self.angle)
        self.order = widgets.Dropdown(options=[], description='Show order',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='330px'))
        self.normal = widgets.Button(description='Normal incidence')
        self.reset = widgets.Button(description='Reset example')
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1302px', height='auto'))
        self.compare = widgets.Checkbox(value=False, description='Compare non-integer values',
                                        indent=False, layout=widgets.Layout(width='250px'))
        self.comparison = widgets.FloatSlider(value=0, min=-1, max=1, step=.05,
            description='Comparison value', readout_format='.3f', continuous_update=False,
            style={'description_width': 'initial'}, layout=widgets.Layout(width='470px'))
        self.use_order = widgets.Button(description='Use selected order',
                                        layout=widgets.Layout(width='170px'))
        self.comparison_readout = widgets.HTML()
        self.comparison_picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1302px', height='auto'))
        self.comparison_box = widgets.VBox([
            widgets.Box([self.comparison, self.use_order], layout=widgets.Layout(
                display='flex', flex_flow='row wrap', align_items='center')),
            self.comparison_readout, self.comparison_picture],
            layout=widgets.Layout(display='none'))
        self.comparison_figure = Figure(figsize=(12.4, 7.5), dpi=105, facecolor='white')
        FigureCanvasAgg(self.comparison_figure)
        self.comparison_data = None
        self._comparison_note = ''
        row = widgets.Box(list(self._controls), layout=widgets.Layout(
            display='flex', flex_flow='row wrap', align_items='center'))
        selection = widgets.Box([self.order, self.normal, self.reset, self.compare], layout=widgets.Layout(
            display='flex', flex_flow='row wrap', align_items='center'))
        self.ui = widgets.VBox([row, selection, self.readout, self.picture, self.comparison_box])
        for control in self._controls:
            control.observe(self._update, names='value')
        self.order.observe(self._select_order, names='value')
        self.normal.on_click(self._normal_incidence)
        self.reset.on_click(self._reset)
        self.compare.observe(self._update_comparison, names='value')
        self.comparison.observe(self._update_comparison, names='value')
        self.use_order.on_click(self._use_selected_order)
        self._update()

    @property
    def paths_ui(self):
        """A later notebook view of the same controls, allocated only once."""
        if self._closed:
            raise RuntimeError('Rerun the explorer creation cell to create a new explorer.')
        if self._paths_ui is None:
            import ipywidgets as widgets

            def row(children):
                return widgets.Box(children, layout=widgets.Layout(
                    display='flex', flex_flow='row wrap', align_items='center'))

            self.path_readout = widgets.HTML()
            self.path_picture = widgets.Image(format='png', layout=widgets.Layout(
                width='100%', max_width='1302px', height='auto'))
            self.path_comparison_row = row([self.comparison, self.use_order])
            self._paths_ui = widgets.VBox([
                row(list(self._controls)), row([self.order, self.normal, self.reset, self.compare]),
                self.path_comparison_row, self.path_readout, self.path_picture])
            self.path_figure = Figure(figsize=(12.4, 7.4), dpi=105, facecolor='white')
            FigureCanvasAgg(self.path_figure)
            self._update_paths()
        return self._paths_ui

    def _update_paths(self):
        if self._closed or self._paths_ui is None:
            return
        signature = (self.wavelength.value, self.period.value, self.angle.value,
                     self.compare.value,
                     self.comparison.value if self.compare.value else self.order.value,
                     self._comparison_note if self.compare.value else '')
        if signature == self._path_signature:
            return
        wave = (self.comparison_data if self.compare.value else cell_comparison_data(
            self.wavelength.value, self.period.value, self.angle.value, self.order.value))
        paths = path_difference_data(wave['period'], wave['angle_deg'], wave['outgoing_angle'])
        self.path_data = dict(wave=wave, geometry=paths)
        self.path_comparison_row.layout.display = 'flex' if self.compare.value else 'none'
        draw_path_connection(self.path_figure, wave, paths)
        self.path_picture.value = figure_png(self.path_figure)
        self._path_signature = signature
        mode = (f'Comparison m_test = {wave["comparison_order"]:+.3f}' if self.compare.value
                else f'Selected order m = {wave["order"]:+d}')
        self.path_readout.value = (
            '<div style="padding:8px 0;line-height:1.7">'
            f'<b>{mode}. Shared period d = {wave["period"]:g} µm.</b> '
            'These controls belong to this explorer.<br>'
            f'{self._comparison_note if self.compare.value else ""}'
            'Follow the routes through A and B from one common incident wavefront '
            'to one common plane perpendicular to the outgoing direction. '
            '<b>ΔL means total route A minus total route B.</b> '
            'At normal incidence the incoming lengths are equal.</div>')

    def _select_order(self, change):
        if self._closed or self._updating:
            return
        self._preferred = change['new']
        self._comparison_follows_order = True
        self._update()

    @property
    def array_ui(self):
        """One-to-many opening illustration with the shared illumination."""
        if self._closed:
            raise RuntimeError('Rerun the explorer creation cell to create a new explorer.')
        if self._array_ui is None:
            import ipywidgets as widgets

            self.openings = widgets.IntSlider(value=1, min=1, max=MAX_OPENINGS, step=1,
                description='Openings N', continuous_update=False,
                style={'description_width': 'initial'}, layout=widgets.Layout(width='400px'))
            self.array_readout = widgets.HTML()
            self.array_picture = widgets.Image(format='png', layout=widgets.Layout(
                width='100%', max_width='1302px', height='auto'))
            parameters = widgets.Box(list(self._controls), layout=widgets.Layout(
                display='flex', flex_flow='row wrap', align_items='center'))
            count_row = widgets.Box([self.openings, self.normal], layout=widgets.Layout(
                display='flex', flex_flow='row wrap', align_items='center'))
            self._array_ui = widgets.VBox([parameters, count_row, self.array_readout, self.array_picture])
            self.array_figure = Figure(figsize=(12.4, 7.4), dpi=105, facecolor='white')
            FigureCanvasAgg(self.array_figure)
            self.openings.observe(self._update_array, names='value')
            self._update_array()
        return self._array_ui

    def _update_array(self, change=None):
        if self._closed or self._updating or self._array_ui is None:
            return
        signature = (self.wavelength.value, self.period.value, self.angle.value, self.openings.value)
        if signature == self._array_signature:
            return
        self.array_data = opening_array_data(*signature, _geometry_cache=self._array_cache)
        draw_opening_array(self.array_figure, self.array_data)
        self.array_picture.value = figure_png(self.array_figure)
        self._array_signature = signature
        self.array_readout.value = (
            '<div style="padding:8px 0;line-height:1.7">'
            f'<b>N = {self.openings.value} illuminated opening(s).</b> '
            'Wavelength, spacing and incidence are shared with sections 5 and 6. '
            'Change only N first; the spatial window stays fixed.<br>'
            '<b>Left:</b> the sum of all spreading contributions. '
            '<b>Right:</b> their far-field directional pattern. '
            'The order dropdown and comparison value above do not filter this sum.<br>'
            'Model: identical narrow openings with equal drive amplitudes; the incident wave '
            'sets their phases. The first wavelength around each source is omitted. '
            'Both views are normalized to compare shapes, not absolute power.</div>')

    def _update(self, change=None):
        if self._closed or self._updating:
            return
        self._updating = True
        try:
            self.data = order_wave_data(self.wavelength.value, self.period.value,
                                        self.angle.value, self._preferred,
                                        _field_cache=self._field_cache)
            data = self.data
            replaced = self._preferred not in data['orders']
            self._preferred = data['selected']
            self.order.options = tuple((f'm = {int(order):+d}; θₘ = {angle:+.2f}°'
                                        + (' (grazing)' if grazing else ''), int(order))
                for order, angle, grazing in zip(data['orders'], data['angles'], data['grazing']))
            self.order.value = data['selected']
            self._sync_comparison()
            draw_order_waves(self.figure, data)
            self.picture.value = figure_png(self.figure)
            index = data['index']
            message = ('The previous order has no real direction at these settings; '
                       'showing m = 0. ' if replaced else '')
            if data['grazing'][index]:
                message += 'This order travels parallel to the grating. '
            self.readout.value = (
                '<div style="padding:8px 0;line-height:1.7">'
                f'<b>{len(data["orders"])} possible real directions. '
                f'Selected m = {data["selected"]:+d}, θₘ = {data["angles"][index]:+.2f}°.</b> '
                f'{message}<br>'
                f'Δφᵢ = kᵢₓd = {data["incident_step"]/np.pi:+.3f}π rad; '
                f'Δφₘ = kₘₓd = {data["outgoing_step"]/np.pi:+.3f}π rad; '
                f'<b>Δφₘ − Δφᵢ = {2*data["selected"]:+d}π rad = 2πm.</b></div>')
        finally:
            self._updating = False
        self._update_comparison()
        self._update_array()

    def _sync_comparison(self):
        """Called while main callbacks are suspended; keep the value real."""
        low, high = cell_comparison_limits(self.wavelength.value, self.period.value,
                                           self.angle.value)
        value = self.order.value if self._comparison_follows_order else self.comparison.value
        clipped = float(np.clip(value, low, high))
        self._comparison_note = ('Comparison value moved to the nearest real-direction limit. '
                                 if abs(clipped-value) > 1e-10 else '')
        # Every incidence range includes zero, so these bounds can change safely.
        self.comparison.min = low
        self.comparison.max = high
        self.comparison.value = clipped

    def _use_selected_order(self, button=None):
        if self._closed:
            return
        self._comparison_follows_order = True
        self._updating = True
        try:
            self._sync_comparison()
        finally:
            self._updating = False
        self._update_comparison()

    def _update_comparison(self, change=None):
        if self._closed or self._updating:
            return
        if change is not None and change['owner'] is self.comparison:
            self._comparison_follows_order = False
            self._comparison_note = ''
        enabled = self.compare.value
        self.comparison_box.layout.display = 'flex' if enabled else 'none'
        if not enabled:
            self.comparison_picture.value = b''
            self.comparison_figure.clear()
            self.comparison_data = None
            self._comparison_signature = None
            self._update_paths()
            return
        signature = (self.wavelength.value, self.period.value, self.angle.value,
                     self.comparison.value, self._comparison_note)
        if signature == self._comparison_signature:
            self._update_paths()
            return
        self.comparison_data = cell_comparison_data(
            self.wavelength.value, self.period.value, self.angle.value, self.comparison.value)
        data = self.comparison_data
        draw_cell_comparison(self.comparison_figure, data)
        self.comparison_picture.value = figure_png(self.comparison_figure)
        self._comparison_signature = signature
        status = (f'The copies join: this is order m = {data["order"]:+d}.' if data['matched']
                  else 'Non-integer comparison: the copies do not join, so this is not a diffraction order.')
        self.comparison_readout.value = (
            '<div style="padding:8px 0;line-height:1.7">'
            f'<b>Shared period d = {data["period"]:g} µm</b>; '
            f'λ = {data["wavelength"]:g} µm; θᵢ = {data["angle_deg"]:g}°. '
            'Same physical coordinate ranges as the outgoing-wave region above.<br>'
            f'{self._comparison_note}<b>{status}</b><br>'
            f'm<sub>test</sub> = {data["comparison_order"]:+.3f}; θ<sub>out</sub> = {data["outgoing_angle"]:+.2f}°. '
            f'Δφ(out) − Δφᵢ = 2π m_test = {data["difference"]/np.pi:+.3f}π rad.<br>'
            '<b>Follow A → B at the same height and instant.</b> '
            'The orange graph measures phase change from A: Δφ<sub>out</sub> = k<sub>out,x</sub>d. '
            'At B, compare the orange end-of-cell phase with the green start-of-next-copy phase. '
            'The green value comes from the illumination: Δφᵢ = kᵢₓd.</div>')
        self._update_paths()

    def _normal_incidence(self, button):
        if not self._closed:
            self.angle.value = 0

    def _reset(self, button=None):
        if self._closed:
            return
        self._updating = True
        try:
            for control, value in zip(self._controls, self._defaults):
                control.value = value
            self._preferred = -1
            self._comparison_follows_order = True
            if self._array_ui is not None:
                self.openings.value = 1
        finally:
            self._updating = False
        self._update()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.order.unobserve(self._select_order, names='value')
        self.normal.on_click(self._normal_incidence, remove=True)
        self.reset.on_click(self._reset, remove=True)
        self.compare.unobserve(self._update_comparison, names='value')
        self.comparison.unobserve(self._update_comparison, names='value')
        self.use_order.on_click(self._use_selected_order, remove=True)
        self.picture.value = b''
        self.comparison_picture.value = b''
        self.figure.clear()
        self.comparison_figure.clear()
        self.figure = None
        self.comparison_figure = None
        self.data = None
        self.comparison_data = None
        self._field_cache.clear()
        self._array_cache.clear()
        self._comparison_signature = self._path_signature = None
        if self._paths_ui is not None:
            self.path_picture.value = b''
            self.path_figure.clear()
            self.path_figure = None
            self.path_data = None
            _close_widget_tree(self._paths_ui)
        if self._array_ui is not None:
            self.openings.unobserve(self._update_array, names='value')
            self.array_picture.value = b''
            self.array_figure.clear()
            self.array_figure = None
            self.array_data = None
            self._array_signature = None
            _close_widget_tree(self._array_ui)
        _close_widget_tree(self.ui)
