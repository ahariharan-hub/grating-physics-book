"""In-memory Plotly views for the 3D grating and standalone Ewald lessons."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .conical import detector_intersections
from .plotting import (geometry_figures, geometry_order_colour, geometry_segments,
                       geometry_wire, geometry_line3d)


INK = '#243444'
ORANGE = '#be641a'
GREEN = '#17806b'
PURPLE = '#83519d'
INCIDENT_CREST = '#df792a'  # Notebook 1: warm crests, blue troughs.
SIGNED_FIELD_COLOURS = 'RdBu_r'  # Notebooks 1, 2 and 4: -1 blue, +1 red.
RADIUS = 4.  # micrometres; a reference distance, not a propagation simulation
DETECTOR_HALF_WIDTH = 6.  # drawn patch of an unbounded reference plane


def _arrow(figure, start, vector, colour, name, *, show=True, width=5,
           row=1, col=1, role=None, legend=True, head_size=.23, update=False):
    """Add a line and arrowhead, retaining the same traces when hidden."""
    vector, start = np.asarray(vector), np.asarray(start)
    end = start+vector
    visible = show and np.linalg.norm(vector) > 1e-12
    points = np.column_stack([start, end]) if visible else np.empty((3, 0))
    if update:
        roles = _trace_roles(figure)
        roles[role].update(x=points[0], y=points[1], z=points[2], name=name,
                           line=dict(color=colour, width=width))
        roles[role+'-head'].update(
            x=[end[0]] if visible else [], y=[end[1]] if visible else [],
            z=[end[2]] if visible else [], u=[vector[0]] if visible else [],
            v=[vector[1]] if visible else [], w=[vector[2]] if visible else [],
            sizeref=head_size, colorscale=[[0, colour], [1, colour]])
        return
    line = geometry_line3d(points, colour, width, name=name)
    line.update(showlegend=legend, hoverinfo='name', meta={'role':role})
    figure.add_trace(line, row=row, col=col)
    figure.add_trace(go.Cone(
        x=[end[0]] if visible else [], y=[end[1]] if visible else [],
        z=[end[2]] if visible else [], u=[vector[0]] if visible else [],
        v=[vector[1]] if visible else [], w=[vector[2]] if visible else [],
        anchor='tip', sizemode='absolute', sizeref=head_size,
        colorscale=[[0, colour], [1, colour]], showscale=False,
        hoverinfo='skip', showlegend=False,
        meta={'role':f'{role}-head'}), row=row, col=col)


def _trace_roles(figure):
    return {trace.meta['role']: trace for trace in figure.data
            if isinstance(trace.meta, dict) and 'role' in trace.meta}


def _grating_bars(data, extent=4.5):
    """The same 50%-open screen in the incident and outgoing spatial views."""
    d = data['period']
    sites = np.arange(np.floor(-extent/d)-1, np.ceil(extent/d)+2)*d
    boundaries = sorted({-extent, extent, *[float(edge) for centre in sites
                         for edge in (centre-d/4, centre+d/4) if -extent < edge < extent]})
    vertices, triangles = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        middle = (left+right)/2
        if abs((middle+d/2) % d-d/2) > d/4:
            start = len(vertices)
            vertices.extend([[left, -extent, 0], [right, -extent, 0],
                             [right, extent, 0], [left, extent, 0]])
            triangles.extend([[start, start+1, start+2], [start, start+2, start+3]])
    vertices, triangles = np.asarray(vertices).reshape(-1, 3), np.asarray(triangles).reshape(-1, 3)
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color='#aab3be', flatshading=True, lighting=dict(ambient=1, diffuse=0),
        name='Grating bars; openings along y', showlegend=True, hoverinfo='name',
        meta={'role':'grating-bars'})


def _grating(figure, data, extent=4.5):
    # Positions use actual micrometres and the current period. The finite drawn
    # patch depicts the location of an indefinitely periodic grating.
    figure.add_trace(_grating_bars(data, extent), row=1, col=1)
    boundary = np.array([[-extent, -extent, 0], [extent, -extent, 0],
                         [extent, extent, 0], [-extent, extent, 0],
                         [-extent, -extent, 0]])
    figure.add_trace(geometry_line3d(boundary.T, '#b9c4ce', 1), row=1, col=1)
    figure.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers', marker=dict(size=4, color=INK),
        name='Grating origin', showlegend=False, hovertemplate='z = 0<extra></extra>'),
        row=1, col=1)


def _spatial_layout(figure, key):
    axis = dict(showbackground=False, nticks=5, showspikes=False,
                gridcolor='#e4e9ef', zerolinecolor='#b8c4cf')
    figure.update_layout(
        template='plotly_white', height=590, font=dict(family='Arial', size=12, color=INK),
        margin=dict(l=15, r=15, t=65, b=70), uirevision=key,
        legend=dict(orientation='h', x=.02, y=-.08),
        scene=dict(xaxis=dict(axis, title='x (µm)', range=[-4.7, 4.7]),
                   yaxis=dict(axis, title='y (µm); grooves', range=[-4.7, 4.7]),
                   zaxis=dict(axis, title='z (µm)', range=[-4.6, 4.6]),
                   aspectmode='cube', dragmode='orbit', uirevision=key,
                   camera=dict(eye=dict(x=1.5, y=-1.8, z=1.1))))
    for annotation in figure.layout.annotations:
        annotation.font.size = 13


def _incident_wavefronts(data, extent=4.5, height=6.):
    """Clip every crest plane s_i.r = n*lambda to the displayed box.

    Intersecting box edges avoids dividing by a small normal component near
    grazing. The grating-coincident plane at normal incidence has no unique
    intersection line and is represented by the uniform surface field instead.
    """
    si, wavelength = data['incident'], data['wavelength']
    corners = np.array([[x, y, z] for x in (-extent, extent)
                        for y in (-extent, extent) for z in (0., height)])
    edges = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]
    offsets = corners @ si
    u = np.cross(si, np.eye(3)[np.argmin(abs(si))])
    u /= np.linalg.norm(u)
    v = np.cross(si, u)
    fronts, footprints = [], []
    low = int(np.ceil(offsets.min()/wavelength-1e-12))
    high = int(np.floor(offsets.max()/wavelength+1e-12))
    for n in range(low, high+1):
        level, points = n*wavelength, []
        for i, j in edges:
            delta = offsets[j]-offsets[i]
            if abs(delta) < 1e-12:
                if abs(offsets[i]-level) < 1e-10:
                    points.extend([corners[i], corners[j]])
            else:
                fraction = (level-offsets[i])/delta
                if -1e-12 <= fraction <= 1+1e-12:
                    points.append(corners[i]+np.clip(fraction, 0, 1)*(corners[j]-corners[i]))
        if not points:
            continue
        points = np.unique(np.round(points, 12), axis=0)
        if len(points) < 3 or np.all(abs(points[:, 2]) < 1e-10):
            continue
        relative = points-points.mean(axis=0)
        points = points[np.argsort(np.arctan2(relative @ v, relative @ u))]
        fronts.append((n, points))
        bottom = points[abs(points[:, 2]) < 1e-10]
        if len(bottom) >= 2:
            distances = np.linalg.norm(bottom[:, None, :]-bottom[None, :, :], axis=2)
            i, j = np.unravel_index(np.argmax(distances), distances.shape)
            footprints.append(bottom[[i, j]])
    return fronts, footprints


def incidence_figure(data):
    """Incident wavefront planes, grating openings and their surface traces."""
    extent, height = 4.5, 6.
    figure = make_subplots(rows=1, cols=2, specs=[[{'type':'scene'}, {'type':'xy'}]],
        column_widths=[.63, .37], horizontal_spacing=.08,
        subplot_titles=['Incident wavefronts above the grating',
                        'Same wave at the surface: z = 0'])
    si, d = data['incident'], data['period']
    x = np.linspace(-extent, extent, 181)
    y = np.linspace(-extent, extent, 181)
    xx, yy = np.meshgrid(x, y)
    phase = data['k']*(si[0]*xx+si[1]*yy)
    field = np.cos(phase)
    opening = abs((xx+d/2) % d-d/2) <= d/4
    figure.add_trace(go.Surface(
        x=xx, y=yy, z=np.where(opening, 0., np.nan), surfacecolor=field,
        cmin=-1, cmax=1, colorscale=SIGNED_FIELD_COLOURS, showscale=False, connectgaps=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        name='Incident field in openings', meta={'role':'incident-opening-field'},
        hovertemplate='x=%{x:.2f} µm<br>y=%{y:.2f} µm<br>uᵢ/A=%{surfacecolor:.3f}<extra></extra>'),
        row=1, col=1)
    # A 50%-open screen makes the extension of the earlier slit picture clear.
    # It locates the grating only; this view does not compute the scattered field.
    figure.add_trace(_grating_bars(data, extent), row=1, col=1)

    fronts, footprints = _incident_wavefronts(data, extent, height)
    vertices, triangles, levels, rims = [], [], [], []
    for n, polygon in fronts:
        start = len(vertices)
        vertices.extend(polygon)
        levels.extend([n]*len(polygon))
        triangles.extend([[start, start+j, start+j+1] for j in range(1, len(polygon)-1)])
        rims.append(np.vstack([polygon, polygon[0]]))
    vertices, triangles = np.asarray(vertices).reshape(-1, 3), np.asarray(triangles).reshape(-1, 3)
    figure.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color=INCIDENT_CREST, opacity=min(.19, 1.3/(len(fronts)+1)), flatshading=True,
        lighting=dict(ambient=1, diffuse=0, specular=0), customdata=levels,
        name='Crest planes: uᵢ/A = +1', legendgroup='wavefronts', showlegend=True,
        meta={'role':'incident-wavefronts'},
        hovertemplate='Incident crest plane<br>phase / 2π = %{customdata:.0f}<extra></extra>'), row=1, col=1)
    rim = geometry_line3d(geometry_segments(rims), INCIDENT_CREST, 2)
    rim.update(legendgroup='wavefronts', meta={'role':'wavefront-rims'})
    figure.add_trace(rim, row=1, col=1)
    foot = geometry_segments(footprints)
    footprint = geometry_line3d(foot, ORANGE, 5, name='Wavefront traces at z = 0')
    footprint.update(showlegend=True, meta={'role':'wavefront-footprints'})
    figure.add_trace(footprint, row=1, col=1)
    # Keep the incoming arrow inside the box, including almost grazing incidence.
    distance = min(.82*extent/max(abs(si[0]), 1e-12),
                   .82*extent/max(abs(si[1]), 1e-12), .90*height/abs(si[2]))
    # End the direction arrow just above the screen so the opaque grating does
    # not hide its tip. Its line still lies on the ray aimed at the origin.
    _arrow(figure, -distance*si, (distance-.45)*si, GREEN, 'Incoming direction', width=7)
    figure.data[-1].sizeref = .9

    figure.add_trace(go.Heatmap(
        x=x, y=y, z=field, zmin=-1, zmax=1, colorscale=SIGNED_FIELD_COLOURS,
        colorbar=dict(title=dict(text='Signed incident field uᵢ/A', side='right'),
                      thickness=12, len=.62, tickvals=[-1, 0, 1], ticktext=['−1', '0', '+1']),
        customdata=phase, meta={'role':'incident-surface-field'},
        hovertemplate='x=%{x:.2f} µm<br>y=%{y:.2f} µm<br>phase=%{customdata:.3f} rad'
                      '<br>uᵢ/A=%{z:.3f}<extra></extra>'), row=1, col=2)
    centres = np.arange(np.ceil(-extent/d), np.floor(extent/d)+1)*d
    grooves = geometry_segments([[[site, -extent, 0], [site, extent, 0]] for site in centres])
    figure.add_trace(go.Scatter(x=grooves[0], y=grooves[1], mode='lines',
        line=dict(color='#ffffff', width=1, dash='dot'), showlegend=False,
        hoverinfo='skip'), row=1, col=2)
    figure.add_trace(go.Scatter(x=foot[0], y=foot[1], mode='lines',
        line=dict(color=ORANGE, width=3), showlegend=False, hoverinfo='skip',
        meta={'role':'wavefront-footprints-map'}), row=1, col=2)
    # Compress the plot area to keep equal physical scales. Letting Plotly
    # constrain ranges makes FigureWidget's partial updates alternately expand
    # y and crop x when a control restores only the range that changed.
    figure.update_xaxes(title='x (µm); across openings', range=[-extent, extent],
                       autorange=False, constrain='domain')
    figure.update_yaxes(title='y (µm); along openings', range=[-extent, extent],
                       scaleanchor='x', scaleratio=1, autorange=False, constrain='domain')
    _spatial_layout(figure, 'conical-incidence-wavefronts')
    figure.update_layout(height=650, margin=dict(l=15, r=25, t=65, b=100),
        legend=dict(orientation='h', x=0, y=-.10, groupclick='togglegroup'),
        scene=dict(zaxis=dict(range=[-.3, height+.2]), aspectmode='data',
                   camera=dict(eye=dict(x=1.45, y=-1.65, z=1.2))))
    return figure


def _wavevector_components(figure, data, *, update=False):
    """Real Cartesian projections; an imaginary kz is never drawn as a ray."""
    index = data['selected_index']
    vector = data['k']*data['components'][index]
    real = vector.real
    evanescent = not data['valid'][index]
    _arrow(figure, np.zeros(3), real, PURPLE,
           'Re(<b>k</b><sub>m</sub>): tangential phase vector' if evanescent else 'Selected wavevector <b>k</b><sub>m</sub>',
           row=2, col=2, role='selected-wavevector', legend=False,
           head_size=.085*data['k'], width=6, update=update)
    colours = ['#d67a17', '#2266a5', '#238c69']
    names = ['k<sub>mx</sub>', 'k<sub>my</sub>',
             'Re(k<sub>mz</sub>)' if evanescent else 'k<sub>mz</sub>']
    projections = np.diag(real)
    for axis, colour, name, component in zip('xyz', colours, names, projections):
        _arrow(figure, np.zeros(3), component, colour,
               f'{name} = {component["xyz".index(axis)]:+.3f} rad/µm',
               row=2, col=2, role=f'component-{axis}', legend=False,
               head_size=.065*data['k'], width=5, update=update)
    # The edges of the Cartesian box connect each projection to the full vector.
    guides = []
    for axis in range(3):
        for bits in ((0, 0), (0, 1), (1, 0), (1, 1)):
            start = np.zeros(3)
            others = [j for j in range(3) if j != axis]
            start[others] = real[others]*bits
            end = start.copy()
            end[axis] = real[axis]
            guides.append([start, end])
    guide = geometry_line3d(geometry_segments(guides), '#9aa6b3', 2, 'dash')
    guide.update(meta={'role':'component-guides'}, hoverinfo='skip', showlegend=False)
    if update:
        _trace_roles(figure)['component-guides'].update(x=guide.x, y=guide.y, z=guide.z)
    else:
        figure.add_trace(guide, row=2, col=2)
    nonzero = abs(real) > 1e-12
    endpoints = projections[nonzero]
    labels = go.Scatter3d(
        x=endpoints[:, 0], y=endpoints[:, 1], z=endpoints[:, 2], mode='text',
        text=np.asarray(names)[nonzero], textfont=dict(size=13, color=np.asarray(colours)[nonzero]),
        textposition='top center', hoverinfo='skip', showlegend=False,
        meta={'role':'component-labels'})
    if update:
        _trace_roles(figure)['component-labels'].update(labels.to_plotly_json())
    else:
        figure.add_trace(labels, row=2, col=2)
    half_span = .65*max(data['k'], np.max(abs(real)))
    axes = dict(autorange=False, showbackground=False,
                gridcolor='#e5eaf0', zerolinecolor='#b5bfcb', nticks=4)
    ranges = [[value/2-half_span, value/2+half_span] for value in real]
    if update:
        for axis, limits in zip('xyz', ranges):
            figure.layout.scene2[axis+'axis'].range = limits
        figure.layout.scene2.zaxis.title.text = names[2]+' (rad/µm)'
        return
    figure.update_layout(scene2=dict(
        xaxis=dict(axes, range=ranges[0], title='k<sub>mx</sub> (rad/µm)'),
        yaxis=dict(axes, range=ranges[1], title='k<sub>my</sub> (rad/µm)'),
        zaxis=dict(axes, range=ranges[2], title=names[2]+' (rad/µm)'),
        aspectmode='cube', camera=dict(eye=dict(x=1.45, y=-1.75, z=.8)),
        dragmode='orbit', uirevision='conical-components'))


def detector_readout(data, distance=3., show_detector=True):
    hits = detector_intersections(data, distance, DETECTOR_HALF_WIDTH)
    index = data['selected_index']
    text = (f'<b>Reference plane: z = −L = −{distance:.2f} µm.</b> '
            'Circles: fixed radius R = 4 µm; open squares: ray–plane intersections. ')
    if not show_detector:
        return text+'Plane and intersection markers are hidden.'
    if hits['reaches'][index]:
        x, y, z = hits['points'][index]
        text += f'Selected m = {data["chosen"]:+d}: (x, y, z) = ({x:+.3f}, {y:+.3f}, {z:+.3f}) µm. '
        if not hits['in_patch'][index]:
            text += '<b>This intersection is outside the drawn patch.</b> '
    elif data['grazing'][index]:
        text += 'The selected grazing ray is parallel to this plane; there is no finite intersection. '
    else:
        text += 'The selected evanescent order has no real outgoing ray or ray–plane intersection. '
    outside = data['labels'][hits['reaches'] & ~hits['in_patch']]
    if len(outside):
        text += 'Beyond the drawn ±6 µm patch: m = '+', '.join(f'{m:+d}' for m in outside)+'. '
    return text+'Marker positions show geometry, not intensity or spot size.'


def _component_title(data):
    index = data['selected_index']
    vector = data['k']*data['components'][index]
    kz = f'{vector[2].real:+.3f}' if data['valid'][index] else f'−{abs(vector[2].imag):.3f}i'
    component_title = (f'Selected m = {data["chosen"]:+d}: wavevector components<br>'
        f'<span style="font-size:12px">k<sub>mx</sub> = {vector[0].real:+.3f}, k<sub>my</sub> = {vector[1].real:+.3f}, k<sub>mz</sub> = {kz} rad/µm</span>')
    if not data['valid'][index]:
        component_title += '<br><sup>Only Re(<b>k</b><sub>m</sub>) is drawn; imaginary k<sub>mz</sub> gives decay.</sup>'
    return component_title


def _ray_lengths(data, distance, show_detector):
    directions = data['directions']
    lengths = np.full(len(directions), RADIUS)
    if show_detector:
        reaches = directions[:, 2] < 0
        travel = -distance/directions[reaches, 2]
        side = DETECTOR_HALF_WIDTH/np.maximum(np.max(abs(directions[reaches, :2]), axis=1), 1e-12)
        lengths[reaches] = np.maximum(RADIUS, np.minimum(travel, side))
    return lengths


def spatial_figure(data, show_cone=False, show_detector=True, detector_distance=3.):
    """Spatial rays and their plane intersections, with selected k projections."""
    index = data['selected_index']
    component_title = _component_title(data)
    figure = make_subplots(rows=2, cols=2,
        specs=[[{'type':'scene', 'rowspan':2}, {'type':'table'}], [None, {'type':'scene'}]],
        column_widths=[.61, .39], row_heights=[.39, .61],
        horizontal_spacing=.05, vertical_spacing=.15,
        subplot_titles=['Outgoing rays and a detector-like plane',
                        'Order candidates (components / k)', component_title])
    _grating(figure, data)
    si, directions = data['incident'], data['directions']
    _arrow(figure, -RADIUS*si, RADIUS*si, GREEN, 'Incident direction <b>k</b><sub>i</sub>/k')
    hits = detector_intersections(data, detector_distance, DETECTOR_HALF_WIDTH)
    # Extend toward the plane, clipping only the drawn ray to the spatial box.
    # Never move an off-patch intersection onto the patch edge.
    ray_lengths = _ray_lengths(data, detector_distance, show_detector)
    rays = geometry_segments([[np.zeros(3), length*point]
                             for length, point in zip(ray_lengths, directions)])
    ray_trace = geometry_line3d(rays, '#7b90a2', 3)
    ray_trace.meta = {'role':'outgoing-rays'}
    figure.add_trace(ray_trace, row=1, col=1)
    colours = [geometry_order_colour(m) for m in data['allowed']]
    symbols = ['diamond' if grazing else 'circle' for grazing in data['grazing'][data['valid']]]
    points = RADIUS*directions
    # Supply one label per point, including blanks: None is omitted from Plotly
    # updates and leaves the previous labels attached to newly positioned points.
    labels = [f'm={m:+d}' if len(points)<=13 or m in (0, data['chosen']) else ''
              for m in data['allowed']]
    figure.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers+text',
        text=labels,
        textposition='top center', textfont=dict(size=11),
        marker=dict(size=5, color=colours, symbol=symbols), showlegend=True,
        name='At radius R = 4 µm', meta={'role':'reference-endpoints'},
        customdata=np.column_stack([data['allowed'], directions]),
        hovertemplate='m=%{customdata[0]:+.0f}<br><b>k</b><sub>m</sub>/k=(%{customdata[1]:.3f}, '
                      '%{customdata[2]:.3f}, %{customdata[3]:.3f})<extra></extra>'),
        row=1, col=1)
    selected_length = ray_lengths[list(data['allowed']).index(data['chosen'])] if data['valid'][index] else RADIUS
    _arrow(figure, np.zeros(3), selected_length*data['components'][index].real, ORANGE,
           'Selected order direction <b>k</b><sub>m</sub>/k', show=bool(data['valid'][index]),
           width=7, role='selected-order-ray')
    angle = np.linspace(0, np.pi, 101)
    arc = np.array([data['rho']*np.cos(angle), np.full_like(angle, si[1]),
                    -data['rho']*np.sin(angle)])
    radius = np.linspace(0, RADIUS, 17)
    cone = radius[:, None, None]*arc.T[None, :, :]
    figure.add_trace(go.Surface(x=cone[:, :, 0], y=cone[:, :, 1], z=cone[:, :, 2],
        visible=show_cone, surfacecolor=np.zeros(cone.shape[:2]),
        colorscale=[[0, '#66b8a7'], [1, '#66b8a7']], opacity=.16,
        showscale=False, hoverinfo='skip', name='Cone of candidate directions',
        meta={'role':'cone-surface'}), row=1, col=1)
    arc_trace = geometry_line3d(RADIUS*arc, GREEN, 4, name='Common angle to grooves')
    arc_trace.update(visible=show_cone, showlegend=True, meta={'role':'cone-arc'})
    figure.add_trace(arc_trace, row=1, col=1)
    axis = geometry_line3d(np.array([[0, 0], [-RADIUS, RADIUS], [0, 0]]),
                           GREEN, 3, 'dash', 'Cone axis: y')
    axis.update(visible=show_cone, showlegend=True, meta={'role':'cone-axis'})
    figure.add_trace(axis, row=1, col=1)
    extent, depth = DETECTOR_HALF_WIDTH, detector_distance
    figure.add_trace(go.Mesh3d(
        x=[-extent, extent, extent, -extent], y=[-extent, -extent, extent, extent], z=[-depth]*4,
        i=[0, 0], j=[1, 2], k=[2, 3], color='#e6b56c', opacity=.12,
        lighting=dict(ambient=1, diffuse=0), flatshading=True,
        visible=show_detector, hoverinfo='skip', showlegend=False,
        meta={'role':'detector-plane'}), row=1, col=1)
    border = geometry_line3d(np.array([[-extent, extent, extent, -extent, -extent],
                                       [-extent, -extent, extent, extent, -extent], [-depth]*5]),
                             ORANGE, 3, name=f'Reference plane z = −{depth:.2f} µm')
    border.update(visible=show_detector, showlegend=True, hoverinfo='name', meta={'role':'detector-border'})
    figure.add_trace(border, row=1, col=1)
    mask = hits['in_patch'] & show_detector
    detector_points, orders = hits['points'][mask], data['labels'][mask]
    figure.add_trace(go.Scatter3d(
        x=detector_points[:, 0], y=detector_points[:, 1], z=detector_points[:, 2],
        mode='markers+text',
        text=[f'{m:+d}' if len(orders)<=9 or m in (0, data['chosen']) else '' for m in orders],
        textposition='bottom center', textfont=dict(size=11),
        marker=dict(size=[8 if m == data['chosen'] else 6 for m in orders], symbol='square-open',
                    color=[geometry_order_colour(m) for m in orders]),
        customdata=orders, name='At the reference plane', showlegend=True,
        meta={'role':'detector-hits'},
        hovertemplate='m=%{customdata:+.0f}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f}) µm<extra></extra>'),
        row=1, col=1)
    fills = ['#fff0d8' if m == data['chosen'] else '#f5f7fa' for m in data['labels']]
    kz_text = [f'{c.real:+.3f}' if abs(c.imag)<1e-12 else f'{c.imag:+.3f}i'
               for c in data['components'][:, 2]]
    figure.add_trace(go.Table(meta={'role':'order-table'},
        columnwidth=[.55, .8, .8, 1., 1.5],
        header=dict(values=['m', 'k<sub>mx</sub>/k', 'k<sub>my</sub>/k', 'k<sub>mz</sub>/k', 'State'], fill_color='#e5edf4',
                    align='left', font=dict(size=11), height=30),
        cells=dict(values=[[f'{m:+d}' for m in data['labels']],
                           [f'{a:+.3f}' for a in data['a']],
                           [f'{b:+.3f}' for b in data['b']], kz_text, data['states']],
                   fill_color=[fills]*5, align='left', font=dict(size=11), height=28)),
        row=1, col=2)
    _spatial_layout(figure, 'conical-spatial')
    # View the outgoing half-space from below the screen so opaque bars do not
    # cover the order endpoints. The notebook retains the user's later rotation.
    figure.layout.scene.camera.eye = dict(x=1.5, y=-1.8, z=-.95)
    _wavevector_components(figure, data)
    figure.update_layout(height=820, margin=dict(b=95), legend=dict(font=dict(size=11)))
    for axis in ('xaxis', 'yaxis'):
        figure.layout.scene[axis].range = [-6.4, 6.4]
    figure.layout.scene.zaxis.range = [-6.4, 4.6]
    figure.layout.scene.aspectmode = 'data'
    figure.update_annotations(font_size=13)
    return figure


def _update_selected_spatial_ray(figure, data, lengths):
    index = data['selected_index']
    real = bool(data['valid'][index])
    length = lengths[np.flatnonzero(data['allowed'] == data['chosen'])[0]] if real else RADIUS
    _arrow(figure, np.zeros(3), length*data['components'][index].real, ORANGE,
           'Selected order direction <b>k</b><sub>m</sub>/k', show=real, width=7,
           role='selected-order-ray', update=True)


def _update_detector_markers(figure, data, distance, show_detector):
    hits = detector_intersections(data, distance, DETECTOR_HALF_WIDTH)
    mask = hits['in_patch'] & show_detector
    points, orders = hits['points'][mask], data['labels'][mask]
    _trace_roles(figure)['detector-hits'].update(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], customdata=orders,
        text=[f'{m:+d}' if len(orders)<=9 or m in (0, data['chosen']) else '' for m in orders],
        marker=dict(size=[8 if m == data['chosen'] else 6 for m in orders],
                    color=[geometry_order_colour(m) for m in orders]))


def update_spatial_controls(figure, data, *, show_cone=False, show_detector=True,
                            detector_distance=3., cone_only=False):
    """Update spatial display controls without rebuilding wavevector geometry."""
    roles = _trace_roles(figure)
    if cone_only:
        for role in ('cone-surface', 'cone-arc', 'cone-axis'):
            roles[role].visible = show_cone
        return
    lengths = _ray_lengths(data, detector_distance, show_detector)
    rays = geometry_segments([[np.zeros(3), length*point]
                              for length, point in zip(lengths, data['directions'])])
    roles['outgoing-rays'].update(x=rays[0], y=rays[1], z=rays[2])
    _update_selected_spatial_ray(figure, data, lengths)
    roles['detector-plane'].update(z=[-detector_distance]*4, visible=show_detector)
    roles['detector-border'].update(z=[-detector_distance]*5, visible=show_detector,
                                    name=f'Reference plane z = −{detector_distance:.2f} µm')
    _update_detector_markers(figure, data, detector_distance, show_detector)


def update_spatial_selection(figure, data, *, show_detector=True, detector_distance=3.):
    """Move the selected ray, component arrows and table emphasis in place."""
    roles = _trace_roles(figure)
    _update_selected_spatial_ray(figure, data, _ray_lengths(data, detector_distance, show_detector))
    _wavevector_components(figure, data, update=True)
    figure.layout.annotations[-1].text = _component_title(data)
    fills = ['#fff0d8' if m == data['chosen'] else '#f5f7fa' for m in data['labels']]
    roles['order-table'].cells.fill.color = [fills]*5
    roles['reference-endpoints'].text = [
        f'm={m:+d}' if len(data['allowed'])<=13 or m in (0, data['chosen']) else ''
        for m in data['allowed']]
    _update_detector_markers(figure, data, detector_distance, show_detector)


def _direction_scale(data, coordinate_scale):
    """Positive multiplier of k_m/k, with labels for the displayed coordinates."""
    if coordinate_scale == 'direction':
        return (1., ('α', 'β', 'γ'), ('α = kₓ/k', 'β = kᵧ/k', 'γ = k_z/k'),
                'dimensionless', 'Direction cosines', '<b>k</b><sub>m</sub>/k')
    if coordinate_scale == 'frequency':
        labels = tuple(f'f<sub>{axis}</sub>' for axis in 'xyz')
        return (1/data['wavelength'], labels, labels, 'cycles/µm',
                'Spatial frequency (Goodman)', '<b>f</b><sub>m</sub>')
    if coordinate_scale == 'wavevector':
        labels = tuple(f'k<sub>{axis}</sub>' for axis in 'xyz')
        return (data['k'], labels, labels, 'rad/µm', 'Wavevector', '<b>k</b><sub>m</sub>')
    raise ValueError('Choose direction, frequency, or wavevector coordinates.')


def _scale_direction_figure(figure, data, coordinate_scale):
    """Convert the whole hemisphere/disk construction, including guides and hover."""
    radius, symbols, axes, units, title, _ = _direction_scale(data, coordinate_scale)
    for trace in figure.data:
        coordinates = 'xyz' if trace.type == 'scatter3d' else 'xy'
        for axis in coordinates:
            values = getattr(trace, axis)
            if values is not None:
                setattr(trace, axis, [None if v is None else radius*v for v in values])
        hover = trace.hovertemplate
        if hover and any(f'{symbol}=' in hover for symbol in ('α', 'β', 'γ')):
            for old, new in zip(('α', 'β', 'γ'), symbols):
                hover = hover.replace(old+'=', new+'=')
            trace.hovertemplate = hover.replace('<extra></extra>', f'<extra>{units}</extra>')
    unit_suffix = '' if coordinate_scale == 'direction' else f' ({units})'
    for axis, label in zip('xyz', axes):
        layout_axis = getattr(figure.layout.scene, axis+'axis')
        layout_axis.range = [radius*v for v in layout_axis.range]
        layout_axis.title.text = label+unit_suffix
        layout_axis.autorange = False
    for axis, label in zip('xy', axes):
        layout_axis = getattr(figure.layout, axis+'axis')
        layout_axis.range = [radius*v for v in layout_axis.range]
        layout_axis.title.text = label+unit_suffix
        layout_axis.autorange = False
        layout_axis.constrain = 'domain'
    figure.layout.annotations[0].text = (
        f'Outgoing hemisphere: {symbols[2]} ≤ 0; radius {radius:.3g}{unit_suffix}')
    figure.layout.annotations[1].text = (
        "Harvey's direction-cosine disk" if coordinate_scale == 'direction'
        else f'{title}: transverse projection')


def direction_scale_readout(data, coordinate_scale='direction'):
    """Show scale, order spacing, and the actual selected (possibly complex) vector."""
    radius, symbols, _, units, title, vector_name = _direction_scale(data, coordinate_scale)
    selected = radius*data['components'][data['selected_index']]
    def fmt(value):
        return f'{value.real:+.4f}' if abs(value.imag) < 1e-12 else f'{value.imag:+.4f}i'
    vector = '('+', '.join(fmt(v) for v in selected)+')'
    text = (f'<b>{title}</b> — {units}. Hemisphere/disk radius = {radius:.4f}; '
            f'neighbouring-order step Δ{symbols[0]} = {radius*data["spacing"]:.4f}; '
            f'common {symbols[1]} = {radius*data["incident"][1]:+.4f}.<br>'
            f'Selected m = {data["chosen"]:+d}: {vector_name} = {vector}. '
            'Axes rescale to fit the same construction; changing coordinate scale preserves directions and order states.')
    if not data['valid'][data['selected_index']]:
        text += ' The normal component is imaginary: it describes decay, with no real hemisphere point.'
    return text


def representation_figures(data, coordinate_scale='direction', *, views=('harvey', 'ewald')):
    """Reuse the established Harvey/Ewald constructions, reflected into -z.

    The old renderer expects a selected real direction on the upper hemisphere.
    Construct its base using a real order, reflect every spatial/normal
    coordinate, then place the actual selection. Evanescent selections retain
    a disk point and a rod, never a fictitious real ray or triangle.
    coordinate_scale affects only the hemisphere/disk; Ewald stays normalized.
    ``views`` requests only the needed figures, returned in the requested order.
    """
    views = tuple(views)
    if not views or len(set(views)) != len(views) or any(key not in ('harvey', 'ewald') for key in views):
        raise ValueError('Request harvey, ewald, or both without duplicates.')
    flip = np.array([1., 1., -1.])
    chosen = data['chosen']
    real_choice = chosen if chosen in data['allowed'] else int(data['allowed'][0])
    base = dict(data, incident=data['incident']*flip,
                directions=data['directions']*flip, changes=data['changes']*flip,
                chosen=real_choice)
    figures = dict(zip(views, geometry_figures(
        base, views=tuple('direction' if key == 'harvey' else 'ewald' for key in views)), strict=True))
    for key, figure in figures.items():
        for trace in figure.data:
            if trace.type in ('scatter3d', 'cone'):
                trace.z = [None if z is None else -z for z in trace.z]
                if trace.type == 'cone':
                    trace.w = [-w for w in trace.w]
            elif key == 'ewald' and trace.type == 'scatter':
                trace.y = [None if y is None else -y for y in trace.y]
        low, high = figure.layout.scene.zaxis.range
        figure.layout.scene.zaxis.range = [-high, -low]
        figure.layout.scene.camera = dict(eye=dict(x=1.5, y=-2., z=1.1))
        figure.layout.height = 580
        figure.layout.margin.b = 80
        for annotation in figure.layout.annotations:
            annotation.font.size = 13
    if 'harvey' in figures:
        harvey = figures['harvey']
        harvey.layout.annotations[0].text = 'Outgoing hemisphere: k_z < 0'
        harvey.layout.scene.zaxis.title.text = 'γ = k_z/k'
        harvey.layout.uirevision = 'conical-harvey'
    if 'ewald' in figures:
        ewald = figures['ewald']
        low, high = ewald.layout.yaxis.range
        ewald.layout.yaxis.range = [-high, -low]
        ewald.layout.uirevision = 'conical-ewald'
        ewald.layout.annotations[0].text = 'Ewald sphere: C = −kᵢ/k, radius 1'

    index = data['selected_index']
    real = bool(data['valid'][index])
    for figure in figures.values():
        for trace in figure.data:
            role = trace.meta.get('role') if isinstance(trace.meta, dict) else None
            if role == 'selected-disk':
                trace.update(x=[data['a'][index]], y=[data['b'][index]],
                             customdata=[[chosen]])
            elif role in ('selected-direction', 'selected-ray', 'selected-ewald',
                          'selected-section', 'outgoing-vector', 'outgoing-head',
                          'change-vector', 'change-head'):
                trace.visible = real
            # The incoming radius remains meaningful even without a real order.
    if 'ewald' in figures:
        x = chosen*data['spacing']
        zlim = ewald.layout.scene.zaxis.range
        rod = geometry_line3d([[x, x], [0, 0], zlim], ORANGE, 5, name='Selected order rod')
        rod.update(showlegend=True, hoverinfo='name', meta={'role':'selected-rod'})
        ewald.add_trace(rod, row=1, col=1)
        ewald.add_trace(go.Scatter(x=[x, x], y=zlim, mode='lines',
            line=dict(color=ORANGE, width=3), showlegend=False, hoverinfo='skip',
            meta={'role':'selected-rod-section'}), row=1, col=2)
        # Label rods at their upper ends, including candidates without intersections.
        ewald.add_trace(go.Scatter(
            x=data['labels']*data['spacing'], y=[zlim[1]-.07]*len(data['labels']),
            mode='text', text=[f'{m:+d}' if len(data['labels'])<=15 or m in (0, chosen) else ''
                               for m in data['labels']],
            textposition='bottom center', textfont=dict(size=10), showlegend=False,
            hoverinfo='skip', meta={'role':'rod-labels'}), row=1, col=2)
    if 'harvey' in figures:
        _scale_direction_figure(harvey, data, coordinate_scale)
    return tuple(figures[key] for key in views)


def update_representation_selection(figure, data, key, coordinate_scale='direction'):
    """Move selection traces while retaining the physical construction and camera."""
    roles = _trace_roles(figure)
    chosen, index = data['chosen'], data['selected_index']
    real = bool(data['valid'][index])
    selected = data['components'][index].real
    colour = geometry_order_colour(chosen)
    if key == 'harvey':
        scale = _direction_scale(data, coordinate_scale)[0]
        point = scale*selected
        roles['selected-disk'].update(x=[point[0]], y=[point[1]], customdata=[[chosen]])
        roles['selected-direction'].visible = real
        roles['selected-ray'].visible = real
        if real:
            roles['selected-direction'].update(x=[point[0]], y=[point[1]], z=[point[2]],
                                                customdata=[[chosen]])
            ray = geometry_segments([[np.zeros(3), point]])
            roles['selected-ray'].update(x=ray[0], y=ray[1], z=ray[2], line=dict(color=colour))
        return
    if key != 'ewald':
        raise ValueError('Choose harvey or ewald.')
    q = selected-data['incident']
    for role in ('selected-ewald', 'selected-section', 'outgoing-vector', 'outgoing-head',
                 'change-vector', 'change-head'):
        roles[role].visible = real
    if real:
        roles['selected-ewald'].update(x=[q[0]], y=[q[1]], z=[q[2]], customdata=[[chosen]])
        roles['selected-section'].update(x=[q[0]], y=[q[2]], customdata=[[chosen]])
        for role, start, vector, color, name in (
                ('outgoing', -data['incident'], selected, colour, f'Outgoing km/k (m={chosen:+d})'),
                ('change', np.zeros(3), q, '#914e97', 'Change Δk/k')):
            xyz = geometry_segments([[start, start+vector]])
            roles[role+'-vector'].update(x=xyz[0], y=xyz[1], z=xyz[2], name=name,
                                         line=dict(color=color))
            end = start+vector
            nonzero = np.linalg.norm(vector)>1e-12
            roles[role+'-head'].update(
                x=[end[0]] if nonzero else [], y=[end[1]] if nonzero else [],
                z=[end[2]] if nonzero else [], u=[vector[0]] if nonzero else [],
                v=[vector[1]] if nonzero else [], w=[vector[2]] if nonzero else [],
                colorscale=[[0, color], [1, color]])
    rod_x = chosen*data['spacing']
    roles['selected-rod'].x = [rod_x, rod_x]
    roles['selected-rod-section'].x = [rod_x, rod_x]
    roles['rod-labels'].text = [f'{m:+d}' if len(data['labels'])<=15 or m in (0, chosen) else ''
                               for m in data['labels']]


def selection_readout(data):
    """Shared physical status without introducing Ewald notation too early."""
    i = data['selected_index']
    selected = data['components'][i]
    def fmt(c):
        return f'{c.real:+.4f}' if abs(c.imag)<1e-12 else f'{c.imag:+.4f}i'
    vector = '('+', '.join(fmt(c) for c in selected)+')'
    propagating = ', '.join(f'{m:+d}' for m in data['labels'][data['valid'] & ~data['grazing']])
    grazing = ', '.join(f'{m:+d}' for m in data['labels'][data['grazing']]) or 'none'
    state = data['states'][i]
    if state == 'evanescent':
        ell = 1/(-selected[2].imag*data['k'])
        note = f'No real outgoing ray. Amplitude decay length ℓ = {ell:.4f} µm.'
    elif state == 'grazing':
        note = 'Grazing: the direction lies in z = 0.'
    else:
        note = 'The selected order propagates into z < 0.'
    theta = np.rad2deg(np.arccos(-data['incident'][2]))
    phi = ('undefined at normal incidence' if np.hypot(*data['incident'][:2])<1e-12
           else f'{np.rad2deg(np.arctan2(data["incident"][1], data["incident"][0])):+.2f}°')
    return (f'<b>m = {data["chosen"]:+d}: {state}</b> &nbsp; <b>k</b><sub>m</sub>/k = {vector}<br>'
            f'{note}<br>λ/d = {data["spacing"]:.3f}; '
            f'k = {data["k"]:.3f} rad/µm; k<sub>my</sub>/k = k<sub>iy</sub>/k = {data["incident"][1]:+.3f}. '
            f'Incident θ<sub>i</sub> = {theta:.2f}° from −z; azimuth φ<sub>i</sub> = {phi}.<br>'
            f'<b>Propagating m:</b> {propagating}. <b>Grazing m:</b> {grazing}. '
            'Candidate positions do not specify amplitudes or powers.')
