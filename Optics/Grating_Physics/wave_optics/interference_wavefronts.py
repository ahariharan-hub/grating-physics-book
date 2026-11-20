"""Combined wavefront, detector and wavevector views for notebook seven.

Coordinates are a wavelength-scale window about the actual detector centre P.
The field and relative phase come from the same model as detector_ui. The
common optical time is chosen so that beam 1 has a crest at P in each snapshot.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .conical_plotting import INK, SIGNED_FIELD_COLOURS, _arrow, _trace_roles
from .plotting import geometry_line3d, geometry_segments
from .detector_interference import detector_field


BEAM_COLOURS = ('#287bb5', '#df792a')
VIEW_OPTIONS = [('Beam 1 field', 'field_1'), ('Beam 2 field', 'field_2'),
                ('Combined intensity', 'intensity')]


def _crest_polygons(direction, wavelength, phase, bounds):
    """Intersect constant-phase planes with a box, as in incidence_figure."""
    direction = np.asarray(direction, dtype=float)
    corners = np.array([[x, y, z] for x in bounds[0]
                        for y in bounds[1] for z in bounds[2]])
    edges = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4)
             if i < (i ^ bit)]
    offsets = corners @ direction / wavelength + phase / (2*np.pi)
    u = np.cross(direction, np.eye(3)[np.argmin(abs(direction))])
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    fronts = []
    for n in range(int(np.ceil(offsets.min()-1e-10)),
                   int(np.floor(offsets.max()+1e-10))+1):
        points = []
        for i, j in edges:
            delta = offsets[j]-offsets[i]
            if abs(delta) < 1e-12:
                if abs(offsets[i]-n) < 1e-10:
                    points.extend([corners[i], corners[j]])
            else:
                fraction = (n-offsets[i])/delta
                if -1e-10 <= fraction <= 1+1e-10:
                    points.append(corners[i]+np.clip(fraction, 0, 1)*(corners[j]-corners[i]))
        if not points:
            continue
        points = np.unique(np.round(points, 12), axis=0)
        # Beam 1's crest at P coincides with the entire detector plane; avoid
        # covering its calculated colour map with a second opaque surface.
        if len(points) < 3 or np.all(abs(points[:, 2]) < 1e-10):
            continue
        relative = points-points.mean(axis=0)
        order = np.argsort(np.arctan2(relative @ v, relative @ u))
        fronts.append((n, points[order]))
    return fronts


def wavefront_data(geometry, *, wavelength_nm=632.8, power_mw=1.,
                   crossing_diameter_mm=.54, mm_per_unit=1.8,
                   optics_phase_deg=0., eo_phase_deg=0., phase_varies=False):
    """Actual detector fields and phase planes in local coordinates, in μm."""
    if not np.isfinite(eo_phase_deg):
        raise ValueError('The EO phase must be finite.')
    parameters = dict(wavelength_nm=wavelength_nm, power_mw=power_mw,
                      crossing_diameter_mm=crossing_diameter_mm,
                      mm_per_unit=mm_per_unit, optics_phase_deg=optics_phase_deg+eo_phase_deg)
    point = detector_field(0., 0., geometry, **parameters)
    wavelength = wavelength_nm*1e-3
    extent = 5*wavelength
    coordinates = np.linspace(-extent, extent, 161)
    xx, yy = np.meshgrid(coordinates, coordinates)
    fields = detector_field(xx*1e-3, yy*1e-3, geometry, **parameters)
    phase_at_p = float(np.angle(np.exp(1j*point['phase_difference'])))
    directions = (geometry['beam1_direction'], geometry['outgoing'])
    # z-z_D is positive on the incoming side of the detector.
    bounds = ((-extent, extent), (-extent, extent), (0., 3.25*wavelength))
    fronts = [_crest_polygons(direction, wavelength, phase, bounds)
              for direction, phase in zip(directions, (0., phase_at_p))]
    profile_x = np.linspace(-extent, extent, 1281)
    profile = detector_field(profile_x*1e-3, 0., geometry, **parameters)
    beta = np.deg2rad(eo_phase_deg)
    for sample in (fields, profile, point):
        # Crest planes/signed fields describe the selected EO phase. A detector
        # may instead average intensities from zero and the selected phase.
        if phase_varies:
            reference = np.abs(sample['field_1'] + sample['field_2']*np.exp(-1j*beta))**2
            sample['recorded_intensity'] = (reference + sample['intensity'])/2
        else:
            sample['recorded_intensity'] = sample['intensity']
    magnitude = float(abs((1+np.exp(1j*beta))/2)) if phase_varies else 1.
    i1, i2 = float(point['intensity_1']), float(point['intensity_2'])
    coherence = magnitude if i1 > 0 and i2 > 0 else None
    visibility = 2*np.sqrt(i1)*np.sqrt(i2)/(i1+i2)*magnitude if i1+i2 > 0 else None
    wavevectors = 2*np.pi/wavelength*np.asarray(directions)
    delta_k = wavevectors[1]-wavevectors[0]
    amplitude_limit = max(1., np.sqrt(power_mw)*(.54/crossing_diameter_mm))
    return dict(coordinates=coordinates, xx=xx, yy=yy, fields=fields,
                point=point, phase_at_p=phase_at_p, directions=directions,
                fronts=fronts, bounds=bounds, wavelength=wavelength,
                profile_x_um=profile_x, profile=profile,
                wavevectors_per_um=wavevectors, delta_k_per_um=delta_k,
                fringe_spacing_um=float(2*np.pi/abs(delta_k[0])),
                phase_at_point_deg=float(np.rad2deg(point['phase_difference']) % 360),
                amplitude_limit=amplitude_limit,
                eo_phase_deg=float(eo_phase_deg), phase_varies=bool(phase_varies),
                coherence_magnitude=coherence, visibility_at_point=visibility,
                intensity_limit=max(4., 4*power_mw*(.54/crossing_diameter_mm)**2),
                theta_deg=float(geometry['theta_deg']))


def _wavevector_inset(figure, data, *, update=False):
    """Common-tail k1/k, k2/k; the tip-to-tip arrow is (k2-k1)/k."""
    roles = _trace_roles(figure) if update else None
    annotation_index = 4

    def add_trace(trace, **position):
        if not update:
            figure.add_trace(trace, **position)
        elif isinstance(trace.meta, dict):
            roles[trace.meta['role']].update(x=trace.x, y=trace.y,
                                           hovertemplate=trace.hovertemplate)
        else:
            index = next(i for i, t in enumerate(figure.data)
                         if t is roles['delta-k-vector'])
            figure.data[index+1].update(x=trace.x, y=trace.y)

    def add_annotation(**properties):
        nonlocal annotation_index
        if update:
            figure.layout.annotations[annotation_index].update(properties)
            annotation_index += 1
        else:
            figure.add_annotation(**properties)

    first, second = (np.asarray(v)[[0, 2]] for v in data['directions'])
    purple = '#83519d'
    origin = np.zeros(2)
    for start, end, colour, role, name in (
            (origin, first, BEAM_COLOURS[0], 'k1-vector', 'k₁ / k'),
            (origin, second, BEAM_COLOURS[1], 'k2-vector', 'k₂ / k'),
            (first, second, purple, 'delta-k-vector', 'Δk / k')):
        vector = end-start
        add_trace(go.Scatter(
            x=[start[0], end[0]], y=[start[1], end[1]], mode='lines',
            line=dict(color=colour, width=2.5), name=name,
            meta={'role': role}, showlegend=False,
            hovertemplate=f'{name}: ({vector[0]:+.4f}, 0, {vector[1]:+.4f})<extra></extra>'),
            row=2, col=1)
        add_annotation(x=end[0], y=end[1], ax=start[0], ay=start[1],
            xref='x2', yref='y2', axref='x2', ayref='y2',
            text='', showarrow=True, arrowhead=3, arrowsize=1.15,
            arrowwidth=2.5, arrowcolor=colour, standoff=0)
    for x, z, label, colour in (
            (.12, -.5, 'k₁', BEAM_COLOURS[0]),
            (second[0]/2-.15, second[1]/2, 'k₂', BEAM_COLOURS[1]),
            (second[0]/2, -1.16, 'Δk', purple),
            (.10, .08, 'O', INK)):
        add_annotation(x=x, y=z, xref='x2', yref='y2', text=label,
                              showarrow=False, font=dict(color=colour, size=14))
    angle = np.linspace(-np.pi/2-np.deg2rad(data['theta_deg']), -np.pi/2, 31)
    add_trace(go.Scatter(
        x=.28*np.cos(angle), y=.28*np.sin(angle), mode='lines',
        line=dict(color='#82909c', width=1), hoverinfo='skip', showlegend=False),
        row=2, col=1)
    middle = np.mean(angle)
    add_annotation(x=.42*np.cos(middle), y=.42*np.sin(middle),
                          xref='x2', yref='y2', text='θ', showarrow=False,
                          font=dict(color='#667580', size=12))
    delta = second-first
    add_annotation(x=.26, y=.25, xref='paper', yref='paper',
        xanchor='left', yanchor='top', align='left', showarrow=False,
        font=dict(size=12, color=INK),
        text=('<b>Δk = k₂ − k₁</b><br>Arrow: tip 1 → tip 2<br><br>'
              f'Δk<sub>x</sub>/k = {delta[0]:+.4f}<br>'
              f'Δk<sub>z</sub>/k = {delta[1]:+.4f}<br>'
              f'Λ<sub>x</sub> = {data["fringe_spacing_um"]:.2f} μm'))


def phase_shifted_data(source, eo_phase_deg=0., *, phase_varies=False):
    """Apply an EO phase to cached zero-EO fields, sharing unchanged arrays."""
    if not np.isfinite(eo_phase_deg):
        raise ValueError('The EO phase must be finite.')
    if source['eo_phase_deg'] != 0 or source['phase_varies']:
        raise ValueError('Supply steady zero-EO reference data.')
    beta = np.deg2rad(eo_phase_deg)
    data = source.copy()
    for key in ('fields', 'profile', 'point'):
        original = source[key]
        sample = original.copy()
        sample['field_2'] = original['field_2']*np.exp(1j*beta)
        sample['phase_difference'] = original['phase_difference']+beta
        sample['intensity'] = np.abs(sample['field_1']+sample['field_2'])**2
        sample['recorded_intensity'] = ((original['intensity']+sample['intensity'])/2
                                        if phase_varies else sample['intensity'])
        data[key] = sample
    data['phase_at_p'] = float(np.angle(np.exp(1j*data['point']['phase_difference'])))
    data['phase_at_point_deg'] = float(np.rad2deg(data['point']['phase_difference']) % 360)
    data['fronts'] = [source['fronts'][0], _crest_polygons(
        source['directions'][1], source['wavelength'], data['phase_at_p'], source['bounds'])]
    magnitude = float(abs((1+np.exp(1j*beta))/2)) if phase_varies else 1.
    data.update(eo_phase_deg=float(eo_phase_deg), phase_varies=bool(phase_varies),
                coherence_magnitude=(magnitude if source['coherence_magnitude'] is not None else None),
                visibility_at_point=(source['visibility_at_point']*magnitude
                                     if source['visibility_at_point'] is not None else None))
    data.pop('surface_values', None)
    return data


def _display_values(data, mode):
    if mode == 'intensity':
        return (data['fields']['recorded_intensity'], 0., data['intensity_limit'],
                [[0, '#000000'], [1, '#ffffff']], 'Relative scalar intensity',
                'Both beams: exposure-averaged intensity' if data['phase_varies']
                else 'Both beams: time-averaged intensity')
    if mode not in ('field_1', 'field_2'):
        raise ValueError('Select field_1, field_2 or intensity.')
    return (data['fields'][mode].real, -data['amplitude_limit'], data['amplitude_limit'],
            SIGNED_FIELD_COLOURS, f'Signed field u{mode[-1]} (reference units)',
            f'Beam {mode[-1]}: signed field at one instant')


def _front_coordinates(fronts):
    vertices, triangles, levels, rims = [], [], [], []
    for n, polygon in fronts:
        start = len(vertices)
        vertices.extend(polygon)
        triangles.extend([[start, start+j, start+j+1] for j in range(1, len(polygon)-1)])
        levels.extend([n]*len(polygon))
        rims.append(np.vstack([polygon, polygon[0]]))
    vertices = np.asarray(vertices).reshape(-1, 3)
    triangles = np.asarray(triangles).reshape(-1, 3)
    return (dict(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                 i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2], customdata=levels),
            dict(zip(('x', 'y', 'z'), geometry_segments(rims), strict=True)))


def update_wavefront_scene(figure, data, *, mode='intensity', change='geometry'):
    """Patch a retained scene; display changes need no new fields or planes."""
    roles = _trace_roles(figure)
    values, low, high, colours, _, _ = _display_values(data, mode)
    data['surface_values'] = values
    surface = roles['detector-surface']
    surface.surfacecolor = values
    if change == 'mode':
        surface.update(cmin=low, cmax=high, colorscale=colours)
    for index, direction in enumerate(data['directions'], start=1):
        if change != 'mode' and (index == 2 or change == 'geometry'):
            mesh, edges = _front_coordinates(data['fronts'][index-1])
            roles[f'beam-{index}-crests'].update(mesh)
            roles[f'beam-{index}-rims'].update(edges)
        if change in ('geometry', 'mode'):
            visible = mode in ('intensity', f'field_{index}')
            roles[f'beam-{index}-crests'].visible = visible
            roles[f'beam-{index}-rims'].visible = visible
            length = 2.65*data['wavelength']
            extent = data['bounds'][1][1]
            tip = np.array([0., (1 if index == 1 else -1)*.65*extent,
                            .20*data['wavelength']])
            _arrow(figure, tip-length*direction, length*direction, BEAM_COLOURS[index-1],
                   f'Beam {index} direction', show=visible, legend=False,
                   head_size=.35*data['wavelength'], width=6,
                   role=f'beam-{index}-direction', update=True)


def update_wavefront_figure(figure, data, *, mode='intensity', change='geometry'):
    """Refresh affected traces in place, preserving camera, axes and layout."""
    roles = _trace_roles(figure)
    values, low, high, colours, colour_title, map_title = _display_values(data, mode)
    with figure.batch_update():
        update_wavefront_scene(figure, data, mode=mode, change=change)
        roles['detector-map'].z = values
        if change == 'mode':
            roles['detector-map'].update(zmin=low, zmax=high, colorscale=colours)
            roles['detector-map'].colorbar.title.text = colour_title
        figure.layout.annotations[0].text = (
            f'Wavefront snapshot · EO at {data["eo_phase_deg"]:g}°'
            if data['eo_phase_deg'] or data['phase_varies'] else 'Wavefronts at the detector')
        figure.layout.annotations[1].text = map_title
        if change != 'mode':
            roles['profile-intensity'].y = data['profile']['recorded_intensity']
            roles['profile-point'].y = [float(data['point']['recorded_intensity'])]
        if change in ('geometry', 'position'):
            for key in ('intensity_1', 'intensity_2'):
                roles['profile-'+key].y = data['profile'][key]
        if change == 'geometry':
            _wavevector_inset(figure, data, update=True)


def wavefront_figure(geometry=None, *, mode='intensity', data=None, **parameters):
    """Shared detector patch, intensity profile and normalized vector triangle."""
    if mode not in dict(VIEW_OPTIONS).values():
        raise ValueError('Select field_1, field_2 or intensity.')
    data = wavefront_data(geometry, **parameters) if data is None else data
    fields = data['fields']
    if mode == 'intensity':
        values = fields['recorded_intensity']
        low, high = 0., data['intensity_limit']
        colours = [[0, '#000000'], [1, '#ffffff']]
        colour_title = 'Relative scalar intensity'
        map_title = ('Both beams: exposure-averaged intensity' if data['phase_varies']
                     else 'Both beams: time-averaged intensity')
    else:
        values = fields[mode].real
        low, high = -data['amplitude_limit'], data['amplitude_limit']
        colours = SIGNED_FIELD_COLOURS
        number = mode[-1]
        colour_title = f'Signed field u{number} (reference units)'
        map_title = f'Beam {number}: signed field at one instant'
    data['surface_values'] = values
    wavefront_title = (f'Wavefront snapshot · EO at {data["eo_phase_deg"]:g}°'
                      if data['eo_phase_deg'] or data['phase_varies']
                      else 'Wavefronts at the detector')
    figure = make_subplots(rows=2, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}]],
        column_widths=[.57, .43], row_heights=[.66, .34],
        horizontal_spacing=.10, vertical_spacing=.16,
        subplot_titles=[wavefront_title, map_title,
                        'Wavevector difference', 'Intensity through P · y = 0'])
    zero = np.zeros_like(data['yy'])
    figure.add_trace(go.Surface(
        x=data['xx'], y=data['yy'], z=zero, surfacecolor=values,
        cmin=low, cmax=high, colorscale=colours, showscale=False,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        name='Detector slice: z = z_D', meta={'role': 'detector-surface'},
        hovertemplate='Detector z = z_D<br>x=%{x:.3f} μm<br>y=%{y:.3f} μm'
                      '<br>value=%{surfacecolor:.3f}<extra></extra>'), row=1, col=1)

    extent = data['bounds'][1][1]
    boundary = np.array([[-extent, -extent, 0], [extent, -extent, 0],
                         [extent, extent, 0], [-extent, extent, 0],
                         [-extent, -extent, 0]])
    rim = geometry_line3d(boundary.T, '#738291', 3, name='Detector: z = z_D')
    rim.update(showlegend=True, meta={'role': 'detector-outline'})
    figure.add_trace(rim, row=1, col=1)
    for index, (fronts, colour, direction) in enumerate(zip(
            data['fronts'], BEAM_COLOURS, data['directions']), start=1):
        visible = mode in ('intensity', f'field_{index}')
        vertices, triangles, levels, rims = [], [], [], []
        for n, polygon in fronts:
            start = len(vertices)
            vertices.extend(polygon)
            triangles.extend([[start, start+j, start+j+1]
                              for j in range(1, len(polygon)-1)])
            levels.extend([n]*len(polygon))
            rims.append(np.vstack([polygon, polygon[0]]))
        vertices = np.asarray(vertices).reshape(-1, 3)
        triangles = np.asarray(triangles).reshape(-1, 3)
        figure.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
            color=colour, opacity=.07, flatshading=True,
            lighting=dict(ambient=1, diffuse=0, specular=0),
            customdata=levels, visible=visible, showlegend=True,
            name=f'Beam {index} crests', legendgroup=f'beam-{index}',
            meta={'role': f'beam-{index}-crests'},
            hovertemplate=f'Beam {index} crest plane'
                          '<br>phase / 2π = %{customdata:.0f}<extra></extra>'),
            row=1, col=1)
        edges = geometry_line3d(geometry_segments(rims), colour, 2)
        edges.update(visible=visible, legendgroup=f'beam-{index}',
                     meta={'role': f'beam-{index}-rims'})
        figure.add_trace(edges, row=1, col=1)
        length = 2.65*data['wavelength']
        tip = np.array([0., (1 if index == 1 else -1)*.65*extent,
                        .20*data['wavelength']])
        _arrow(figure, tip-length*direction, length*direction, colour,
               f'Beam {index} direction', show=visible, legend=False,
               head_size=.35*data['wavelength'], width=6,
               role=f'beam-{index}-direction')

    figure.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers+text', text=['P'],
        textposition='top center', textfont=dict(color='#18b7c3'),
        marker=dict(size=3, color='#18b7c3'), showlegend=False,
        hovertemplate='Detector centre P = (0, 0, z_D)<extra></extra>'), row=1, col=1)
    figure.add_trace(go.Heatmap(
        x=data['coordinates'], y=data['coordinates'], z=values,
        zmin=low, zmax=high, colorscale=colours,
        colorbar=dict(title=dict(text=colour_title, side='right'),
                      thickness=13, len=.51, y=.74),
        meta={'role': 'detector-map'},
        hovertemplate='x=%{x:.3f} μm<br>y=%{y:.3f} μm'
                      '<br>value=%{z:.3f}<extra></extra>'), row=1, col=2)
    figure.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers+text', text=['P'], textposition='top right',
        textfont=dict(color='#18b7c3'),
        marker=dict(size=8, color='#18b7c3', symbol='cross'),
        showlegend=False, hoverinfo='skip'), row=1, col=2)
    _wavevector_inset(figure, data)
    for key, colour, dash, name in (
            ('intensity_1', BEAM_COLOURS[0], 'dash', 'Beam 1 alone'),
            ('intensity_2', BEAM_COLOURS[1], 'dot', 'Beam 2 alone'),
            ('intensity', INK, 'solid', 'Both beams')):
        figure.add_trace(go.Scatter(
            x=data['profile_x_um'],
            y=data['profile']['recorded_intensity' if key == 'intensity' else key], mode='lines',
            line=dict(color=colour, width=2, dash=dash), name=name,
            showlegend=True, meta={'role': 'profile-'+key},
            hovertemplate='x=%{x:.3f} μm<br>I=%{y:.4f}<extra>%{fullData.name}</extra>'),
            row=2, col=2)
    figure.add_trace(go.Scatter(
        x=[0], y=[float(data['point']['recorded_intensity'])], mode='markers',
        marker=dict(color='#18b7c3', size=7), showlegend=False,
        name='Intensity at P', meta={'role': 'profile-point'},
        hovertemplate='P: I=%{y:.4f}<extra></extra>'), row=2, col=2)
    axis = dict(showbackground=False, showspikes=False, nticks=5,
                gridcolor='#e4e9ef', zerolinecolor='#b8c4cf')
    figure.update_layout(
        template='plotly_white', height=850,
        font=dict(family='Arial', size=12, color=INK),
        margin=dict(l=45, r=25, t=55, b=100),
        uirevision='two-arm-wavefronts',
        legend=dict(orientation='h', x=.04, y=-.10, itemclick=False, itemdoubleclick=False),
        scene=dict(
            xaxis=dict(axis, title='x (μm)', range=[-extent, extent]),
            yaxis=dict(axis, title='y (μm)', range=[-extent, extent]),
            zaxis=dict(axis, title='z − z_D (μm)', range=[-.12*extent, data['bounds'][2][1]]),
            aspectmode='data', dragmode='orbit', uirevision='two-arm-wavefronts',
            camera=dict(eye=dict(x=2., y=-2.35, z=1.85),
                        up=dict(x=0, y=0, z=1))))
    figure.update_xaxes(title='x (μm)', range=[-extent, extent],
                       autorange=False, constrain='domain', row=1, col=2)
    figure.update_yaxes(title='y (μm)', range=[-extent, extent],
                       scaleanchor='x', scaleratio=1, autorange=False,
                       constrain='domain', row=1, col=2)
    # Keep the vector triangle to the left of its explanation in this cell.
    figure.update_xaxes(title='kₓ / k', range=[-.60, .35], domain=[.035, .20],
                       autorange=False, constrain='domain', showgrid=False,
                       zeroline=False, tickvals=[-.5, 0], row=2, col=1)
    figure.update_yaxes(title='k<sub>z</sub> / k', range=[-1.30, .18], scaleanchor='x2',
                       scaleratio=1, autorange=False, constrain='domain',
                       showgrid=False, zeroline=False, tickvals=[-1, -.5, 0],
                       row=2, col=1)
    figure.layout.annotations[2].x = .15
    figure.update_xaxes(title='x (μm)', range=[-extent, extent], autorange=False,
                       matches='x', row=2, col=2)
    figure.update_yaxes(title='Relative intensity', range=[0, data['intensity_limit']*1.06],
                       autorange=False, row=2, col=2)
    return figure, data
