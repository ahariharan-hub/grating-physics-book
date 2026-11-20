"""Signed path differences for the notebook's transmitted x-z geometry."""

import numpy as np


def path_difference_data(period, incident_angle, outgoing_angle):
    """Compare complete routes from an incident plane to an outgoing plane.

    A=(0,0), B=(d,0). Both unit directions point towards negative z.
    The reference planes are perpendicular to their respective directions.
    delta_length is route A minus route B; its outgoing contribution is
    d*sin(theta_out), and its incoming contribution is -d*sin(theta_i).
    No finite observation distance or scattering amplitude is assumed.
    """
    if (not np.all(np.isfinite([period, incident_angle, outgoing_angle]))
            or period <= 0 or abs(incident_angle) >= 90 or abs(outgoing_angle) > 90):
        raise ValueError('Use d > 0, |incident angle| < 90 and |outgoing angle| <= 90.')
    positions = np.array([[0., 0.], [period, 0.]])

    def direction(angle):
        radians = np.deg2rad(angle)
        return np.array([np.sin(radians), -np.cos(radians)])

    incident, outgoing = direction(incident_angle), direction(outgoing_angle)
    incoming_projection, outgoing_projection = positions @ incident, positions @ outgoing
    margin = .65*period
    incident_plane = incoming_projection.min()-margin
    outgoing_plane = outgoing_projection.max()+margin
    incoming_lengths = incoming_projection-incident_plane
    outgoing_lengths = outgoing_plane-outgoing_projection
    sources = positions-incoming_lengths[:, None]*incident
    ends = positions+outgoing_lengths[:, None]*outgoing
    incident_projection = float(incoming_projection[1]-incoming_projection[0])
    outgoing_projection = float(outgoing_projection[1]-outgoing_projection[0])
    return dict(period=float(period), positions=positions, incident=incident, outgoing=outgoing,
                sources=sources, ends=ends, incoming_lengths=incoming_lengths,
                outgoing_lengths=outgoing_lengths, route_lengths=incoming_lengths+outgoing_lengths,
                incident_projection=incident_projection, outgoing_projection=outgoing_projection,
                delta_length=outgoing_projection-incident_projection)


def draw_path_geometry(axis, data):
    """Draw ray directions and the two projection triangles at the grating."""
    blue, orange, green, grey, ink = '#2866a3', '#b85c17', '#23734d', '#89939e', '#243444'
    period = data['period']
    positions = data['positions']
    box = dict(facecolor='white', edgecolor='none', alpha=.94, pad=2)
    axis.axhline(0, color=grey, lw=2, zorder=1)

    # Endpoints on each grey line share one reference plane. Only the
    # incident plane is necessarily a wavefront for a noninteger comparison.
    for points in (data['sources'], data['ends']):
        axis.plot(points[:, 0], points[:, 1], '--', color=grey, lw=1, alpha=.75)

    for point, source, end in zip(positions, data['sources'], data['ends']):
        axis.annotate('', point, source,
                      arrowprops=dict(arrowstyle='->', color=green, lw=1.6, shrinkA=0, shrinkB=0))
        axis.annotate('', end, point,
                      arrowprops=dict(arrowstyle='->', color=blue, lw=1.6, shrinkA=0, shrinkB=0))

    # Highlight actual extra segments, not the horizontal separation d.
    for incoming, projection, direction, color, label in (
            (True, data['incident_projection'], data['incident'], green, r'$|\ell_i|$'),
            (False, data['outgoing_projection'], data['outgoing'], orange, r'$|\ell_{\rm out}|$')):
        if abs(projection) < 1e-10:
            continue
        if not incoming and abs(data['incident_projection']) < 1e-10:
            label = r'$|\Delta L|$'
        longer = (1 if projection > 0 else 0) if incoming else (0 if projection > 0 else 1)
        point, other = positions[longer], positions[1-longer]
        q = point+(-1 if incoming else 1)*abs(projection)*direction
        axis.plot([point[0], q[0]], [point[1], q[1]], color=color, lw=4,
                  solid_capstyle='round', zorder=4)
        axis.plot(*q, 'o', color=color, ms=4, zorder=5)
        # A line perpendicular to the rays compares their path lengths.
        perpendicular = np.array([-direction[1], direction[0]])
        joins = np.array([q, other])
        low, high = np.argmin(joins @ perpendicular), np.argmax(joins @ perpendicular)
        start, end = joins[low]-.1*period*perpendicular, joins[high]+.1*period*perpendicular
        axis.plot([start[0], end[0]], [start[1], end[1]], '--', color=color, lw=1.1, alpha=.8)
        midpoint = (point+q)/2
        axis.annotate(label, midpoint, xytext=(-26 if longer == 0 else 17, 0),
                      textcoords='offset points', color=color, fontsize=11,
                      ha='center', va='center', bbox=box, zorder=7)
        # Right-angle marker; omitted at the degenerate grazing limit.
        side = other-q
        if np.linalg.norm(side) > 1e-9:
            side = side/np.linalg.norm(side)*.055*period
            ray = (point-q)/np.linalg.norm(point-q)*.055*period
            corner = np.array([q+side, q+side+ray, q+ray])
            axis.plot(corner[:, 0], corner[:, 1], color=color, lw=.8)

    for point, label in zip(positions, ('A', 'B')):
        axis.plot(*point, 'o', color=ink, markeredgecolor='white', ms=7, zorder=8)
        axis.annotate(label, point, xytext=(0, 10), textcoords='offset points',
                      ha='center', fontsize=11, bbox=box, zorder=9)
    axis.text(.5, .98, 'From a common incident wavefront', color=green, ha='center', va='top', transform=axis.transAxes)
    axis.text(.5, .02, 'To a common outgoing reference plane', color=blue, ha='center', va='bottom', transform=axis.transAxes)
    points = np.vstack([positions, data['sources'], data['ends']])
    lower, upper = points.min(axis=0), points.max(axis=0)
    padding = .27*period
    axis.set(xlim=(lower[0]-padding, upper[0]+padding),
             ylim=(lower[1]-padding, upper[1]+padding), aspect='equal',
             xlabel='x (µm)', ylabel='z (µm)', title='Two routes through neighbouring openings')
    axis.title.set_fontsize(11)
    axis.spines[['top', 'right']].set_visible(False)
    if abs(data['outgoing'][1]) < 1e-9:
        axis.text(.5, .13, 'Grazing: outgoing paths lie along the grating.', ha='center',
                  fontsize=9, transform=axis.transAxes, bbox=box)
