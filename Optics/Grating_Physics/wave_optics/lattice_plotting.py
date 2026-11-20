"""Coordinate plots of 2D order geometry, independent of collection optics."""
import numpy as np
from matplotlib.patches import Circle


def coordinate_plot(ax, data, scale='direction', annotate=False, title=None):
    """Plot every supplied transverse candidate and its propagation boundary."""
    if scale == 'direction':
        points, radius, names = data['components'][:, :2].real, 1., (r'$\alpha$', r'$\beta$')
    elif scale == 'frequency':
        points, radius = data['f'][:, :2].real, data['n_out']/data['wavelength0']
        names = ('f_x (cycles/um)', 'f_y (cycles/um)')
    elif scale == 'wavevector':
        points, radius = data['k'][:, :2].real, data['n_out']*data['k0']
        names = ('k_x (rad/um)', 'k_y (rad/um)')
    else:
        raise ValueError('Use direction, frequency, or wavevector.')
    for status, color, marker in [('propagating','tab:blue','o'),
                                   ('grazing','tab:purple','D'), ('evanescent','.6','x')]:
        keep = data['status'] == status
        ax.scatter(*points[keep].T, s=22, color=color, marker=marker, label=status.capitalize())
    ax.add_patch(Circle((0,0), radius, fill=False, color='black', linestyle='--'))
    if annotate:
        for mn, xy in zip(data['labels'], points):
            if max(abs(mn)) <= 1:
                ax.annotate(f'({int(mn[0])},{int(mn[1])})',xy,
                            xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel=names[0], ylabel=names[1], title=title or scale.capitalize(),
           aspect='equal', xlim=(-1.7*radius,1.7*radius),ylim=(-1.7*radius,1.7*radius))
    ax.grid(alpha=.2)
