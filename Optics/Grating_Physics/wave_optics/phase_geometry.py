"""Optional phase-offset geometry in notebook seven's laboratory xyz frame."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc


PROJECTION_INTRO = r"""**Trace the two arms**

Take $S=(0,0,0)$, $M=(h,0,0)$ and detector centre $P=(0,0,z_D)$, with $z_D<0$.
Both beams have $k_y=0$. Suppressing the common source phase, their phases
at $(x,y,z_D)$ are

$$
\begin{aligned}
\Phi_1&=k_{1x}x+k_{1z}z_D-\omega t+\phi_{1,\mathrm{optics}},\\
\Phi_2&=kh+k_{2x}(x-h)+k_{2z}z_D-\omega t+\phi_{2,\mathrm{optics}}.
\end{aligned}
$$

Beam 2 gains $kh$ travelling from the splitter to the mirror along $+x$.
Subtracting the phases gives

$$
\Delta\phi_0=(k-k_{2x})h+\Delta\phi_{\mathrm{optics}}
=kh(1+\sin\theta)+\Delta\phi_{\mathrm{optics}},
$$

where $\Delta\phi_{\mathrm{optics}}=\phi_{2,\mathrm{optics}}-\phi_{1,\mathrm{optics}}$.

**Geometric reading at the detector centre**

The wavefront through $P$ meets beam 2's axis at $F$, with $FP$ perpendicular
to its propagation direction. This construction uses $x=y=0$.

**Try:** at zero mirror tilt, move $z_D$ to $-3.5$ to reveal the projection
triangle; return to $-6.9$ to make $F$ and $P$ coincide.
"""

PROJECTION_EXPLANATION = r"""Points $F$ and $P$ share beam 2's phase.
The phase advance from the mirror is $k\,MF$, where

$$
MF=\hat{\mathbf s}_2\cdot\overrightarrow{MP}
=h\sin\theta-z_D\cos\theta,
\qquad \hat{\mathbf s}_2=(-\sin\theta,0,-\cos\theta).
$$

Thus $k\,MF=-k_{2x}h+k_{2z}z_D$. This projected distance equals
$|MP|=\sqrt{h^2+z_D^2}$ when $P$ lies on the beam axis.

**Wavefront intersections along $z$**

Beam 2 has $k_{2z}=-k\cos\theta$. Its phase decreases as $z$ increases;
the positive spacing between successive wavefront intersections is

$$
|k_{2z}|s=2\pi
\quad\Longrightarrow\quad s=\frac{\lambda}{|\cos\theta|}.
$$

This is one beam's wavefront spacing along $z$. The detector's interference
fringe spacing is $\Lambda_x$, across $x$.

**Check against the geometric arm lengths**

When the axes meet at $P$, $z_D=-h\cot\theta$, and

$$
\Delta\Phi(P)=k\left[h+\sqrt{h^2+z_D^2}+z_D\right]
+\Delta\phi_{\mathrm{optics}}.
$$

The direct-arm length is $-z_D>0$, so the bracket is the reflected-arm
length minus the direct-arm length.
"""


def phase_projection(geometry):
    """Project M-to-P onto the reflected direction; all vectors are xyz."""
    origin = np.asarray(geometry['splitter'])
    mirror = np.asarray(geometry['mirror'])-origin
    point = np.asarray(geometry['detector'])-origin
    direction = np.asarray(geometry['outgoing'])
    displacement = point-mirror
    projected = float(np.dot(displacement, direction))
    foot = mirror+projected*direction
    return dict(mirror=mirror, point=point, foot=foot, direction=direction,
                projected_distance=projected,
                geometric_distance=float(np.linalg.norm(displacement)),
                transverse_distance=float(np.linalg.norm(point-foot)),
                z_spacing_over_wavelength=float(1/abs(direction[2])))


def draw_phase_geometry(geometry):
    """Draw the projection and longitudinal wavefront spacing in the xz view."""
    data = phase_projection(geometry)
    side = lambda vector: np.asarray(vector)[..., [0, 2]]
    mirror, point, foot, direction = (side(data[key]) for key in
                                     ('mirror', 'point', 'foot', 'direction'))
    normal = np.array([-direction[1], direction[0]])
    theta = geometry['theta_deg']
    orange, blue, purple = '#d97635', '#267b9b', '#8966a4'
    ink, muted = '#304051', '#74808b'
    fig, (ax, waves) = plt.subplots(1, 2, figsize=(11, 7),
                                   gridspec_kw={'width_ratios': [1, 1.15]})
    fig.subplots_adjust(left=.055, right=.96, bottom=.23, top=.86, wspace=.32)
    for panel in (ax, waves):
        panel.set_aspect('equal')
        panel.axis('off')
    ax.set(xlim=(-1.3, 3.4), ylim=(-10.7, 1.5))
    ax.set_title('1. Mirror to detector centre', fontsize=12, color=ink, pad=18)
    ax.plot([0, mirror[0]], [0, 0], color=orange, lw=2)
    ax.annotate('', xy=(0, -10.3), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=blue, lw=1.5))
    ax.text(.15, -10.45, '$-z$', color=blue, fontsize=11)
    ax.text(-.2, .25, '$S$', ha='right', color=ink, fontsize=11)
    ax.text(mirror[0]/2, .30, '$h$', ha='center', color=ink, fontsize=12)
    ax.plot([-1.1, 2.8], [point[1]]*2, ':', color=muted, lw=1)
    ax.text(-.35, point[1]+.25, r'$z_D$', color=muted,
            ha='right', va='bottom', fontsize=11)
    end = mirror+(-10.3-mirror[1])/direction[1]*direction
    ax.plot(*np.array([mirror, end]).T, '--', color=orange, lw=1, alpha=.6)
    ax.plot(*np.array([mirror, point]).T, '--', color=muted, lw=1.6,
            label='Geometric segment MP')
    ax.plot(*np.array([mirror, foot]).T, color=orange, lw=2.7,
            label='Projected distance MF')
    front = foot+np.array([-1.1, 1.3])[:, None]*normal
    ax.plot(*front.T, color=purple, lw=1.4, label='Wavefront through P')
    edge = mirror+np.array([-.28, .28])[:, None]*side(geometry['tangent'])
    ax.plot(*edge.T, color=ink, lw=3)
    ax.text(mirror[0], .65, '$M$', ha='center', color=ink, fontsize=11)
    ax.plot([mirror[0]]*2, [0, -2], ':', color=muted, lw=1)
    ax.add_patch(Arc(mirror, 2.1, 2.1, theta1=-90-theta, theta2=-90, color=purple))
    a = np.deg2rad(-90-theta/2)
    ax.text(*(mirror+1.5*np.array([np.cos(a), np.sin(a)])), r'$\theta$',
            color=purple, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=.5))
    ax.plot(*point, 'o', color=blue, ms=5, zorder=6)
    if data['transverse_distance'] < 1e-7:
        ax.annotate('$P=F$', xy=point, xytext=(10, -17),
                    textcoords='offset points', color=blue, fontsize=11)
    else:
        ax.plot(*foot, 'o', color=purple, ms=4, zorder=6)
        sign = np.sign(np.dot(point-foot, normal))
        size = min(.16, data['transverse_distance']*.35)
        corner = np.array([foot-size*direction,
                           foot-size*direction+sign*size*normal,
                           foot+sign*size*normal])
        ax.plot(*corner.T, color=purple, lw=1)
        ax.annotate('$F$', xy=foot, xytext=(10, 8), textcoords='offset points',
                    color=purple, fontsize=11)
        ax.annotate('$P$', xy=point, xytext=(-17, -16), textcoords='offset points',
                    color=blue, fontsize=11)
    ax.legend(loc='upper left', bbox_to_anchor=(-.16, -.03), frameon=False, fontsize=9)

    waves.set(xlim=(-.75, 4.4), ylim=(-.65, 5.6))
    waves.set_title('2. Beam 2 wavefronts along z', fontsize=12, color=ink, pad=18)
    wavelength = .9
    spacing = wavelength/abs(direction[1])
    crossings = .4+np.arange(5)*spacing
    x = np.array([-.12, 1.3])
    for z in crossings:
        waves.plot(x, z-direction[0]/direction[1]*x, color=orange, lw=2)
    waves.annotate('', xy=(0, 5.05), xytext=(0, -.1),
                   arrowprops=dict(arrowstyle='->', color=blue, lw=1.5))
    waves.text(0, 5.25, '$z$', ha='center', color=blue, fontsize=12)
    waves.plot(np.zeros_like(crossings), crossings, 'o', color=orange, ms=4)
    for z in crossings[:2]:
        waves.plot([-.03, -.42], [z, z], ':', color=muted, lw=1)
    waves.annotate('', xy=(-.37, crossings[1]), xytext=(-.37, crossings[0]),
                   arrowprops=dict(arrowstyle='<->', color=ink, lw=1))
    waves.text(-.62, np.mean(crossings[:2]), '$s$', va='center', color=ink, fontsize=12)
    start = np.array([.7, crossings[3]-direction[0]/direction[1]*.7])
    stop = start+wavelength*direction
    waves.annotate('', xy=stop, xytext=start,
                   arrowprops=dict(arrowstyle='<->', color=ink, lw=1))
    waves.text(*((start+stop)/2+.18*normal), r'$\lambda$', color=ink, fontsize=12)
    origin = np.array([3.25, 4.8])
    tip = origin+2*direction
    z_tip = np.array([origin[0], tip[1]])
    for endpoint, colour in ((tip, orange), (z_tip, blue)):
        waves.annotate('', xy=endpoint, xytext=origin,
                       arrowprops=dict(arrowstyle='->', color=colour, lw=1.5))
    waves.plot([tip[0], z_tip[0]], [tip[1]]*2, ':', color=muted)
    waves.text(3.45, 3.75, r'$k_{2z}$', color=blue, fontsize=12)
    waves.text(2.3, 3.5, r'$\mathbf{k}_2$', color=orange, fontsize=12)
    waves.text(2.7, 2.15, r'$k_{2z}=-k\cos\theta$', ha='center', color=blue, fontsize=11)
    fig.text(.055, .955, rf'Same apparatus: $z_D={point[1]:.2f}$, '
             rf'$\theta={theta:.2f}^\circ$', fontsize=12, color=ink)
    fig.text(.055, .08, rf'$MF={data["projected_distance"]:.3f}$; '
             rf'$|MP|={data["geometric_distance"]:.3f}$ (drawing units).', fontsize=10, color=ink)
    fig.text(.055, .035, 'F and P lie on the same wavefront; at alignment F=P.',
             fontsize=9, color=muted)
    fig.text(.60, .08, r'$s=\lambda/|\cos\theta|$', fontsize=12, color=ink)
    fig.text(.60, .035, 'Spacing enlarged; same propagation direction.', fontsize=9, color=muted)
    return fig, data
