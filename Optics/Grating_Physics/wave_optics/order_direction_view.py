"""Direction-only grating geometry for the angle explorer in lesson 9.3.

No scattering solve is performed. Theta is measured from the downward normal;
positive angles point toward +x. Lengths share any one consistent unit.
"""
import io
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def order_direction_data(theta_deg, *, wavelength=.633, period=1.5,
                         n_incident=1., n_output=1., orders=tuple(range(-3, 4))):
    """Return signed transverse direction cosines, kinds, and real angles.

    Evanescent angles are NaN. Grazing orders are separate from propagating
    orders; numerical equality to cutoff is classified within roundoff.
    """
    values = np.asarray([theta_deg, wavelength, period, n_incident, n_output], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values[1:] <= 0) or abs(theta_deg) >= 90:
        raise ValueError('Use positive finite optical inputs and -90 < incidence < 90 deg.')
    m = np.asarray(orders)
    if m.ndim != 1 or not len(m) or not np.issubdtype(m.dtype, np.integer):
        raise ValueError('orders must be a nonempty one-dimensional integer sequence.')
    if len(np.unique(m)) != len(m):
        raise ValueError('Order labels must be distinct.')
    s = n_incident/n_output*np.sin(np.deg2rad(theta_deg))+m*wavelength/(n_output*period)
    grazing = np.isclose(np.abs(s), 1., rtol=0., atol=64*np.finfo(float).eps)
    kinds = np.where(grazing, 'grazing', np.where(np.abs(s) < 1., 'propagating', 'evanescent'))
    theta = np.full(s.shape, np.nan)
    real = kinds != 'evanescent'
    theta[real] = np.rad2deg(np.arcsin(np.clip(s[real], -1., 1.)))
    return dict(orders=m, transverse=s, kinds=kinds, theta_deg=theta)


def order_directions_png(theta_deg=0., *, wavelength=.633, period=1.5,
                         n_incident=1., n_output=1., orders=tuple(range(-3, 4))):
    """PNG showing an angle sweep, a polar direction diagram and order status.

    Every propagating output arrow has unit radius. Its length carries no
    information about amplitude or power. Grazing orders get rim markers only;
    evanescent orders never receive a real direction arrow.
    """
    data = order_direction_data(theta_deg, wavelength=wavelength, period=period,
                               n_incident=n_incident, n_output=n_output, orders=orders)
    fig = Figure(figsize=(14, 4.8), dpi=110)
    fig.subplots_adjust(left=.065, right=.99, top=.84, bottom=.15, wspace=.35)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.25, 1.15, .7))
    sweep = fig.add_subplot(grid[0, 0])
    polar = fig.add_subplot(grid[0, 1], projection='polar')
    status = fig.add_subplot(grid[0, 2])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    scan = np.linspace(-80., 80., 321)
    sweep.axhspan(-1, 1, color='grey', alpha=.15)
    sweep.axhline(-1, color='grey', ls=':', lw=1)
    sweep.axhline(1, color='grey', ls=':', lw=1)
    sweep.axvline(theta_deg, color='black', ls='--', lw=1)
    for j, m in enumerate(data['orders']):
        color = colors[j % len(colors)]
        curve = n_incident/n_output*np.sin(np.deg2rad(scan))+m*wavelength/(n_output*period)
        sweep.plot(scan, curve, color=color, lw=1.3, label=f'm={m}')
        sweep.plot(theta_deg, data['transverse'][j], 'o', color=color, ms=5)
    sweep.set(xlim=(-80, 80), ylim=(-2.2, 2.2), xlabel='Incidence angle (deg)',
              ylabel=r'$\sigma_{x,m}=k_{x,m}/(n_o k_0)$', title=f'Selected incidence: {theta_deg:+.1f} deg')
    sweep.legend(fontsize=8, ncol=2, loc='upper left')
    polar.set_theta_zero_location('S')
    polar.set_theta_direction(1)
    polar.set_ylim(0, 1.32)
    polar.set_yticks([])
    polar.set_thetagrids([-90, -60, -30, 0, 30, 60, 90],
                        labels=['-90', '-60', '-30', '0', '+30', '+60', '+90'])
    polar.tick_params(axis='x', pad=1, labelsize=9)
    polar.set_title(r'Real directions: horizontal projection $\sigma_{x,m}$'+'\n0 deg = downward normal', pad=16)
    arc = np.linspace(-np.pi/2, np.pi/2, 200)
    polar.fill_between(arc, 0, 1, color='#e9f3fa', zorder=0)
    polar.plot(arc, np.ones_like(arc), color='grey', lw=1)
    for edge in (-np.pi/2, np.pi/2):
        polar.plot([edge, edge], [0, 1.3], color='black', lw=2)
    polar.plot([0, 0], [0, 1.2], color='grey', ls=':', lw=1)
    incoming = np.deg2rad(theta_deg)+np.pi
    polar.annotate('', xy=(incoming, .03), xytext=(incoming, 1.),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    polar.text(incoming, 1.14, 'incident', ha='center', va='center', fontsize=9,
               bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1))
    for j, (m, angle, kind) in enumerate(zip(data['orders'], data['theta_deg'], data['kinds'])):
        color = colors[j % len(colors)]
        if kind == 'evanescent':
            continue
        direction = np.deg2rad(angle)
        if kind == 'propagating':
            polar.annotate('', xy=(direction, 1.), xytext=(direction, .02),
                           arrowprops=dict(arrowstyle='->', lw=2, color=color))
        else:
            polar.plot(direction, 1., 's', ms=7, color=color)
        polar.text(direction, 1.13, f'm={m}', color=color, ha='center', va='center', fontsize=9,
                   bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1))
    # Polar plot() calls can autoscale to the first drawn arc; retain both half-spaces.
    polar.set_xlim(-np.pi, np.pi)
    status.axis('off')
    status.set_title('Orders shown', loc='left')
    for j, (m, angle, kind) in enumerate(zip(data['orders'], data['theta_deg'], data['kinds'])):
        detail = f'{angle:+.1f} deg' if kind == 'propagating' else kind
        status.text(0, .95-j*.09, f'm={m:+d}: {detail}', color=colors[j % len(colors)],
                    transform=status.transAxes, fontsize=10)
    status.text(0, .19, 'Evanescent: no real angle.\nGrazing: square on the rim.\n\nEqual arrow lengths show\ndirection only, not power.',
                va='top', transform=status.transAxes, fontsize=9)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    return buffer.getvalue()


class OrderDirectionExplorer:
    """Owned widgets: re-render geometry without invoking a scattering solver."""
    def __init__(self, theta_deg=25., **optics):
        import ipywidgets as widgets
        self.optics = optics
        self.angle = widgets.FloatSlider(value=theta_deg, min=-80., max=80., step=.1,
                                         description='Incidence (deg)', continuous_update=False,
                                         readout_format='.1f', style={'description_width': 'initial'},
                                         layout=widgets.Layout(width='600px'))
        self.image = widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1540px'))
        self.note = widgets.HTML('<b>Direction only:</b> drag incidence; no FMM solve is needed. '
                                 'Black line = grating; positive transmitted angles point toward +x.')
        self.ui = widgets.VBox([self.angle, self.note, self.image])
        self.angle.observe(self._update, names='value')
        self._update()

    def _update(self, change=None):
        self.image.value = order_directions_png(self.angle.value, **self.optics)

    def close(self):
        self.angle.unobserve(self._update, names='value')
        for widget in (self.angle, self.image, self.note, self.ui):
            widget.close()
