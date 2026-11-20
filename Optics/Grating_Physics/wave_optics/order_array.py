"""A scalar narrow-opening array: spreading contributions and far-field sum.

Each opening is a prescribed, equally driven cylindrical-wave source. The
spatial illustration uses exp(i*k*r)/sqrt(r), the outgoing cylindrical-wave
approximation away from a source. Disks r < lambda are omitted. The far-field
curve is the exact array factor of these identical sources, normalized by N².
It excludes finite opening width, polarization, coupling and efficiencies.
"""

import numpy as np

from .diffraction import diffraction_orders


MAX_OPENINGS = 21


def opening_positions(period, count):
    """Add neighbours on fixed grating sites while retaining the opening x=0."""
    if not np.isfinite(period) or period <= 0:
        raise ValueError('Period must be positive and finite.')
    if not np.isfinite(count) or count != int(count) or not 1 <= count <= MAX_OPENINGS:
        raise ValueError(f'Use an integer number of openings from 1 to {MAX_OPENINGS}.')
    count = int(count)
    return (np.arange(count)-(count-1)//2)*period


def spreading_field(x, z, positions, wavelength, incident_angle):
    """Coherent outgoing sum with the incident phase at each opening.

    Time convention is exp(-i*omega*t). At a far point R*s, each contribution
    has phase k*R + (kix-k*sx)*x_j, matching the earlier path/seam convention.
    A common cylindrical asymptotic phase is omitted; it shifts every wave
    equally. Omitted source neighbourhoods are returned as NaNs.
    """
    x, z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(z, dtype=float))
    k = 2*np.pi/wavelength
    kix = k*np.sin(np.deg2rad(incident_angle))
    field = np.zeros(x.shape, dtype=complex)
    excluded = np.zeros(x.shape, dtype=bool)
    for source in positions:
        distance = np.hypot(x-source, z)
        excluded |= distance < wavelength
        # Values in the source neighbourhood are not displayed or interpreted.
        safe_distance = np.maximum(distance, wavelength)
        field += np.exp(1j*(k*safe_distance+kix*source))/np.sqrt(safe_distance)
    field[excluded] = complex(np.nan, np.nan)
    return field


def far_field_amplitude(angles, positions, wavelength, incident_angle):
    """Complex array factor divided by N; no single-opening envelope."""
    k = 2*np.pi/wavelength
    mismatch = k*(np.sin(np.deg2rad(angles))-np.sin(np.deg2rad(incident_angle)))
    return np.exp(-1j*np.asarray(mismatch)[..., None]*positions).mean(axis=-1)


def opening_array_data(wavelength=2., period=4., incident_angle=0., count=1,
                       *, _geometry_cache=None):
    positions = opening_positions(period, count)
    if (not np.all(np.isfinite([wavelength, incident_angle])) or wavelength <= 0
            or abs(incident_angle) >= 90):
        raise ValueError('Use a positive wavelength and |incidence angle| < 90 degrees.')
    k = 2*np.pi/wavelength
    kix = k*np.sin(np.deg2rad(incident_angle))
    # Retain only one grid and one angular geometry per explorer. Fields remain
    # streamed source by source, never a sources x z x x allocation.
    cache = {} if _geometry_cache is None else _geometry_cache
    grid_key = (wavelength, period)
    if 'grid' not in cache or cache['grid'][0] != grid_key:
        # Same physical window for every N: changing N never zooms the picture.
        extent = (MAX_OPENINGS-1)/2*period+2*wavelength
        samples = max(241, int(np.ceil(2*extent/(wavelength/10)))+1)
        x = np.linspace(-extent, extent, samples)
        z = np.linspace(-2*extent, 0, samples)
        cache['grid'] = (grid_key, x, z)
    _, x, z = cache['grid']
    field = spreading_field(x[None, :], z[:, None], positions, wavelength, incident_angle)
    scale = float(np.nanmax(np.abs(field)))
    order_key = (wavelength, period, incident_angle)
    if 'orders' not in cache or cache['orders'][0] != order_key:
        result = diffraction_orders(wavelength, period, abs(incident_angle),
                                    180 if incident_angle < 0 else 0)
        order_angles = np.rad2deg(np.arcsin(np.clip(result['a'][result['valid']], -1, 1)))
        orders = result['allowed'].copy()
        grazing = result['grazing'][result['valid']].copy()
        angles = np.unique(np.r_[np.linspace(-90, 90, 7201), order_angles])
        cache['orders'] = (order_key, orders, order_angles, grazing, angles)
    _, orders, order_angles, grazing, angles = cache['orders']
    amplitude = far_field_amplitude(angles, positions, wavelength, incident_angle)
    return dict(wavelength=float(wavelength), period=float(period), incident_angle=float(incident_angle),
                count=int(count), positions=positions, source_phases=kix*positions,
                x=x, z=z, field=field, field_scale=scale, snapshot=field.real/scale,
                orders=orders, order_angles=order_angles,
                grazing=grazing,
                angles=angles, amplitude=amplitude, intensity=np.abs(amplitude)**2,
                incident_step=kix*period)


def draw_opening_array(figure, data):
    """An actual spatial sum beside its separate far-field angular limit."""
    from matplotlib.colors import SymLogNorm
    from .grating_intro import _uniform_field_image

    blue, orange, ink, grey = '#2866a3', '#b85c17', '#243444', '#89939e'
    figure.clear()
    grid = figure.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, .55])
    field = figure.add_subplot(grid[:, 0])
    pattern = figure.add_subplot(grid[0, 1])
    explanation = figure.add_subplot(grid[1, 1])
    figure.subplots_adjust(left=.065, right=.96, top=.84, bottom=.19, wspace=.30, hspace=.46)
    count = data['count']
    figure.suptitle(f'From one opening to many: N = {count}', fontsize=14)
    figure.text(.5, .91,
                f'd = {data["period"]:g} µm; λ = {data["wavelength"]:g} µm; '
                f'θᵢ = {data["incident_angle"]:+.1f}°; neighbouring drive phase = '
                f'{data["incident_step"]/(2*np.pi):+.3f} cycles', ha='center', fontsize=10, color=ink)
    picture = _uniform_field_image(field, data['x'], data['z'], data['snapshot'],
                                    cmap='RdBu_r',
                                    norm=SymLogNorm(linthresh=.12, vmin=-1, vmax=1))
    field.axhline(0, color=grey, lw=2)
    field.scatter(data['positions'], np.zeros(count), s=20, color=orange,
                  edgecolor='white', linewidth=.6, zorder=5, clip_on=False)
    field.set(xlim=(data['x'][0], data['x'][-1]), ylim=(data['z'][0], 0), aspect='equal',
              xlabel='x (µm)', ylabel='z (µm)', title='Sum of the spreading contributions at t = 0')
    field.title.set_fontsize(10)
    field.text(.03, .03, 'Same physical window for every N', transform=field.transAxes,
               fontsize=8.5, color=ink, bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
    color_axis = figure.add_axes([.095, .10, .31, .014])
    bar = figure.colorbar(picture, cax=color_axis, orientation='horizontal', ticks=[-1, 0, 1],
                          label='Signed summed field / peak magnitude (enhanced contrast)')
    bar.set_ticklabels(['−1', '0', '+1'])

    pattern.plot(data['angles'], data['intensity'], color=blue, lw=1.8)
    if count > 1:
        for order, angle, grazing in zip(data['orders'], data['order_angles'], data['grazing']):
            pattern.axvline(angle, color=orange, lw=.9, ls='--', alpha=.75)
            pattern.text(angle, 1.06, f'{order:+d}' + ('*' if grazing else ''),
                          ha='center', va='bottom', fontsize=9, color=orange)
    pattern.set(xlim=(-93, 93), ylim=(-.02, 1.23), xticks=np.arange(-90, 91, 30),
                yticks=[0, .5, 1], xlabel='Outgoing angle θ from −z (°)',
                ylabel='Normalized far-field intensity', title='Which directions reinforce?')
    pattern.title.set_fontsize(11)
    pattern.spines[['top', 'right']].set_visible(False)
    pattern.grid(axis='y', alpha=.2)
    explanation.axis('off')
    if count == 1:
        message = ('One narrow opening: a spreading contribution.\n'
                   'No neighbours yet, so no array interference.\n'
                   'This ideal source has a broad, uniform angular response.')
    elif count == 2:
        message = ('Two openings: their fields add.\n'
                   'Whole-cycle path differences reinforce;\n'
                   'half-cycle differences cancel in the far field.')
    else:
        message = (f'{count} openings: more contributions must agree.\n'
                   'The principal peaks narrow around the same order angles.\n'
                   'A finite array still has angular width and smaller lobes.')
    explanation.text(0, .98, message, transform=explanation.transAxes, va='top',
                      fontsize=10, color=ink, linespacing=1.65)
    detail = ('Orange dashed lines: integer orders of the periodic array.\n'
              '* Grazing direction, parallel to the grating.' if count > 1
              else 'The period d locates additional openings when N increases.')
    explanation.text(0, .28, detail, transform=explanation.transAxes, va='top',
                      fontsize=9, color=orange if count > 1 else ink, linespacing=1.6)
    figure.text(.5, .017,
                'Ideal narrow sources. White source neighbourhoods are omitted. Both views are normalized; order efficiencies are not shown.',
                ha='center', fontsize=8.5, color=ink)


class OpeningArrayExplorer:
    """Standalone source-array controls, independent of the order explorer."""

    def __init__(self):
        import ipywidgets as widgets
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        self._closed = False
        self._updating = False
        self._geometry_cache = {}
        style = {'description_width': 'initial'}
        self.period = widgets.FloatSlider(value=4, min=1, max=4, step=.1,
            description='Spacing d (µm)', continuous_update=False, style=style,
            layout=widgets.Layout(width='360px'))
        self.wavelength = widgets.FloatSlider(value=2, min=1, max=4, step=.1,
            description='λ in medium (µm)', continuous_update=False, style=style,
            layout=widgets.Layout(width='360px'))
        self.angle = widgets.FloatSlider(value=0, min=-60, max=60, step=5,
            description='θᵢ from −z (°)', continuous_update=False, style=style,
            layout=widgets.Layout(width='360px'))
        self.openings = widgets.IntSlider(value=1, min=1, max=MAX_OPENINGS, step=1,
            description='Openings N', continuous_update=False, style=style,
            layout=widgets.Layout(width='360px'))
        self.normal = widgets.Button(description='Normal incidence')
        self.reset = widgets.Button(description='Reset example')
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1302px', height='auto'))
        self._controls = (self.wavelength, self.period, self.angle, self.openings)

        def row(children):
            return widgets.Box(children, layout=widgets.Layout(
                display='flex', flex_flow='row wrap', align_items='center'))

        self.ui = widgets.VBox([row([self.period, self.wavelength, self.angle]),
                               row([self.openings, self.normal, self.reset]),
                               self.readout, self.picture])
        self.figure = Figure(figsize=(12.4, 7.4), dpi=105, facecolor='white')
        FigureCanvasAgg(self.figure)
        for control in self._controls:
            control.observe(self._update, names='value')
        self.normal.on_click(self._normal_incidence)
        self.reset.on_click(self._reset)
        self._update()

    def _update(self, change=None):
        if self._closed or self._updating:
            return
        from .plotting import figure_png
        self.data = opening_array_data(*(control.value for control in self._controls),
                                       _geometry_cache=self._geometry_cache)
        draw_opening_array(self.figure, self.data)
        self.picture.value = figure_png(self.figure)
        self.readout.value = (
            '<div style="padding:8px 0;line-height:1.7">'
            f'<b>N = {self.openings.value} illuminated opening(s).</b> '
            'Change only N first; the spatial window stays fixed.<br>'
            '<b>Left:</b> the sum of all spreading contributions. '
            '<b>Right:</b> their normalized far-field directional pattern.<br>'
            'Model: identical narrow openings with equal drive amplitudes; the incident wave '
            'sets their phases. The first wavelength around each source is omitted. '
            'Both views are normalized to compare shapes, not absolute power.</div>')

    def _normal_incidence(self, button=None):
        if not self._closed:
            self.angle.value = 0

    def _reset(self, button=None):
        if self._closed:
            return
        self._updating = True
        try:
            for control, value in zip(self._controls, (2., 4., 0., 1)):
                control.value = value
        finally:
            self._updating = False
        self._update()

    def close(self):
        if self._closed:
            return
        from .grating_intro import _close_widget_tree
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.normal.on_click(self._normal_incidence, remove=True)
        self.reset.on_click(self._reset, remove=True)
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        self._geometry_cache.clear()
        _close_widget_tree(self.ui)
