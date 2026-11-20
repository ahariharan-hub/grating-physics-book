"""Phase matching and finite-array interference for diffraction teaching.

The periodic direction is x; +s points into the transmitted medium. Signed
angles run from +s toward +x. Wavelength is in the common exterior medium.
The tiled field is a compatibility construction, not a scattering solution.
"""

import numpy as np

from .diffraction import diffraction_orders


def _wave_components(wavelength, period, incidence_deg, angle_deg):
    angles = np.asarray(angle_deg, dtype=float)
    if (not np.all(np.isfinite([wavelength, period, incidence_deg]))
            or not np.all(np.isfinite(angles))):
        raise ValueError('Lengths and angles must be finite.')
    if wavelength <= 0 or period <= 0:
        raise ValueError('Wavelength and period must be positive.')
    if abs(incidence_deg) >= 90 or np.any(np.abs(angles) > 90):
        raise ValueError('Require |incidence| < 90 and |outgoing angle| <= 90 degrees.')
    k = 2 * np.pi / wavelength
    return (k * np.sin(np.deg2rad(incidence_deg)),
            k * np.sin(np.deg2rad(angles)),
            k * np.cos(np.deg2rad(angles)))


def phase_step(wavelength, period, incidence_deg, angle_deg):
    """Return delta = (kx_out - kx_in)d in radians, without wrapping.

    Adjacent far-field contributions have ratio exp(-1j * delta).
    """
    kxi, kx, _ = _wave_components(wavelength, period, incidence_deg, angle_deg)
    return (kx - kxi) * period


def arrival_phasors(wavelength, period, incidence_deg, angle_deg, periods):
    """Equal-amplitude contributions at one far-field point, relative to A.

    The final axis indexes openings j=0,...,N-1. With exp(-i*omega*t),
    the real contribution from j is cos(omega*t + j*delta).
    """
    if not np.isfinite(periods) or periods < 2 or periods != int(periods):
        raise ValueError('The number of illuminated periods must be an integer >= 2.')
    delta = np.asarray(phase_step(wavelength, period, incidence_deg, angle_deg))
    return np.exp(-1j * delta[..., None] * np.arange(int(periods)))


def array_intensity(wavelength, period, incidence_deg, angle_deg, periods):
    """Return |sum(exp(-ij*delta))/N|^2, excluding the element envelope.

    A direct coherent sum avoids the removable 0/0 singularity at orders.
    Normalization is to the N-element principal peak, not transmitted power.
    """
    return np.abs(arrival_phasors(wavelength, period, incidence_deg,
                                 angle_deg, periods).mean(axis=-1)) ** 2


def wavefront_fields(x, s, wavelength, period, incidence_deg, angle_deg):
    """Return the continuous trial plane wave and its Floquet-tiled version.

    In cell j (jd <= x < (j+1)d), the tiled phase is
    kx*(x-jd) + ks*s + kxi*jd. The tiles obey the incident translation rule
    but only join as a global plane wave when (kx-kxi)d = 2*pi*m.
    Arrays x and s must be broadcastable; angle_deg is scalar.
    """
    if np.ndim(angle_deg) != 0:
        raise ValueError('Use one outgoing angle for the wavefront view.')
    kxi, kx, ks = _wave_components(wavelength, period, incidence_deg, angle_deg)
    x, s = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(s, dtype=float))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(s)):
        raise ValueError('Coordinates must be finite.')
    cell_start = np.floor(x / period) * period
    continuous = np.exp(1j * (kx * x + ks * s))
    tiled = np.exp(1j * (kx * (x - cell_start) + ks * s + kxi * cell_start))
    return continuous, tiled


class OrderFormationExplorer:
    """Arrival contributions first; optional wavefront view for Fourier study.

    Both modes share the same far-field array calculation and controls.
    Separate instances have independent parameters and resource ownership.
    """

    def __init__(self, view='arrivals'):
        if view not in ('arrivals', 'wavefronts'):
            raise ValueError("Choose view='arrivals' or view='wavefronts'.")
        import ipywidgets as widgets
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self._closed = False
        self._busy = False
        self.view = view
        self._scan_cache = None
        self._wavefront_cache = None
        self._wavefront_x = (np.arange(750) + .5) / 150 - 2
        self._wavefront_s = (np.arange(360) + .5) / 150

        def slider(value, low, high, step, label):
            return widgets.FloatSlider(value=value, min=low, max=high, step=step,
                description=label, readout_format='.2f', continuous_update=False,
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='310px'))

        self.wavelength = slider(1, .5, 2, .05, 'λ in medium (µm)')
        self.period = slider(2.5, .5, 5, .1, 'Period d (µm)')
        self.incidence = slider(0, -60, 60, 1, 'Incident θᵢ (°)')
        self.angle = slider(0, -90, 90, .1, 'Observation θ (°)')
        self.periods = widgets.IntSlider(value=2, min=2, max=32,
            description='Illuminated N', continuous_update=False,
            style={'description_width': 'initial'}, layout=widgets.Layout(width='310px'))
        self.order = widgets.Dropdown(options=[('m = 0', 0)], value=0,
            description='Follow direction', style={'description_width': 'initial'},
            layout=widgets.Layout(width='310px'))
        self.reset = widgets.Button(description='Reset example')
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1100px', height='auto'))
        self._controls = (self.wavelength, self.period, self.incidence,
                          self.angle, self.periods, self.order)
        controls = widgets.Box(list(self._controls) + [self.reset],
            layout=widgets.Layout(display='flex', flex_flow='row wrap'))
        self.ui = widgets.VBox([controls, self.readout, self.picture])
        self.figure = Figure(figsize=(10.8, 7.2), dpi=115, facecolor='white')
        FigureCanvasAgg(self.figure)
        for control in self._controls:
            control.observe(self._update, names='value')
        self.reset.on_click(self._reset)
        self._update()

    def _reset(self, button=None):
        if self._closed:
            return
        self._busy = True
        try:
            self.wavelength.value, self.period.value = 1, 2.5
            self.incidence.value, self.periods.value = 0, 2
            # Refresh options before selecting an order that may have vanished.
            self.order.options = [('m = 0', 0)]
            self.order.value = 0
        finally:
            self._busy = False
        self._update()

    def _update(self, change=None):
        if self._closed or self._busy:
            return
        self._busy = True
        try:
            self._calculate_and_draw(change)
        finally:
            self._busy = False

    def _calculate_and_draw(self, change):
        from .plotting import figure_png

        wavelength, period, incidence = (self.wavelength.value, self.period.value,
                                          self.incidence.value)
        orders = diffraction_orders(wavelength, period, abs(incidence),
                                    180 if incidence < 0 else 0)
        labels = orders['allowed']
        alphas = np.clip(orders['a'][orders['valid']], -1, 1)
        order_angles = np.rad2deg(np.arcsin(alphas))
        selected = self.order.value
        note = ''
        if change is not None and change['owner'] is self.angle:
            selected = None
        elif selected is not None and selected not in labels:
            note = f'Order m = {selected:+d} is now evanescent; keeping the observation angle. '
            selected = None
        self.order.options = [('Free angle', None)] + [
            (f'm = {m:+d}' + (' (grazing)' if abs(a) >= 1 - 1e-12 else ''), int(m))
            for m, a in zip(labels, alphas)]
        self.order.value = selected
        if selected is not None:
            self.angle.value = float(order_angles[np.flatnonzero(labels == selected)[0]])
        angle, count = self.angle.value, self.periods.value
        delta = float(phase_step(wavelength, period, incidence, angle))
        cycles = delta / (2 * np.pi)
        mismatch = float(abs(np.expm1(1j * delta)))
        matched = mismatch < 1e-9
        contributions = arrival_phasors(wavelength, period, incidence, angle, count)
        # Moving the observation cursor does not change the angular pattern.
        # Keep exact order angles in the cached curve and insert the cursor's
        # exact sample below, preserving the previous plotting convention.
        scan_key = (wavelength, period, incidence, count)
        if self._scan_cache is None or self._scan_cache[0] != scan_key:
            scan = np.unique(np.r_[np.linspace(-90, 90, 3601), order_angles])
            intensity = array_intensity(wavelength, period, incidence, scan, count)
            self._scan_cache = (scan_key, scan, intensity)
        _, scan, intensity = self._scan_cache
        current = float(abs(contributions.mean()) ** 2)
        cursor_index = np.searchsorted(scan, angle)
        if cursor_index == len(scan) or scan[cursor_index] != angle:
            scan = np.insert(scan, cursor_index, angle)
            intensity = np.insert(intensity, cursor_index, current)
        self.data = dict(delta=delta, cycles=cycles, mismatch=mismatch, matched=matched,
                         contributions=contributions, scan=scan, intensity=intensity,
                         current=current, orders=labels, order_angles=order_angles)
        status = (f'All contributions arrive in phase: m = {int(round(cycles)):+d}.'
                  if matched else 'Neighboring contributions arrive with different phases.')
        detail = (f'Adjacent total path difference / λ = {cycles:+.5f}; '
                  f'phase difference = {np.rad2deg(delta):+.2f}° (before wrapping).')
        if self.view == 'wavefronts':
            status = ('Cell copies join at an order direction.' if matched
                      else 'Cell copies have a phase mismatch at their seams.')
            detail = (f'Δφ / 2π = {cycles:+.5f} cycles; '
                      f'complex seam mismatch = {mismatch:.4f} (0 means agreement).')
        self.readout.value = (
            f'<p>{note}<b>{status}</b> θ = {angle:+.4f}°; '
            f'{detail}<br>N = {count}; '
            f'normalized array intensity at this angle = {current:.4f}.</p>')
        self.figure.clear()
        grid = self.figure.add_gridspec(2, 2, height_ratios=[1, .85])
        left = self.figure.add_subplot(grid[0, 0])
        right = self.figure.add_subplot(grid[0, 1])
        bottom = self.figure.add_subplot(grid[1, :])
        self.figure.subplots_adjust(left=.08, right=.96, top=.90, bottom=.09,
                                     hspace=.52, wspace=.29)
        if self.view == 'arrivals':
            self._draw_arrivals(left, right, angle, incidence, count, contributions)
        else:
            self._draw_wavefronts(left, right, wavelength, period, incidence, angle)
        bottom.plot(scan, intensity, color='#2866a3', lw=1.5)
        for m, order_angle in zip(labels, order_angles):
            bottom.axvline(order_angle, color='#b7bdc5', lw=.7, zorder=0)
        # Dense order labels obscure the curve: retain all lines, label a subset.
        label_stride = max(1, int(np.ceil(len(labels) / 9)))
        for index, (m, order_angle) in enumerate(zip(labels, order_angles)):
            if index % label_stride == 0 or m == 0:
                bottom.text(order_angle, 1.035, f'{m:+d}' if m else '0', fontsize=9,
                            ha='left' if order_angle < -85 else 'right' if order_angle > 85 else 'center')
        bottom.plot(angle, current, 'o', color='#b85c17', markersize=6, clip_on=False)
        bottom.axvline(angle, color='#b85c17', lw=1, linestyle='--')
        bottom.set(xlim=(-90, 90), ylim=(0, 1.13), xlabel='Observation angle θ (degrees)',
                   ylabel='Normalized array intensity', yticks=[0, .5, 1],
                   title=f'{count} illuminated openings — integer labels mark order directions')
        bottom.title.set_fontsize(11)
        bottom.grid(axis='y', color='#e4e7eb', lw=.7)
        self.picture.value = figure_png(self.figure)

    def _draw_arrivals(self, geometry, waves, angle, incidence, count, contributions):
        """Schematic converging paths and actual far-field time traces at P.

        Geometry compresses the distance to P for legibility; phases use the
        Fraunhofer path differences, never distances measured from this sketch.
        """
        from matplotlib.patches import Arc

        colors = ['#2866a3', '#b85c17']
        positions = np.linspace(-1, 1, count)
        theta, theta_i = np.deg2rad([angle, incidence])
        point = 3 * np.array([np.sin(theta), np.cos(theta)])
        geometry.axhline(0, color='#a9afb7', lw=1)
        geometry.plot([0, 0], [0, 3.1], '--', color='#a9afb7', lw=.8)
        geometry.text(.1, 2.4, '+s', fontsize=9)
        geometry.text(3.3, -.12, '+x', fontsize=9, ha='right', va='top')
        # Draw later openings first so A and B remain identifiable in a dense array.
        for j in reversed(range(count)):
            xx = positions[j]
            color = colors[j] if j < 2 else '#a9afb7'
            geometry.plot([xx, point[0]], [0, point[1]], color=color,
                          lw=1.5 if j < 2 else .7, alpha=1 if j < 2 else .6,
                          linestyle='--' if j == 1 else '-')
            geometry.plot(xx, 0, 'o', color=color, markersize=5 if j < 2 else 3)
            if j < 2:
                geometry.annotate('AB'[j], (xx, 0), xytext=(-6 if j == 0 else 6, -14),
                                  textcoords='offset points', ha='center', fontsize=10,
                                  color=color)
        # Representative incident rays show which side supplies the illumination.
        for xx in (-.8, .8):
            tail = np.array([xx, 0]) - .72 * np.array([np.sin(theta_i), np.cos(theta_i)])
            geometry.annotate('', (xx, -.03), tail,
                              arrowprops=dict(arrowstyle='->', color='#555555', lw=1))
        geometry.text(0, -1.04, f'Incident plane wave: θᵢ = {incidence:g}°',
                      fontsize=9, ha='center')
        geometry.plot(*point, 'o', color='#222222', markersize=5)
        geometry.annotate('P: far field\n(distance compressed)', point,
                          xytext=(0, 12), textcoords='offset points',
                          ha='center', fontsize=9)
        if abs(angle) > 1e-9:
            low, high = sorted([90, 90 - angle])
            geometry.add_patch(Arc((0, 0), 1, 1, theta1=low, theta2=high,
                                   color='#555555', lw=1))
            geometry.text(.75*np.sin(theta/2), .75*np.cos(theta/2), 'θ', fontsize=10)
        geometry.set(xlim=(-4, 4), ylim=(-1.25, 3.95), aspect='equal',
                     title=f'{count} equally spaced openings → the same point P')
        geometry.title.set_fontsize(11)
        geometry.axis('off')

        time = np.linspace(0, 2, 501)
        traces = (contributions[:, None] * np.exp(-2j * np.pi * time)).real
        self.data['time'] = time
        self.data['arrival_traces'] = traces
        for j in range(2, count):
            waves.plot(time, traces[j], color='#a9afb7', lw=.7, alpha=.4,
                       label='Other openings' if j == 2 else None)
        waves.plot(time, traces[0], color=colors[0], lw=1.8, label='From A')
        waves.plot(time, traces[1], color=colors[1], linestyle='--', lw=1.8, label='From B')
        waves.plot(time, traces.mean(axis=0), color='#222222', lw=1.5,
                   linestyle=':', label='Sum / N')
        waves.axhline(0, color='#d1d5db', lw=.7, zorder=0)
        waves.set(xlim=(0, 2), ylim=(-1.15, 1.15), xlabel='Time t / T (relative to A)',
                  ylabel='Field at P / single-opening amplitude',
                  title='Compare arrivals at P over one shared clock', yticks=[-1, 0, 1])
        waves.title.set_fontsize(11)
        waves.legend(loc='upper center', bbox_to_anchor=(.5, 1.30),
                     ncol=2, fontsize=8, frameon=False)
        waves.tick_params(labelsize=9)

    def _draw_wavefronts(self, left, right, wavelength, period, incidence, angle):
        # This optional construction is introduced only after Fourier periodicity.
        key = (wavelength, period, incidence, angle)
        if self._wavefront_cache is None or self._wavefront_cache[0] != key:
            continuous, tiled = wavefront_fields(
                self._wavefront_x[None, :] * period,
                self._wavefront_s[:, None] * period,
                wavelength, period, incidence, angle)
            self._wavefront_cache = (key, continuous, tiled)
        _, continuous, tiled = self._wavefront_cache
        self.data.update(continuous=continuous, tiled=tiled)
        for axis, field, title in (
                (left, continuous, 'One continuous plane-wave component'),
                (right, tiled, 'Cell copies with the incident phase offset')):
            picture = axis.imshow(field.real, origin='lower', extent=(-2, 3, 0, 2.4),
                                  cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest',
                                  aspect='equal')
            for edge in range(-1, 3):
                axis.axvline(edge, color='#333333', linestyle='--', lw=.8, alpha=.65)
            theta = np.deg2rad(angle)
            axis.annotate('', (.5 + .8 * np.sin(theta), 1.2 + .8 * np.cos(theta)),
                          (.5, 1.2), arrowprops=dict(arrowstyle='->', color='black', lw=2))
            axis.set(title=title, xlabel='x / d', ylabel='s / d (transmission ↑)',
                     xticks=[-2, -1, 0, 1, 2, 3], yticks=[0, 1, 2])
            axis.tick_params(labelsize=9)
            axis.title.set_fontsize(11)
        colorbar = self.figure.colorbar(picture, ax=[left, right], shrink=.85,
                                        pad=.025, fraction=.025, ticks=[-1, 0, 1])
        colorbar.set_label('Re(U)', fontsize=9)

    def close(self):
        """Release this explorer's callbacks, image buffer, figure and widgets."""
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.reset.on_click(self._reset, remove=True)
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        self._scan_cache = self._wavefront_cache = None
        self._wavefront_x = self._wavefront_s = None
        closed = set()

        def close_tree(widget):
            if widget is None or id(widget) in closed:
                return
            closed.add(id(widget))
            for child in getattr(widget, 'children', ()):
                close_tree(child)
            close_tree(getattr(widget, 'layout', None))
            close_tree(getattr(widget, 'style', None))
            widget.close()

        close_tree(self.ui)
