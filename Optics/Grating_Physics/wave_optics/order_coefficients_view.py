"""Notebook 7.4 presentation helpers; reuse existing scattering physics unchanged."""
from io import BytesIO
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import ipywidgets as w

BLUE, ORANGE = '#2866a3', '#b85c17'


def png(figure):
    buffer = BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    return buffer.getvalue()


class OrderCoefficientExplorer:
    """One owned widget tree; no scattering solves or file writes on interaction."""
    def __init__(self):
        self.closed = False
        self._resetting = False
        self.pair = w.Dropdown(options=[('0, +1', (0, 1)), ('-1, +1', (-1, 1))], description='Orders')
        style = {'description_width': '100px'}
        self.a = w.FloatSlider(value=1, min=0, max=1.5, step=.05, description='Magnitude m', continuous_update=False, style=style)
        self.b = w.FloatSlider(value=1, min=0, max=1.5, step=.05, description='Magnitude p', continuous_update=False, style=style)
        self.phase = w.FloatSlider(value=0, min=-180, max=180, step=5, description='Phase p (deg)', continuous_update=False, style=style)
        self.depth = w.FloatSlider(value=0, min=0, max=12, step=.25, description='Depth s (um)', continuous_update=False, style=style)
        self.button = w.Button(description='Reset')
        self.picture = w.Image(format='png', layout=w.Layout(width='100%'))
        self.readout = w.HTML()
        self.controls = (self.pair, self.a, self.b, self.phase, self.depth)
        self.ui = w.VBox([w.HBox([self.pair, self.button]),
            w.HBox([self.a, self.b]), w.HBox([self.phase, self.depth]), self.readout, self.picture])
        for control in self.controls:
            control.observe(self.update, names='value')
        self.button.on_click(self.reset)
        self.update()

    def sample(self):
        # Fixed wavelength 1 um, pitch 4 um, normal incidence into -z.
        orders = np.array(self.pair.value)
        kx = orders * 2*np.pi/4
        kz = -np.sqrt((2*np.pi)**2-kx**2)
        coefficients = np.array([self.a.value, self.b.value*np.exp(1j*np.deg2rad(self.phase.value))])
        x = np.linspace(-8, 8, 801)
        fields = coefficients[:, None]*np.exp(1j*(kx[:, None]*x-kz[:, None]*self.depth.value))
        return x, kx, kz, coefficients, fields

    def update(self, change=None):
        if self.closed or self._resetting:
            return
        x, kx, kz, coefficients, fields = self.sample()
        fig = Figure(figsize=(10, 5.4), layout='constrained')
        axes = fig.subplots(2, 2)
        colors = (BLUE, ORANGE)
        for j, color in enumerate(colors):
            ax = axes[0, 0]
            ax.annotate('', xy=(kx[j]/(2*np.pi), kz[j]/(2*np.pi)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
            ax.text(kx[j]/(2*np.pi)+.03, kz[j]/(2*np.pi), f'order {self.pair.value[j]:+d}', color=color)
            ax = axes[0, 1]
            c = coefficients[j]
            ax.annotate('', xy=(c.real, c.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
            ax.plot(c.real, c.imag, 'o', color=color, markersize=10 if j == 0 else 4,
                    markerfacecolor='none' if j == 0 else color)
            ax.annotate(f'C({self.pair.value[j]:+d})', (c.real, c.imag),
                        xytext=(-8, 12 if j == 0 else -16), textcoords='offset points',
                        ha='right', color=color, fontsize=8)
            axes[1, 0].plot(x, fields[j].real, color=color, label=f'Re U({self.pair.value[j]:+d})')
        axes[0, 0].set(title='Kinematics: fixed directions for selected pair', xlabel='k_x/k', ylabel='k_z/k',
                       xlim=(-.5, .6), ylim=(-1.15, .1), aspect='equal')
        axes[0, 1].set(title='Dynamics variables: prescribed C at z_ref=0', xlabel='Re C', ylabel='Im C',
                       xlim=(-1.7, 1.7), ylim=(-1.7, 1.7), aspect='equal')
        for ax in axes[0]:
            ax.axhline(0, color='0.8', lw=.7); ax.axvline(0, color='0.8', lw=.7)
        axes[1, 0].set(title='Individual signed fields at t=0', xlabel='x (um)', ylabel='Re U', ylim=(-1.6,1.6))
        axes[1, 0].legend(fontsize=8)
        intensity = abs(fields.sum(axis=0))**2
        axes[1, 1].plot(x, intensity, color='#243444', label='Coherent intensity')
        for j, color in enumerate(colors):
            axes[1, 1].plot(x, abs(fields[j])**2, '--', color=color, lw=1, label=f'I({self.pair.value[j]:+d})')
        axes[1, 1].set(title='Add fields before squaring', xlabel='x (um)', ylabel='Relative I', ylim=(0,9.2))
        axes[1, 1].legend(fontsize=8)
        total = sum(abs(coefficients)**2)
        visibility = 2*abs(coefficients.prod())/total if total else None
        self.readout.value = ('<b>Prescribed coefficients, not a scattering solution.</b> '
            'Wavelength 1 um; pitch 4 um; z=-s. Common polarization, stable phase. '
            + (f'Ideal phase-scan visibility: {visibility:.3f}.' if visibility is not None else 'Both fields zero: visibility undefined.'))
        self.picture.value = png(fig)

    def reset(self, button=None):
        if self.closed:
            return
        self._resetting = True
        try:
            for control, value in zip(self.controls, ((0,1), 1., 1., 0., 0.)):
                control.value = value
        finally:
            self._resetting = False
        self.update()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for control in self.controls:
            control.unobserve(self.update, names='value')
        self.button.on_click(self.reset, remove=True)
        def close_tree(widget):
            for child in getattr(widget, 'children', ()):
                close_tree(child)
            widget.close()
        self.picture.value = b''
        close_tree(self.ui)


def coefficient_comparison_figure(solution):
    """Compare at the same exit plane and incident phase; remove trivial air carrier."""
    c = solution.configuration
    orders = np.arange(-4, 5)
    index = np.array([np.flatnonzero(solution.orders == m)[0] for m in orders])
    k = 2*np.pi/c.wavelength
    physical = solution.at_reference_plane(-c.thickness)[index]
    # Incident amplitude is 1 at z=0; ideal mask at z=-h sees exp(i*k*h).
    # Removing that common carrier sets incident phase zero at the exit in both models.
    physical = physical*np.exp(-1j*k*c.thickness)
    ideal = c.open_fraction*np.sinc(orders*c.open_fraction)
    fig = Figure(figsize=(10, 5.5), layout='constrained')
    axes = fig.subplots(2, 2)
    for values, label, color, marker in [(ideal, 'Ideal exit mask', BLUE, 'o'), (physical, 'Physical scalar absorber', ORANGE, 's')]:
        axes[0, 0].plot(orders, abs(values), marker+'-', color=color, label=label)
        valid = abs(values)>1e-6
        axes[0, 1].plot(orders[valid], np.angle(np.asarray(values, complex)[valid]), marker, color=color, label=label)
    axes[0, 0].set(title='Same pitch: different coefficient magnitudes', xlabel='Order m', ylabel='|C_m|')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title='Coefficient phase (zeros omitted)', xlabel='Order m', ylabel='arg C_m (rad)', ylim=(-3.4,3.4))
    for duty in (.3, .5, .7):
        axes[1, 0].plot(orders, duty*np.sinc(orders*duty), 'o-', label=f'D_open={duty:.1f}')
    axes[1, 0].axhline(0,color='0.7',lw=.7)
    axes[1, 0].set(title='Ideal mask: signs change with open fraction', xlabel='Order m', ylabel='Signed C_m')
    axes[1, 0].legend(fontsize=8)
    # Collection is geometry; marker area conveys coefficient magnitude, not power.
    na = .6
    coordinates = orders*c.wavelength/c.period
    inside = abs(coordinates)<1
    for m, coordinate, value in zip(orders[inside], coordinates[inside], physical[inside]):
        accepted = abs(coordinate)<=na
        axes[1, 1].scatter(coordinate, 0, s=25+450*abs(value), color=BLUE if accepted else ORANGE)
        axes[1, 1].text(coordinate, .12, str(m), ha='center', fontsize=8)
    from matplotlib.patches import Circle
    axes[1, 1].add_patch(Circle((0,0),na,fill=False,color='0.5'))
    axes[1, 1].set(title='NA=0.6: area scales with |C|, not efficiency', xlabel='n alpha', ylabel='n beta',
                   xlim=(-1.05,1.05),ylim=(-1.05,1.05),aspect='equal')
    fig.suptitle('Exit plane z=-h; same incident normalization and common phase reference. Air on both sides.',fontsize=10)
    return fig


def flux_figure():
    fig = Figure(figsize=(6, 2.7), layout='constrained')
    ax = fig.subplots()
    angles = np.array([0.,30.,60.])
    ax.bar(np.arange(3)-.17, np.ones(3), width=.34, label='|C/A_i| squared')
    ax.bar(np.arange(3)+.17, np.cos(np.deg2rad(angles)), width=.34, label='Normal scalar efficiency')
    ax.set(xticks=np.arange(3), xticklabels=['0 deg','30 deg','60 deg'],ylabel='Relative value',
           title='Equal amplitudes, different normal flux; incident angle zero')
    ax.legend(fontsize=8)
    return fig
