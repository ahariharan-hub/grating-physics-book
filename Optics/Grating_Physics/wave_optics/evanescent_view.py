"""Notebook four: selectable evanescent orders, signed fields, and decay.

All plots are rendered in memory with a fixed linear colour scale. Their
unit surface amplitude illustrates one component, not a scattering solution.
"""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .evanescent import evanescent_order, order_phasor, decay_envelope
from .plotting import figure_png
from .grating_intro import _close_widget_tree


INK = '#243444'
BLUE = '#2866a3'
PURPLE = '#7951a3'
AMBER = '#aa650b'
ORANGE = '#b85c17'
COLOURS = dict(propagating=BLUE, grazing=AMBER, evanescent=PURPLE)
MARKERS = dict(propagating='o', grazing='D', evanescent='x')


def draw_evanescent_order(figure, data, *, tau=0., depth=2.):
    """Draw sampled integer candidates, one signed field, and its depth cut."""
    figure.clear()
    grid = figure.add_gridspec(2, 2, height_ratios=[.72, 2.1], width_ratios=[1.1, 1])
    ruler = figure.add_subplot(grid[0, :])
    field = figure.add_subplot(grid[1, 0])
    profile = figure.add_subplot(grid[1, 1])
    figure.subplots_adjust(left=.075, right=.96, top=.86, bottom=.15,
                           wspace=.34, hspace=.60)
    figure.suptitle(f'Order m = {data["order"]:+d}: {data["status"]}',
                    fontsize=16, color=COLOURS[data['status']], y=.98)
    caption = figure.text(.5, .928,
        f'λ = {data["wavelength"]:g} µm    d = {data["period"]:g} µm    '
        f'θᵢ = {data["angle_deg"]:+g}°    t/T = {tau:.3f}',
        ha='center', fontsize=10, color=INK)

    # This is a one-dimensional component axis, never a real-vector arrow
    # assigned to an evanescent harmonic. The seven labels are only a sample.
    candidates = [evanescent_order(data['wavelength'], data['period'], data['angle_deg'], m)
                  for m in range(-3, 4)]
    limits = [d['alpha'] for d in candidates] + [-1, 1]
    margin = max(.35, .07*(max(limits)-min(limits)))
    ruler.axvspan(-1, 1, color=BLUE, alpha=.09)
    ruler.axhline(0, color='#b2bbc3', lw=1)
    for boundary in (-1, 1):
        ruler.axvline(boundary, color=AMBER, ls=':', lw=1.2)
    for candidate in candidates:
        xx, status = candidate['alpha'], candidate['status']
        ruler.plot(xx, 0, marker=MARKERS[status], color=COLOURS[status], ms=7, mew=1.8)
        if candidate['order'] == data['order']:
            ruler.plot(xx, 0, 'o', ms=15, mfc='none', mec=INK, mew=1.3)
        ruler.text(xx, .28, f'm={candidate["order"]:+d}', ha='center', fontsize=9,
                   color=COLOURS[status])
    ruler.set(xlim=(min(limits)-margin, max(limits)+margin), ylim=(-.25, .65),
              yticks=[], xlabel='Order component kₘₓ / k; shaded band: −1 to +1')
    ruler.spines[['top', 'right', 'left']].set_visible(False)
    ruler.set_title('Sample of integer candidates (m = −3 … +3); ring = selected order',
                    fontsize=10, loc='left', pad=8)
    ruler.legend(handles=[Line2D([], [], ls='', marker=MARKERS[s], color=COLOURS[s],
                                 label=s.capitalize()) for s in COLOURS],
                 loc='lower right', bbox_to_anchor=(1, 1.17), ncol=3,
                 fontsize=8, frameon=False)

    x = np.linspace(-depth, depth, 481)
    z = np.linspace(-depth, 0, 301)
    wave = order_phasor(x[None, :], z[:, None], data)
    snapshot = (wave*np.exp(-2j*np.pi*tau)).real
    # Uniform samples are an image; preserve the cell edges of shading='auto'.
    dx, dz = x[1]-x[0], z[1]-z[0]
    picture = field.imshow(snapshot, origin='lower', interpolation='nearest',
        extent=(x[0]-dx/2, x[-1]+dx/2, z[0]-dz/2, z[-1]+dz/2),
        cmap='RdBu_r', vmin=-1, vmax=1)
    field.axhline(0, color=INK, lw=2, clip_on=False)
    field.axvline(0, color='#444444', ls=':', lw=1)
    # Preserve physical aspect ratio; depth is an explicit view control.
    field.set(xlim=(-depth, depth), ylim=(-depth, 0), aspect='equal', xlabel='x (µm)',
              ylabel='z (µm); outgoing side below 0', title='Signed field u / A')
    field.title.set_fontsize(11)
    field.text(.02, .97, 'Grating surface: z = 0', transform=field.transAxes,
               va='top', fontsize=8, color=INK,
               bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=2))
    colour_axis = figure.add_axes([.11, .065, .32, .014])
    figure.colorbar(picture, cax=colour_axis, orientation='horizontal',
                    ticks=[-1, 0, 1], label='Same linear field scale in every state')

    s = np.linspace(0, depth, 601)
    envelope = decay_envelope(s, data)
    cut_wave = order_phasor(0., -s, data)
    cut = (cut_wave*np.exp(-2j*np.pi*tau)).real
    profile.plot(s, envelope, color=ORANGE, lw=2.3, label='Amplitude |U| / A')
    profile.plot(s, -envelope, color=ORANGE, lw=1, ls=':', label='Negative envelope')
    cut_line, = profile.plot(s, cut, color=BLUE, lw=1.8, ls='--', label='Signed field at x = 0')
    profile.axhline(0, color='#b2bbc3', lw=.8)
    if data['status'] == 'evanescent' and data['decay_length'] <= depth:
        ell = data['decay_length']
        profile.plot(ell, np.exp(-1), 'o', color=PURPLE, ms=5)
        profile.annotate('1/e at ℓ', (ell, np.exp(-1)), xytext=(9, 15),
                         textcoords='offset points', fontsize=9, color=PURPLE)
    profile.set(xlim=(0, depth), ylim=(-1.1, 1.15), xlabel='Distance s = −z (µm)',
                ylabel='Field / surface amplitude A', title='Follow the dotted x = 0 line')
    profile.title.set_fontsize(11)
    profile.spines[['top', 'right']].set_visible(False)
    profile.grid(alpha=.15)
    profile.legend(loc='lower right', fontsize=8, framealpha=.95)
    figure.text(.72, .025, 'One component with A = 1 at the surface.\n'
                'The envelope is independent of time. No power is plotted.',
                ha='center', fontsize=9, color=INK, linespacing=1.5)
    return dict(wave=wave, cut_wave=cut_wave, picture=picture,
                cut_line=cut_line, caption=caption)


def transition_figure():
    """Same order and colour scale on either side of the normal-incidence cutoff."""
    figure = Figure(figsize=(12, 4.5), dpi=110, facecolor='white')
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    figure.subplots_adjust(left=.06, right=.97, top=.78, bottom=.23, wspace=.25)
    figure.suptitle('Follow m = +1: λ = 2 µm, normal incidence, t = 0', fontsize=14)
    x, z = np.linspace(-2, 2, 481), np.linspace(-2, 0, 301)
    for axis, period in zip(axes, (3., 2., 1.5)):
        data = evanescent_order(2., period, 0., 1)
        picture = axis.pcolormesh(x, z, order_phasor(x[None, :], z[:, None], data).real,
                                  cmap='RdBu_r', vmin=-1, vmax=1, shading='auto',
                                  rasterized=True)
        axis.axhline(0, color=INK, lw=2, clip_on=False)
        axis.set(xlim=(-2, 2), ylim=(-2, 0), aspect='equal', xlabel='x (µm)',
                 title=f'd = {period:g} µm: {data["status"]}\nkₘₓ/k = {data["alpha"]:.3f}')
        axis.title.set_fontsize(10)
        axis.title.set_color(COLOURS[data['status']])
    axes[0].set_ylabel('z (µm)')
    colour_axis = figure.add_axes([.36, .105, .28, .024])
    figure.colorbar(picture, cax=colour_axis, orientation='horizontal', ticks=[-1, 0, 1],
                    label='Signed field u / A; A = 1 at the surface in every panel')
    return figure


class EvanescentOrderExplorer:
    """Keep the integer selection when its propagation classification changes."""

    def __init__(self):
        import ipywidgets as widgets

        self._closed = False
        self._updating = False
        self._plot_signature = None
        self._artists = None
        style = {'description_width': 'initial'}

        def slider(description, value, minimum, maximum, step):
            return widgets.FloatSlider(description=description, value=value, min=minimum,
                max=maximum, step=step, continuous_update=False, style=style,
                readout_format='.2f', layout=widgets.Layout(width='330px'))

        self.wavelength = slider('λ in medium (µm)', 2, 1, 4, .05)
        self.period = slider('Period d (µm)', 2, 1, 4.5, .05)
        self.angle = slider('θᵢ from −z (°)', 30, -60, 60, 5)
        self.order = widgets.Dropdown(options=[(f'm = {m:+d}', m) for m in range(-3, 4)],
            value=1, description='Order', style=style, layout=widgets.Layout(width='170px'))
        self.time = slider('Time t/T', 0, 0, 1, .025)
        self.time.readout_format = '.3f'
        self.depth = slider('Depth shown (µm)', 2, .25, 4, .25)
        self.play = widgets.Play(value=0, min=0, max=40, interval=400,
                                 description='Play one cycle', repeat=False)
        self._clock_link = widgets.link((self.play, 'value'), (self.time, 'value'),
                                       (lambda frame: frame/40, lambda tau: int(round(tau*40))))
        self.reset = widgets.Button(description='Notebook 3 example', layout=widgets.Layout(width='175px'))
        self.propagating = widgets.Button(description='Propagating example', layout=widgets.Layout(width='175px'))
        self.grazing = widgets.Button(description='Grazing example')
        self.evanescent = widgets.Button(description='Evanescent example', layout=widgets.Layout(width='175px'))
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1320px', height='auto'))
        self._controls = (self.wavelength, self.period, self.angle, self.order, self.time, self.depth)

        def row(children):
            return widgets.Box(children, layout=widgets.Layout(display='flex', flex_flow='row wrap',
                                                                align_items='center'))

        self.ui = widgets.VBox([row([self.wavelength, self.period, self.angle]),
            row([self.order, self.time, self.play, self.depth]),
            row([self.reset, self.propagating, self.grazing, self.evanescent]),
            self.readout, self.picture])
        self.figure = Figure(figsize=(12.4, 7.5), dpi=105, facecolor='white')
        FigureCanvasAgg(self.figure)
        for control in self._controls:
            control.observe(self._update, names='value')
        self._buttons = (self.reset, self.propagating, self.grazing, self.evanescent)
        for button in self._buttons:
            button.on_click(self._preset)
        self._update()

    def _update(self, change=None):
        if self._closed or self._updating:
            return
        signature = (self.wavelength.value, self.period.value, self.angle.value,
                     self.order.value, self.depth.value)
        if signature != self._plot_signature:
            self.data = evanescent_order(*signature[:4])
            self._artists = draw_evanescent_order(
                self.figure, self.data, tau=self.time.value, depth=self.depth.value)
            self._plot_signature = signature
        else:
            rotation = np.exp(-2j*np.pi*self.time.value)
            self._artists['picture'].set_data((self._artists['wave']*rotation).real)
            self._artists['cut_line'].set_ydata((self._artists['cut_wave']*rotation).real)
            caption = self._artists['caption']
            caption.set_text(caption.get_text().rsplit('t/T = ', 1)[0]
                             + f't/T = {self.time.value:.3f}')
        data = self.data
        self.picture.value = figure_png(self.figure)
        if data['status'] == 'evanescent':
            detail = (f'k<sub>mz</sub> = −i {data["kappa"]:.3f} µm⁻¹; '
                      f'amplitude decay length ℓ = {data["decay_length"]:.3f} µm. '
                      'No real outgoing angle.')
        elif data['status'] == 'grazing':
            detail = 'k<sub>mz</sub> = 0; parallel to the grating. No decay and no travel away from it.'
        else:
            detail = (f'k<sub>mz</sub> = {data["kz"].real:+.3f} rad/µm; '
                      f'outgoing angle = {data["outgoing_angle"]:+.2f}° from −z. '
                      'The amplitude stays constant.')
        self.readout.value = (
            '<div style="padding:8px 0;line-height:1.7">'
            f'<b>Selected m = {data["order"]:+d}: {data["status"]}.</b> '
            f'kₘₓ/k = {data["alpha"]:+.3f}; kₘₓ = {data["kx"]:+.3f} rad/µm. {detail}<br>'
            f'Extra phase per period: (kₘₓ − kᵢₓ)d = 2π × ({data["order"]:+d}). '
            'This integer condition still holds in every state.<br>'
            '<b>Time moves the signed field; the orange amplitude envelope stays fixed.</b> '
            'The seven order labels are a sample. The selected order remains selected across grazing.</div>')

    def _preset(self, button):
        if self._closed:
            return
        self._updating = True
        try:
            self.play.playing = False
            period, angle = ((2., 30.) if button is self.reset else
                             (3., 0.) if button is self.propagating else
                             (2., 0.) if button is self.grazing else (1.5, 0.))
            for control, value in zip(self._controls, (2., period, angle, 1, 0., 2.)):
                control.value = value
        finally:
            self._updating = False
        self._update()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.play.playing = False
        self._clock_link.unlink()
        for control in self._controls:
            control.unobserve(self._update, names='value')
        for button in self._buttons:
            button.on_click(self._preset, remove=True)
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        self._artists = self._plot_signature = None
        _close_widget_tree(self.ui)
