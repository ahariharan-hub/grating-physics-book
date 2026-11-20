"""Thin-mask exit-field views for Notebook 9.1.

The main example reuses the binary mask and incident wave from Lesson 2.2.
A smooth three-harmonic mask is retained for an optional exact order-sum check.
No FFT, propagation, or material-scattering calculation is performed here.
"""
from io import BytesIO

import ipywidgets as widgets
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .grating_intro import GratingSurfaceExplorer, amplitude_grating, _close_widget_tree
from .plotting import figure_png


class ThinMaskExitFieldExplorer:
    """Add the thin-mask product to the Lesson 2.2 view over a fixed 10 µm.

    Period and open duty fraction set the geometry. Wavelength, angle,
    coordinates, mask and phase are shared with the incoming-wave view.
    The period control uses nanometres; plotted lengths use micrometres; A_i=1.
    The comparison repeats the central cell without its inter-cell Bloch phase.
    """

    def __init__(self):
        self._closed = False
        self.surface = GratingSurfaceExplorer(field_width_um=10., adjustable_duty=True)
        # Open on an oblique example so the PBC comparison is visible at once.
        self.surface.angle.value = 30
        self.period = self.surface.period
        self.duty_cycle = self.surface.duty_cycle
        self.wavelength = self.surface.wavelength
        self.angle = self.surface.angle
        self._controls = self.surface._controls
        self.figure = Figure(figsize=(10, 7), dpi=115, facecolor='white')
        FigureCanvasAgg(self.figure)
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1100px', height='auto'))
        self.ui = widgets.VBox([
            widgets.HTML('<b>1. Incident wave and mask over a fixed 10 µm field</b>'),
            self.surface.ui,
            widgets.HTML('<b>2. Exit field and the ordinary-PBC cell copy</b>'),
            self.picture, self.readout,
        ])
        for control in self._controls:
            control.observe(self._update, names='value')
        self._update()

    def _update(self, change=None):
        if self._closed:
            return
        # The surface view registered its callbacks first, so its data already
        # describe these controls. Reuse that same mask and incident phase.
        source = self.surface.data
        x = source['x']
        transmission = source['transmission']
        incident = np.exp(1j*source['surface_phase'])
        field = transmission*incident
        period = source['period_um']
        local_x = (x+period/2) % period-period/2
        # Tile the actual reference cell [-d/2,d/2) with no phase weighting
        # between copies. This constructs a different illumination unless
        # exp(i*kx*d)=1; it is not another solution for the same plane wave.
        pbc_field = transmission*np.exp(1j*source['k_vector'][0]*local_x)
        phase = np.full(x.shape, np.nan)
        pbc_phase = np.full(x.shape, np.nan)
        opened = transmission > 0
        phase[opened] = np.angle(field[opened])
        pbc_phase[opened] = np.angle(pbc_field[opened])
        self.data = dict(x=x, transmission=transmission, U_i=incident,
                         U_exit=field, amplitude=abs(field), phase=phase,
                         local_x_um=local_x, U_ordinary_pbc=pbc_field,
                         phase_ordinary_pbc=pbc_phase,
                         periodic_factor=field*np.exp(-1j*source['surface_phase']),
                         kx_i=source['k_vector'][0], phase_step=source['phase_step'])
        for name in ('period_um', 'opening_width_um', 'closed_width_um',
                     'field_width_um', 'spatial_frequency_per_um', 'G_rad_per_um',
                     'periods_in_field', 'duty_cycle'):
            self.data[name] = source[name]

        self.figure.clear()
        axes = self.figure.subplots(2, 2)
        self.figure.subplots_adjust(left=.08, right=.98, bottom=.13, top=.91,
                                    hspace=.65, wspace=.25)
        mask_ax, field_ax, amplitude_ax, phase_ax = axes.flat
        width = source['opening_width_um']
        half_field = source['field_width_um']/2
        centres = period*np.arange(-int(np.ceil(half_field/period))-1,
                                    int(np.ceil(half_field/period))+2)
        for ax in axes.flat:
            for centre in centres:
                ax.axvspan(centre-width/2, centre+width/2,
                           color='#dff4fc', zorder=0)
            ax.set(xlim=(-half_field, half_field), xlabel='x (µm)')
            ax.axhline(0, color='#888888', lw=.6, zorder=0)
            ax.grid(axis='y', alpha=.15)
        mask_ax.step(x, transmission, where='mid', color='#222222', lw=1.8)
        mask_ax.set(title='Mask: 1 in openings, 0 behind opaque strips',
                    ylabel='Transmission t(x)', ylim=(-.12, 1.22))
        for ax in (field_ax, phase_ax):
            for edge in (-period/2, period/2):
                ax.axvline(edge, color='#777777', ls=':', lw=1)
        field_ax.plot(x, field.real, color='#2674af', lw=1.8, label='Actual Re')
        field_ax.plot(x, field.imag, color='#d46b19', lw=1.8, label='Actual Im')
        field_ax.plot(x, pbc_field.real, '--', color='#2674af', lw=1.4, label='PBC copy Re')
        field_ax.plot(x, pbc_field.imag, '--', color='#d46b19', lw=1.4, label='PBC copy Im')
        field_ax.set(title='Complex field: actual vs ordinary-PBC copy',
                     ylabel='Field components', ylim=(-1.22, 1.22))
        field_ax.legend(loc='upper center', bbox_to_anchor=(.5, -.23),
                        fontsize=8, ncol=2, borderaxespad=0)
        amplitude_ax.step(x, abs(field), where='mid', color='#222222', lw=1.8)
        amplitude_ax.set(title=r'Same amplitude for both fields: $|U|=t$',
                         ylabel='Exit-field amplitude', ylim=(-.12, 1.22))
        # Avoid joining +pi to -pi with a line across a wrapped phase jump.
        for values, color, style, label in (
                (phase, '#14758c', '-', 'Actual phase'),
                (pbc_phase, '#a1439b', '--', 'PBC copy phase')):
            plotted_phase = values.copy()
            jumps = np.flatnonzero(abs(np.diff(values)) > np.pi)+1
            plotted_phase[jumps] = np.nan
            phase_ax.plot(x, plotted_phase, style, color=color, lw=1.8, label=label)
        phase_ax.set(title='Wrapped phase: actual vs ordinary-PBC copy',
                     ylabel='Exit phase (rad)', ylim=(-1.18*np.pi, 1.18*np.pi),
                     yticks=[-np.pi, 0, np.pi], yticklabels=['−π', '0', 'π'])
        phase_ax.legend(loc='upper center', bbox_to_anchor=(.5, -.23),
                        fontsize=8, ncol=2, borderaxespad=0)
        phase_ax.text(.02, .04, 'No phase where the field is zero',
                      transform=phase_ax.transAxes, fontsize=8, color='#555555')
        self.picture.value = figure_png(self.figure)

        factor = np.exp(1j*source['phase_step'])
        status = ('The ordinary-PBC copy matches the actual field for this cell phase.'
                  if np.isclose(factor, 1., atol=1e-12, rtol=0)
                  else 'The actual field has a Bloch phase offset; ordinary PBC changes the field.')
        self.readout.value = (
            '<div style="padding:8px 0"><b>Solid = actual field; dashed = ordinary-PBC copy.</b> '
            'The dashed curves repeat the central cell −d/2 ≤ x &lt; d/2 '
            '(gray dotted edges) with identical complex values in every cell. '
            'Blue bands are openings. <b>Both fields have the same amplitude.</b><br>'
            f'p = d = {period:.2f} µm; '
            f'k<sub>x,i</sub> = {source["k_vector"][0]:+.3f} rad/µm; '
            f'Δφ<sub>cell</sub> = k<sub>x,i</sub>p = {source["phase_step"]:+.3f} rad; '
            f'Δφ<sub>field</sub> = k<sub>x,i</sub>L = '
            f'{source["k_vector"][0]*source["field_width_um"]:+.3f} rad.<br>'
            'At corresponding opening centres: U<sub>exit</sub>(0) = 1; '
            f'U<sub>exit</sub>(p) = {factor.real:+.3f} {factor.imag:+.3f}i; '
            'the ordinary-PBC copy has U(p) = 1. '
            f'<b>{status}</b><br>'
            'Removing the incident phase gives '
            'U<sub>exit</sub>(x) exp(−ik<sub>x,i</sub>x) = t(x), '
            'which repeats after p.</div>')

    def close(self):
        """Release both owned views and their observers; safe on cell reruns."""
        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        self.surface.close()
        self.picture.value = b''
        self.figure.clear()
        self.figure = None
        self.data = None
        _close_widget_tree(self.ui)


def _png(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format='png', facecolor='white')
    figure.clear()
    return buffer.getvalue()


def exit_field_periodicity_png(period_nm=1000., incident_amplitude=1., phase_step=np.pi/2,
                               opening_fraction=.4):
    """Compare a normally illuminated repeating field with a tilted Bloch field."""
    x = np.linspace(0, 3*period_nm, 2401)
    u = x/period_nm
    mask = amplitude_grating(x, period_nm, period_nm*opening_fraction)
    figure = Figure(figsize=(11.4, 7.6), dpi=110)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    figure.subplots_adjust(left=.08, right=.98, bottom=.08, top=.91,
                           hspace=.40, wspace=.22)
    for column, step in enumerate((0., phase_step)):
        incident_phase = step*u
        incident = incident_amplitude*np.exp(1j*incident_phase)
        field = mask*incident
        for ax in axes[:, column]:
            for boundary in (0, 1, 2, 3):
                ax.axvline(boundary, color='#db8655', lw=1, alpha=.6)
            ax.set_xlim(0, 3)
            ax.set_xticks([0, 1, 2, 3])
            ax.grid(axis='y', alpha=.12)
        axes[0, column].plot(u, mask, color='#222222', lw=2)
        axes[0, column].set(ylabel='Mask transmission t(x)', ylim=(0, 1.15),
                            title='Same mask repeats in every cell')
        axes[1, column].plot(u, incident_phase, color='#2674af', lw=2)
        axes[1, column].set(ylabel='Incident phase (rad)',
                            title='Phase continues across the cell boundaries')
        axes[1, column].set_ylim(-.2, max(1., 3*abs(phase_step))+.25)
        axes[2, column].plot(u, field.real, color='#2674af', lw=1.9, label='Re U_exit')
        axes[2, column].plot(u, field.imag, '--', color='#d46b19', lw=1.9, label='Im U_exit')
        values = incident_amplitude*np.exp(1j*step*np.arange(4))
        axes[2, column].plot(np.arange(4), values.real, 'o', color='#2674af', ms=5)
        axes[2, column].plot(np.arange(4), values.imag, 'x', color='#d46b19', ms=6)
        axes[2, column].set(ylabel='Complex exit field', xlabel='Position x / p',
                            ylim=(-1.2*incident_amplitude, 1.2*incident_amplitude),
                            title='Compare the field at x = 0, p, 2p, 3p')
        axes[2, column].legend(fontsize=8, loc='lower left')
    figure.text(.29, .965, 'A. Normal incidence: exact repetition after p',
                ha='center', fontsize=12, weight='bold')
    figure.text(.76, .965, 'B. Tilted incidence: a phase step between cells',
                ha='center', fontsize=12, weight='bold')
    return _png(figure)


def exit_field_reconstruction_png(period_nm=1000., incident_amplitude=1., phase_step=np.pi/2):
    """Show the same exit field as a local product and an exact three-order sum."""
    x = np.linspace(0, period_nm, 701)
    G = 2*np.pi/period_nm
    incident_kx = phase_step/period_nm
    modes = np.array([-1, 0, 1])
    coefficients = incident_amplitude*np.array([.2, .6, .2])
    product = incident_amplitude*(.6+.4*np.cos(G*x))*np.exp(1j*incident_kx*x)
    reconstruction = np.exp(1j*x[:, None]*(incident_kx+modes*G)) @ coefficients
    figure = Figure(figsize=(11.4, 3.8), dpi=110)
    FigureCanvasAgg(figure)
    spatial, spectral = figure.subplots(1, 2)
    figure.subplots_adjust(left=.075, right=.98, bottom=.23, top=.87, wspace=.25)
    for quantity, color, label in ((np.real, '#2674af', 'Real part'),
                                    (np.imag, '#d46b19', 'Imaginary part')):
        spatial.plot(x/period_nm, quantity(product), color=color, lw=2, label=label+' of product')
        spatial.plot(x[::35]/period_nm, quantity(reconstruction)[::35], 'o',
                     mfc='none', color=color, ms=5, label=label+' of order sum')
    spatial.set(title='Same field, two equivalent calculations', xlabel='Position x / p',
                 ylabel='U_exit components')
    figure.legend(*spatial.get_legend_handles_labels(), fontsize=7, ncol=2,
                   loc='lower center', bbox_to_anchor=(.285, .01))
    spatial.grid(alpha=.15)
    spectral.vlines(modes, 0, abs(coefficients), color='#2674af', lw=2)
    spectral.plot(modes, abs(coefficients), 'o', color='#2674af', ms=7)
    spectral.set(title='Order-space data used for the reconstruction', xlabel='Order label m',
                  ylabel='Coefficient magnitude |C_m|', xticks=[-1, 0, 1], xlim=(-1.5, 1.5),
                  ylim=(0, .85*incident_amplitude))
    for m, coefficient in zip(modes, coefficients):
        spectral.text(m, abs(coefficient)+.04*incident_amplitude,
                       f'{coefficient:g}', ha='center', color='#2674af')
    spectral.grid(alpha=.15)
    return _png(figure)
