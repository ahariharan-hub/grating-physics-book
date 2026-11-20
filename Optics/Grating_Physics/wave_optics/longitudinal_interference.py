"""Sample notebook seven's two fields on x=y=0 and add their real values.

The common carrier is restored after evaluating the existing apparatus model.
Choose the observation instant so beam 1 is at a crest at the detector centre.
This changes neither relative phase nor intensity. No figures on import.
"""

import numpy as np

from .detector_interference import detector_field


def longitudinal_fields(offset_um, geometry, *, wavelength_nm=632.8,
                        power_mw=1., crossing_diameter_mm=.54,
                        mm_per_unit=1.8, optics_phase_deg=0.):
    """Return phasors and one real-field snapshot along z through P.

    offset_um is z-z_D in micrometres; both amplitudes and relative phase
    come from the same two arms used for the detector calculation.
    """
    offset = np.asarray(offset_um, dtype=float)
    centre_mm = (geometry['detector'][2] - geometry['splitter'][2]) * mm_per_unit
    sampled = detector_field(
        0., 0., geometry, z_mm=centre_mm + offset * 1e-3,
        wavelength_nm=wavelength_nm, power_mw=power_mw,
        crossing_diameter_mm=crossing_diameter_mm,
        mm_per_unit=mm_per_unit, optics_phase_deg=optics_phase_deg)
    carrier = np.exp(-2j * np.pi * offset / (wavelength_nm * 1e-3))
    first = sampled['field_1'] * carrier
    second = sampled['field_2'] * carrier
    return dict(offset_um=offset, field_1=first, field_2=second,
                real_1=first.real, real_2=second.real,
                real_sum=(first + second).real,
                intensity=sampled['intensity'],
                phase_difference=sampled['phase_difference'])


def draw_longitudinal_interference(geometry, *, wavelength_nm=632.8,
                                   power_mw=1., crossing_diameter_mm=.54,
                                   mm_per_unit=1.8, optics_phase_deg=0.):
    """Draw the individual fields and their point-by-point sum on one z cut."""
    import matplotlib.pyplot as plt

    parameters = dict(wavelength_nm=wavelength_nm, power_mw=power_mw,
                      crossing_diameter_mm=crossing_diameter_mm,
                      mm_per_unit=mm_per_unit, optics_phase_deg=optics_phase_deg)
    # Display 24 carrier periods: enough to see agreement change along z,
    # while resolving both sinusoids at the actual experimental wavelength.
    half_width = 12 * wavelength_nm * 1e-3
    offset = np.linspace(-half_width, half_width, 4001)
    data = longitudinal_fields(offset, geometry, **parameters)
    point = longitudinal_fields(0., geometry, **parameters)
    amplitude_limit = max(1., np.sqrt(power_mw) * .54 / crossing_diameter_mm)
    blue, orange, ink, teal = '#267b9b', '#d97635', '#304051', '#1495a3'
    fig, (individual, total) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.subplots_adjust(left=.075, right=.975, top=.79, bottom=.13, hspace=.43)
    for axis in (individual, total):
        axis.axhline(0, color='#bac5ce', lw=.8)
        axis.axvline(0, color=teal, ls=':', lw=1)
        axis.set_xlim(-half_width, half_width)
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_color('#a1adb7')
        axis.tick_params(labelsize=9, colors='#62717e')
        axis.annotate('$P$', xy=(0, .97), xycoords=('data', 'axes fraction'),
                      xytext=(5, -3), textcoords='offset points', va='top',
                      fontsize=11, color=teal,
                      bbox=dict(facecolor='white', edgecolor='none', pad=.6))
    for key, color, label in (('real_1', blue, 'Beam 1'),
                              ('real_2', orange, 'Beam 2 sampled along z')):
        individual.plot(offset, data[key], color=color, lw=1.45, label=label)
        individual.plot(0, point[key], 'o', color=color, ms=4)
    individual.set_ylim(-1.15 * amplitude_limit, 1.15 * amplitude_limit)
    individual.set_ylabel('Real field', fontsize=11, color=ink)
    individual.set_title('1. Both waves on the same observation line',
                         loc='left', fontsize=11, color=ink, pad=10)
    individual.legend(loc='lower right', bbox_to_anchor=(1, 1.015), ncol=2,
                       frameon=False, fontsize=10)
    total.plot(offset, data['real_sum'], color=ink, lw=1.55)
    total.plot(0, point['real_sum'], 'o', color=teal, ms=4)
    total.set_ylim(-2.2 * amplitude_limit, 2.2 * amplitude_limit)
    total.set_ylabel('Real field sum', fontsize=11, color=ink)
    total.set_xlabel(r'$z-z_D$ ($\mu$m), along $x=y=0$', fontsize=11, color=ink)
    total.set_title(r'2. Add at each point: $u=u_1+u_2$',
                    loc='left', fontsize=11, color=ink, pad=10)
    wavelength_um = wavelength_nm * 1e-3
    period_2 = wavelength_um / abs(geometry['outgoing'][2])
    phase_at_point = float(np.rad2deg(point['phase_difference']) % 360)
    fig.text(.075, .95, 'Sample both waves along z, then add their fields',
             fontsize=15, color=ink)
    fig.text(.075, .89,
             rf'$\lambda={wavelength_um:.3f}\,\mu\mathrm{{m}}$; '
             rf'$s=\lambda/\cos\theta={period_2:.3f}\,\mu\mathrm{{m}}$; '
             rf'$\Delta\Phi(P)={phase_at_point:.1f}^\circ$ (modulo $360^\circ$).',
             fontsize=11, color=ink)
    fig.text(.075, .025,
             'One instant, chosen so beam 1 peaks at P. Field curves use the same '
             'amplitude units; the detector records cycle-averaged intensity.',
             fontsize=10, color='#62717e')
    data.update(point=point, period_2_um=period_2,
                phase_at_point_deg=phase_at_point)
    return fig, data
