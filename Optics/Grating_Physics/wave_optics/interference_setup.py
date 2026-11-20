"""Notebook seven's beam-splitter experiment, propagating towards negative z.

All geometry uses right-handed laboratory (x,y,z) coordinates, with S at
the origin. The Matplotlib side view plots x horizontally and z vertically;
positive y points into the page. Beam widths indicate geometric contours,
while linked detector views calculate scalar fields on z=z_D.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon, Rectangle


def cylindrical_overlap_volume(diameter_mm, theta_deg):
    """Equal circular cylinders whose axes intersect at the full angle theta.

    Returns mm^3 for a diameter in mm. Cylinders must extend through their
    entire intersection; the result describes a chosen beam-width contour.
    See J. H. Hubbell (1965), doi:10.6028/jres.069c.017.
    """
    if not np.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError('The diameter at the crossing must be finite and positive.')
    if not np.isfinite(theta_deg) or not 0 < theta_deg < 180:
        raise ValueError('Intersecting cylinder axes require 0 < theta < 180 degrees.')
    return float(2 * diameter_mm**3 / (3 * np.sin(np.deg2rad(theta_deg))))


def two_beam_geometry(mirror_tilt_deg=0.0, detector_z=-6.9, beam_half_width=.15):
    """Reflect beam 2 and intersect a detector in the plane z=z_D<0.

    Positions are 3D vectors in schematic units. Beam 1 propagates along -z;
    the mirror is at (h,0,0), and beam 2 approaches from positive x.
    Positive mirror tilt is counterclockwise in the x-horizontal, z-up view.
    """
    if not np.isfinite(mirror_tilt_deg) or abs(mirror_tilt_deg) > 2:
        raise ValueError('Use a finite mirror tilt between -2 and +2 degrees.')
    if not np.isfinite(detector_z) or not -10 <= detector_z <= -3.5:
        raise ValueError('Use a detector z-position between -10 and -3.5 schematic units.')
    if not np.isfinite(beam_half_width) or beam_half_width <= 0:
        raise ValueError('Use a finite positive beam half-width.')
    splitter = np.zeros(3)
    mirror = np.array([2.1, 0., 0.])
    detector = np.array([0., 0., detector_z])
    aligned_target = np.array([0., 0., -6.9])
    beam1_direction = np.array([0., 0., -1.])
    incoming = np.array([1., 0., 0.])
    reference = (aligned_target-mirror)/np.linalg.norm(aligned_target-mirror)
    tangent = incoming+reference
    tangent /= np.linalg.norm(tangent)
    angle = np.deg2rad(mirror_tilt_deg)
    c, s = np.cos(angle), np.sin(angle)
    # A counterclockwise rotation in the displayed xz plane is about -y.
    tangent = np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]]) @ tangent
    outgoing = 2*np.dot(incoming, tangent)*tangent-incoming
    hit = mirror+(detector_z-mirror[2])/outgoing[2]*outgoing
    half_width = float(beam_half_width)
    footprint_1 = (-half_width, half_width)
    footprint_2 = (hit[0]-half_width/abs(outgoing[2]),
                   hit[0]+half_width/abs(outgoing[2]))
    overlap = (max(footprint_1[0], footprint_2[0]),
               min(footprint_1[1], footprint_2[1]))
    detector_interval = (-.65, .65)
    # n2.r measures transverse distance in the crossing plane.
    normal = np.array([-outgoing[2], 0., outgoing[0]])
    overlap_outline = np.array([
        [x, 0., (offset-normal[0]*(x-mirror[0]))/normal[2]]
        for x, offset in ((-half_width, -half_width), (-half_width, half_width),
                          (half_width, half_width), (half_width, -half_width))
    ])
    axis_crossing = mirror-mirror[0]/outgoing[0]*outgoing

    def intercepted(interval):
        low, high = max(interval[0], detector_interval[0]), min(interval[1], detector_interval[1])
        return (low, high) if high > low else None

    return dict(splitter=splitter, mirror=mirror, detector=detector,
                detector_z=float(detector_z), beam1_hit=detector.copy(),
                beam1_direction=beam1_direction, tangent=tangent,
                incoming=incoming, outgoing=outgoing, hit=hit,
                theta_deg=float(np.rad2deg(np.arccos(np.clip(-outgoing[2], -1, 1)))),
                theta_reference_deg=float(np.rad2deg(np.arccos(-reference[2]))),
                footprint_1=footprint_1, footprint_2=footprint_2,
                beam_half_width=half_width, overlap=overlap,
                has_overlap=overlap[1] > overlap[0],
                overlap_outline=overlap_outline, axis_crossing=axis_crossing,
                detector_interval=detector_interval,
                sampled_footprints=(intercepted(footprint_1), intercepted(footprint_2)),
                sampled_overlap=intercepted(overlap))


def _detector_note(geometry):
    """Describe intercepted schematic widths, including separated spots."""
    first, second = geometry['sampled_footprints']
    if geometry['sampled_overlap'] is not None:
        return 'Shared beam region\non detector'
    if first is not None and second is not None:
        return 'Both beams on detector,\nin separate regions'
    if first is not None:
        return 'Beam 1 footprint\nonly on detector'
    if second is not None:
        return 'Beam 2 footprint\nonly on detector'
    return 'Detector outside both\nillustrated footprints'


def _beam_vertices(start, end, start_surface, end_surface, half_width):
    direction = (end-start)/np.linalg.norm(end-start)
    transverse = np.array([-direction[1], direction[0]])

    def boundary(point, surface, offset):
        normal = np.array([-surface[1], surface[0]])
        shift = -offset*np.dot(transverse, normal)/np.dot(direction, normal)
        return point+offset*transverse+shift*direction

    return [boundary(start, start_surface, -half_width),
            boundary(end, end_surface, -half_width),
            boundary(end, end_surface, half_width),
            boundary(start, start_surface, half_width)]


def _eo_label(phase, varies):
    return ((rf'Added phase: $0^\circ\leftrightarrow{phase:.0f}^\circ$'
             '\nSwitches during exposure') if varies else
            (rf'Added phase: $\beta={phase:.0f}^\circ$' '\nHeld steady during exposure'))


def draw_two_plane_wave_setup(*, wavelength_nm=632.8, power_mw=1.0,
                             diameter_mm=.54, mirror_tilt_deg=0.,
                             detector_z=-6.9, beam_half_width=.15,
                             ax=None, compact=False, phase_step_deg=None, phase_varies=False,
                             _artists=None):
    """Draw the xz apparatus, optionally adding an ideal EO phase modulator.

    A phase_step_deg value shows the added EO phase; phase_varies selects
    alternation between zero and that phase during exposure. None keeps
    the original apparatus sketch. Ray geometry is
    unchanged: the modulator is represented as a pure phase element.
    """
    if phase_step_deg is not None and (
            not np.isfinite(phase_step_deg) or not 0 <= phase_step_deg <= 180):
        raise ValueError('Use a finite modulator phase step from 0 to 180 degrees.')
    ink, muted = '#243444', '#667580'
    orange, purple = '#ed7134', '#8966a4'
    geometry = two_beam_geometry(mirror_tilt_deg, detector_z, beam_half_width)
    side = lambda vector: np.asarray(vector)[..., [0, 2]]
    splitter, mirror, detector = (side(geometry[key]) for key in ('splitter', 'mirror', 'detector'))
    outgoing, tangent = side(geometry['outgoing']), side(geometry['tangent'])
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 8.2), facecolor='white')
        fig.subplots_adjust(left=.035, right=.985, bottom=.06, top=.98)
    else:
        fig = ax.figure
    ax.set(xlim=(-2.5, 10.1), ylim=(-10.9, 3.7), aspect='equal')
    ax.set_axis_off()
    splitter_tangent = np.array([1., -1.])/np.sqrt(2)
    horizontal = np.array([1., 0.])
    artists = {} if _artists is None else _artists
    artists.update(beams=[], surfaces=[], directions=[], footprints=[])

    def beam(start, end, start_surface, end_surface, sampling_point=None):
        shade = ax.add_patch(Polygon(_beam_vertices(
            start, end, start_surface, end_surface, beam_half_width),
            closed=True, facecolor=orange, edgecolor='none', alpha=.16, zorder=1))
        stop = end if sampling_point is None else sampling_point
        line, = ax.plot(*np.array([start, stop]).T, color=orange, lw=2.4, zorder=2)
        continuation = None
        if sampling_point is not None:
            continuation, = ax.plot(*np.array([stop, end]).T, color=orange, lw=1.3, ls='--', alpha=.7)
        delta = stop-start
        arrow = ax.annotate('', xy=start+.64*delta, xytext=start+.47*delta,
                    arrowprops=dict(arrowstyle='-|>', color=orange, lw=2, mutation_scale=14),
                    zorder=3)
        artists['beams'].append((shade, line, continuation, arrow))

    end_z = -10.55
    beam(np.array([0., 1.75]), splitter, horizontal, splitter_tangent)
    beam(splitter, np.array([0., end_z]), splitter_tangent, horizontal,
         sampling_point=detector)
    beam(splitter, mirror, splitter_tangent, tangent)
    beam2_end = mirror+(end_z-mirror[1])/outgoing[1]*outgoing
    beam(mirror, beam2_end, tangent, horizontal, sampling_point=side(geometry['hit']))
    artists['overlap'] = ax.add_patch(Polygon(side(geometry['overlap_outline']), closed=True,
                         facecolor=purple, edgecolor='#684688', alpha=.35, lw=1.1))
    artists['overlap_note'] = ax.annotate('Overlap volume\n(side view)', xy=side(geometry['axis_crossing']),
                xytext=(3.4, -5.5), ha='left', va='center', color='#684688', fontsize=10,
                arrowprops=dict(arrowstyle='-', color=purple, lw=1))

    ax.add_patch(Rectangle((-.85, 1.75), 1.7, 1.4,
                           facecolor='#195e79', edgecolor=ink, lw=1.5, zorder=4))
    ax.text(0, 2.45, 'Collimated\nlaser', ha='center', va='center',
            color='white', fontsize=11, zorder=5)

    def optical_surface(centre, surface, length, thickness, colour):
        normal = np.array([-surface[1], surface[0]])
        a, b = centre-length*surface, centre+length*surface
        patch = ax.add_patch(Polygon([a, b, b+thickness*normal, a+thickness*normal],
                             closed=True, facecolor=colour, edgecolor='none', zorder=4))
        line, = ax.plot(*np.array([a, b]).T, color=ink, lw=2.4, zorder=5)
        artists['surfaces'].append((patch, line))

    optical_surface(splitter, splitter_tangent, .38, .08, '#d7ebf2')
    optical_surface(mirror, tangent, .48, .12, '#c6cdd3')
    ax.text(-.45, .20, 'Beam splitter\n$S=(0,0,0)$', ha='right', va='bottom', fontsize=10, color=ink)
    artists['mirror_text'] = ax.text(mirror[0], .9, 'Mirror $M=(h,0,0)$\n'+rf'$\delta={mirror_tilt_deg:+.2f}^\circ$',
            ha='center', va='bottom', fontsize=10, color=ink)
    ax.annotate('', xy=(mirror[0], -.70), xytext=(0, -.70),
                arrowprops=dict(arrowstyle='<->', color=muted, lw=1))
    ax.text(mirror[0]/2, -.93, '$h$', ha='center', va='top', fontsize=12, color=muted)

    if phase_step_deg is not None:
        teal = '#007b87'
        centre_x = mirror[0]/2
        modulator = Rectangle((centre_x-.36, -.25), .72, .50,
                              facecolor='#d6f1ee', edgecolor=teal, lw=1.5, zorder=6)
        modulator.set_gid('phase-modulator')
        ax.add_patch(modulator)
        # The optical axis continues through the crystal without deflection.
        ax.plot([centre_x-.36, centre_x+.36], [0, 0], color=orange, lw=2.4, zorder=7)
        # Electrical connection to the phase control, routed above M.
        ax.plot([centre_x, centre_x, 3.45], [.25, .64, .64],
                color=teal, lw=1.2, zorder=6)
        ax.plot(centre_x, .25, 'o', ms=3, color=teal, zorder=7)
        ax.text(3.65, 1.30, 'EO phase modulator', color=teal,
                fontsize=11, fontweight='bold', ha='left', va='top')
        artists['eo_label'] = ax.text(3.65, .76, _eo_label(phase_step_deg, phase_varies),
                ha='left', va='top', color=ink, fontsize=10, linespacing=1.6)

    artists['detector_plane'], = ax.plot([-1.2, 3.2], [detector_z]*2, ':', color='#aebbc4', lw=1, zorder=0)
    low, high = geometry['detector_interval']
    artists['detector'] = ax.add_patch(Rectangle((low, detector_z-.07), high-low, .14,
                           facecolor='#d7ebf2', edgecolor=ink, lw=1.4, zorder=4))
    for index, footprint in enumerate((*geometry['sampled_footprints'], geometry['sampled_overlap'])):
        lo, hi = (0., 0.) if footprint is None else footprint
        patch = ax.add_patch(Rectangle((lo, detector_z-.07), hi-lo, .14,
            facecolor=purple if index == 2 else orange, alpha=1. if index == 2 else .5,
            edgecolor='none', zorder=6 if index == 2 else 5, visible=footprint is not None))
        artists['footprints'].append(patch)
    artists['detector_text'] = ax.text(-.95, detector_z, 'Detector\n'+rf'$z_D={detector_z:.2f}$',
            ha='right', va='center', fontsize=10, color=ink)
    artists['point'], = ax.plot(0, detector_z, 'o', color='#007b87', ms=4, zorder=8)
    artists['point_text'] = ax.annotate('$P$', xy=(0, detector_z), xytext=(6, -17),
                textcoords='offset points', color='#007b87', fontsize=11, zorder=9)
    artists['x_arrow'] = ax.annotate('', xy=(1., detector_z-.48), xytext=(0, detector_z-.48),
                arrowprops=dict(arrowstyle='->', color='#007b87', lw=1))
    artists['x_text'] = ax.text(1.10, detector_z-.48, '$x$', color='#007b87', va='center', fontsize=12)
    ax.text(-.35, -2.5, 'Beam 1', ha='right', va='center', color=ink, fontsize=11)
    beam2_label = mirror+(-2.8-mirror[1])/outgoing[1]*outgoing
    artists['beam2_text'] = ax.text(beam2_label[0]+.35, -2.8, 'Beam 2', color=ink, fontsize=11)
    if not compact:
        artists['detector_note'] = ax.annotate(_detector_note(geometry), xy=detector, xytext=(3.4, -8.1),
                    ha='left', va='center', fontsize=10, color=ink,
                    arrowprops=dict(arrowstyle='-', color=muted, lw=1))

    # Direction inset uses the same x-horizontal, z-up convention.
    inset_shift = 1.6 if phase_step_deg is not None else 0.
    origin = np.array([6.6+inset_shift, .9])
    ax.text(6.4+inset_shift, 2.0, 'Arrival directions', ha='center', fontsize=11, color=muted)
    for direction, label in ((np.array([0., -1.]), '1'), (outgoing, '2')):
        endpoint = origin+2*direction
        arrow = ax.annotate('', xy=endpoint, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=ink, lw=1.5))
        label_artist = ax.text(endpoint[0]+.12, endpoint[1]-.22, label, fontsize=10, color=ink)
        artists['directions'].append((arrow, label_artist))
    theta = geometry['theta_deg']
    artists['angle'] = ax.add_patch(Arc(origin, 1.35, 1.35, theta1=-90-theta, theta2=-90,
                     color=purple, lw=1.3))
    angle = np.deg2rad(-90-theta/2)
    artists['angle_text'] = ax.text(*(origin+1.05*np.array([np.cos(angle), np.sin(angle)])),
            r'$\theta$', ha='center', va='center', color=purple, fontsize=12)

    if not compact:
        specs = ('Example source (continuous wave)\n'
                 rf'$\lambda={wavelength_nm:g}\,\mathrm{{nm}}$'
                 '\n'+rf'$P={power_mw:g}\,\mathrm{{mW}}$'
                 '\n'+rf'$D_{{1/e^2}}={diameter_mm:g}\,\mathrm{{mm}}$'
                 '\nLinear polarization along $y$')
        ax.text(4.8, -2.2, specs, ha='left', va='top', fontsize=10.5, color=ink,
                linespacing=1.7, bbox=dict(boxstyle='round,pad=.55',
                                         facecolor='#f2f6f8', edgecolor='#dde5ea'))
    origin = np.array([7.9, -7.2])
    for delta, label, offset in (([1.4, 0], '$+x$', [.2, 0]),
                                 ([0, 1.4], '$+z$', [0, .25])):
        end = origin+delta
        ax.annotate('', xy=end, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=muted, lw=1.1))
        ax.text(*(end+offset), label, ha='center', va='center', fontsize=11, color=muted)
    ax.plot(*origin, 'o', ms=9, mfc='white', mec=muted)
    ax.plot(*origin, 'x', ms=5, color=muted)
    ax.text(origin[0], origin[1]-.45, '$+y$ into page', ha='center', fontsize=10, color=muted)
    ax.text(6.6, -9.4, 'Propagation: $-z$\nDetector plane: $z=z_D<0$',
            ha='center', fontsize=11, color=ink, linespacing=1.6)
    if not compact:
        fig.text(.045, .02, 'Beam-width contours; dashed axes continue beyond the detector. '
                 'Shading in this sketch indicates geometry, not intensity.',
                 fontsize=9, color=muted)
    return fig, ax


class _SetupSketch:
    """Retained Matplotlib artists and a cached background for the EO label."""

    def __init__(self, **parameters):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        self.artists = {}
        self.figure, self.axis = draw_two_plane_wave_setup(**parameters, _artists=self.artists)
        plt.close(self.figure)  # Owned here, rather than by pyplot's display registry.
        self.figure.set_dpi(110)
        self.canvas = FigureCanvasAgg(self.figure)
        self._background = None
        self._crop = None
        if 'eo_label' in self.artists:
            self.artists['eo_label'].set_animated(True)

    def set_geometry(self, geometry, tilt):
        a = self.artists
        side = lambda vector: np.asarray(vector)[..., [0, 2]]
        splitter, mirror, detector, outgoing, tangent = (
            side(geometry[key]) for key in ('splitter', 'mirror', 'detector', 'outgoing', 'tangent'))
        z = geometry['detector_z']
        horizontal = np.array([1., 0.])
        splitter_tangent = np.array([1., -1.])/np.sqrt(2)
        end = mirror+(-10.55-mirror[1])/outgoing[1]*outgoing
        beam_specs = (
            (np.array([0., 1.75]), splitter, horizontal, splitter_tangent, splitter),
            (splitter, np.array([0., -10.55]), splitter_tangent, horizontal, detector),
            (splitter, mirror, splitter_tangent, tangent, mirror),
            (mirror, end, tangent, horizontal, side(geometry['hit'])))
        for artists, (start, end, s1, s2, stop) in zip(a['beams'], beam_specs, strict=True):
            shade, line, continuation, arrow = artists
            shade.set_xy(_beam_vertices(start, end, s1, s2, geometry['beam_half_width']))
            line.set_data(*np.array([start, stop]).T)
            if continuation is not None:
                continuation.set_data(*np.array([stop, end]).T)
            arrow.xy = start+.64*(stop-start)
            arrow.set_position(start+.47*(stop-start))
        normal = np.array([-tangent[1], tangent[0]])
        first, second = mirror-.48*tangent, mirror+.48*tangent
        a['surfaces'][1][0].set_xy([first, second, second+.12*normal, first+.12*normal])
        a['surfaces'][1][1].set_data(*np.array([first, second]).T)
        a['mirror_text'].set_text('Mirror $M=(h,0,0)$\n'+rf'$\delta={tilt:+.2f}^\circ$')
        a['overlap'].set_xy(side(geometry['overlap_outline']))
        a['overlap_note'].xy = side(geometry['axis_crossing'])
        a['detector_plane'].set_ydata([z, z])
        a['detector'].set_y(z-.07)
        for patch, interval in zip(a['footprints'],
                (*geometry['sampled_footprints'], geometry['sampled_overlap']), strict=True):
            patch.set_visible(interval is not None)
            if interval is not None:
                patch.set_bounds(interval[0], z-.07, interval[1]-interval[0], .14)
        a['detector_text'].set_position((-.95, z))
        a['detector_text'].set_text('Detector\n'+rf'$z_D={z:.2f}$')
        a['point'].set_data([0.], [z])
        a['point_text'].xy = (0., z)
        a['x_arrow'].xy = (1., z-.48)
        a['x_arrow'].set_position((0., z-.48))
        a['x_text'].set_y(z-.48)
        beam_label = mirror+(-2.8-mirror[1])/outgoing[1]*outgoing
        a['beam2_text'].set_x(beam_label[0]+.35)
        a['detector_note'].xy = detector
        a['detector_note'].set_text(_detector_note(geometry))
        origin = np.array([8.2 if 'eo_label' in a else 6.6, .9])
        end = origin+2*outgoing
        a['directions'][1][0].xy = end
        a['directions'][1][1].set_position(end+np.array([.12, -.22]))
        theta = geometry['theta_deg']
        a['angle'].theta1 = -90-theta
        angle = np.deg2rad(-90-theta/2)
        a['angle_text'].set_position(origin+1.05*np.array([np.cos(angle), np.sin(angle)]))
        self._background = None

    def set_phase(self, phase, varies):
        self.artists['eo_label'].set_text(_eo_label(phase, varies))

    def png(self):
        from io import BytesIO
        from PIL import Image

        label = self.artists.get('eo_label')
        if self._background is None:
            self.canvas.draw()
            self._background = self.canvas.copy_from_bbox(self.figure.bbox)
            if self._crop is None:
                # Fixed crop avoids repeated tight-layout passes. All moving
                # elements stay within the established apparatus bounds.
                box = self.figure.get_tightbbox(self.canvas.get_renderer()).padded(.1)
                left, bottom, right, top = box.extents*self.figure.dpi
                height = self.canvas.get_width_height()[1]
                self._crop = tuple(round(v) for v in (left, height-top, right, height-bottom))
        else:
            self.canvas.restore_region(self._background)
        if label is not None:
            self.axis.draw_artist(label)
        pixels = Image.fromarray(np.asarray(self.canvas.buffer_rgba()))
        with BytesIO() as buffer:
            pixels.crop(self._crop).save(buffer, format='PNG', compress_level=3)
            return buffer.getvalue()

    def close(self):
        self._background = None
        self.artists.clear()
        self.figure.clear()


class MirrorTiltExplorer:
    """Mirror and detector controls, with a Matplotlib sketch in memory."""

    def __init__(self, *, wavelength_nm=632.8, power_mw=1.0, diameter_mm=.54,
                 crossing_diameter_mm=.54, mm_per_unit=1.8, optics_phase_deg=0.):
        import ipywidgets as widgets

        # Validate the physical assumption before allocating widgets.
        cylindrical_overlap_volume(crossing_diameter_mm, 90)
        if not np.isfinite(mm_per_unit) or mm_per_unit <= 0:
            raise ValueError('The length scale must be finite and positive.')
        if not np.isfinite(wavelength_nm) or wavelength_nm <= 0:
            raise ValueError('The wavelength must be finite and positive.')
        if not np.isfinite(power_mw) or power_mw < 0:
            raise ValueError('The source power must be finite and nonnegative.')
        if not np.isfinite(optics_phase_deg):
            raise ValueError('The optical phase offset must be finite.')
        self._closed = False
        self._detector_ui = None
        self._detector_modulated_ui = None
        self._exposure_view = None
        self._switching_view = None
        self._switching_dirty = True
        self._modulated_ui = None
        self._phase_geometry_ui = None
        self._phase_geometry_dirty = True
        self._field_cache = None
        self._eo_cache = None
        self._setup_sketch = None
        self._modulated_sketch = None
        self._longitudinal_ui = None
        self._overview_dirty = True
        self._modulated_overview_dirty = True
        self.detector_modulated_data = None
        self.modulated_overview_data = None
        self.overview_data = None
        self.detector_data = None
        self.phase_geometry_data = None
        self.longitudinal_data = None
        self.wavefront_data = None
        self.crossing_diameter_mm = float(crossing_diameter_mm)
        self.mm_per_unit = float(mm_per_unit)
        self.optics_phase_deg = float(optics_phase_deg)
        self.beam_half_width = self.crossing_diameter_mm / (2*self.mm_per_unit)
        self.source = dict(wavelength_nm=wavelength_nm, power_mw=power_mw,
                           diameter_mm=diameter_mm)
        self.tilt = widgets.FloatSlider(
            value=0, min=-2, max=2, step=.05, readout_format='.2f',
            description='Mirror tilt δ (°)', continuous_update=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='450px', max_width='100%'))
        self.detector_z = widgets.FloatSlider(
            value=-6.9, min=-10, max=-3.5, step=.05, readout_format='.2f',
            description='Detector z-position (schematic units)',
            continuous_update=False, style={'description_width': 'initial'},
            layout=widgets.Layout(width='650px', max_width='100%'))
        self.eo_phase = widgets.FloatSlider(
            value=0, min=0, max=180, step=5, readout_format='.0f',
            description='EO phase β (°)', continuous_update=False,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='550px', max_width='100%'))
        self.phase_varies = widgets.Checkbox(
            value=False, description='Phase varies during exposure', indent=False,
            layout=widgets.Layout(width='auto'))
        self.readout = widgets.HTML()
        self.picture = widgets.Image(format='png', layout=widgets.Layout(
            width='100%', max_width='1100px', height='auto'))
        self.ui = widgets.VBox([self.readout, self.picture])
        # A single owner releases the setup and all later views together.
        self._owned_ui = widgets.VBox([
            self.ui, self.tilt, self.detector_z, self.eo_phase, self.phase_varies])
        self._controls = (self.tilt, self.detector_z)
        for control in self._controls:
            control.observe(self._update, names='value')
        for control in (self.eo_phase, self.phase_varies):
            control.observe(self._update_eo_views, names='value')
        self._update()

    def _update(self, change=None):
        if self._closed:
            return
        self.geometry = two_beam_geometry(self.tilt.value, self.detector_z.value,
                                          self.beam_half_width)
        self._geometry_change = ('position' if change is not None and
                                 change['owner'] is self.detector_z else 'geometry')
        self._field_cache = None
        self._eo_cache = None
        theta = self.geometry['theta_deg']
        self.volume_mm3 = cylindrical_overlap_volume(self.crossing_diameter_mm, theta)
        change_theta = theta - self.geometry['theta_reference_deg']
        sampling = _detector_note(self.geometry).replace('\n', ' ')
        self.readout.value = (
            f'<b>Crossing angle θ = {theta:.2f}°</b> &nbsp; '
            f'Change in θ = {change_theta:+.2f}°<br>'
            f'<b>Cylindrical overlap estimate: {self.volume_mm3:.3f} mm<sup>3</sup></b> '
            f'(assumed diameter at crossing: {self.crossing_diameter_mm:g} mm)<br>'
            f'Detector z<sub>D</sub> = {self.detector_z.value:.2f} schematic units; more negative is farther from the splitter. '
            f'{sampling} (schematic widths).')
        if self._setup_sketch is None:
            self._setup_sketch = _SetupSketch(
                **self.source, mirror_tilt_deg=self.tilt.value,
                detector_z=self.detector_z.value, beam_half_width=self.beam_half_width)
        else:
            self._setup_sketch.set_geometry(self.geometry, self.tilt.value)
        self.picture.value = self._setup_sketch.png()
        if self._detector_ui is not None:
            self._update_detector()
        if self._exposure_view is not None:
            self._exposure_view.set_source(self._exposure_fields())
        self._update_eo_views()
        if self._phase_geometry_ui is not None:
            self._phase_geometry_dirty = True
            if self._phase_geometry_ui.selected_index == 0:
                self._update_phase_geometry()
        if self._longitudinal_ui is not None:
            self._update_longitudinal()

    @property
    def detector_ui(self):
        """One linked wavefront, detector, profile and wavevector display."""
        import ipywidgets as widgets
        import plotly.graph_objects as go
        from .interference_wavefronts import VIEW_OPTIONS

        if self._closed:
            raise RuntimeError('Rerun the apparatus cell before displaying this view.')
        if self._detector_ui is None:
            self.wavefront_mode = widgets.ToggleButtons(
                options=VIEW_OPTIONS, value='intensity', description='Show:',
                style={'button_width': '160px'})
            self.wavefront_readout = widgets.HTML()
            figure, self.wavefront_data = self._make_wavefront_figure()
            self.detector_data = self.wavefront_data
            self.wavefront_figure = go.FigureWidget(figure)
            self.overview_picture = widgets.Image(format='png', layout=widgets.Layout(
                width='100%', max_width='750px', height='auto'))
            self.detector_overview = widgets.Accordion(
                children=[self.overview_picture], selected_index=None)
            self.detector_overview.set_title(0, 'Full detector and enlarged patch')
            self._detector_ui = widgets.VBox([
                self.tilt, self.detector_z, self.wavefront_mode, self.wavefront_figure,
                self.wavefront_readout, self.detector_overview])
            self._owned_ui.children += (self._detector_ui,)
            self.wavefront_mode.observe(self._update_detector, names='value')
            self.detector_overview.observe(self._open_overview, names='selected_index')
            self._wavefront_note()
        return self._detector_ui

    @property
    def wavefront_ui(self):
        """Compatibility alias: both names return the same combined widget."""
        return self.detector_ui

    def _make_wavefront_figure(self):
        from .interference_wavefronts import wavefront_figure

        return wavefront_figure(mode=self.wavefront_mode.value, data=self._exposure_fields().copy())

    def _wavefront_note(self):
        data = self.wavefront_data
        point = data['point']
        self.wavefront_readout.value = (
            f'<b>{10*data["wavelength"]:.2f} μm square at P</b>; '
            f'θ = {data["theta_deg"]:.2f}°; '
            f'fringe spacing = {data["fringe_spacing_um"]:.2f} μm.<br>'
            f'At P: ΔΦ = {data["phase_at_point_deg"]:.1f}° modulo 360°; '
            f'I<sub>1</sub> = {point["intensity_1"]:.3f}, '
            f'I<sub>2</sub> = {point["intensity_2"]:.3f}, '
            f'<b>I = {point["intensity"]:.3f}</b> (relative scalar intensity).')

    def _update_detector(self, change=None):
        from .interference_wavefronts import update_wavefront_figure

        if self._closed:
            return
        self.wavefront_data = self._exposure_fields().copy()
        self.detector_data = self.wavefront_data
        update_wavefront_figure(self.wavefront_figure, self.wavefront_data,
                               mode=self.wavefront_mode.value,
                               change=self._geometry_change if change is None else 'mode')
        self._wavefront_note()
        # Switching signed-field/intensity mode leaves the overview unchanged.
        if change is None:
            self._overview_dirty = True
            if self.detector_overview.selected_index == 0:
                self._update_overview()

    def _open_overview(self, change):
        if not self._closed and change['new'] == 0 and self._overview_dirty:
            self._update_overview()

    def _exposure_fields(self):
        """Reuse the detector patch, even when its display has not been opened."""
        from .interference_wavefronts import wavefront_data

        if self._field_cache is None:
            self._field_cache = wavefront_data(
                self.geometry, wavelength_nm=self.source['wavelength_nm'],
                power_mw=self.source['power_mw'],
                crossing_diameter_mm=self.crossing_diameter_mm,
                mm_per_unit=self.mm_per_unit, optics_phase_deg=self.optics_phase_deg)
        return self._field_cache

    def _modulated_fields(self):
        from .interference_wavefronts import phase_shifted_data

        if self._eo_cache is None or self._eo_cache['eo_phase_deg'] != self.eo_phase.value:
            self._eo_cache = phase_shifted_data(self._exposure_fields(), self.eo_phase.value)
        return self._eo_cache

    @property
    def exposure_ui(self):
        """Compatibility comparison sharing detector_modulated_ui's EO controls."""
        from .phase_exposure import PhaseStepExposureView

        if self._closed:
            raise RuntimeError('Rerun the apparatus cell before displaying this view.')
        if self._exposure_view is None:
            self._exposure_view = PhaseStepExposureView(
                self._exposure_fields(), phase_step=self.eo_phase, vary_phase=self.phase_varies)
            self._owned_ui.children += (self._exposure_view.ui,)
        return self._exposure_view.ui

    @property
    def exposure_data(self):
        """Current exposure arrays and readouts, or None before display/after close."""
        return None if self._exposure_view is None else self._exposure_view.data

    @property
    def modulated_ui(self):
        """Apparatus drawing with the EO modulator; controls live at the detector."""
        import ipywidgets as widgets

        if self._closed:
            raise RuntimeError('Rerun the apparatus cell before displaying this view.')
        if self._modulated_ui is None:
            self.modulated_picture = widgets.Image(format='png', layout=widgets.Layout(
                width='100%', max_width='1100px', height='auto'))
            self._modulated_ui = widgets.VBox([self.readout, self.modulated_picture])
            self._owned_ui.children += (self._modulated_ui,)
            self._update_modulated_setup()
        return self._modulated_ui

    def _update_modulated_setup(self, change=None):
        if self._closed:
            return
        if self._modulated_sketch is None:
            self._modulated_sketch = _SetupSketch(
                **self.source, mirror_tilt_deg=self.tilt.value,
                detector_z=self.detector_z.value, beam_half_width=self.beam_half_width,
                phase_step_deg=self.eo_phase.value, phase_varies=self.phase_varies.value)
        elif change is None:
            self._modulated_sketch.set_geometry(self.geometry, self.tilt.value)
        self._modulated_sketch.set_phase(self.eo_phase.value, self.phase_varies.value)
        self.modulated_picture.value = self._modulated_sketch.png()

    def _update_eo_views(self, change=None):
        """EO changes affect the Section 4 views, with geometry held fixed."""
        if self._closed:
            return
        if self._modulated_ui is not None:
            self._update_modulated_setup(change)
        # Opening the exposure lesson changes the apparatus label, while the
        # four-panel comparison above it always describes a steady EO phase.
        if change is not None and change['owner'] is self.phase_varies:
            return
        if self._detector_modulated_ui is not None:
            self._update_detector_modulated(physical_change=self._geometry_change if change is None else 'phase')
            self._switching_dirty = True
            if self.exposure_options.selected_index == 0:
                self._update_switching_view()

    @property
    def detector_modulated_ui(self):
        """Detector view with mirror, position and EO controls beside the data."""
        import ipywidgets as widgets
        import plotly.graph_objects as go
        from .interference_wavefronts import VIEW_OPTIONS

        if self._closed:
            raise RuntimeError('Rerun the apparatus cell before displaying this view.')
        if self._detector_modulated_ui is None:
            self.modulated_mode = widgets.ToggleButtons(
                options=VIEW_OPTIONS, value='intensity', description='Show:',
                style={'button_width': '160px'})
            self.modulated_readout = widgets.HTML()
            figure, self.detector_modulated_data = self._make_modulated_figure()
            self.modulated_figure = go.FigureWidget(figure)
            self.modulated_overview_picture = widgets.Image(format='png', layout=widgets.Layout(
                width='100%', max_width='750px', height='auto'))
            self.modulated_overview = widgets.Accordion(
                children=[self.modulated_overview_picture], selected_index=None)
            self.modulated_overview.set_title(0, 'Full detector and enlarged patch')
            self.exposure_content = widgets.VBox()
            self.exposure_options = widgets.Accordion(
                children=[self.exposure_content], selected_index=None)
            self.exposure_options.set_title(0, 'Optional: phase changes during exposure')
            self._detector_modulated_ui = widgets.VBox([
                self.tilt, self.detector_z, self.eo_phase, self.modulated_mode,
                self.modulated_figure, self.modulated_readout,
                self.modulated_overview, self.exposure_options])
            self._owned_ui.children += (self._detector_modulated_ui,)
            self.modulated_mode.observe(self._update_detector_modulated, names='value')
            self.modulated_overview.observe(self._open_modulated_overview, names='selected_index')
            self.exposure_options.observe(self._open_exposure, names='selected_index')
            self._modulated_note()
        return self._detector_modulated_ui

    def _make_modulated_figure(self):
        from .interference_wavefronts import wavefront_figure

        return wavefront_figure(mode=self.modulated_mode.value, data=self._modulated_fields().copy())

    def _modulated_note(self):
        data = self.detector_modulated_data
        point = data['point']
        description = 'Steady EO phase: orange crest planes and fringes shift together.'
        coherence = ('|γ<sub>12</sub>| undefined: an arm is dark at P.'
                     if data['coherence_magnitude'] is None else
                     f'|γ<sub>12</sub>| = {data["coherence_magnitude"]:.3f}; '
                     f'local visibility at P: {data["visibility_at_point"]:.3f}.')
        self.modulated_readout.value = (
            f'<b>EO phase β = {self.eo_phase.value:.0f}°</b>; '
            f'θ = {data["theta_deg"]:.2f}°; fringe spacing = {data["fringe_spacing_um"]:.2f} μm.<br>'
            f'{description}<br>'
            f'At P: I<sub>1</sub> = {point["intensity_1"]:.3f}, '
            f'I<sub>2</sub> = {point["intensity_2"]:.3f}, '
            f'<b>recorded I = {point["recorded_intensity"]:.3f}</b>. {coherence}')

    def _update_detector_modulated(self, change=None, *, physical_change='geometry'):
        from .interference_wavefronts import update_wavefront_figure

        if self._closed:
            return
        self.detector_modulated_data = self._modulated_fields().copy()
        update_wavefront_figure(self.modulated_figure, self.detector_modulated_data,
                               mode=self.modulated_mode.value,
                               change=physical_change if change is None else 'mode')
        self._modulated_note()
        if change is None:
            self._modulated_overview_dirty = True
            if self.modulated_overview.selected_index == 0:
                self._update_modulated_overview()

    def _open_modulated_overview(self, change):
        if not self._closed and change['new'] == 0 and self._modulated_overview_dirty:
            self._update_modulated_overview()

    def _open_exposure(self, change):
        if self._closed:
            return
        if change['new'] == 0:
            self._update_switching_view()
            self.phase_varies.value = True
        else:
            if self._switching_view is not None:
                self._switching_view.pause()
            self.phase_varies.value = False

    def _update_switching_view(self):
        from .phase_exposure import PhaseSwitchingExposureView

        parameters = dict(source=self._exposure_fields(), selected=self._modulated_fields())
        if self._switching_view is None:
            self._switching_view = PhaseSwitchingExposureView(
                self.geometry, self.eo_phase.value, **parameters)
            self.exposure_content.children = (self._switching_view.ui,)
        elif self._switching_dirty:
            self._switching_view.set_parameters(
                self.geometry, self.eo_phase.value, **parameters)
        self._switching_dirty = False

    def _update_modulated_overview(self):
        from io import BytesIO
        from .detector_interference import draw_detector_overview

        figure, self.modulated_overview_data = draw_detector_overview(
            self.geometry,
            patch_half_extent_mm=self.detector_modulated_data['bounds'][0][1]*1e-3,
            wavelength_nm=self.source['wavelength_nm'], power_mw=self.source['power_mw'],
            crossing_diameter_mm=self.crossing_diameter_mm, mm_per_unit=self.mm_per_unit,
            optics_phase_deg=self.optics_phase_deg, eo_phase_deg=self.eo_phase.value,
            phase_varies=False)
        try:
            with BytesIO() as buffer:
                figure.savefig(buffer, format='png', dpi=110)
                self.modulated_overview_picture.value = buffer.getvalue()
        finally:
            plt.close(figure)
        self._modulated_overview_dirty = False

    def _update_overview(self):
        from io import BytesIO
        from .detector_interference import draw_detector_overview

        figure, self.overview_data = draw_detector_overview(
            self.geometry,
            patch_half_extent_mm=self.wavefront_data['bounds'][0][1]*1e-3,
            wavelength_nm=self.source['wavelength_nm'],
            power_mw=self.source['power_mw'],
            crossing_diameter_mm=self.crossing_diameter_mm,
            mm_per_unit=self.mm_per_unit, optics_phase_deg=self.optics_phase_deg)
        try:
            with BytesIO() as buffer:
                figure.savefig(buffer, format='png', dpi=110)
                self.overview_picture.value = buffer.getvalue()
        finally:
            plt.close(figure)
        self._overview_dirty = False

    @property
    def phase_geometry_ui(self):
        """Optional arm-phase and projection explanation, initially collapsed."""
        import ipywidgets as widgets
        from .phase_geometry import PROJECTION_INTRO, PROJECTION_EXPLANATION

        if self._closed:
            raise RuntimeError('Rerun the apparatus cell before displaying this view.')
        if self._phase_geometry_ui is None:
            def markdown_output(source):
                return widgets.Output(outputs=({
                    'output_type': 'display_data',
                    'data': {'text/markdown': source, 'text/plain': source},
                    'metadata': {},
                },))

            self.phase_geometry_picture = widgets.Image(
                format='png', layout=widgets.Layout(
                    width='100%', max_width='1400px', height='auto'))
            explanation = widgets.VBox([
                markdown_output(PROJECTION_INTRO),
                self.phase_geometry_picture,
                markdown_output(PROJECTION_EXPLANATION),
            ])
            self._phase_geometry_ui = widgets.Accordion(
                children=[explanation], selected_index=None)
            self._phase_geometry_ui.set_title(
                0, 'Where does the phase offset come from?')
            self._owned_ui.children += (self._phase_geometry_ui,)
            self._phase_geometry_ui.observe(self._open_phase_geometry, names='selected_index')
        return self._phase_geometry_ui

    def _open_phase_geometry(self, change):
        if not self._closed and change['new'] == 0 and self._phase_geometry_dirty:
            self._update_phase_geometry()

    def _update_phase_geometry(self):
        from io import BytesIO
        from .phase_geometry import draw_phase_geometry

        if self._closed:
            return
        figure, self.phase_geometry_data = draw_phase_geometry(self.geometry)
        try:
            with BytesIO() as buffer:
                figure.savefig(buffer, format='png', dpi=110)
                self.phase_geometry_picture.value = buffer.getvalue()
        finally:
            plt.close(figure)
        self._phase_geometry_dirty = False

    @property
    def longitudinal_ui(self):
        """Individual fields and their sum along z, using the setup controls."""
        import ipywidgets as widgets

        if self._closed:
            raise RuntimeError('Rerun the apparatus cell before displaying this view.')
        if self._longitudinal_ui is None:
            self.longitudinal_picture = widgets.Image(
                format='png', layout=widgets.Layout(
                    width='100%', max_width='1400px', height='auto'))
            self._longitudinal_ui = widgets.VBox([self.longitudinal_picture])
            self._owned_ui.children += (self._longitudinal_ui,)
            self._update_longitudinal()
        return self._longitudinal_ui

    def _update_longitudinal(self):
        from io import BytesIO
        from .longitudinal_interference import draw_longitudinal_interference

        if self._closed:
            return
        figure, self.longitudinal_data = draw_longitudinal_interference(
            self.geometry, wavelength_nm=self.source['wavelength_nm'],
            power_mw=self.source['power_mw'],
            crossing_diameter_mm=self.crossing_diameter_mm,
            mm_per_unit=self.mm_per_unit, optics_phase_deg=self.optics_phase_deg)
        try:
            with BytesIO() as buffer:
                figure.savefig(buffer, format='png', dpi=110)
                self.longitudinal_picture.value = buffer.getvalue()
        finally:
            plt.close(figure)

    def close(self):
        """Detach the control and release its display; safe to call twice."""
        from .grating_intro import _close_widget_tree

        if self._closed:
            return
        self._closed = True
        for control in self._controls:
            control.unobserve(self._update, names='value')
        for control in (self.eo_phase, self.phase_varies):
            control.unobserve(self._update_eo_views, names='value')
        self.picture.value = b''
        if self._modulated_ui is not None:
            self.modulated_picture.value = b''
        if self._detector_modulated_ui is not None:
            self.modulated_mode.unobserve(self._update_detector_modulated, names='value')
            self.modulated_overview.unobserve(self._open_modulated_overview, names='selected_index')
            self.modulated_overview_picture.value = b''
            self.exposure_options.unobserve(self._open_exposure, names='selected_index')
        if self._switching_view is not None:
            self._switching_view.close(close_widgets=False)
        if self._exposure_view is not None:
            self._exposure_view.close(close_widgets=False)
        if self._detector_ui is not None:
            self.wavefront_mode.unobserve(self._update_detector, names='value')
            self.detector_overview.unobserve(self._open_overview, names='selected_index')
            self.overview_picture.value = b''
        if self._phase_geometry_ui is not None:
            self._phase_geometry_ui.unobserve(self._open_phase_geometry, names='selected_index')
            self.phase_geometry_picture.value = b''
        if self._longitudinal_ui is not None:
            self.longitudinal_picture.value = b''
        self.detector_data = None
        self.detector_modulated_data = None
        self.modulated_overview_data = None
        self.phase_geometry_data = None
        self.longitudinal_data = None
        self.wavefront_data = None
        self.geometry = None
        self.overview_data = None
        self._field_cache = None
        self._eo_cache = None
        for sketch in (self._setup_sketch, self._modulated_sketch):
            if sketch is not None:
                sketch.close()
        _close_widget_tree(self._owned_ui)
