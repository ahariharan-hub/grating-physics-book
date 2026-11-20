"""Detector slices of the beam-splitter experiment in notebook seven.

The phase is traced from the splitter through the fixed mirror centre. Equal
power arms have prescribed constant-width Gaussian envelopes and plane-wave
phase fronts. No diffraction, curvature or camera responsivity is simulated.
Intensity uses a fixed reference: the on-axis scalar intensity in one arm
for a 1 mW source, ideal 50:50 splitting and a 0.54 mm 1/e^2 diameter.
"""

import numpy as np


def detector_field(x_mm, y_mm, geometry, *, wavelength_nm=632.8,
                   power_mw=1., crossing_diameter_mm=.54,
                   mm_per_unit=1.8, optics_phase_deg=0., z_mm=None):
    """Scalar fields at (x,y,z_D), or supplied z_mm, in fixed reference units.

    The common phase of beam 1 has been factored out of U1 and U2. Their
    relative phase retains the actual arm propagation and optical offset.
    Beam 2's transverse distance is measured perpendicular to its axis.
    Optional signed z_mm coordinates are measured from the splitter; they allow
    a longitudinal cut of the same fields without moving the detector.
    """
    parameters = np.asarray([wavelength_nm, power_mw, crossing_diameter_mm,
                             mm_per_unit, optics_phase_deg], dtype=float)
    if (not np.all(np.isfinite(parameters)) or power_mw < 0 or
            min(wavelength_nm, crossing_diameter_mm, mm_per_unit) <= 0):
        raise ValueError('Use finite parameters, positive lengths and nonnegative power.')
    if z_mm is None:
        z_mm = (geometry['detector'][2] - geometry['splitter'][2]) * mm_per_unit
    x, y, z = np.broadcast_arrays(np.asarray(x_mm, dtype=float),
                                  np.asarray(y_mm, dtype=float),
                                  np.asarray(z_mm, dtype=float))
    if not all(np.all(np.isfinite(coordinate)) for coordinate in (x, y, z)):
        raise ValueError('Observation coordinates must be finite.')
    sine, cosine = -geometry['outgoing'][0], -geometry['outgoing'][2]
    h = (geometry['mirror'][0] - geometry['splitter'][0]) * mm_per_unit
    k = 2*np.pi / (wavelength_nm*1e-6)  # rad/mm
    phase = k*(h*(1+sine) - x*sine + z*(1-cosine)) + np.deg2rad(optics_phase_deg)
    radius = crossing_diameter_mm/2
    transverse_2 = (x-h)*cosine - z*sine
    # Each arm carries half the source power. The radius factor keeps the
    # reference intensity fixed even when the assumed beam diameter changes.
    peak_amplitude = np.sqrt(power_mw) * (.27/radius)
    amplitude_1 = peak_amplitude * np.exp(-(x*x + y*y)/radius**2)
    amplitude_2 = peak_amplitude * np.exp(-(transverse_2**2 + y*y)/radius**2)
    u1 = amplitude_1.astype(complex)
    u2 = amplitude_2*np.exp(1j*phase)
    return dict(phase_difference=phase, field_1=u1, field_2=u2,
                intensity_1=amplitude_1**2, intensity_2=amplitude_2**2,
                intensity=np.abs(u1+u2)**2)


def detector_image(geometry, *, half_extent_mm, pixels=240, **parameters):
    """Area-average the calculated intensity over each rectangular image bin.

    The y Gaussian separates from the x interference profile. Gauss-Legendre
    quadrature resolves the carrier inside each bin before averaging, rather
    than sampling only at bin centres when a bin spans multiple fringes.
    This is display binning, not a specification of an experimental camera.
    """
    if not np.isfinite(half_extent_mm) or half_extent_mm <= 0:
        raise ValueError('The image half-extent must be finite and positive.')
    if not isinstance(pixels, (int, np.integer)) or pixels < 8:
        raise ValueError('Use at least eight pixels per image dimension.')
    # Validate optical parameters before sizing the quadrature.
    detector_field(0., 0., geometry, **parameters)
    wavelength_mm = parameters.get('wavelength_nm', 632.8)*1e-6
    radius = parameters.get('crossing_diameter_mm', .54)/2
    step = 2*half_extent_mm/pixels
    carrier_span = 2*np.pi*abs(geometry['outgoing'][0])*step/wavelength_mm
    order = max(16, int(np.ceil(carrier_span))+8, int(np.ceil(8*step/radius)))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(-half_extent_mm, half_extent_mm, pixels+1)
    centres = (edges[1:]+edges[:-1])/2
    samples = centres[:, None] + step/2*nodes
    along_x = detector_field(samples, 0., geometry, **parameters)['intensity']
    x_average = along_x @ (weights/2)
    y_average = np.exp(-2*samples**2/radius**2) @ (weights/2)
    # Rows are y and columns are x, matching the detector's front view.
    return dict(intensity=y_average[:, None]*x_average[None, :],
                centres_mm=centres, edges_mm=edges, bin_width_mm=step,
                quadrature_order=order)


def draw_detector_overview(geometry, *, patch_half_extent_mm,
                           wavelength_nm=632.8, power_mw=1.,
                           crossing_diameter_mm=.54, mm_per_unit=1.8,
                           optics_phase_deg=0., eo_phase_deg=0., phase_varies=False):
    """Full detector with the combined view's central patch marked to scale."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not np.isfinite(patch_half_extent_mm) or patch_half_extent_mm <= 0:
        raise ValueError('The patch half-extent must be finite and positive.')
    if not np.isfinite(eo_phase_deg):
        raise ValueError('The EO phase must be finite.')
    parameters = dict(wavelength_nm=wavelength_nm, power_mw=power_mw,
                      crossing_diameter_mm=crossing_diameter_mm,
                      mm_per_unit=mm_per_unit, optics_phase_deg=optics_phase_deg+eo_phase_deg)
    half_detector = geometry['detector_interval'][1]*mm_per_unit
    full = detector_image(geometry, half_extent_mm=half_detector, **parameters)
    if phase_varies:
        parameters['optics_phase_deg'] = optics_phase_deg
        reference = detector_image(geometry, half_extent_mm=half_detector, **parameters)
        full['intensity'] = (full['intensity']+reference['intensity'])/2
    limit = max(4., 4*power_mw*(.54/crossing_diameter_mm)**2)
    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor='white')
    fig.subplots_adjust(left=.13, right=.86, bottom=.19, top=.88)
    artist = ax.imshow(full['intensity'], origin='lower', cmap='gray',
                       vmin=0, vmax=limit, interpolation='nearest',
                       extent=(-half_detector, half_detector,
                               -half_detector, half_detector), aspect='equal')
    ax.set(xlabel='x (mm)', ylabel='y (mm)', title=(
        'Full detector · exposure and area averaged' if phase_varies else
        'Full detector · area averaged'))
    radius = patch_half_extent_mm
    ax.add_patch(Rectangle((-radius, -radius), 2*radius, 2*radius,
                           fill=False, edgecolor='#20ccd8', lw=1))
    ax.annotate(f'P: {2*radius*1000:.2f} μm patch', xy=(0, 0),
                xytext=(.16*half_detector, .42*half_detector),
                fontsize=10, color='#20ccd8',
                arrowprops=dict(arrowstyle='-', color='#20ccd8', lw=.8))
    colorbar = fig.colorbar(artist, ax=ax, fraction=.047, pad=.035)
    colorbar.set_label('Relative scalar intensity', fontsize=10)
    fig.text(.13, .055,
             f'Image bins: {full["bin_width_mm"]*1000:.2f} μm. '
             'The marked patch is enlarged above.\n'
             'The overview and close-up use the same intensity scale.',
             fontsize=10, color='#52616e')
    return fig, dict(full=full, patch_half_extent_mm=radius, intensity_limit=limit)


def draw_detector_experiment(geometry, *, wavelength_nm=632.8, power_mw=1.,
                            crossing_diameter_mm=.54, mm_per_unit=1.8,
                            optics_phase_deg=0.):
    """Return the full detector, resolved close-up and line cut only."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    parameters = dict(wavelength_nm=wavelength_nm, power_mw=power_mw,
                      crossing_diameter_mm=crossing_diameter_mm,
                      mm_per_unit=mm_per_unit, optics_phase_deg=optics_phase_deg)
    half_detector = geometry['detector_interval'][1]*mm_per_unit
    full = detector_image(geometry, half_extent_mm=half_detector, **parameters)
    zoom_half = .010  # mm: fixed 20 micrometre window centred at P.
    zoom = detector_image(geometry, half_extent_mm=zoom_half, pixels=256, **parameters)
    x = np.linspace(-zoom_half, zoom_half, 1201)
    profile = detector_field(x, 0., geometry, **parameters)
    point = detector_field(0., 0., geometry, **parameters)
    fringe_spacing = wavelength_nm*1e-3 / abs(geometry['outgoing'][0])
    peak_scale = power_mw*(.54/crossing_diameter_mm)**2
    # Shared scale is independent of detector position and mirror alignment.
    upper_intensity = max(4., 4*peak_scale)
    fig = plt.figure(figsize=(14, 4.6), facecolor='white')
    grid = fig.add_gridspec(1, 3, left=.055, right=.98, bottom=.28, top=.85,
                           wspace=.47, width_ratios=[1, 1, 1.45])
    overview = fig.add_subplot(grid[0, 0])
    detail = fig.add_subplot(grid[0, 1])
    trace = fig.add_subplot(grid[0, 2])
    images = []
    for axis, data, extent, unit, title in (
            (overview, full, half_detector, 'mm', 'Full detector · area averaged'),
            (detail, zoom, zoom_half*1000, 'μm', '20 μm close-up at P')):
        artist = axis.imshow(data['intensity'], origin='lower', cmap='gray',
                             vmin=0, vmax=upper_intensity, interpolation='nearest',
                             extent=(-extent, extent, -extent, extent), aspect='equal')
        images.append(artist)
        axis.set(xlabel=f'x ({unit})', ylabel=f'y ({unit})', title=title)
        axis.tick_params(labelsize=9)
        axis.plot(0, 0, '+', ms=6, mew=1, color='#20ccd8')
    overview.add_patch(Rectangle((-zoom_half, -zoom_half), 2*zoom_half, 2*zoom_half,
                                 fill=False, edgecolor='#20ccd8', lw=1))
    overview.annotate('P', xy=(0, 0), xytext=(.22, .2), color='#20ccd8',
                      fontsize=10, arrowprops=dict(arrowstyle='-', color='#20ccd8', lw=.8))
    colorbar = fig.colorbar(images[0], ax=overview, fraction=.047, pad=.035)
    colorbar.set_label('Relative scalar intensity', fontsize=9)
    colorbar.ax.tick_params(labelsize=9)
    for key, color, style, label in (
            ('intensity_1', '#2866a3', '--', 'Beam 1 alone'),
            ('intensity_2', '#b85c17', ':', 'Beam 2 alone'),
            ('intensity', '#243444', '-', 'Both beams')):
        trace.plot(x*1000, profile[key], color=color, ls=style, lw=1.7, label=label)
    trace.plot(0, point['intensity'], 'o', color='#1495a3', ms=5)
    trace.set(xlim=(-10, 10), ylim=(0, upper_intensity*1.06), xlabel='x (μm)',
              ylabel='Relative scalar intensity', title=r'Across $P$: $y=0,\ z=z_D$')
    trace.spines[['top', 'right']].set_visible(False)
    trace.grid(alpha=.15)
    trace.legend(loc='upper center', bbox_to_anchor=(.5, -.27),
                 ncol=3, frameon=False, fontsize=9)
    fig.suptitle('One detector plane, viewed at two scales', fontsize=14, y=.985)
    fig.text(.055, .025,
             f'Full-view bins: {full["bin_width_mm"]*1000:.2f} μm. '
             f'Fringe spacing: {fringe_spacing:.2f} μm. '
             'The close-up resolves the fringes; both images share one intensity scale.',
             fontsize=10, color='#52616e')
    return fig, dict(full=full, zoom=zoom, profile=profile, profile_x_mm=x,
                     point=point, fringe_spacing_um=float(fringe_spacing),
                     phase_at_point_deg=float(np.rad2deg(point['phase_difference']) % 360),
                     intensity_limit=upper_intensity)
