"""Independent analytical checks and owned-widget lifecycle checks for 7.3."""
import gc
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from numpy.testing import assert_allclose
from wave_optics.gaussian_order_beams import (
    GaussianOrderConfiguration, GaussianOrderExperiment, line_overlap)
from wave_optics.gaussian_order_beams_view import GaussianOrderBeamsExplorer, beam_shell, DETECTOR_CMAP
from wave_optics.gaussian_fringe_diagnostics import fringe_diagnostics
from wave_optics.gaussian_notebook_setup import create_gaussian_explorer


def test_width_rayleigh_and_amplitude():
    m = GaussianOrderExperiment()
    c = m.configuration
    assert_allclose(c.rayleigh, np.pi*36.)
    q, axis = m.basis(1)
    for distance in (0., c.rayleigh, 2*c.rayleigh):
        width = c.waist*np.sqrt(1+(distance/c.rayleigh)**2)
        points = np.array([distance*axis, distance*axis+width*q])
        values = m.field(points, 1)
        assert_allclose(abs(values[1]/values[0]), np.exp(-1), atol=1e-14)
        assert_allclose(abs(values[0]), c.waist/width, atol=1e-14)


def test_gouy_curvature_and_time_convention():
    m = GaussianOrderExperiment()
    c = m.configuration
    q, axis = m.basis(-1)
    distance, transverse = c.rayleigh, 2.
    origin = distance*axis
    center, offaxis = m.field(np.array([origin, origin+transverse*q]), -1)
    assert_allclose(center/abs(center)*np.exp(-1j*m.k*distance), np.exp(-1j*np.pi/4), atol=1e-13)
    expected = m.k*transverse**2/(4*c.rayleigh)
    assert_allclose((offaxis/center)/abs(offaxis/center), np.exp(1j*expected), atol=2e-13)


@pytest.mark.parametrize('order', [-2, -1, 0, 1, 2])
def test_rotation_and_phase_gradient(order):
    m = GaussianOrderExperiment()
    q, axis = m.basis(order)
    assert_allclose(np.cross(q, [0, 1, 0]), axis, atol=1e-15)
    assert_allclose(q@axis, 0., atol=1e-15)
    assert_allclose(m.local_coordinates(3*q+5*axis+[0, 2, 0], order), [3, 2, 5])
    assert_allclose(m.k*axis[0], order*m.G, atol=1e-15)
    assert axis[2] < 0
    # Gouy phase changes magnitude slightly, but the actual centre gradient
    # remains parallel to the prescribed transmitted wavevector.
    h = 1e-5
    offsets = np.eye(3)*h
    gradient = np.angle(m.field(offsets, order)/m.field(-offsets, order))/(2*h)
    assert_allclose(gradient, (m.k-1/m.configuration.rayleigh)*axis, atol=2e-10)
    carrier = np.exp(1j*m.k*m.local_coordinates(offsets, order)[..., 2])
    assert_allclose(np.angle(carrier)/h, m.k*axis, atol=1e-12)


def test_coherent_sum_phase_and_zero_arm():
    m = GaussianOrderExperiment()
    x = np.linspace(-80, 80, 1601)
    a = m.sample(x, [0, 40, 80])
    b = m.sample(x, [0, 40, 80], coefficients=(1, 1j))
    assert_allclose(a['intensity'], a['intensities'].sum(axis=0)+a['cross'], atol=2e-15)
    assert_allclose(a['overlap'], b['overlap'], atol=2e-15)
    assert np.max(abs(a['intensity']-b['intensity'])) > .5
    for coefficients in ((0, 1), (1, 0), (0, 0)):
        z = m.sample(x, [40], coefficients=coefficients)
        assert np.all(np.isnan(z['overlap']))
        assert_allclose(z['cross'], 0)
        assert_allclose(z['intensity'], z['intensities'].sum(axis=0))


def test_overlap_known_integral_strength_and_separation():
    x = np.linspace(-100, 100, 10001)
    a, b = np.exp(-x*x/64), np.exp(-(x-20)**2/64)
    assert_allclose(line_overlap(a, b, x), np.exp(-400/128), atol=1e-14)
    assert_allclose(line_overlap(1e-200*a, 2*b*1j, x), line_overlap(a, b, x), atol=1e-14)
    assert_allclose(line_overlap(a, a, x), 1., atol=1e-14)
    m = GaussianOrderExperiment()
    d = m.sample(x, np.arange(81))
    assert np.all((d['overlap'] >= 0) & (d['overlap'] <= 1))
    assert d['overlap'][0] > .99
    assert d['overlap'][-1] < .05
    assert np.all(np.diff(d['overlap']) < 0)


def test_default_detector_quadrature_refinement_and_extension():
    m = GaussianOrderExperiment()
    depth = np.arange(81)
    base = m.sample(np.linspace(-80, 80, 1601), depth)
    fine = m.sample(np.linspace(-80, 80, 3201), depth)
    wide = m.sample(np.linspace(-160, 160, 3201), depth)
    assert_allclose(base['overlap'], fine['overlap'], atol=1e-12)
    assert_allclose(base['overlap'], wide['overlap'], atol=1e-12)
    assert_allclose(base['intensity'], fine['intensity'][:, ::2], atol=1e-14)


def test_local_broad_beam_plane_wave_limit():
    m = GaussianOrderExperiment(GaussianOrderConfiguration(waist=10000))
    x = np.linspace(-2, 2, 401)
    depths = np.array([0., 10.])
    d = m.sample(x, depths, coefficients=(1, 1j))
    plane = []
    for order, coefficient in ((0, 1), (1, 1j)):
        axis = m.basis(order)[1]
        plane.append(coefficient*np.exp(1j*m.k*(axis[0]*x[None, :]-axis[2]*depths[:, None])))
    assert_allclose(d['fields'], plane, atol=1e-7)
    assert_allclose(d['intensity'], abs(sum(plane))**2, atol=4e-7)


def test_rejections_and_no_fft_dependency():
    for kw in ({'waist': 2}, {'wavelength': 0}, {'period': np.nan}):
        with pytest.raises(ValueError):
            GaussianOrderConfiguration(**kw)
    m = GaussianOrderExperiment()
    for order in (4, 3, .5, np.nan):
        with pytest.raises(ValueError):
            m.field([0, 0, 0], order)
    with pytest.raises(ValueError):
        m.sample([-1, 0, 1], [0], pair=(0, 0))
    with pytest.raises(ValueError):
        m.sample([-1, 0, 1], [-1])
    with patch('numpy.fft.fft', side_effect=AssertionError('FFT forbidden')):
        assert np.isfinite(m.sample([-1, 0, 1], [0])['intensity']).all()


def test_widget_updates_cache_reset_and_close():
    with patch('wave_optics.gaussian_order_beams_view.figure_png', return_value=b'png'):
        v = GaussianOrderBeamsExplorer()
        try:
            ids = tuple(map(id, (*v.images, *v.profiles)))
            data, count, cross = v.data, v.model_updates, v.cross
            v.detector.value = 80
            assert v.data is data and v.cross is cross and v.model_updates == count
            v.phase.value = 90
            assert v.data is data and v.model_updates == count
            for control, value in ((v.wavelength, .6), (v.pitch, 8.), (v.waist, 4.)):
                before = v.updates
                control.value = value
                assert v.updates == before+1
            v.pair.value = v.pair.options[-1][1]
            before_reset = v.updates
            v.reset_button.click()
            assert v.updates == before_reset+1
            assert v.pair.value == (0, 1)
            assert_allclose([v.wavelength.value, v.pitch.value, v.waist.value,
                             v.detector.value, v.phase.value], [1, 4, 6, 40, 0])
            assert ids == tuple(map(id, (*v.images, *v.profiles)))
            for control in v._inputs:
                assert control._trait_notifiers['value']['change'].count(v._changed) == 1
            v.pair.value = (-2, 2)
            v.wavelength.value = 1.2
            assert v.pair.value == (0, 1)  # now the old pair is outside 30°.
            before = v.updates
            v.close(); v.close(); v.waist.value = 7
            assert v.updates == before and v.data is None and v.cross is None
        finally:
            v.close()


def test_notebook_reexecution_owns_widgets():
    from ipywidgets.widgets.widget import _instances
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root/'7_3_Finite_Diffraction_Beams_and_Experimental_Recombination.ipynb').read_text(encoding='utf-8'))
    builder = ''.join(next(c['source'] for c in notebook['cells'] if c['id'] == 'n73-explorer'))
    baseline = set(_instances)
    ns = dict(create_gaussian_explorer=create_gaussian_explorer, display=lambda x: None)
    counts = []
    with patch('wave_optics.gaussian_order_beams_view.figure_png', return_value=b'png'):
        try:
            for _ in range(3):
                old = ns.get('gaussian_orders')
                exec(builder, ns)
                if old:
                    assert old._closed
                gc.collect()
                counts.append(len(set(_instances)-baseline))
            assert counts == [counts[0]]*3
        finally:
            ns['gaussian_orders'].close()
            gc.collect()
    assert set(_instances) == baseline


@pytest.mark.parametrize('order', [-2, 0, 2])
def test_3d_shell_is_a_rotated_radius_contour(order):
    model = GaussianOrderExperiment()
    faces = beam_shell(model, order)
    points = np.concatenate(faces)
    assert np.all(points[:, 2] <= 1e-12)
    assert np.all(points[:, 2] >= -80-1e-12)
    # The clipping edges interpolate display-mesh vertices; test original
    # contour vertices away from those two display planes.
    interior = points[(points[:, 2] < -1e-8) & (points[:, 2] > -80+1e-8)]
    local = model.local_coordinates(interior, order)
    c = model.configuration
    width = c.waist*np.sqrt(1+(local[:, 2]/c.rayleigh)**2)
    assert_allclose(np.hypot(local[:, 0], local[:, 1]), width, atol=2e-14)


def test_3d_detector_slice_phase_cache_and_arrow_directions():
    with patch('wave_optics.gaussian_order_beams_view.figure_png', return_value=b'png'):
        v = GaussianOrderBeamsExplorer()
        try:
            # The fitted detector keeps its x samples on the existing xz grid.
            center = len(v.detector_y)//2
            indices=np.rint((v.detector_x-v.x[0])/(v.x[1]-v.x[0])).astype(int)
            assert np.all((indices>=0)&(indices<len(v.x)))
            assert_allclose(v.detector_fields[:, center],
                            v.data['fields'][:, 40, indices], atol=5e-14)
            fields, count = v.detector_fields, v.model_updates
            individual = abs(fields)**2
            v.phase.value = 90
            assert v.detector_fields is fields and v.model_updates == count
            assert_allclose(v.detector_intensity, abs(fields[0]+1j*fields[1])**2, atol=2e-15)
            assert_allclose(abs(v.detector_fields)**2, individual)
            expected = v.combined[40, indices]
            assert_allclose(v.detector_intensity[center], expected, atol=8e-14)
            for arrow, order in zip(v.direction_arrows, v.pair.value):
                start, tip = arrow._segments3d[0]
                assert_allclose((tip-start)/np.linalg.norm(tip-start), v.model.basis(order)[1])
            assert np.all(v.detector_display_intensity >= 0)
            assert np.all(v.detector_display_intensity <= 4)
            v.detector.value = 60
            assert v.detector_fields is not fields
            assert v.model_updates == count
            assert_allclose(v._detector_points[..., 2], -60)
        finally:
            v.close()
        assert v.detector_fields is None and v.detector_intensity is None


def test_optional_sections_render_only_when_open_and_dirty():
    with patch('wave_optics.gaussian_order_beams_view.figure_png', return_value=b'png') as render:
        v = GaussianOrderBeamsExplorer()
        try:
            assert render.call_count == 0  # main view is a live Plotly widget
            v.phase.value = 90
            assert render.call_count == 0
            v.sections.selected_index = 0
            assert render.call_count == 1
            v.sections.selected_index = None
            v.sections.selected_index = 0
            assert render.call_count == 1  # unchanged cached section image
        finally:
            v.close()


def test_incident_contour_points_toward_grating_and_controls_are_compact():
    m = GaussianOrderExperiment()
    points = np.concatenate(beam_shell(m, 0, z_limits=(0., 32.)))
    assert np.all(points[:, 2] >= -1e-12)
    assert np.all(points[:, 2] <= 32+1e-12)
    expected = m.configuration.waist*np.sqrt(1+(points[:, 2]/m.configuration.rayleigh)**2)
    assert_allclose(np.hypot(points[:, 0], points[:, 1]), expected, atol=.01)
    with patch('wave_optics.gaussian_order_beams_view.figure_png', return_value=b'png'):
        v = GaussianOrderBeamsExplorer()
        try:
            start, tip = v.incident_arrow._segments3d[0]
            assert_allclose((np.array(tip)-start)/np.linalg.norm(np.array(tip)-start), [0, 0, -1])
            assert len(v.readout.value) < 180
            assert '<br>' not in v.readout.value
            assert len(v.control_panel.children) == 2
            for control in v._inputs:
                assert control.style.description_width == '90px'
                assert control.layout.width == '100%'
            assert_allclose(v.figure.get_size_inches(), [12, 7.5])
            assert_allclose(np.ptp(v.detector_x), np.ptp(v.detector_y))
            lengths = [np.ptp(v.geometry_axis.get_xlim()), np.ptp(v.geometry_axis.get_ylim()),
                       np.ptp(v.geometry_axis.get_zlim())]
            assert_allclose(lengths, [lengths[0]]*3)
            assert_allclose(v.geometry_axis.get_box_aspect()/v.geometry_axis.get_box_aspect()[0], 1.)
        finally:
            v.close()


def test_detector_palette_uses_dark_background_and_increasing_brightness():
    colors = DETECTOR_CMAP(np.linspace(0, 1, 257))[:, :3]
    brightness = colors@np.array([.2126, .7152, .0722])
    assert np.all(np.diff(brightness) >= 0)
    assert brightness[0] < .02 and brightness[-1] > .9


def test_live_3d_widget_retains_camera_traces_and_detector_signal():
    from plotly.graph_objects import FigureWidget
    v = GaussianOrderBeamsExplorer()
    try:
        assert isinstance(v.picture, FigureWidget)
        assert v.picture in v.ui.children
        assert v.picture.layout.scene.dragmode == 'orbit'
        assert v.picture.layout.scene.aspectmode == 'cube'
        assert v.picture.layout.scene.uirevision == 'gaussian-orders-camera'
        traces = tuple(map(id,v.picture.data))
        v.picture.layout.scene.camera.eye = dict(x=2.3,y=1.1,z=.8)
        for control,value in ((v.phase,90),(v.detector,60),(v.waist,8)):
            control.value=value
            assert v.picture.layout.scene.camera.eye.x == 2.3
            assert traces == tuple(map(id,v.picture.data))
        assert_allclose(v.interactive.roles['screen'].z,v.detector_intensity)
        assert_allclose(v.interactive.roles['depth-marker'].x,[60,60])
        assert_allclose(v.interactive.roles['detector'].surfacecolor,v.detector_display_intensity)
        v.reset_button.click()
        assert v.picture.layout.scene.camera.eye.x == 1.45
    finally:
        v.close()


def test_fringe_gradient_plane_waves_phase_invariance_and_missing_overlap():
    x=np.linspace(-30,30,1201)
    envelope=np.exp(-x*x/64)
    a=envelope.astype(complex)
    b=envelope*np.exp(1j*2*np.pi*x/4)
    f=fringe_diagnostics(a,b,x,4)
    assert_allclose(f['spacing'][f['mask']],4,atol=2e-12)
    shifted=fringe_diagnostics(a,1j*b,x,4)
    assert_allclose(shifted['spacing'],f['spacing'],atol=2e-12,equal_nan=True)
    missing=fringe_diagnostics(a,np.zeros_like(a),x,4)
    assert missing['center'] is None and np.isnan(missing['local'])
    with pytest.raises(ValueError):
        fringe_diagnostics(a,b,x[::-1],4)


def test_gaussian_local_spacing_matches_analytic_phase_derivative():
    model=GaussianOrderExperiment()
    x=np.linspace(-40,40,8001)
    data=model.sample(x,[40,80])
    c=model.configuration
    for row,depth in enumerate((40,80)):
        f=fringe_diagnostics(*data['fields'][:,row],x,4)
        xc=x[f['center']]
        derivatives=[]
        for order in (0,1):
            direction=model.basis(order)[1]
            sine,cosine=direction[0],-direction[2]
            q=-cosine*xc+sine*depth
            ell=sine*xc+cosine*depth
            denominator=ell**2+c.rayleigh**2
            derivative=(model.k*sine + model.k/2*(
                -2*q*cosine*ell/denominator
                +q*q*sine*(c.rayleigh**2-ell**2)/denominator**2)
                -sine*c.rayleigh/denominator)
            derivatives.append(derivative)
        expected=2*np.pi/abs(derivatives[1]-derivatives[0])
        assert_allclose(f['local'],expected,rtol=1e-8)
    assert f['local'] > 5.9  # curvature is appreciable, not always a tiny correction.
    broad=GaussianOrderExperiment(GaussianOrderConfiguration(waist=10000))
    d=broad.sample(x,[80])
    f=fringe_diagnostics(*d['fields'][:,0],x,4)
    assert_allclose(f['local'],4,atol=1e-6)


def test_spacing_rulers_follow_pair_and_hide_without_overlap():
    v=GaussianOrderBeamsExplorer()
    try:
        f=v.fringes
        assert f['nominal']==4
        assert_allclose(f['local'],4.50265,atol=1e-5)
        ruler=v.interactive.roles['nominal-ruler']
        assert_allclose(ruler.x[-1]-ruler.x[0],4)
        cached=v.fringes
        v.phase.value=90
        assert v.fringes is cached
        v.pair.value=(-1,1)
        assert v.fringes['nominal']==2
        v.waist.value=4
        v.pair.value=(-2,2)
        v.detector.value=80
        assert v.fringes['center'] is None
        assert not v.interactive.roles['nominal-ruler'].visible
        assert not v.interactive.roles['local-ruler'].visible
    finally:
        v.close()


def test_fitted_square_windows_follow_beam_footprints_without_changing_overlap():
    from wave_optics.gaussian_order_interactive_view import fitted_view_bounds
    model=GaussianOrderExperiment()
    near=fitted_view_bounds(model,(0,1),0)
    default=fitted_view_bounds(model,(0,1),40)
    far=fitted_view_bounds(model,(0,1),80)
    assert near['side'] < default['side'] < far['side'] < 128
    assert default['side'] == pytest.approx(41.6)
    v=GaussianOrderBeamsExplorer()
    try:
        data=v.data
        assert np.ptp(v.picture.layout.xaxis.range) == pytest.approx(default['side'])
        assert np.ptp(v.picture.layout.yaxis.range) == pytest.approx(default['side'])
        assert_allclose(v.interactive.roles['grating'].x,default['x'])
        assert_allclose(v.interactive.roles['border'].x[:2],default['x'])
        initial_x=v.detector_x
        v.phase.value=90
        assert v.detector_x is initial_x
        v.detector.value=80
        assert v.data is data
        assert v.view_bounds['side'] > default['side']
        for x in v.detector_x[::max(1,len(v.detector_x)//10)]:
            assert abs((x-v.x[0])/.1-round((x-v.x[0])/.1)) < 1e-10
        assert (len(v.detector_x)-1)%16==0 and (len(v.detector_y)-1)%8==0
        # Moving the display crop never alters the full-line overlap stack.
        assert_allclose(v.data['overlap'],data['overlap'])
    finally:
        v.close()


def test_section_aids_follow_shared_controls_and_skip_phase_only_renders():
    v=GaussianOrderBeamsExplorer()
    try:
        aids=v.lesson_aids
        assert aids.renders==[0,0]
        first=aids.coordinates_view();second=aids.overlap_view()
        assert first.value[:8]==b'\x89PNG\r\n\x1a\n'
        assert second.value[:8]==b'\x89PNG\r\n\x1a\n'
        assert aids.renders==[1,1]
        snapshot=(first.value,second.value)
        for control,value in ((v.detector,28),(v.wavelength,.8),(v.pitch,5.),
                              (v.waist,4.),(v.pair,(-1,1))):
            before=list(aids.renders)
            control.value=value
            assert aids.renders==[before[0]+1,before[1]+1]
            assert (first.value,second.value)!=snapshot
            snapshot=(first.value,second.value)
        before=list(aids.renders)
        v.phase.value=90
        assert aids.renders==before
        assert (first.value,second.value)==snapshot
        # Redisplaying either panel keeps the same widget and cached image.
        assert aids.coordinates_view() is first and aids.overlap_view() is second
        assert aids.renders==before
    finally:
        v.close()
    assert aids._closed and aids.explorer is None


def test_notebook_setup_replaces_cached_old_class_and_closes_previous():
    import wave_optics.gaussian_order_beams_view as cached
    class LegacyExplorer:
        closed=False
        def close(self):self.closed=True
    old=LegacyExplorer()
    # Reproduce a long-lived module cache with the pre-lesson-aids class.
    cached.GaussianOrderBeamsExplorer=LegacyExplorer
    fresh=create_gaussian_explorer(old)
    try:
        assert old.closed
        assert hasattr(fresh,'lesson_aids')
        assert not isinstance(fresh,LegacyExplorer)
        assert fresh.lesson_aids.explorer is fresh
    finally:
        fresh.close()


def test_optional_coordinate_panel_refreshes_only_when_open():
    with patch('wave_optics.gaussian_lesson_figures.figure_png',return_value=b'png'):
        v=GaussianOrderBeamsExplorer()
        try:
            aids=v.lesson_aids
            panel=aids.coordinates_panel()
            assert panel.selected_index is None and aids.renders[0]==0
            v.wavelength.value=.8
            assert aids.renders[0]==0
            panel.selected_index=0
            assert aids.renders[0]==1
            assert aids._keys[0]==(v._key,v.detector.value)
            panel.selected_index=None
            v.waist.value=4
            assert aids.renders[0]==1
            panel.selected_index=0
            assert aids.renders[0]==2
            v.phase.value=90
            assert aids.renders[0]==2
        finally:
            v.close()
        assert panel._trait_notifiers['selected_index']['change'].count(aids._coordinate_visibility)==0


def test_main_calculation_uses_current_controls_without_optional_setup():
    root=Path(__file__).resolve().parents[2]
    notebook=json.loads((root/'7_3_Finite_Diffraction_Beams_and_Experimental_Recombination.ipynb').read_text(encoding='utf-8'))
    source=''.join(next(c['source'] for c in notebook['cells'] if c['id']=='n73-overlap-calculation'))
    v=GaussianOrderBeamsExplorer()
    try:
        v.wavelength.value=.8
        v.pair.value=(-1,1)
        v.detector.value=28
        v.phase.value=90
        ns=dict(gaussian_orders=v,np=np,line_overlap=line_overlap)
        exec(source,ns)
        expected=v.data['fields'][:,[0,28,80]].copy()
        expected[1]*=1j
        assert_allclose(ns['sample']['fields'],expected,atol=2e-15)
        assert ns['beam_model'] is v.model
    finally:
        v.close()
