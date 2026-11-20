"""Physical kernels, approximation steps, display scales and owned controls."""
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from ipywidgets.widgets.widget import _instances
from numpy.testing import assert_allclose
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg

from wave_optics import propagation_kernels as kernels
from wave_optics.green_kirchhoff import response


def test_full_kernel_is_the_normal_derivative_of_the_3d_green_response():
    # The established 3D gradient supplies an independent normalization/sign check.
    lam, s = .7, 2.3
    x = np.array([-1., .4, 2.]); xi = np.array([.2, -.3, 1.2])
    y, eta = .8, -.6
    actual = kernels.kernel_values(x, xi, y=y, eta=eta, distance=s, wavelength=lam)
    for j in range(3):
        _, gradient = response(np.array([x[j], y, s]), np.array([xi[j], eta, 0.]), 2*np.pi/lam)
        assert_allclose(actual['kernels'][0, j], -2*gradient[2], atol=2e-15)
    assert_allclose(actual['reduced'], abs(actual['reduced'])*np.exp(1j*actual['phase']), atol=3e-14)


def test_full_kernel_integral_agrees_with_independent_angular_spectrum():
    from wave_optics.propagation_comparison import boundary_field, asm_cut
    s, dx = 80., .125
    q = np.arange(-30+dx/2, 30, dx)
    xi, eta = np.meshgrid(q, q)
    boundary = boundary_field(xi, eta)
    points = np.array([0., 8., 19.])
    integral = [np.sum(kernels.kernel_values(x, xi, eta=eta, distance=s)['kernels'][0]*boundary)*dx**2
                for x in points]
    assert_allclose(integral, asm_cut(points, distance=s), atol=2e-11, rtol=0)


def test_fraunhofer_omits_exactly_the_boundary_quadratic_phase():
    x, xi, y, eta, s, lam = .8, np.linspace(-3, 3, 11), -.4, .6, 2., .9
    d = kernels.kernel_values(x, xi, y=y, eta=eta, distance=s, wavelength=lam)
    omitted = np.pi*(xi*xi+eta*eta)/(lam*s)
    assert_allclose(d['kernels'][1]/d['kernels'][2], np.exp(1j*omitted), atol=1e-14)
    assert_allclose(d['phase'][1]-d['phase'][2], omitted, atol=1e-14)
    central = kernels.kernel_values(2., 0., distance=1.)
    assert_allclose(central['kernels'][1], central['kernels'][2], atol=1e-15)
    assert abs(central['kernels'][0]-central['kernels'][1]) > .1


def test_translation_distinguishes_convolution_from_the_far_field_mapping():
    x, xi, s, a = .3, .8, 4., .7
    before = kernels.kernel_values(x, xi, distance=s)
    after = kernels.kernel_values(x+a, xi+a, distance=s)
    assert_allclose(after['kernels'][:2], before['kernels'][:2], atol=1e-15)
    assert_allclose(after['kernels'][2]/before['kernels'][2],
                    np.exp(-1j*np.pi*(2*a*xi+a*a)/s), atol=1e-15)
    assert abs(after['kernels'][2]-before['kernels'][2]) > .1


def test_inverse_area_units_and_common_scaling():
    d = kernels.kernel_values(.8, -.4, y=.3, eta=.6, distance=2.7, wavelength=.8)
    t = 3.1
    scaled = kernels.kernel_values(.8*t, -.4*t, y=.3*t, eta=.6*t, distance=2.7*t, wavelength=.8*t)
    assert_allclose(scaled['kernels']*t*t, d['kernels'], atol=2e-15)
    assert_allclose(scaled['reduced'], d['reduced'], atol=2e-14)
    assert_allclose(d['reduced'], 1j*.8*2.7*np.exp(-1j*2*np.pi*2.7/.8)*d['kernels'], atol=2e-15)


def test_fixed_transverse_positions_approach_paraxial_and_far_field_limits():
    near = kernels.kernel_state(1., 3., -3.)['selected']
    far = kernels.kernel_state(100., 3., -3.)['selected']
    assert near['phase_error'][1] > 10*2*np.pi  # retain whole turns
    assert max(abs(far['phase_error'][1:])) < .3
    assert_allclose(np.exp(1j*near['phase_error']),
                    near['kernels']/near['kernels'][0]/abs(near['kernels']/near['kernels'][0]), atol=2e-14)
    assert max(abs(far['kernels'])) < max(abs(near['kernels']))/90


def test_phase_fronts_follow_the_full_kernel_and_obliquity_weight():
    for outer in (12., 124.):
        radii, stride = kernels.phase_front_radii(outer)
        assert len(radii) <= 12 and np.all((radii > 0) & (radii <= outer))
        turns = (2*np.pi*radii-np.arctan(2*np.pi*radii))/(2*np.pi)
        assert_allclose(turns, np.rint(turns), atol=2e-14)
        assert_allclose(np.diff(turns), stride, atol=2e-14)
        for theta in (0., .6, 1.2):
            # Points on the same radius have equal phase, but different strength.
            for radius in radii[::3]:
                s = radius*np.cos(theta)
                actual = kernels.kernel_values(radius*np.sin(theta), 0., distance=s)['kernels'][0]
                assert abs(np.angle(actual)) < 2e-13
                expected = np.cos(theta)/radius*np.sqrt(1+1/(2*np.pi*radius)**2)
                assert_allclose(abs(actual), expected, atol=2e-15)


def test_real_patch_response_keeps_absolute_phase_and_spreading():
    data = kernels.kernel_state()
    w = data['selected']['kernels']
    for fraction, expected in [(0., w.real), (.25, w.imag), (.5, -w.real), (.75, -w.imag)]:
        index = np.flatnonzero(np.isclose(data['time_cycles'], fraction))[0]
        assert_allclose(data['real_response'][:, index], expected, atol=2e-15)
    far = kernels.kernel_state(100.)
    assert np.max(abs(far['real_response'])) < np.max(abs(data['real_response']))/20
    assert not data['real_response'].flags.writeable


def test_each_live_control_changes_the_relevant_geometry_or_kernel():
    baseline = set(_instances)
    panel = kernels.kernel_explorer()
    try:
        old = panel.data
        for name, value in [('distance', 10.), ('observation', 2.), ('boundary', -2.)]:
            png = bytes(panel.image.value)
            panel.controls[name].value = value
            assert panel.data is not old and bytes(panel.image.value) != png
            assert not np.allclose(panel.data['selected']['kernels'], old['selected']['kernels'])
            old = panel.data
        png = bytes(panel.image.value)
        panel.controls['view'].value = 'Absolute weights'
        assert panel.data is old and bytes(panel.image.value) != png
        assert '$' not in panel.equation.value and '\\' not in panel.equation.value
        Image.open(BytesIO(panel.image.value)).verify()
        for view in kernels.VIEWS:
            fig = kernels.kernel_figure(panel.data, view)
            try:
                FigureCanvasAgg(fig).draw()
                geo = fig.axes[0]
                for horizontal, transverse in [(0., panel.data['boundary']),
                                               (panel.data['distance'], panel.data['observation'])]:
                    assert geo.get_xlim()[0] < horizontal < geo.get_xlim()[1]
                    assert geo.get_ylim()[0] < transverse < geo.get_ylim()[1]
                assert_allclose(fig.axes[1].get_ylim(), [-1.15, 1.15])
                assert_allclose(fig.axes[2].get_ylim(), [0, 1.15])
                if view == 'Patch response':
                    assert_allclose(fig.axes[1].get_xlim(), [0, 2])
                    for i in range(3):
                        assert_allclose(fig.axes[1].lines[2*i].get_ydata(), panel.data['real_response'][i])
                else:
                    assert_allclose(fig.axes[1].get_xlim(), [-1.15, 1.15])
                    key = 'reduced' if view == 'Compare shapes' else 'kernels'
                    for annotation, weight in zip(fig.axes[1].texts, panel.data['selected'][key]):
                        assert_allclose(annotation.xy, [weight.real, weight.imag])
            finally:
                fig.clear()
    finally:
        panel.close(); panel.close()
    assert set(_instances) == baseline


def test_failed_initial_render_cleans_up_all_owned_widgets():
    baseline = set(_instances)
    with patch.object(kernels, 'kernel_figure', side_effect=RuntimeError('render failed')):
        with pytest.raises(RuntimeError):
            kernels.kernel_explorer()
    assert set(_instances) == baseline


def test_invalid_inputs_are_rejected():
    for args in [dict(distance=0), dict(wavelength=-1), dict(wavelength=True), dict(x=1j),
                 dict(eta=np.nan), dict(x=np.inf)]:
        base = dict(x=0., xi=0.)
        base.update(args)
        with pytest.raises(ValueError):
            kernels.kernel_values(**base)
    for args in [dict(distance=.5), dict(boundary=4.), dict(observation=[1., 2.]), dict(observation=True)]:
        with pytest.raises((ValueError, TypeError)):
            kernels.kernel_state(**args)
    kernels.kernel_state(1., 1., 1.)
    with pytest.raises(ValueError):
        kernels.kernel_state(True, 1., 1.)
