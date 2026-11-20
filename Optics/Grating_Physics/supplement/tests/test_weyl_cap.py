"""Weyl normalization/branches, grating boundary data, curved-cap sum and UI."""
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose
from PIL import Image
from ipywidgets.widgets.widget import _instances

from wave_optics.weyl_cap import (
    K,SOURCE_DISTANCE,CAP_CENTER,CAP_RADIUS,CAP_QUADRATURE_ORDER,cap_points,reference_field,weyl_modes,
    reconstructed_field,cap_profile,selected_orders,weyl_cap_explorer,cap_figure,
    _spectrum_density,
)


def test_no_grating_reconstructs_the_sphere_in_full_3d():
    points=np.array([[0.,0.,0.],[.4,-.3,.8],[1.1,.6,2.]])
    assert_allclose(reconstructed_field(points),reference_field(points),atol=1.2e-10,rtol=0)
    profile=cap_profile()
    assert_allclose(profile['ratio'],1,atol=1e-10,rtol=0)
    # A source-centred wave has varying raw phase on the OBJECT-centred cap.
    assert np.ptp(np.unwrap(np.angle(profile['field']))) > .5


def test_cap_is_curved_and_centered_on_the_reference_object_point():
    p=cap_points([-40,0,40],[0,15,90])
    assert_allclose(CAP_CENTER,[0.,0.,0.],atol=0)
    assert_allclose(np.linalg.norm(p-CAP_CENTER,axis=1),CAP_RADIUS,atol=1e-14)
    assert_allclose(cap_points(0),[0.,0.,CAP_RADIUS],atol=0)
    assert np.all(p[:,2]>0)
    assert np.ptp(p[:,2])>0
    assert np.ptp(np.linalg.norm(p+[0,0,SOURCE_DISTANCE],axis=1)) > .1


@pytest.mark.parametrize('period',[1.2,2.,4.])
def test_thin_grating_boundary_values_and_quadrature_convergence(period):
    points=np.array([[0,0,0],[.2,-.3,0],[2.1,.4,0]],float)
    expected=(.5+.5*np.cos(K*points[:,0]/period))*reference_field(points)
    assert_allclose(reconstructed_field(points,True,period),expected,atol=1.2e-10,rtol=0)
    normal=cap_profile(True,period,CAP_QUADRATURE_ORDER)
    refined=cap_profile(True,period,96)
    assert_allclose(normal['ratio'],refined['ratio'],atol=3e-8,rtol=0)
    assert np.ptp(abs(normal['field']))>.001


def test_outgoing_dispersion_and_selected_order_shifts():
    vectors,coefficients=weyl_modes(True,2.)
    assert_allclose(np.sum(vectors*vectors,axis=1),K*K,atol=2e-12)
    assert np.all(vectors[:,2].real<=0) and np.all(vectors[:,2].imag<=0)
    assert np.any(vectors[:,2].imag<0)
    assert np.all(np.isfinite(coefficients))
    data=selected_orders(30,True,2.)
    assert_allclose(np.diff(data['alpha']),.5,atol=1e-15)
    assert_allclose(data['alpha'],[0,.5,1],atol=1e-15)


def test_cap_grating_field_obeys_cartesian_helmholtz():
    p=np.array([.3,-.2,2.]);h=1e-4
    u=reconstructed_field(p,True,2.)
    lap=sum((reconstructed_field(p+h*e,True,2.)-2*u+reconstructed_field(p-h*e,True,2.))/h**2 for e in np.eye(3))
    assert_allclose(lap+K*K*u,0,atol=2e-7)


def test_component_selection_does_not_change_full_field_but_grating_does():
    baseline=set(_instances);panel=weyl_cap_explorer()
    try:
        panel.controls['stage'].value=3
        panel.controls['grating'].value=False
        initial=panel.data['profile']
        old_image=bytes(panel.image.value)
        panel.controls['input_angle'].value=-30
        assert panel.data['profile'] is initial
        assert bytes(panel.image.value)!=old_image
        assert panel.controls['period'] not in panel.children[0].children
        panel.controls['grating'].value=True
        assert panel.controls['period'] in panel.children[0].children
        assert not np.allclose(panel.data['profile']['field'],initial['field'])
        old=panel.data['profile']
        panel.controls['period'].value=2.4
        assert not np.allclose(panel.data['profile']['field'],old['field'])
        current=panel.data['profile'];panel.controls['cap_angle'].value=30
        assert panel.data['profile'] is current
        assert_allclose(panel.data['selected_field'],current['field'][70],atol=0)
        assert '$' not in panel.equation.value and '\\' not in panel.equation.value
        Image.open(BytesIO(panel.image.value)).verify()
    finally:panel.close();panel.close()
    assert set(_instances)==baseline


@pytest.mark.parametrize('source_x,source_distance',[(-1.,1.),(.6,1.2),(1.,2.5)])
def test_moved_source_reconstruction_mask_and_fixed_cap(source_x,source_distance):
    points=np.array([[.2,-.3,0.],[1.1,.6,2.],[0.,0.,4.5]])
    expected=reference_field(points,source_x,source_distance)
    assert_allclose(reconstructed_field(points,source_x=source_x,source_distance=source_distance),
                    expected,atol=2e-10,rtol=0)
    plane=points[:1]
    masked=(.5+.5*np.cos(K*plane[:,0]/4.))*reference_field(plane,source_x,source_distance)
    assert_allclose(reconstructed_field(plane,True,4.,48,source_x,source_distance),masked,atol=2e-10,rtol=0)
    normal=cap_profile(True,4.,CAP_QUADRATURE_ORDER,source_x,source_distance)
    refined=cap_profile(True,4.,96,source_x,source_distance)
    assert_allclose(normal['ratio'],refined['ratio'],atol=3e-8,rtol=0)
    assert_allclose(normal['points'],cap_profile()['points'],atol=0)
    assert_allclose(normal['reference'],cap_profile()['reference'],atol=0)
    sphere=cap_profile(False,2.,48,source_x,source_distance)
    assert_allclose(sphere['field'],reference_field(sphere['points'],source_x,source_distance),atol=2e-10,rtol=0)
    assert not np.allclose(sphere['ratio'],1)


def test_translation_phase_and_coherent_grating_spectrum():
    kx=K*np.array([.2,.6,1.3]);ky=K*np.array([.1,-.2,.3])
    kappa=np.sqrt((K*K-kx*kx-ky*ky).astype(complex))
    base=_spectrum_density(kx,ky,False,2.,0.,1.5)
    assert_allclose(base,1j/(2*kappa)*np.exp(1j*kappa*1.5),atol=2e-17)
    moved=_spectrum_density(kx,ky,False,2.,.7,1.5)
    assert_allclose(moved,base*np.exp(-1j*kx*.7),atol=2e-17)
    assert_allclose(abs(moved),abs(base),atol=2e-17)
    farther=_spectrum_density(kx,ky,False,2.,0.,2.)
    assert_allclose(farther,base*np.exp(1j*kappa*.5),atol=2e-17)
    shifted=sum(c*_spectrum_density(kx-m*2*np.pi/2.,ky,False,2.,.7,1.5)
                for m,c in [(-1,.25),(0,.5),(1,.25)])
    assert_allclose(_spectrum_density(kx,ky,True,2.,.7,1.5),shifted,atol=2e-17)


def test_evanescent_grazing_split_outside_moved_source_truncation():
    # At this distance/period the outgoing radial grazing transition exceeds
    # beta_max. Integration must stop at beta_max, without negative intervals.
    normal=cap_profile(True,1.2,CAP_QUADRATURE_ORDER,.7,2.5)
    refined=cap_profile(True,1.2,96,.7,2.5)
    assert_allclose(normal['field'],refined['field'],atol=3e-10,rtol=0)


def test_four_steps_expose_relevant_controls_and_changing_objects():
    baseline=set(_instances);panel=weyl_cap_explorer()
    try:
        c=panel.controls
        initial=panel.data['plane_field'].copy();old_image=bytes(panel.image.value)
        assert panel.data['stage']==0 and panel.data['profile'] is None
        assert c['input_angle'] not in panel.children[0].children
        assert c['grating'] not in panel.children[0].children
        c['source_x'].value=.7
        assert not np.allclose(panel.data['plane_field'],initial)
        assert bytes(panel.image.value)!=old_image
        c['stage'].value=1
        assert c['input_angle'] in panel.children[0].children
        weight=panel.data['spectral_weight'];image=panel.data['spectrum_image'].copy()
        c['source_x'].value=-.4
        assert not np.isclose(panel.data['spectral_weight'],weight)
        assert_allclose(panel.data['spectrum_image'],image,atol=1e-14)
        old_image=bytes(panel.image.value);c['input_angle'].value=-40
        assert bytes(panel.image.value)!=old_image
        c['stage'].value=2
        assert c['period'] in panel.children[0].children
        assert c['cap_angle'] not in panel.children[0].children
        mask=panel.data['mask'].copy();image=panel.data['spectrum_image'].copy()
        c['period'].value=2.4
        assert not np.allclose(panel.data['mask'],mask)
        assert not np.allclose(panel.data['spectrum_image'],image)
        c['stage'].value=3
        profile=panel.data['profile'];point=panel.data['point'].copy()
        c['source_distance'].value=2.
        assert_allclose(panel.data['point'],point,atol=0)
        assert not np.allclose(panel.data['profile']['field'],profile['field'])
        c['cap_angle'].value=30
        assert not np.allclose(panel.data['point'],point)
        assert '$' not in panel.equation.value and '\\' not in panel.equation.value
        assert '?' not in panel.equation.value
        Image.open(BytesIO(panel.image.value)).verify()
    finally:panel.close()
    assert set(_instances)==baseline


def test_retained_cap_explorer_and_failed_render_cleanup():
    # The cap application is retained for later lessons, not notebook 8.2.
    baseline=set(_instances)
    panel=weyl_cap_explorer()
    panel.close();panel.close()
    assert set(_instances)==baseline
    with patch('wave_optics.weyl_cap.cap_figure',side_effect=RuntimeError('render failed')):
        with pytest.raises(RuntimeError):weyl_cap_explorer()
    assert set(_instances)==baseline


def test_invalid_inputs():
    for args in [dict(period=0),dict(period=np.nan),dict(order=3),dict(order=True),dict(grating='yes')]:
        with pytest.raises(ValueError):weyl_modes(**args)
    with pytest.raises(ValueError):cap_points(80)
    with pytest.raises(ValueError):reconstructed_field([0,0,-.1])
    with pytest.raises(ValueError):reconstructed_field([0,0,1j])
    with pytest.raises(ValueError):selected_orders(80,False,2)
    for args in [dict(source_x=1.1),dict(source_distance=.9),dict(source_distance=np.nan),dict(source_x=True)]:
        with pytest.raises(ValueError):weyl_modes(**args)
    with pytest.raises(ValueError):cap_figure(stage=1.5)
