"""Closed-surface physics and lifecycle of the linked 8.1 demonstration."""
from io import BytesIO
from unittest.mock import patch
import numpy as np
import pytest
from numpy.testing import assert_allclose
from PIL import Image
from ipywidgets.widgets.widget import _instances

from wave_optics.green_kirchhoff import (
    response, sphere_terms, closure_terms, stage_figure, linked_explorer, SOURCE,
)


def test_full_gradient_and_cartesian_helmholtz():
    q=np.array([.4,-.2,1.3]);h=1e-4;g,grad=response(q,SOURCE)
    fd=np.array([(response(q+h*e,SOURCE)[0]-response(q-h*e,SOURCE)[0])/(2*h) for e in np.eye(3)])
    assert_allclose(fd,grad,rtol=1e-7,atol=1e-9)
    lap=sum((response(q+h*e,SOURCE)[0]-2*g+response(q-h*e,SOURCE)[0])/h**2 for e in np.eye(3))
    assert_allclose(lap+(2*np.pi)**2*g,0,atol=4e-8)


@pytest.mark.parametrize('y',[-.6,.3,.6])
def test_surface_deformation_preserves_complex_field(y):
    p=np.array([0.,y,1.3]);values=[sphere_terms(p,r) for r in [1.5,2.1,2.8]]
    for t in values:assert_allclose(t['total'],t['reference'],rtol=3e-11,atol=1e-13)
    assert not np.allclose(values[0]['rings'],values[-1]['rings'])
    # A different source and off-axis P verify actual 3D geometry, not a planar derivative.
    t=sphere_terms([.2,y,1.3],2.,source=[-.7,.8,-3.],order=64)
    assert_allclose(t['total'],t['reference'],rtol=3e-11)


def test_exclusion_signs_total_at_finite_radius_and_separate_limits():
    p=np.array([0.,.3,1.3])
    for e in [.6,.2,.04]:
        t=sphere_terms(p,e,center=p,inner=True)
        assert_allclose(t['total'],-t['reference'],rtol=3e-12)
    large=sphere_terms(p,.04,center=p,inner=True)
    small=sphere_terms(p,.004,center=p,inner=True)
    assert abs(small['value']/small['reference']+1)<abs(large['value']/large['reference']+1)/80
    assert abs(small['slope']/small['reference'])<abs(large['slope']/large['reference'])/80


def test_quadrature_convergence_and_closing_surface():
    p=np.array([0.,.3,1.3])
    coarse=sphere_terms(p,2.8,order=16);fine=sphere_terms(p,2.8,order=48)
    assert abs(fine['total']-fine['reference'])<abs(coarse['total']-coarse['reference'])/1000
    errors=[]
    for r in [4.,8.,14.]:
        t=closure_terms(p,r);u=closure_terms(p,r,order=128)
        assert_allclose(t['total'],u['total'],atol=3e-12)
        errors.append(abs(t['total']))
    assert errors[0]>errors[1]>errors[2]


def test_invalid_domains_are_rejected():
    for kwargs in [dict(p=[0,0,4],radius=1),dict(p=[0,0,1],radius=5),
                   dict(p=[0,0,1],radius=0),dict(p=[0,0,1],radius=2,order=4),
                   dict(p=[0,0,1],radius=.1,inner=True)]:
        with pytest.raises(ValueError):sphere_terms(**kwargs)
    with pytest.raises(ValueError):response(SOURCE,SOURCE)


def test_failed_render_releases_widgets():
    baseline=set(_instances)
    with patch('wave_optics.green_kirchhoff.stage_figure',side_effect=RuntimeError('failed')):
        with pytest.raises(RuntimeError):linked_explorer()
    assert set(_instances)==baseline


@pytest.mark.parametrize('i',range(6))
def test_extreme_stage_states_render(i):
    f,_=stage_figure(i,y=-.6,radius=2.8,angle=6.2,epsilon=.04,closure=14)
    try:
        stream=BytesIO();f.savefig(stream,format='png',dpi=70)
        Image.open(stream).verify()
    finally:f.clear()
