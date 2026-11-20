"""Physics, input contracts, rendering and rerun lifecycle for notebook 8.1."""
from collections import Counter
from io import BytesIO
from itertools import product
from pathlib import Path
from unittest.mock import patch
import gc

import nbformat
import numpy as np
import pytest
from PIL import Image
from numpy.testing import assert_allclose
from ipywidgets.widgets.widget import _instances

from wave_optics.propagation_interactive import (
    green, green_normal, patches, picture, propagation_explorer, poisson_arago_field,
    screen_profile,
)


def test_green_normal_and_source_free_helmholtz():
    s=2.3; y=np.linspace(-3,3,31); h=1e-5
    assert_allclose((green(s-h,y)-green(s+h,y))/(2*h),
                    green_normal(s,y),rtol=1e-8,atol=1e-9)
    p=np.array([.7,-.3,2.1]); h=1e-4
    def field(p):
        return green(p[2],np.hypot(p[0],p[1]))
    lap=sum((field(p+h*e)-2*field(p)+field(p-h*e))/h**2 for e in np.eye(3))
    assert_allclose(lap+(2*np.pi)**2*field(p),0,atol=5e-7)


def test_poisson_arago_axis_symmetry_and_radial_convergence():
    s=25.;radius=5.
    expected=np.exp(2j*np.pi*s+1j*np.pi*radius**2/s)
    assert_allclose(poisson_arago_field(0,s,radius),expected,atol=2e-13)
    x=np.linspace(-10,10,51)
    values=poisson_arago_field(x,s,radius)
    assert_allclose(values,values[::-1],atol=1e-14)
    assert_allclose(values,poisson_arago_field(x,s,radius,count=240),atol=2e-13)
    assert_allclose(abs(poisson_arago_field(0,s,radius))**2,1,atol=1e-13)
    assert_allclose(poisson_arago_field(x,s,radius=1e-5),np.exp(2j*np.pi*s),atol=2e-10)
    with pytest.raises(ValueError):poisson_arago_field(x,s=0)


def test_patch_integral_symmetry_convergence_and_helmholtz():
    assert_allclose(patches(3,.7,3,1)[1].sum(),patches(3,-.7,3,-1)[1].sum(),atol=1e-14)
    values=[patches(.8,3,6,4,n)[1].sum() for n in (41,81,161)]
    assert abs(values[1]-values[2]) < abs(values[0]-values[1])
    # At x=y=0, square symmetry gives d_xx U=d_yy U. Each integrated
    # kernel must obey the source-free PDE, including its near-field term.
    h=1e-4; s=2.3
    def field(s,y):return patches(s,y,3,0)[1].sum()
    u=field(s,0)
    lap=2*(field(s,h)-2*u+field(s,-h))/h**2
    lap+=(field(s+h,0)-2*u+field(s-h,0))/h**2
    assert_allclose(lap+(2*np.pi)**2*u,0,atol=2e-6)


@pytest.mark.parametrize('s,y,width,ramp',[
    (.8,3,6,4),(.8,0,6,0),(8,3,.8,4),(3,.5,3,0),
])
def test_midpoints_against_gauss_legendre_integral(s,y,width,ramp):
    nodes,weights=np.polynomial.legendre.leggauss(100)
    q=nodes*width/2; w=weights*width/2
    eta,xi=np.meshgrid(q,q,indexing='ij')
    kernel=2*green_normal(s,np.hypot(y-eta,xi))
    reference=np.sum(np.exp(1j*ramp*eta)*kernel*w[:,None]*w[None,:])
    # Absolute complex-field accuracy: relative error is meaningless near zeros.
    assert_allclose(patches(s,y,width,ramp)[1].sum(),reference,rtol=0,atol=.004)


@pytest.mark.parametrize('fn,args',[
    (green,(0,0)),(green,(np.nan,0)),(green,(1,1j)),
    (green,(1,0,-1)),(green_normal,(-1,0)),
    (patches,(1,0,0,0)),(patches,(1,0,1,np.inf)),
    (patches,(1,0,1,0,0)),(patches,(1,0,1,0,3.5)),
    (patches,(1,0,1,0,True)),
    (picture,('unknown',)),(picture,('Patch sum',1,0,1,0,'RS I',-1)),
    (picture,('Patch sum',1,0,1,0,'RS I',41)),
    (picture,('Boundary choice',1,0,1,0,'unknown')),
])
def test_reject_invalid_inputs(fn,args):
    with pytest.raises(ValueError):fn(*args)


@pytest.mark.parametrize('choice,zero_line',[('RS I',0),('RS II',1)])
def test_image_cancellation_on_complete_plane(choice,zero_line):
    fig=picture('Boundary choice',s=.8,y=3,width=6,choice=choice)
    try:
        lines=fig.axes[1].lines
        assert_allclose(lines[zero_line].get_ydata(),0,atol=0)
        assert np.all(lines[1-zero_line].get_ydata()>0)
    finally:fig.clear()


def test_plot_row_chain_matches_displayed_rs_integral():
    fig=picture('Patch sum',s=2.5,y=.7,width=3,ramp=1.2,row=10)
    try:
        chain=fig.axes[2].lines[0]
        plotted=chain.get_xdata()+1j*chain.get_ydata()
        _,rows=patches(2.5,.7,3,1.2)
        assert_allclose(np.diff(plotted),rows,atol=1e-15)
        assert_allclose(plotted[-1],rows.sum(),atol=1e-15)
    finally:fig.clear()


def test_screen_curve_marker_and_phasor_sum_are_the_same_calculation():
    fig=picture('Patch sum',s=2.5,y=.75,width=3,ramp=1.2,row=10)
    try:
        data=fig.propagation_data
        profile=data['profile']
        index=np.flatnonzero(np.isclose(profile['y'],.75))[0]
        expected=patches(2.5,.75,3,1.2)[1].sum()
        assert_allclose(profile['field'][index],expected,atol=2e-15)
        assert_allclose(data['selected_field'],expected,atol=2e-15)
        curve,reference,marker=fig.axes[3].lines
        assert_allclose(curve.get_ydata()[index],abs(expected)**2,atol=3e-15)
        assert_allclose(marker.get_ydata(),abs(expected)**2,atol=3e-15)
        assert_allclose(reference.get_ydata(),1)
        assert data['intensity_reference']==1
    finally:fig.clear()


def test_profile_symmetry_phase_changes_and_fixed_normalization():
    normal=screen_profile(3,3,0)
    ramp=screen_profile(3,3,2)
    reverse=screen_profile(3,3,-2)
    assert_allclose(normal['intensity'],normal['intensity'][::-1],atol=1e-14)
    assert_allclose(ramp['intensity'],reverse['intensity'][::-1],atol=1e-14)
    # The two input fields have identical intensity; their output cuts differ.
    assert np.max(abs(normal['intensity']-ramp['intensity'])) > .5
    half=screen_profile(3,3,0,amplitude=.5)
    assert_allclose(half['intensity'],.25*normal['intensity'],atol=1e-15)
    assert half['intensity_reference']==normal['intensity_reference']==1
    assert normal['intensity'].max()>1  # No independent peak normalization.
    assert_allclose(screen_profile(3,3,0,amplitude=0)['intensity'],0,atol=0)
    assert not normal['intensity'].flags.writeable


@pytest.mark.parametrize('s,width,ramp',[(3,3,0),(.8,6,0),(.8,6,4),(8,6,4),(8,.8,0)])
def test_full_screen_cut_quadrature_and_fringe_sampling(s,width,ramp):
    coarse=screen_profile(s,width,ramp)
    refined=screen_profile(s,width,ramp,count=81)
    reference=screen_profile(s,width,ramp,count=161)
    err_coarse=np.max(abs(coarse['intensity']-reference['intensity']))
    err_refined=np.max(abs(refined['intensity']-reference['intensity']))
    assert err_refined<err_coarse
    assert err_coarse<.011  # Absolute units of the fixed incident intensity.
    assert err_refined<.0022
    # Check intervening screen points, not just the same sampled locations.
    dense=screen_profile(s,width,ramp,positions=np.linspace(-8,8,641),count=81)
    interpolation=np.interp(dense['y'],refined['y'],refined['intensity'])
    assert np.max(abs(interpolation-dense['intensity'])) < .004


@pytest.mark.parametrize('kwargs',[
    dict(positions=[0]),dict(positions=[[0,1]]),dict(positions=[0,0]),
    dict(positions=[1,0]),dict(positions=[0,np.nan]),dict(amplitude=-1),
])
def test_invalid_screen_profile_inputs(kwargs):
    with pytest.raises(ValueError):screen_profile(3,3,0,**kwargs)


def test_moving_point_or_selecting_row_does_not_reintegrate_or_change_profile():
    panel=propagation_explorer(stage='Patch sum')
    try:
        original=panel.data['profile']
        curve=original['intensity'].copy()
        with patch('wave_optics.propagation_interactive.screen_profile',wraps=screen_profile) as integrate:
            panel.controls['y'].value=.75
            value=panel.data['selected_field']
            panel.controls['row'].value=0
            assert panel.data['selected_field']==value
            assert panel.data['profile'] is original
            assert_allclose(panel.data['profile']['intensity'],curve,atol=0)
            integrate.assert_not_called()
            panel.controls['amplitude'].value=.5
            assert integrate.call_count==1
            assert_allclose(panel.data['profile']['intensity'],.25*curve,atol=1e-15)
    finally:panel.close()


def test_all_control_corners_render_finite_svg():
    for s,y,width,ramp in product((.8,8),(-8,8),(.8,6),(-4,4)):
        fig=picture('Patch sum',s=s,y=y,width=width,ramp=ramp,row=40)
        try:
            for ax in fig.axes:
                for line in ax.lines:
                    assert np.all(np.isfinite(line.get_xydata()))
            data=BytesIO();fig.savefig(data,format='svg')
            assert b'<svg' in data.getvalue()
        finally:fig.clear()


def test_notebook_cell_reruns_close_only_owned_widgets():
    from ipywidgets import FloatSlider
    external=FloatSlider()
    baseline=set(_instances)
    notebook=nbformat.read(Path(__file__).resolve().parents[2]/
        '8_2_Rayleigh_Sommerfeld_Propagation.ipynb',as_version=4)
    source=next(c.source for c in notebook.cells if c.cell_type=='code' and
                '_propagation_panel = propagation_explorer(' in c.source)
    namespace={}; old=[]; counts=None
    try:
        with patch('IPython.display.display'):
            for repeat in range(3):
                exec(compile(source,'interactive-cell','exec'),namespace)
                panel=namespace['_propagation_panel']
                owned={key for key in _instances if key not in baseline}
                current=Counter(type(_instances[key]).__name__ for key in owned)
                if counts is None:counts=current
                assert current==counts
                assert baseline<=set(_instances)
                for previous in old:
                    assert previous._closed
                    for control in previous.controls.values():
                        assert previous._update_callback not in control._trait_notifiers['value']['change']
                # Real, successful callback serialization in each stage.
                panel.controls['stage'].value='Boundary choice'
                for choice in ('Kirchhoff','RS I','RS II'):
                    panel.controls['choice'].value=choice
                panel.controls['stage'].value='Patch sum'
                image=bytes(panel.children[-1].value)
                panel.controls['row'].value=40
                assert bytes(panel.children[-1].value)!=image
                assert panel.children[-1].format == 'png'
                assert bytes(panel.children[-1].value).startswith(b'\x89PNG\r\n\x1a\n')
                with Image.open(BytesIO(bytes(panel.children[-1].value))) as decoded:
                    decoded.verify()
                old.append(panel)
        panel.close();panel.close()
        gc.collect()
        assert set(_instances)==baseline
    finally:
        if '_propagation_panel' in namespace:namespace['_propagation_panel'].close()
        external.layout.close();external.style.close();external.close()


@pytest.mark.parametrize('stage',[None,'Point response','Boundary choice','Patch sum'])
def test_failed_initial_render_releases_widget_tree(stage):
    baseline=set(_instances)
    with patch('wave_optics.propagation_interactive.picture',side_effect=RuntimeError('render failed')):
        with pytest.raises(RuntimeError,match='render failed'):propagation_explorer(stage)
    assert set(_instances)==baseline


def test_two_8_2_experiments_are_fixed_and_rerun_independently():
    baseline=set(_instances)
    notebook=nbformat.read(Path(__file__).resolve().parents[2]/
        '8_2_Rayleigh_Sommerfeld_Propagation.ipynb',as_version=4)
    cells=[c for c in notebook.cells if c.cell_type=='code' and 'propagation_explorer(' in c.source]
    assert len(cells)==2
    names=('_boundary_choice_panel','_propagation_panel')
    namespace={}
    try:
        with patch('IPython.display.display'):
            for cell in cells:exec(compile(cell.source,cell.id,'exec'),namespace)
            initial=set(_instances)-baseline
            for cell,name,stage in zip(cells,names,('Boundary choice','Patch sum')):
                old=namespace[name]
                assert old.controls['stage'].value==stage
                assert old.controls['stage'].layout.display=='none'
                assert old.controls['stage'].disabled
                # Frontends need not honor hidden-layout CSS: the selector
                # must be absent from the rendered tree, not merely styled.
                def descendants(widget):
                    for child in getattr(widget,'children',()):
                        yield child
                        yield from descendants(child)
                assert old.controls['stage'] not in list(descendants(old))
                others=[namespace[n] for n in names if n!=name]
                exec(compile(cell.source,cell.id,'exec'),namespace)
                assert old._closed
                assert not any(getattr(p,'_closed',False) for p in others)
                assert len(set(_instances)-baseline)==len(initial)
    finally:
        for name in names:
            if name in namespace:namespace[name].close()
    assert set(_instances)==baseline
