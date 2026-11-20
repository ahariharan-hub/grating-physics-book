"""Full 3D Weyl reconstruction and the focused notebook 8.2 interaction."""
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import ipywidgets as widgets
from ipywidgets.widgets.widget import _instances
import nbformat
import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
import pytest

from wave_optics import weyl_reconstruction as weyl
from wave_optics.weyl_cap import reference_field, reconstructed_field


@pytest.mark.parametrize('source_x',[-1.,0.,1.])
def test_numerical_map_reconstructs_sphere_on_fixed_absolute_scale(source_x):
    grid=weyl.field_map(source_x)
    points=np.stack([grid['xx'],np.zeros_like(grid['xx']),grid['ss']],axis=-1)
    assert_allclose(grid['field'],reference_field(points,source_x),atol=1.2e-10,rtol=0)
    for name in ('x','s','ss','xx','field','phase'):
        assert not grid[name].flags.writeable
    # A meridional display is still a 3D spherical field, with 1/R spreading.
    assert_allclose(abs(grid['field'][:,48]),
                    1/(4*np.pi*np.hypot(source_x,grid['s']+1.5)),atol=1.2e-10,rtol=0)


def test_azimuth_integrated_map_agrees_with_independent_3d_sum():
    points=np.array([[0.,0.,0.],[.4,-.3,.8],[1.1,.6,2.]])
    for source_x,depth in [(-.7,1.),(.8,2.5)]:
        fast=weyl.weyl_field(points,source_x,depth)
        independent=reconstructed_field(points,source_x=source_x,source_distance=depth)
        assert_allclose(fast,independent,atol=2e-14,rtol=0)
        assert_allclose(fast,reference_field(points,source_x,depth),atol=1.7e-10,rtol=0)


def test_single_component_has_flat_fronts_and_is_not_the_spherical_target():
    data=weyl.reconstruction_state(0)
    assert data['count']==1
    phase=np.unwrap(np.unwrap(np.angle(data['field']),axis=0),axis=1)
    assert np.max(abs(np.diff(phase,n=2,axis=0)))<4e-14
    assert np.max(abs(np.diff(phase,n=2,axis=1)))<4e-14
    assert np.max(abs(np.diff(data['grid']['phase'],n=2,axis=1)))>1e-3
    assert not np.allclose(data['field'],data['target'])


def test_full_3d_component_sums_converge_without_moving_the_target():
    states=[weyl.reconstruction_state(step) for step in range(4)]
    assert [d['count'] for d in states]==[1,512,2048,4608]
    assert states[1]['error']>.1
    assert states[2]['error']<3e-5
    assert states[3]['error']<1.2e-10
    for data in states:
        assert_allclose(data['target'],states[0]['target'],atol=0)
        assert data['grid'] is states[0]['grid']
        for name in ('field','target','vectors','coefficients'):
            assert not data[name].flags.writeable
    # The last picture is an actual 3D plane-wave sum, independently agreeing
    # with the Bessel-integrated map; azimuthal ky modes have not been dropped.
    final=states[-1]
    assert np.any(abs(final['vectors'][:,1])>1.)
    assert np.any(final['vectors'][:,2].imag<0.)
    assert_allclose(final['field'],final['grid']['field'],atol=3e-14,rtol=0)


def test_live_controls_update_geometry_and_mathematical_objects():
    baseline=set(_instances)
    panel=weyl.reconstruction_explorer()
    try:
        assert len(panel.controls)==1
        for value in (1,2,3):
            old_image=bytes(panel.image.value);old=panel.data
            panel.controls['step'].value=value
            assert bytes(panel.image.value)!=old_image
            assert not np.allclose(panel.data['field'],old['field'])
            assert_allclose(panel.data['target'],old['target'],atol=0)
            Image.open(BytesIO(panel.image.value)).verify()
        assert '$' not in panel.equation.value and '\\' not in panel.equation.value
        assert '\ufffd' not in panel.equation.value
    finally:
        panel.close();panel.close()
    assert set(_instances)==baseline


def test_plots_use_the_changed_field_and_a_shared_fixed_phase_scale():
    for step in (0,1,3):
        data=weyl.reconstruction_state(step)
        fig=weyl.reconstruction_figure(data)
        try:
            for ax,field in zip(fig.axes[:2],[data['field'],data['target']]):
                mesh=ax.collections[0]
                assert_allclose(mesh.get_array(),np.angle(field),atol=0)
                assert_allclose(mesh.get_clim(),[-np.pi,np.pi],atol=0)
            assert_allclose(fig.axes[0].get_xlim(),fig.axes[1].get_xlim())
            assert_allclose(fig.axes[0].get_ylim(),fig.axes[1].get_ylim())
        finally:fig.clear()


def test_notebook_cell_replaces_own_panel_and_preserves_settings():
    book=nbformat.read(Path(__file__).resolve().parents[2]/
                       '8_2_Rayleigh_Sommerfeld_Propagation.ipynb',4)
    code=next(c.source for c in book.cells if c.id=='8-2-weyl-reconstruction-controls')
    baseline=set(_instances);sibling=widgets.HTML('Other teaching panel');ns={}
    try:
        with patch('IPython.display.display'):
            exec(code,ns)
            old=ns['_weyl_reconstruction_panel'];count=len(_instances)
            old.controls['step'].value=3
            exec(code,ns)
        new=ns['_weyl_reconstruction_panel']
        assert old._closed and len(_instances)==count
        assert sibling.model_id in _instances
        assert {name:c.value for name,c in new.controls.items()}==dict(step=3)
    finally:
        if '_weyl_reconstruction_panel' in ns:ns['_weyl_reconstruction_panel'].close()
        sibling.layout.close();sibling.style.close()
        sibling.close()
    assert set(_instances)==baseline


def test_invalid_initial_state_and_failed_render_release_widgets():
    baseline=set(_instances)
    for step in (True,np.bool_(False),-1,4,1.2,1j,np.nan,[0.]):
        with pytest.raises(ValueError):weyl.reconstruction_explorer(step=step)
        assert set(_instances)==baseline
    for points in ([0,0,-.1],[0,0,1j],[0,np.nan,1],[0,1]):
        with pytest.raises(ValueError):weyl.weyl_field(points)
    with pytest.raises(ValueError):weyl.field_map(True)
    with patch.object(weyl,'reconstruction_figure',side_effect=RuntimeError('render failed')):
        with pytest.raises(RuntimeError):weyl.reconstruction_explorer()
    assert set(_instances)==baseline
