"""The live lesson controls change the mathematical objects they teach."""
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
from ipywidgets.widgets.widget import _instances

from wave_optics.green_kirchhoff import (
    concept_explorer, reconstruction_explorer, response, SOURCE, stage_figure,
)


def state(stage, **kwargs):
    fig, data = stage_figure(stage, **kwargs)
    fig.clear()
    return data


def test_helmholtz_selected_point_and_wavelength_change_actual_balance():
    a=state(0,y=-.6,wavelength=1.)
    b=state(0,y=.6,wavelength=1.)
    c=state(0,y=-.6,wavelength=.8)
    assert a['selected_radius'] != b['selected_radius']
    assert a['field'] != b['field']
    assert a['field'] != c['field']
    assert a['k'] != c['k']
    for value in (a,b,c):
        assert_allclose(value['laplacian']+value['k']**2*value['field'],0,atol=1e-12)


def test_green_geometry_magnitude_and_arrow_use_a_stable_scale():
    a=state(1,angle=.1,y=-.6)
    b=state(1,angle=2.8,y=.6)
    assert a['separation'] != b['separation']
    assert abs(a['response']) != abs(b['response'])
    assert a['phasor_limit'] == b['phasor_limit']
    for value in (a,b):
        assert_allclose(abs(value['response']),1/(4*np.pi*value['separation']),rtol=1e-13)
        assert abs(value['response']) < value['phasor_limit']


def test_distance_restores_point_response_and_boundary_weight_patterns():
    from wave_optics.propagation_interactive import green, green_normal
    near=state(1,distance=.6,angle=3.)
    far=state(1,distance=3.,angle=3.)
    assert near['distance']!=far['distance']
    assert near['response']!=far['response']
    assert near['phasor_limit']==far['phasor_limit']
    for distance in (.6,1.3,3.):
        t=state(5,distance=distance)
        assert_allclose(t['green_weight'],abs(green(distance,.3-t['weight_eta'])),atol=1e-15)
        assert_allclose(t['normal_weight_profile'],abs(green_normal(distance,.3-t['weight_eta']))/(2*np.pi),atol=1e-15)


def test_main_field_has_more_spatial_oscillations_at_shorter_wavelength():
    short=state(0,wavelength=.7);long=state(0,wavelength=1.5)
    crossings=lambda z:np.count_nonzero(np.diff(np.signbit(z.real)))
    assert crossings(short['profile_field'])>crossings(long['profile_field'])
    assert_allclose(short['profile_laplacian']+short['k']**2*short['profile_field'],0,atol=1e-12)


def test_cell_accounting_pairs_equal_and_opposite_face_terms():
    values=[state(2,accounting=i) for i in range(4)]
    for value in values:
        assert abs(value['face_term'])>0
        assert_allclose(value['face_term']+value['opposite_term'],0,atol=1e-14)
    assert not values[0]['remaining_boundaries']
    assert values[3]['remaining_boundaries']
    assert values[0]['displayed_terms']==(values[0]['face_term'],)
    assert values[1]['displayed_terms']==(values[1]['face_term'],values[1]['opposite_term'])
    assert_allclose(sum(values[1]['displayed_terms']),0,atol=1e-14)
    assert values[2]['displayed_terms']==(0j,)


def test_shrinking_sphere_changes_actual_area_weight_and_complex_terms():
    a=state(3,epsilon=.6)
    b=state(3,epsilon=.04)
    assert_allclose(a['area']/b['area'],(.6/.04)**2)
    assert abs(b['normal_weight'])>abs(a['normal_weight'])
    assert a['inner']['value'] != b['inner']['value']
    assert a['inner']['slope'] != b['inner']['slope']
    for value in (a,b):
        assert_allclose(value['inner']['total'],-value['inner']['reference'],rtol=3e-12)


def test_surface_transformation_changes_contour_without_moving_detector():
    a=state(4,morph=0.,closure=6.)
    b=state(4,morph=.5,closure=6.)
    c=state(4,morph=1.,closure=6.)
    d=state(4,morph=1.,closure=14.)
    for v in (b,c,d): assert_allclose(v['p'],a['p'],atol=0)
    assert a['contour'].shape==b['contour'].shape==c['contour'].shape
    assert not np.allclose(a['contour'],b['contour'])
    assert not np.allclose(b['contour'],c['contour'])
    assert np.max(d['contour'][:,0])>np.max(c['contour'][:,0])
    assert 'closure' not in a or a['closure'] is None
    assert 'closure' not in b or b['closure'] is None
    assert 'closure' in c and c['closure'] is not None


def test_kirchhoff_width_and_point_change_region_and_assumed_integrand():
    # The point stays fixed while moving an aperture edge across it.
    hidden=state(5,width=2.,boundary_eta=.6,reveal=False)
    opening=state(5,width=2.,boundary_eta=.6,reveal=True)
    blocked=state(5,width=.8,boundary_eta=.6,reveal=True)
    moved=state(5,width=2.,boundary_eta=.2,reveal=True)
    assert hidden['prescribed_value'] is None
    assert hidden['prescribed_slope'] is None
    assert hidden['value_term'] is None and hidden['slope_term'] is None
    assert opening['region']=='A' and blocked['region']=='B'
    assert_allclose(opening['q'],blocked['q'],atol=0)
    assert_allclose(opening['prescribed_value'],opening['incident_value'])
    assert_allclose(opening['prescribed_slope'],opening['incident_slope'])
    assert blocked['prescribed_value']==0 and blocked['prescribed_slope']==0
    assert blocked['value_term']==0 and blocked['slope_term']==0
    assert abs(opening['value_term'])>0 and abs(opening['slope_term'])>0
    assert moved['value_term']!=opening['value_term']
    assert moved['slope_term']!=opening['slope_term']
    # Verify the retained derivative is normal to the plane, not a radial slope.
    _,grad=response(opening['q'],SOURCE)
    assert_allclose(opening['incident_slope'],4*np.pi*grad[2],rtol=1e-13)


def test_each_live_panel_owns_only_relevant_controls_and_decodable_plot():
    expected=[{'y','wavelength'},{'distance','y','angle'},{'accounting'},{'epsilon'},
              {'morph','closure'},{'distance','width','boundary_eta','reveal'}]
    baseline=set(_instances)
    panels=[]
    try:
        for stage,keys in enumerate(expected):
            panel=concept_explorer(stage);panels.append(panel)
            assert set(panel.controls)==keys
            Image.open(BytesIO(panel.image.value)).verify()
            assert panel.equation.value
        panels[4].controls['morph'].value=0.
        assert panels[4].controls['closure'].layout.display=='none'
        panels[4].controls['morph'].value=1.
        assert panels[4].controls['closure'].layout.display!='none'
    finally:
        for panel in panels:panel.close();panel.close()
    assert set(_instances)==baseline


def test_separate_reconstruction_controls_change_surface_contributions():
    baseline=set(_instances)
    panel=reconstruction_explorer()
    try:
        assert set(panel.controls)=={'y','radius'}
        before=panel.data
        panel.controls['radius'].value=2.8
        after=panel.data
        # This is the dedicated known-field check, not a hidden conceptual-stage control.
        assert_allclose(before['reference'],after['reference'],atol=0)
        assert not np.allclose(before['rings'],after['rings'])
        assert_allclose(after['total'],after['reference'],rtol=3e-11)
    finally:panel.close()
    assert set(_instances)==baseline


def test_plain_html_readouts_do_not_depend_on_widget_math_rendering():
    import ipywidgets as widgets
    baseline=set(_instances)
    for stage in range(6):
        panel=concept_explorer(stage)
        try:
            assert isinstance(panel.equation,widgets.HTML)
            assert not isinstance(panel.equation,widgets.HTMLMath)
            initial=panel.equation.value
            updates=[('wavelength',.8),('distance',2.),('accounting',1),
                     ('epsilon',.04),('morph',1.),('reveal',True)]
            key,value=updates[stage];panel.controls[key].value=value
            assert panel.equation.value!=initial
            for text in (initial,panel.equation.value):
                assert not any(token in text for token in ('$',r'\quad',r'\nabla',r'\(',r'\)'))
                assert '<b>' in text and '<div' in text
            assert panel.children[2] is panel.image  # numbers follow the plot
        finally:panel.close()
    check=reconstruction_explorer()
    try:
        assert '$' not in check.equation.value and '\\' not in check.equation.value
    finally:check.close()
    assert set(_instances)==baseline


def test_short_main_and_companion_preserve_live_concepts_and_clean_reruns():
    root=Path(__file__).resolve().parents[2]
    notebook=nbformat.read(root/'8_1_Physical_Pictures_Behind_Scalar_Field_Propagation.ipynb',4)
    companion=nbformat.read(root/'supplement/8_1_Boundary_Mathematics_and_History.ipynb',4)
    main_cells=[c for c in notebook.cells if c.cell_type=='code' and 'concept_explorer(' in c.source]
    extra_cells=[c for c in companion.cells if c.cell_type=='code' and 'concept_explorer(' in c.source]
    check_cells=[c for c in companion.cells if c.cell_type=='code' and 'reconstruction_explorer(' in c.source]
    assert len(main_cells)==3 and len(extra_cells)==3 and len(check_cells)==1
    assert not any(c.cell_type=='code' and 'reconstruction_explorer(' in c.source for c in notebook.cells)
    concept_cells=main_cells+extra_cells
    # Exercise actual notebook code twice: replacing one panel must release its tree.
    from wave_optics import green_kirchhoff
    baseline=set(_instances);ns={'green_kirchhoff':green_kirchhoff,'display':lambda *_:None}
    try:
        with patch('IPython.display.display'):
            for cell in concept_cells+check_cells:exec(cell.source,ns)
            first_count=len(set(_instances)-baseline)
            for cell in concept_cells+check_cells:exec(cell.source,ns)
            assert len(set(_instances)-baseline)==first_count
            assert baseline<=set(_instances)
    finally:
        for key,value in list(_instances.items()):
            if key not in baseline and hasattr(value,'controls') and hasattr(value,'close'):
                value.close()
    assert set(_instances)==baseline
