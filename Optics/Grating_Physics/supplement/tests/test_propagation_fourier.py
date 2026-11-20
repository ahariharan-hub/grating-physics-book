"""Section 6: angular-coordinate conversion, physical branches and live panels."""
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import pytest
from ipywidgets.widgets.widget import _instances
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.testing import assert_allclose
from PIL import Image

from wave_optics import propagation_fourier as fourier
from wave_optics.propagation_kernels import kernel_values


def test_angular_frequency_units_and_outgoing_three_dimensional_branches():
    k = 2*np.pi/.8
    kx = k*np.array([0., .3, 1.2]); ky = .4*k
    d = fourier.multiplier_values(kx, ky, distance=.7, wavelength=.8)
    assert_allclose(d['kappa']**2+kx**2+ky**2, k*k, atol=2e-14)
    assert np.all(d['kappa'].real >= 0) and np.all(d['kappa'].imag >= 0)
    assert_allclose(abs(d['H'][0,:2]), 1., atol=2e-16)
    assert_allclose(abs(d['H'][0,2]), np.exp(-.7*np.sqrt(kx[2]**2+ky**2-k*k)))
    edge = fourier.multiplier_values([0.,2*np.pi,4*np.pi], distance=1.)
    assert_allclose(edge['H'][0], [1.,1.,np.exp(-2*np.pi*np.sqrt(3))], atol=3e-16)
    assert np.isnan(edge['phase_error'][-1])


def test_identity_composition_and_length_unit_scaling():
    kx = np.array([.1,2.,8.]); ky = -.7
    a = fourier.multiplier_values(kx,ky,distance=1.2)
    b = fourier.multiplier_values(kx,ky,distance=2.1)
    c = fourier.multiplier_values(kx,ky,distance=3.3)
    assert_allclose(a['H']*b['H'],c['H'],atol=5e-15)
    assert_allclose(fourier.multiplier_values(kx,ky,distance=0.)['H'],1.)
    t = 3.7
    scaled = fourier.multiplier_values(kx/t,ky/t,distance=3.3*t,wavelength=t)
    assert_allclose(scaled['H'],c['H'],atol=6e-15)
    assert_allclose(c['reduced'],c['H']*np.exp(-1j*2*np.pi*3.3),atol=2e-16)


def test_phase_error_sign_retains_whole_turns_and_paraxial_limit():
    d = fourier.multiplier_values(2*np.pi*np.array([0.,.01,.6,1.]),distance=100.)
    assert d['phase_error'][0] == 0
    assert d['phase_error'][-1] > 40*2*np.pi
    assert np.all(d['phase_error'] >= 0)
    assert_allclose(np.exp(1j*d['phase_error']),d['H'][1]/d['H'][0],atol=1e-13)
    assert_allclose(d['phase_error'][1],2*np.pi*100*.01**4/8,rtol=6e-5)


def test_fraunhofer_coordinate_selects_the_existing_kernel_fourier_weight():
    x,y,s,lam = 2.,-.7,30.,.8
    xi,eta = np.array([-.8,0.,1.4]),np.array([.2,-.3,.5])
    frequency = fourier.far_field_coordinate(x,y,distance=s,wavelength=lam)
    weights = kernel_values(x,xi,y=y,eta=eta,distance=s,wavelength=lam)['kernels'][2]
    central = kernel_values(x,0.,y=y,eta=0.,distance=s,wavelength=lam)['kernels'][2]
    assert_allclose(weights/central,np.exp(-1j*(frequency[0]*xi+frequency[1]*eta)),atol=2e-15)
    # Fixed direction selects the same coordinate at every observation distance.
    assert_allclose(fourier.far_field_coordinate(2*x,2*y,distance=2*s,wavelength=lam),frequency)


def test_live_controls_change_the_selected_mathematical_objects():
    baseline = set(_instances)
    panels = []
    try:
        p = fourier.multiplier_explorer(); panels.append(p)
        for name,value in [('component',.7),('distance',12.),('component',1.15)]:
            previous = bytes(p.image.value); weights = p.data['selected']['H'].copy()
            p.controls[name].value = value
            assert bytes(p.image.value) != previous
            assert not np.allclose(p.data['selected']['H'],weights)
            Image.open(BytesIO(p.image.value)).verify()
        assert 'evanescent' in p.equation.value
        assert abs(p.data['selected']['H'][0]) < 1e-10
        assert_allclose(abs(p.data['selected']['H'][1]),1.,atol=2e-16)
        assert '$' not in p.equation.value and '\\' not in p.equation.value
        m = fourier.mapping_explorer(); panels.append(m)
        for direction in (-.12,0.,.12):
            previous = bytes(m.image.value)
            m.controls['direction'].value = direction
            assert bytes(m.image.value) != previous
            assert_allclose(m.data['observation']/m.data['distance'],direction,atol=1e-16)
            assert_allclose(m.data['frequency'],[direction,0.],atol=1e-16)
        assert not getattr(p,'_closed',False)
    finally:
        for p in panels:p.close(); p.close()
    assert set(_instances) == baseline


def test_displayed_curves_and_arrows_keep_their_actual_values_and_scales():
    for s,u in [(1.,-.5),(100.,1.25)]:
        data = fourier.multiplier_state(s,u)
        fig = fourier.multiplier_figure(data)
        try:
            FigureCanvasAgg(fig).draw()
            phasor,magnitude,error = fig.axes[1:]
            assert_allclose(phasor.get_xlim(),[-1.15,1.15])
            assert_allclose(phasor.get_ylim(),[-1.15,1.15])
            for arrow,weight in zip(phasor.texts,data['selected']['reduced']):
                assert_allclose(arrow.xy,[weight.real,weight.imag])
            assert_allclose(magnitude.get_ylim(),[0.,1.12])
            assert_allclose(magnitude.lines[0].get_ydata(),abs(data['profile']['H'][0]))
            assert_allclose(magnitude.lines[2].get_ydata(),abs(data['profile']['H'][1]))
            assert_allclose(error.lines[0].get_ydata(),data['profile']['phase_error'])
        finally:fig.clear()
    for u in (-.12,.12):
        fig = fourier.mapping_figure(fourier.mapping_state(u))
        try:
            FigureCanvasAgg(fig).draw()
            assert_allclose(fig.axes[1].lines[2].get_xdata(),[u])
        finally:fig.clear()


def test_notebook_cells_replace_only_their_own_panels_and_retain_controls():
    book = nbformat.read(Path(__file__).resolve().parents[2]/'8_3_ASM_Fresnel_and_Fraunhofer_Propagation.ipynb',4)
    cells = [c for c in book.cells if c.cell_type=='code' and
             ('propagation_fourier.multiplier_explorer(' in c.source or
              'propagation_fourier.mapping_explorer(' in c.source)]
    assert len(cells) == 2
    names = ['_fourier_multiplier_panel','_fourier_mapping_panel']
    ns = {}; baseline = set(_instances)
    try:
        with patch('IPython.display.display'):
            for c in cells:exec(c.source,ns)
            count = len(_instances)
            ns[names[0]].controls['component'].value = .65
            ns[names[1]].controls['direction'].value = -.09
            for cell,name in zip(cells,names):
                old = ns[name]; exec(cell.source,ns)
                assert old._closed and len(_instances)==count
                assert all(not getattr(ns[n],'_closed',False) for n in names)
            assert ns[names[0]].controls['component'].value == .65
            assert ns[names[1]].controls['direction'].value == -.09
    finally:
        for n in names:
            if n in ns:ns[n].close()
    assert set(_instances)==baseline


def test_invalid_inputs_and_failed_initial_render_release_widgets():
    for args in [dict(kx=True),dict(kx=1j),dict(kx=np.nan),dict(kx=0.,distance=-1.),
                 dict(kx=0.,wavelength=0.)]:
        with pytest.raises(ValueError):fourier.multiplier_values(**args)
    for function,args in [(fourier.multiplier_state,dict(component=1.3)),
                          (fourier.multiplier_state,dict(distance=.5)),
                          (fourier.mapping_state,dict(direction=.2)),
                          (fourier.far_field_coordinate,dict(x=0.,distance=0.))]:
        with pytest.raises(ValueError):function(**args)
    baseline = set(_instances)
    for constructor,renderer in [(fourier.multiplier_explorer,'multiplier_figure'),
                                 (fourier.mapping_explorer,'mapping_figure')]:
        with patch.object(fourier,renderer,side_effect=RuntimeError('render failed')):
            with pytest.raises(RuntimeError):constructor()
        assert set(_instances)==baseline
