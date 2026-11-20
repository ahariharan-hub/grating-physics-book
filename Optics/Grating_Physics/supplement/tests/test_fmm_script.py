"""Runnable FMM companion: physical output selection, GUI solve ownership and CLI."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import os
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'Wave_solver_py_scripts/numerical_explorer/A1_1_scalar_fmm_grating.py'
spec = importlib.util.spec_from_file_location('scalar_fmm_playground', SCRIPT)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def test_filtered_profile_is_projection_of_one_solved_field():
    model = app.solve(app.Configuration(M=8))
    x = np.linspace(-.7,.7,71)
    z, C, propagating, full, selected = app.output_profile(model,.05,x)
    assert_allclose(full,model.reconstruct(x,[z])[0],atol=2e-14)
    assert_allclose(selected,C[propagating] @ np.exp(1j*model.kx[propagating,None]*x),atol=2e-14)
    assert np.max(abs(full-selected))>1e-5


def test_apply_depth_reset_and_colorbar_lifecycle():
    config=app.Configuration(M=6,wavelength=.633,thickness=.333)
    model=app.solve(config)
    explorer=app.Explorer(model,.125)
    try:
        population=len(explorer.figure.axes)
        assert len(explorer.axes)==9
        assert len(explorer.propagation_figure.axes)==7
        assert len(explorer.axes[2].containers)==2
        assert explorer.sliders[0].val==.633
        assert explorer.sliders[3].val==.333
        assert explorer.sliders[5].val==.125
        with patch.object(app,'solve',wraps=app.solve) as solve:
            explorer.sliders[1].set_val(1.8)
            assert solve.call_count==0
            explorer.update()
            assert solve.call_count==1
            explorer.sliders[5].set_val(.25)
            explorer.update()
            assert solve.call_count==1
            sampled_x=explorer.axes[6].lines[0].get_xdata()
            expected=explorer.solution.reconstruct(
                sampled_x,[-explorer.solution.configuration.thickness])[0]
            c=explorer.solution.configuration
            expected/=np.exp(2j*np.pi/c.wavelength*c.n_incident*c.thickness)
            assert_allclose(explorer.axes[6].lines[0].get_ydata(),expected.real,atol=2e-13)
            assert_allclose(explorer.axes[7].lines[0].get_ydata(),abs(expected),atol=2e-13)
            assert len(explorer.figure.axes)==population
            assert len(explorer.axes)==9
            assert len(explorer.propagation_figure.axes)==7
            assert len(explorer.axes[2].containers)==2
            explorer.restore()
            assert solve.call_count==2
            assert explorer.solution.configuration==config
            assert explorer.depth==.125
            assert len(explorer.figure.axes)==population
            assert len(explorer.axes)==9
            assert len(explorer.propagation_figure.axes)==7
            assert len(explorer.axes[2].containers)==2
    finally:
        explorer.close();explorer.close();plt.close(explorer.figure)


def test_uniform_limit_and_convergence_cutoff_validation():
    config=app.Configuration(M=3,n_ridge=1.2,n_groove=1.2,n_incident=1.2,n_output=1.2)
    model=app.solve(config)
    assert_allclose(model.reflection,0,atol=2e-13)
    assert_allclose(model.transmission[3],np.exp(2j*np.pi/config.wavelength*1.2*config.thickness),atol=2e-13)
    rows=app.convergence(config,[2,3],model)
    assert max(row[2] for row in rows)<1e-12
    with pytest.raises(ValueError,match='include all'):
        app.convergence(app.Configuration(M=3),[0,3])
    with pytest.raises(ValueError,match='at least one'):
        app.convergence(config,[])


def test_script_runs_outside_repository_without_writing_files(tmp_path):
    result=subprocess.run([sys.executable,'-c',
        "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
        "source=p.read_text().replace('interactive = True', 'interactive = False'); "
        "exec(compile(source,str(p),'exec'),dict(__name__='__main__',__file__=str(p)))",
        str(SCRIPT)],cwd=tmp_path,text=True,capture_output=True,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=90)
    assert result.returncode==0,result.stderr
    assert f'lambda0={app.wavelength:g}' in result.stdout
    assert list(tmp_path.iterdir())==[]


def test_binary_fft_matches_analytic_coefficients_and_evanescent_decay():
    config=app.Configuration(M=8)
    model=app.solve(config)
    x=np.linspace(-config.period,config.period,71)
    opening=1-config.duty_cycle
    # Opening is centered at d/2, whereas the ridge is centered at zero.
    carrier=np.exp(2j*np.pi/config.wavelength*config.n_incident*config.thickness)
    analytic=opening*np.sinc(model.orders*opening)*np.exp(-1j*np.pi*model.orders)*carrier
    expected=analytic*np.exp(1j*model.beta_output*.2)
    coefficients,field,efficiency,error=app.thin_fft_comparison(model,.2,x,samples=4096)
    assert error<2e-6
    assert_allclose(coefficients,expected,atol=2e-6)
    assert_allclose(field,expected@np.exp(1j*model.kx[:,None]*x),atol=1e-5)
    flux=model.beta_output.real/(2*np.pi/config.wavelength*config.n_incident)
    assert_allclose(efficiency,flux*abs(analytic)**2,atol=2e-6)
    near=app.thin_fft_comparison(model,0,x,samples=4096)[0]
    evanescent=model.beta_output.imag>0
    assert np.all(abs(coefficients[evanescent])<=abs(near[evanescent])+1e-15)


def test_binary_mask_remains_an_aperture_even_for_uniform_fmm_material():
    config=app.Configuration(M=3,n_ridge=1.2,n_groove=1.2,n_incident=1.2,n_output=1.2)
    model=app.solve(config)
    x=np.linspace(-1,1,11)
    coefficients,field,efficiency,error=app.thin_fft_comparison(model,.4,x)
    expected=(1-config.duty_cycle)*np.exp(2j*np.pi/config.wavelength*1.2*(config.thickness+.4))
    assert_allclose(coefficients[3],expected,atol=1e-13)
    assert abs(coefficients[2])>.1
    assert sum(efficiency)<1
    assert_allclose(abs(model.transmission[3]),1,atol=1e-13)
    with pytest.raises(ValueError,match='greater than'):
        app.thin_fft_comparison(model,0,x,samples=6)
    unequal=app.solve(app.Configuration(M=3,n_output=1.33))
    with pytest.raises(ValueError,match='equal exterior'):
        app.thin_fft_comparison(unequal,0,x)
    fig,axes=plt.subplots(3,3)
    try:
        app.draw(unequal,.1,fig,axes.ravel(),[])
        assert any('equal exterior' in text.get_text() for text in axes.ravel()[5].texts)
    finally:
        plt.close(fig)


def test_order_directions_use_each_medium_and_exclude_evanescent_angles():
    model=app.solve(app.Configuration(wavelength=.193,period=.6,thickness=.1,
                                    duty_cycle=2/3,n_output=1.33,M=12))
    figure,axes=plt.subplots(1,2)
    try:
        rows=app.draw_order_directions(model,figure,axes)
        for side,m,dx,dz,angle,power in rows:
            index=1. if side=='R' else 1.33
            assert_allclose(dx,m*.193/(index*.6),atol=1e-14)
            assert_allclose(dx*dx+dz*dz,1,atol=1e-14)
            assert dz>0 if side=='R' else dz<0
            assert_allclose(angle,np.degrees(np.arcsin(dx)),atol=1e-12)
        assert max(abs(row[1]) for row in rows if row[0]=='R')==3
        assert max(abs(row[1]) for row in rows if row[0]=='T')==4
        app.draw_order_directions(model,figure,axes)
        assert len(figure.axes)==2
        assert len(axes[1].tables)==1
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_coefficient_panels_compare_exit_complex_amplitudes_by_order():
    model=app.solve(app.Configuration(wavelength=.193,period=.6,thickness=.1,duty_cycle=2/3,M=12))
    figure,axes=plt.subplots(1,2)
    try:
        app.draw_order_coefficients(model,figure,axes)
        selected=np.ones(model.orders.shape,dtype=bool)
        assert_allclose(axes[0].containers[0].markerline.get_xdata(),model.orders[selected])
        assert_allclose(axes[0].containers[0].markerline.get_ydata(),abs(model.transmission[selected]))
        fft=app.thin_fft_comparison(model,0.,[0.])[0][selected]
        assert_allclose(axes[0].containers[1].markerline.get_ydata(),abs(fft))
        reference=app.incident_exit_reference(model.configuration)
        assert_allclose(axes[1].lines[0].get_ydata(),app.wrapped_phase(model.transmission[selected]/reference),equal_nan=True)
        mask_phase=axes[1].lines[1].get_ydata()
        finite=mask_phase[np.isfinite(mask_phase)]
        assert np.all(np.minimum(abs(finite),abs(finite-np.pi))<1e-11)
        assert axes[1].lines[0].get_linestyle()=='None'
        assert axes[1].lines[1].get_linestyle()=='None'
        app.draw_order_coefficients(model,figure,axes)
        assert len(axes[0].containers)==2
        assert len(axes[1].lines)==2
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_plot_limits_change_view_without_filtering_coefficients():
    model=app.solve(app.Configuration(M=8))
    figure,axes=plt.subplots(3,3)
    try:
        with patch.object(app,'order_xlim',(-2.5,2.5)), patch.object(app,'magnitude_ylim',(0,.8)), \
             patch.object(app,'phase_ylim',(-2,2)), patch.object(app,'efficiency_ylim',(0,.7)):
            app.draw(model,.1,figure,axes.ravel(),[])
            for axis in (axes.ravel()[2],axes.ravel()[3],axes.ravel()[4]):
                assert_allclose(axis.get_xlim(),[-2.5,2.5])
            assert_allclose(axes.ravel()[2].get_ylim(),[0,.8])
            assert_allclose(axes.ravel()[3].get_ylim(),[-2,2])
            assert_allclose(axes.ravel()[4].get_ylim(),[0,.7])
            assert len(axes.ravel()[2].containers[0].markerline.get_xdata())==17
    finally:
        plt.close(figure)


def test_field_direction_overlay_starts_at_interfaces_with_outgoing_vectors():
    model=app.solve(app.Configuration(wavelength=.193,period=.6,thickness=.1,duty_cycle=2/3,M=12))
    figure,axis=plt.subplots()
    try:
        artists=app.overlay_field_directions(model,axis,(0,.6),(-.3,.1))
        arrows=artists[:-1:2]
        labels=artists[1:-1:2]
        assert len(arrows)==10  # Three strongest reflected; all seven transmitted.
        for arrow,label in zip(arrows,labels):
            side=label.get_text()[0];m=int(label.get_text()[1:])
            origin=np.array(arrow.xyann);tip=np.array(arrow.xy)
            vector=tip-origin
            assert_allclose(origin,[.3,0 if side=='R' else -.1])
            assert vector[1]>0 if side=='R' else vector[1]<0
            direction=vector/np.linalg.norm(vector)
            assert_allclose(direction[0],m*.193/.6,atol=1e-13)
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_large_propagation_view_matches_outgoing_modal_field_and_reuses_axes():
    model=app.solve(app.Configuration(M=8,n_output=1.33))
    figure,axes=plt.subplots(1,2)
    colorbars=[]
    try:
        data=app.draw_propagation_view(model,figure,axes,colorbars)
        for j in (0,len(data['distance'])//2,-1):
            expected=app.output_profile(model,data['distance'][j],data['x'])[4]
            assert_allclose(data['field'][j],expected,atol=2e-12)
        assert np.all(np.isin(data['orders'],model.orders[model.output_kinds=='propagating']))
        selected_index=int(data['selected_order']+model.configuration.M)
        assert_allclose(abs(data['selected']),abs(model.transmission[selected_index]),atol=1e-13)
        footer_count=len(figure.texts)
        app.draw_propagation_view(model,figure,axes,colorbars)
        assert len(figure.axes)==4
        assert len(figure.texts)==footer_count
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_propagation_player_changes_time_and_order_without_resolving():
    model=app.solve(app.Configuration(wavelength=.193,period=.6,thickness=.1,duty_cycle=2/3,M=8))
    figure,axes=plt.subplots(1,2)
    player=None
    try:
        data=app.draw_propagation_view(model,figure,axes,[])
        player=app.PropagationPlayer(model,figure,axes,data)
        with patch.object(app,'solve',side_effect=AssertionError('No new scattering solve')):
            player.time_slider.set_val(.25)
            assert_allclose(axes[0].images[0].get_array(),data['field'].imag.T,atol=1e-12)
            player.order_slider.set_val(-1)
            assert data['selected_order']==-1
            j=int(np.flatnonzero(model.orders==-1)[0])
            expected=model.transmission[j]*np.exp(1j*(
                data['distance'][:,None]*model.beta_output[j]+model.kx[j]*data['x'][None,:]))
            assert_allclose(axes[1].images[0].get_array(),expected.imag.T/abs(model.transmission[j]),atol=1e-12)
            player.toggle();assert player.running
            player.tick();assert_allclose(player.time_slider.val,.275)
            player.reset();assert not player.running
            assert player.time_slider.val==0
            assert data['selected_order']==app.propagation_order
    finally:
        if player is not None:
            player.close();player.close()
        plt.close(figure)


def test_transmission_profiles_compare_complex_ratios_at_the_same_exit_plane():
    model=app.solve(app.Configuration(wavelength=.193,period=.6,thickness=.1,duty_cycle=2/3,M=8))
    figure,axes=plt.subplots(1,3)
    try:
        data=app.draw_transmission_profiles(model,axes)
        expected=model.reconstruct(data['x'],[-model.configuration.thickness])[0]
        expected/=data['incident_reference']
        assert_allclose(data['effective_transmission'],expected,atol=2e-13)
        assert_allclose(axes[0].lines[0].get_ydata(),expected.real,atol=2e-13)
        assert_allclose(axes[0].lines[1].get_ydata(),expected.imag,atol=2e-13)
        assert_allclose(axes[1].lines[0].get_ydata(),abs(expected),atol=2e-13)
        assert set(np.unique(data['ideal_transmission']))=={0.,1.}
        edge=.5*model.configuration.duty_cycle*model.configuration.period
        assert_allclose(axes[1].lines[1].get_xdata()[1:3],[edge,edge])
        assert_allclose(axes[1].lines[1].get_ydata(),[0,0,1,1,0,0])
        assert np.all(data['ideal_phase'][data['ideal_transmission']>0]==0)
        assert np.all(np.isnan(data['ideal_phase'][data['ideal_transmission']==0]))
        finite=np.isfinite(data['phase'])
        assert_allclose(data['phase'][finite],np.angle(expected[finite]),atol=2e-13)
        app.draw_transmission_profiles(model,axes)
        assert [len(ax.lines) for ax in axes]==[4,2,2]
        uniform=app.solve(app.Configuration(M=3,n_ridge=1.2,n_groove=1.2,n_incident=1.2,n_output=1.2))
        uniform_data=app.draw_transmission_profiles(uniform,axes)
        assert_allclose(uniform_data['effective_transmission'],1,atol=2e-13)
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_order_and_spatial_transmission_use_identical_reference():
    model=app.solve(app.Configuration(wavelength=.193,period=.6,thickness=.105,duty_cycle=2/3,M=12))
    c=model.configuration
    x=np.arange(512)*c.period/512
    effective=model.reconstruct(x,[-c.thickness])[0]/app.incident_exit_reference(c)
    recovered=effective@np.exp(-1j*x[:,None]*model.kx)/x.size
    assert_allclose(recovered,model.transmission/app.incident_exit_reference(c),atol=3e-13)


def test_phase_zeros_and_equivalent_branch_endpoints_are_handled_consistently():
    phase=app.wrapped_phase(np.array([1.,-1.,-1.-1e-15j,0.]))
    assert_allclose(phase[:3],[0,np.pi,np.pi],atol=1e-13)
    assert np.isnan(phase[3])
    crossed=app.wrapped_phase(np.exp(1j*np.array([3.1,-3.1])),break_wraps=True)
    assert np.isnan(crossed[1])
