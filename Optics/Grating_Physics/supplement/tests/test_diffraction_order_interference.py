"""Independent order geometry, field identities and Notebook 7.2A lifecycle checks."""
import gc
from pathlib import Path
import json
import unittest
from unittest.mock import patch
from ipywidgets.widgets.widget import _instances

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.diffraction_order_interference_view import (
    DiffractionOrderInterferenceExplorer, selected_vectors, transmitted_orders)
from wave_optics.superposition import two_wave_field


def profile(x, vectors, z=-3., ratio=1., phase_deg=0.):
    points=np.column_stack((x,np.zeros_like(x),np.full_like(x,z)))
    return two_wave_field(points,vectors,amplitude2=ratio,
        phase_difference_deg=phase_deg)['relative_intensity']


class OrderInterferencePhysicsTests(unittest.TestCase):
    def test_initial_example_has_required_vectors_period_and_fringe_locations(self):
        v=selected_vectors(2,2,30,(-1,0))
        assert_allclose(v,[[-np.pi/2,0,-np.sqrt(3)*np.pi/2],
            [np.pi/2,0,-np.sqrt(3)*np.pi/2]],atol=1e-14)
        self.assertAlmostEqual(2*np.pi/abs(v[0,0]-v[1,0]),2)
        assert_allclose(profile(np.array([-2.,-1.,0.,1.,2.]),v),[4,0,4,0,4],atol=1e-14)

    def test_signed_incidence_and_all_allowed_vectors_obey_dispersion(self):
        for wavelength,pitch,angle in ((2,2,30),(2,4,0),(1,5,-45),(3,6,60)):
            with self.subTest(wavelength=wavelength,pitch=pitch,angle=angle):
                labels,v=transmitted_orders(wavelength,pitch,angle)
                k=2*np.pi/wavelength
                assert_allclose(v[:,0],k*np.sin(np.deg2rad(angle))+labels*2*np.pi/pitch)
                assert_allclose(np.linalg.norm(v,axis=1),k,atol=1e-14)
                assert_allclose(v[:,1],0)
                self.assertTrue(np.all(v[:,2]<0))

    def test_conical_kinematics_matches_incident_transverse_momentum(self):
        for angle in (-50,0,35):
            for azimuth in (-150,-45,30,90,180):
                labels,v=transmitted_orders(1,4,angle,azimuth)
                k=2*np.pi
                theta,phi=np.deg2rad([angle,azimuth])
                assert_allclose(v[:,0],k*np.sin(theta)*np.cos(phi)+labels*2*np.pi/4,atol=1e-13)
                assert_allclose(v[:,1],k*np.sin(theta)*np.sin(phi),atol=1e-13)
                assert_allclose(np.linalg.norm(v,axis=1),k,atol=1e-13)
                self.assertTrue(np.all(v[:,2]<0))

    def test_grazing_and_evanescent_orders_cannot_be_selected(self):
        labels,_=transmitted_orders(2,2,0)
        assert_allclose(labels,[0])  # ±1 are grazing, not transmitted into z<0.
        for pair in ((0,1),(-1,0),(0,2),(0,0),(.5,0)):
            with self.subTest(pair=pair),self.assertRaises(ValueError):
                selected_vectors(2,2,0,pair)
        for angle in (90,-90,np.nan):
            with self.assertRaises(ValueError):transmitted_orders(2,2,angle)

    def test_transverse_spacing_is_independent_of_wavelength_and_incidence_for_fixed_pair(self):
        for wavelength in (1.,1.5,2.,2.5):
            for angle in (-20,0,20):
                v=selected_vectors(wavelength,4,angle,(-1,0))
                self.assertAlmostEqual(v[0,0]-v[1,0],-np.pi/2)
                self.assertAlmostEqual(2*np.pi/abs(v[0,0]-v[1,0]),4)

    def test_numerically_measured_noncommensurate_period_matches_pitch_rule(self):
        for pitch,pair in ((3.7,(-1,0)),(4.7,(-1,1)),(5.3,(0,2))):
            v=selected_vectors(1,pitch,17,pair)
            x=np.linspace(-30,30,60001)
            intensity=profile(x,v,z=-2.7,phase_deg=37)
            peaks=np.flatnonzero((intensity[1:-1]>intensity[:-2]) & (intensity[1:-1]>=intensity[2:]))+1
            self.assertGreater(peaks.size,4)
            self.assertAlmostEqual(np.diff(x[peaks]).mean(),pitch/abs(pair[0]-pair[1]),delta=.002)

    def test_symmetric_detector_invariance_and_asymmetric_translation(self):
        x=np.linspace(-8,8,1001)
        symmetric=selected_vectors(2,2,30,(-1,0))
        assert_allclose(profile(x,symmetric,z=-1),profile(x,symmetric,z=-7),atol=3e-14)
        asymmetric=selected_vectors(2,4,0,(0,1))
        delta=asymmetric[0]-asymmetric[1]
        dz=-2.5
        shift=-delta[2]/delta[0]*dz
        self.assertGreater(abs(shift),.1)
        assert_allclose(profile(x+shift,asymmetric,z=-1+dz),profile(x,asymmetric,z=-1),atol=2e-14)

    def test_coefficient_phase_translates_without_changing_period_or_visibility(self):
        v=selected_vectors(2,2,30,(-1,0))
        x=np.linspace(-8,8,1001)
        phase=75.
        shift=-np.deg2rad(phase)/(v[1,0]-v[0,0])
        assert_allclose(profile(x+shift,v,phase_deg=phase),profile(x,v),atol=2e-14)

    def test_numerical_visibility_matches_amplitude_imbalance(self):
        v=selected_vectors(2,2,30,(-1,0))
        for ratio in (0,.2,.5,1,2):
            intensity=profile(np.linspace(-1,1,4097),v,ratio=ratio)
            measured=(intensity.max()-intensity.min())/(intensity.max()+intensity.min())
            self.assertAlmostEqual(measured,2*ratio/(1+ratio**2),places=13)


    def test_wavefront_vertices_obey_crest_phase_and_clip_only_outgoing_box(self):
        from wave_optics.diffraction_order_detector_view import clipped_order_crests
        for v in (np.array([0.,0.,-3.]),np.array([2.,0.,-2.]),np.array([-3.,0.,-1e-8])):
            for phase in (0.,-.7,np.pi):
                polygons=clipped_order_crests(v,phase)
                self.assertTrue(polygons)
                for polygon in polygons:
                    phases=polygon@v+phase
                    assert_allclose(phases,np.full(len(polygon),phases[0]),atol=1e-10)
                    self.assertAlmostEqual(phases[0]/(2*np.pi),round(phases[0]/(2*np.pi)),places=10)
                    self.assertTrue(np.all(abs(polygon[:,0])<=8+1e-10))
                    self.assertTrue(np.all(abs(polygon[:,1])<=8+1e-10))
                    self.assertTrue(np.all((polygon[:,2]>=-8-1e-10)&(polygon[:,2]<=1e-10)))


class OrderInterferenceViewTests(unittest.TestCase):
    def setUp(self):
        self.view=DiffractionOrderInterferenceExplorer()

    def tearDown(self):
        self.view.close()
        gc.collect()

    def test_conical_orders_share_ky_and_detector_translation_preserves_phase(self):
        v=self.view
        ids=tuple(map(id,v.detector_view.figure.data))
        v.azimuth.value=30
        vectors=v.data['vectors']
        k=2*np.pi/v.wavelength.value
        assert_allclose(vectors[:,1]/k,[.25,.25])
        assert_allclose(np.linalg.norm(vectors,axis=1),k)
        assert_allclose(v.data['intensity'],np.broadcast_to(v.data['profile'],v.data['intensity'].shape),atol=1e-13)
        delta=v.data['delta']
        normal_period=2*np.pi/np.linalg.norm(delta[:2])
        self.assertAlmostEqual(v.data['normal_period_um'],normal_period)
        points=np.array([[.13,.27,-3.],[.7,-.3,-3.]])
        tangent=np.array([-delta[1],delta[0],0.])
        initial=two_wave_field(points,vectors)['relative_intensity']
        assert_allclose(two_wave_field(points+tangent,vectors)['relative_intensity'],initial,atol=1e-13)
        dz=-2.
        shift=-delta[2]*dz*delta[:2]/np.dot(delta[:2],delta[:2])
        assert_allclose(two_wave_field(points+np.r_[shift,dz],vectors)['relative_intensity'],initial,atol=1e-13)
        live=v.detector_view
        for suffix in ('m','n'):
            self.assertTrue(live.roles['ky-'+suffix].visible)
        self.assertFalse(live.roles['delta-y'].visible)
        incoming=live.roles['incident']
        points=np.array([incoming.x,incoming.y,incoming.z]).T
        direction=(points[1]-points[0])/2
        assert_allclose(direction,[np.sqrt(3)/4,.25,-np.sqrt(3)/2],atol=1e-13)
        assert_allclose(vectors[:,0],k*direction[0]+np.array(v.pair.value)*2*np.pi/v.pitch.value)
        for index,suffix in enumerate(('m','n')):
            mesh=live.roles['crests-'+suffix]
            vertices=np.array([mesh.x,mesh.y,mesh.z]).T
            cycles=vertices@vectors[index]/(2*np.pi)
            assert_allclose(cycles,np.rint(cycles),atol=1e-10)
        self.assertEqual(ids,tuple(map(id,live.figure.data)))
        self.assertEqual(live.x.size,live.y.size)
        assert_allclose(live.x,live.y)
        border=live.roles['border']
        self.assertAlmostEqual(np.ptp(border.x),np.ptp(border.y))
        grating_outline=next(t for t in live.figure.data
            if t.type=='scatter3d' and len(t.x)==5
            and np.all(np.asarray(t.z)==0))
        assert_allclose(np.ptp(grating_outline.x),16.)
        assert_allclose(np.ptp(grating_outline.y),16.)
        v.reset()
        self.assertEqual(v.azimuth.value,0)

    def test_conical_azimuth_excludes_nonpropagating_pairs_and_recovers(self):
        v=self.view
        v.azimuth.value=90
        self.assertIsNone(v.data)
        self.assertTrue(np.isnan(v.detector_view.intensity).all())
        v.reset()
        self.assertIsNotNone(v.data)
        for azimuth in (np.nan,np.inf):
            with self.assertRaises(ValueError):
                selected_vectors(2,2,30,(-1,0),azimuth)

    def check_algebra(self):
        v=self.view
        roles=v.detector_view.roles
        m,n=v.data['vectors']
        zero=np.zeros(3)
        paths={'k-m':(zero,m),'k-n':(zero,n),
            'kx-m':(zero,[m[0],0,0]),'kz-m':([m[0],0,0],m),
            'kx-n':(zero,[n[0],0,0]),'kz-n':([n[0],0,0],n),
            'delta':(m,n),'delta-x':(m,[n[0],0,m[2]]),
            'delta-z':([n[0],0,m[2]],n)}
        for role,(start,end) in paths.items():
            trace=roles[role]
            if np.linalg.norm(np.asarray(end)-start)>1e-12:
                self.assertTrue(trace.visible)
                assert_allclose(np.array([trace.x,trace.y,trace.z]).T,[start,end])
            else:self.assertFalse(trace.visible)
        assert_allclose(v.data['delta'],n-m)
        self.assertAlmostEqual(v.data['period_um'],v.pitch.value/abs(v.pair.value[0]-v.pair.value[1]))

    def test_direction_arrows_and_right_angle_markers_match_wavefront_normals(self):
        v=self.view
        for phase in (0,90,-180):
            v.phase.value=phase
            for index,suffix in enumerate(('m','n')):
                vector=v.data['vectors'][index]
                roles=v.detector_view.roles
                arrow=roles['order-'+suffix]
                endpoints=np.array([arrow.x,arrow.y,arrow.z]).T
                normal=endpoints[1]-endpoints[0]
                normal/=np.linalg.norm(normal)
                assert_allclose(normal,vector/np.linalg.norm(vector),atol=1e-13)
                cue=roles['normal-cue-'+suffix]
                points=np.array([cue.x,cue.y,cue.z]).T
                tangent=points[1]-points[0]
                self.assertAlmostEqual(np.dot(normal,tangent),0,places=13)
                self.assertAlmostEqual(np.dot(points[1]-points[0],points[2]-points[1]),0,places=13)
                coefficient_phase=0 if index==0 else np.deg2rad(phase)
                crest_phase=points[3]@vector+coefficient_phase
                self.assertAlmostEqual(crest_phase/(2*np.pi),round(crest_phase/(2*np.pi)),places=12)

    def test_wavefronts_follow_coefficient_phase_not_detector_position(self):
        v=self.view
        live=v.detector_view
        def vertices(role):
            mesh=live.roles[role]
            return np.array([mesh.x,mesh.y,mesh.z]).T.copy()
        m=vertices('crests-m');n=vertices('crests-n')
        identities=tuple(map(id,live.figure.data))
        v.detector.value=-6
        assert_allclose(vertices('crests-m'),m)
        assert_allclose(vertices('crests-n'),n)
        v.phase.value=90
        assert_allclose(vertices('crests-m'),m)
        moved=vertices('crests-n')
        phases=moved@v.data['vectors'][1]+np.pi/2
        assert_allclose(phases/(2*np.pi),np.rint(phases/(2*np.pi)),atol=1e-10)
        self.assertEqual(identities,tuple(map(id,live.figure.data)))
        v.ratio.value=0
        self.assertFalse(live.roles['crests-n'].visible)
        v.angle.value=0
        self.assertFalse(live.roles['crests-m'].visible)
        v.reset()
        self.assertTrue(live.roles['crests-n'].visible)

    def test_phase_control_is_second_minus_first_and_matches_arriving_correlation(self):
        v=self.view
        v.pitch.value=4;v.angle.value=0;v.pair.value=(0,1)
        v.ratio.value=.7;v.phase.value=90;v.detector.value=-4
        coefficients=v.data['coefficients']
        self.assertAlmostEqual(np.angle(coefficients[1]*coefficients[0].conjugate()),np.pi/2)
        points=np.stack((v.detector_view.xx,v.detector_view.yy,
            np.full_like(v.detector_view.xx,-4.)),axis=-1)
        fields=[c*np.exp(1j*(points@k)) for c,k in zip(coefficients,v.data['vectors'])]
        gamma=fields[0].conjugate()*fields[1]/.7
        expected=1+.7**2+2*.7*gamma.real
        assert_allclose(v.data['intensity'],expected,atol=2e-14)
        assert_allclose(v.data['delta'],v.data['vectors'][1]-v.data['vectors'][0])

    def test_3d_algebra_signed_components_and_difference(self):
        self.check_algebra()
        v=self.view
        v.pitch.value=4;v.angle.value=0;v.pair.value=(0,1)
        self.check_algebra()
        v.angle.value=-20
        self.check_algebra()
        v.wavelength.value=1
        self.check_algebra()

    def test_xy_detector_matches_scalar_profile_and_ruler_maxima(self):
        v=self.view
        v.pitch.value=4;v.angle.value=0;v.pair.value=(0,1)
        v.ratio.value=.5;v.phase.value=90;v.detector.value=-5
        live=v.detector_view
        expected=profile(live.x,v.data['vectors'],z=-5,ratio=.5,phase_deg=90)
        assert_allclose(live.intensity,np.broadcast_to(expected,live.intensity.shape),atol=1e-13)
        assert_allclose(live.roles['detector'].z,-5)
        for x in (live.roles['ruler'].x[0],live.roles['ruler'].x[-1]):
            self.assertAlmostEqual(profile(np.array([x]),v.data['vectors'],z=-5,ratio=.5,phase_deg=90)[0],2.25)

    def test_controls_retain_traces_camera_and_one_update(self):
        v=self.view
        ids=tuple(map(id,v.detector_view.figure.data))
        v.detector_view.figure.layout.scene2.camera.eye.x=2.8
        for control,value in ((v.pitch,4),(v.wavelength,1.5),(v.angle,-30),
                (v.ratio,.5),(v.phase,90),(v.detector,-5)):
            count=v.updates
            control.value=value
            self.assertEqual(v.updates,count+1)
            self.assertEqual(ids,tuple(map(id,v.detector_view.figure.data)))
            self.assertEqual(v.detector_view.figure.layout.scene2.camera.eye.x,2.8)
        count=v.geometry_evaluations
        v.phase.value=45;v.ratio.value=1;v.detector.value=-3
        self.assertEqual(v.geometry_evaluations,count)
        self.assertFalse(hasattr(v,'images'))
        self.assertFalse(hasattr(v,'views'))

    def test_no_pair_zero_amplitude_and_reset(self):
        v=self.view
        v.ratio.value=0
        self.assertFalse(v.detector_view.roles['ruler'].visible)
        v.angle.value=0
        self.assertIsNone(v.data)
        self.assertTrue(np.isnan(v.detector_view.intensity).all())
        for role in ('k-m','k-n','delta','delta-x','delta-z','order-m'):
            self.assertFalse(v.detector_view.roles[role].visible)
        before=v.updates
        v.reset_button.click()
        self.assertEqual(v.updates,before+1)
        self.assertTrue(np.isfinite(v.detector_view.intensity).all())
        self.check_algebra()

    def test_readout_matches_asymmetric_example(self):
        v=self.view
        v.pitch.value=6;v.wavelength.value=1;v.angle.value=0;v.pair.value=(0,5);v.ratio.value=1.3
        self.check_algebra()
        for text in ('1.200 µm','0.967','-6.283','+5.236','-3.473','+2.810','<table','<details'):
            self.assertIn(text,v.readout.value)

    def test_close_is_idempotent_and_detaches_callbacks(self):
        v=self.view
        before=v.updates
        v.close();v.close()
        for control in v._controls:
            self.assertNotIn(v._changed,control._trait_notifiers.get('value',{}).get('change',[]))
        v.phase.value=90
        self.assertEqual(v.updates,before)

    def test_new_notebook_replaces_only_its_owned_explorer(self):
        root=Path(__file__).resolve().parents[2]
        b=json.loads((root/'7_2_Interference_of_Diffraction_Orders_from_a_Grating.ipynb').read_text(encoding='utf-8'))
        builder=''.join(next(c['source'] for c in b['cells'] if c['id']=='n72-explorer'))
        namespace={'DiffractionOrderInterferenceExplorer':DiffractionOrderInterferenceExplorer,'display':lambda ui:None}
        baseline=set(_instances)
        populations=[]
        try:
            for _ in range(3):
                old=namespace.get('order_interference')
                exec(builder,namespace)
                if old is not None:self.assertTrue(old._closed)
                new=namespace['order_interference']
                for control in new._controls:
                    self.assertEqual(control._trait_notifiers['value']['change'].count(new._changed),1)
                gc.collect()
                populations.append(len(set(_instances)-baseline))
            self.assertEqual(populations,[populations[0]]*3)
            self.assertFalse(self.view._closed)
        finally:
            namespace['order_interference'].close()
            gc.collect()
            self.assertEqual(set(_instances),baseline)


if __name__=='__main__':unittest.main()
