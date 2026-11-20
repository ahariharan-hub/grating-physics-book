"""Independent physics, FFT convergence and widget lifecycle checks for 7.3A."""
import gc
from pathlib import Path
import json
import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose
from ipywidgets.widgets.widget import _instances
from wave_optics.finite_grating_beams import (
    FiniteGratingConfiguration,FiniteGratingExperiment,intensity_overlap)
from wave_optics.finite_grating_beams_view import FiniteGratingBeamsExplorer


class FiniteBeamPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=FiniteGratingExperiment()
        cls.data=cls.model.selected_fields(depths=np.array([0.,40.,80.]))

    def test_gaussian_boundary_and_binary_mask(self):
        m=self.model
        assert_allclose(abs(m.incident),np.exp(-m.x**2/64))
        self.assertEqual(set(m.mask.real),{0,1})
        local=(m.x+2)%4-2
        assert_allclose(m.mask,abs(local)<1)
        assert_allclose(m.boundary,m.mask*m.incident)
        assert_allclose(m.propagate_spectrum(m.spectrum,[0])[0],m.boundary,atol=5e-16)

    def test_spectral_peaks_and_gaussian_width(self):
        for radius in (4.,8.):
            m=FiniteGratingExperiment(replace(self.model.configuration,radius=radius))
            for order in (0,1,-1):
                window=m.window(order)
                center=m.k_x_i+order*m.G
                peak=m.k_x[window][np.argmax(abs(m.spectrum[window]))]
                self.assertAlmostEqual(peak,center,delta=2*np.pi/(m.size*m.dx))
            # Isolated incident spectrum: spectral-intensity sigma=1/w0.
            weights=abs(np.fft.fft(m.incident))**2
            sigma=np.sqrt(np.sum((m.k_x-m.k_x_i)**2*weights)/sum(weights))
            self.assertAlmostEqual(sigma,1/radius,places=10)

    def test_evanescent_decay_and_plane_wave_limit(self):
        m=self.model
        for index in (20,np.flatnonzero(m.k_x>1.2*m.k)[0]):
            spectrum=np.zeros(m.size,complex);spectrum[index]=m.size
            s=np.array([0.,1.,2.])
            field=m.propagate_spectrum(spectrum,s)
            beta=np.sqrt(complex(m.k*m.k-m.k_x[index]**2))
            expected=np.exp(1j*2*np.pi*index*np.arange(m.size)/m.size)[None,:]*np.exp(1j*beta*s[:,None])
            assert_allclose(field,expected,atol=5e-12)
            if beta.imag>0:self.assertLess(abs(field[-1,0]),abs(field[0,0]))
        self.assertTrue(np.all(m.signed_k_z.real<=0))
        self.assertTrue(np.all(m.signed_k_z.imag<=0))

    def test_beam_centroid_propagates_toward_positive_x_for_positive_order(self):
        m=self.model;d=self.data
        intensity=abs(d['fields'][1])**2
        centers=np.sum(intensity*m.x,axis=1)/np.sum(intensity,axis=1)
        predicted=80*.25/np.sqrt(1-.25**2)
        self.assertAlmostEqual(centers[0],0,places=10)
        self.assertAlmostEqual(centers[-1],predicted,delta=.2)

    def test_filter_disjointness_reconstruction_and_cross_term(self):
        m=self.model;d=self.data
        self.assertFalse(np.any(d['windows'][0]&d['windows'][1]))
        self.assertFalse(np.any(d['windows'][:,~m.propagating]))
        reconstructed=m.propagate_spectrum(d['spectra'].sum(axis=0),d['depth'])
        assert_allclose(reconstructed,d['total'],atol=5e-16)
        assert_allclose(abs(d['total'])**2,abs(d['fields'][0])**2+abs(d['fields'][1])**2+d['cross'],atol=5e-16)
        with self.assertRaises(ValueError):m.selected_fields((0,0),[0])
        with self.assertRaises(ValueError):m.window(2)  # Zero ideal coefficient.

    def test_overlap_bounds_phase_invariance_and_separation(self):
        m=self.model;d=self.data
        self.assertTrue(np.all((d['overlap']>=0)&(d['overlap']<=1)))
        self.assertGreater(d['overlap'][0],.99)
        self.assertLess(d['overlap'][-1],.12)
        assert_allclose(intensity_overlap(d['fields'][0],d['fields'][1]*1j,m.dx),d['overlap'])
        assert_allclose(intensity_overlap(d['fields'][0],d['fields'][0],m.dx),1.)
        self.assertTrue(np.isnan(intensity_overlap(np.zeros(3),np.ones(3),1.)))

    def test_lossless_spectral_power_parseval_and_flux(self):
        m=self.model;d=self.data
        assert_allclose(d['integrated_intensity'],np.repeat(d['integrated_intensity'][:,:1],3,axis=1),atol=3e-15)
        self.assertAlmostEqual(d['full_integrated_intensity'][0],sum(abs(m.boundary)**2)*m.dx,places=12)
        self.assertLessEqual(d['full_integrated_intensity'][-1],d['full_integrated_intensity'][0]+1e-12)
        weights=-m.signed_k_z.real/m.k
        flux=np.sum(weights*abs(np.fft.fft(d['full'],axis=-1))**2,axis=-1)/m.size*m.dx
        assert_allclose(flux,d['diagnostics']['downward_flux'],atol=3e-15)

    def test_resolution_and_padding_convergence_and_edge_safety(self):
        m=self.model
        def sample(configuration):
            fine=FiniteGratingExperiment(configuration)
            d=fine.selected_fields(depths=np.array([0.,80.]))
            return fine,d
        fine,f=sample(replace(m.configuration,samples_per_period=128))
        wide,w=sample(replace(m.configuration,minimum_window=1024.))
        coarse=self.data
        for field in range(2):
            # Compare intensities on physical coordinates; finer cells are offset.
            target=abs(coarse['fields'][field,-1])**2
            higher=np.interp(m.x,fine.x,abs(f['fields'][field,-1])**2)
            padded=np.interp(m.x,wide.x,abs(w['fields'][field,-1])**2)
            roi=abs(m.x)<64
            self.assertLess(np.linalg.norm(higher[roi]-target[roi])/np.linalg.norm(higher[roi]),.01)
            self.assertLess(np.linalg.norm(padded[roi]-target[roi])/np.linalg.norm(padded[roi]),1e-4)
        self.assertLess(coarse['diagnostics']['edge_fraction'],1e-4)
        self.assertAlmostEqual(f['overlap'][-1],coarse['overlap'][-1],delta=.002)
        self.assertAlmostEqual(w['overlap'][-1],coarse['overlap'][-1],delta=.002)

    def test_isolation_warning_and_handoff_metadata(self):
        m=FiniteGratingExperiment(FiniteGratingConfiguration(radius=2,period=6))
        d=m.selected_fields(depths=np.array([0.,20.]))
        self.assertTrue(any('overlap' in warning for warning in d['warnings']))
        far=m.selected_fields(depths=np.array([0.,80.]))
        self.assertTrue(any('padding convergence' in warning for warning in far['warnings']))
        self.assertEqual(d['metadata']['output_half_space'],'z<=0')
        self.assertTrue(np.iscomplexobj(d['fields']))
        self.assertTrue(np.iscomplexobj(d['spectra']))
        self.assertEqual(d['fields'].shape,(2,2,m.size))
        assert_allclose(d['z'],-d['depth'])

    def test_configuration_validation(self):
        for kw in ({'radius':0},{'angle_deg':90},{'samples_per_period':12},{'minimum_window':np.inf},{'wavelength':.01}):
            with self.assertRaises(ValueError):FiniteGratingConfiguration(**kw)
        with self.assertRaises(ValueError):self.model.propagate_spectrum(self.model.spectrum,[-1])


class FiniteBeamWidgetTests(unittest.TestCase):
    def setUp(self):
        self.render=patch('wave_optics.finite_grating_beams_view.figure_png',return_value=b'png')
        self.render.start();self.view=FiniteGratingBeamsExplorer()
    def tearDown(self):
        self.view.close();self.render.stop();gc.collect()

    def test_controls_reset_caching_and_artist_identity(self):
        v=self.view
        ids=tuple(map(id,(*v.images,*v.profiles)))
        original=v.data
        count=v.model_updates
        v.detector.value=40
        self.assertIs(v.data,original);self.assertEqual(v.model_updates,count)
        for control,value in ((v.wavelength,.8),(v.pitch,5.),(v.radius,6.),(v.angle,5.)):
            old=v.updates;control.value=value;self.assertEqual(v.updates,old+1)
        v.pair.value=v.pair.options[-1][1]
        v.reset_button.click()
        self.assertEqual(v.pair.value,(0,1))
        assert_allclose([v.wavelength.value,v.pitch.value,v.radius.value,v.angle.value,v.detector.value],[1,4,8,0,60])
        self.assertEqual(ids,tuple(map(id,(*v.images,*v.profiles))))
        for control in v._inputs:self.assertEqual(control._trait_notifiers['value']['change'].count(v._changed),1)
        before=v.updates;v.close();v.close();v.radius.value=9;self.assertEqual(v.updates,before)

    def test_notebook_reexecution_closes_only_owned_widgets(self):
        root=Path(__file__).resolve().parents[2]
        b=json.loads((root/'Appendix_2_Numerical_Diffraction_of_Finite_Grating_Illumination.ipynb').read_text(encoding='utf-8'))
        builder=''.join(next(c['source'] for c in b['cells'] if c['id']=='n73-explorer'))
        baseline=set(_instances)
        ns={'FiniteGratingBeamsExplorer':FiniteGratingBeamsExplorer,'display':lambda x:None}
        populations=[]
        try:
            for _ in range(3):
                old=ns.get('finite_grating');exec(builder,ns)
                if old:self.assertTrue(old._closed)
                gc.collect();populations.append(len(set(_instances)-baseline))
            self.assertEqual(populations,[populations[0]]*3)
            self.assertFalse(self.view._closed)
        finally:
            ns['finite_grating'].close();gc.collect();self.assertEqual(set(_instances),baseline)
