"""Physical and interactive checks for the introductory propagation lesson."""

import gc
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from wave_optics import scalar_propagation_view as views
from wave_optics.scalar_theory import slit_propagation


class ScalarPropagationLessonTests(unittest.TestCase):
    def setUp(self):
        self.explorers = []

    def tearDown(self):
        for view in self.explorers:
            view.close()
            view.close()
            self.assertIsNone(view.data)
            self.assertIsNone(view.figure)
            self.assertIsNone(view._slit_artists)
            self.assertEqual(len(view._cache), 0)
            for control, default in view._defaults:
                control.value = default  # Disposed observers must not render.
        self.explorers.clear()
        gc.collect()

    def explorer(self, stage):
        view = views.ScalarPropagationExplorer(stage)
        self.explorers.append(view)
        self.assertTrue(bytes(view.picture.value).startswith(b'\x89PNG'))
        return view

    def test_complex_spectrum_reconstructs_slit_and_matches_fourier_integral(self):
        view = self.explorer('spectrum')
        for width in (10, 20, 40):
            view.width.value = width
            data = view.data
            assert_allclose(data['reconstructed'], data['boundary_field'], atol=1e-14)
            fx = data['fx']
            central = abs(fx) < .3
            # Compare the normalized continuous spectrum, including its sign,
            # against the analytic aperture integral, not another FFT.
            assert_allclose(data['spectrum'][central],
                width*np.sinc(width*fx[central]), atol=.002, rtol=0)
            self.assertAlmostEqual(data['spectrum'][len(fx)//2].real, width)
        original = view.data['spectrum']
        view.keep_phase.value = False
        self.assertIs(view.data['spectrum'], original)
        self.assertGreater(view.data['reconstruction_error'], .5)
        self.assertLess(np.max(abs(view.data['reconstructed'].imag)), 1e-13)
        # No clipped peaks in the phase-removal demonstration.
        lower, upper = view.figure.axes[2].get_ylim()
        self.assertGreater(view.data['reconstructed'].real.min(), lower)
        self.assertLess(view.data['reconstructed'].real.max(), upper)
        view.reset_button.click()
        self.assertEqual(view.width.value, 20)
        self.assertTrue(view.keep_phase.value)
        self.assertLess(view.data['reconstruction_error'], 1e-14)

    def test_component_plot_has_phase_grazing_and_exponential_decay(self):
        view = self.explorer('component')
        z = view.data['z_over_wavelength']
        assert_allclose(view.data['transfer'], np.exp(2j*np.pi*.8*z), atol=1e-14)
        view.alpha.value = 1
        assert_allclose(view.data['transfer'], 1, atol=0, rtol=0)
        self.assertIn('Grazing', view.readout.value)
        view.alpha.value = 1.25
        assert_allclose(view.data['transfer'], np.exp(-2*np.pi*.75*z), atol=1e-14)
        self.assertIn('Evanescent', view.readout.value)
        view.reset_button.click()
        self.assertEqual(view.alpha.value, .6)

    def test_observation_controls_reuse_map_and_match_direct_plane(self):
        view = self.explorer('slit')
        original = view.data['field']
        axes = tuple(view.figure.axes)
        with patch.object(views, 'slit_propagation', wraps=slit_propagation) as compute:
            for distance in (0, 20, 200, 1000):
                view.distance.value = distance
                self.assertIs(view.data['field'], original)
                self.assertEqual(tuple(view.figure.axes), axes)
                direct = slit_propagation(distances_um=[distance])
                assert_allclose(view.data['selected_field'], direct['field'][0], atol=2e-11)
                assert_allclose(view._slit_artists[1].get_ydata(), direct['intensity'][0], atol=2e-11)
                assert_allclose(view._slit_artists[0].get_ydata(), [distance, distance])
            compute.assert_not_called()
            # Values set in Python need not fall on a slider's step grid.
            view.distance.value = 17
            self.assertEqual(compute.call_count, 1)
            direct = slit_propagation(distances_um=[17])
            assert_allclose(view.data['selected_field'], direct['field'][0], atol=2e-11)
        view.distance.value = 1000
        original_profile = view.data['selected_intensity'].copy()
        view.wavelength.value = .8
        self.assertIsNot(view.data['field'], original)
        self.assertGreater(np.max(abs(view.data['selected_intensity']-original_profile)), .05)
        view.width.value = 10
        data = view.data
        center = np.argmin(abs(data['x_um']))
        expected = np.sinc(10*data['x_um']/(.8*1000))**2
        assert_allclose(data['selected_intensity']/data['selected_intensity'][center], expected,
                        atol=.01, rtol=0)
        self.assertEqual(view.figure.axes[1].images[0].get_clim(), (0, 2))
        self.assertEqual(view.figure.axes[2].get_ylim(), (0, 2))
        view.distance.value = 0
        assert_allclose(view.data['selected_field'], view.data['boundary_field'], atol=0, rtol=0)
        view.reset_button.click()
        self.assertEqual((view.width.value, view.wavelength.value, view.distance.value), (20, .6, 500))


if __name__ == '__main__':
    unittest.main()
