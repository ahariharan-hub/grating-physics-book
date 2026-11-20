"""Scientific display and widget lifecycle contracts for the Chapter 1 comparison."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from wave_optics.beam_comparison_view import BeamComparison


class BeamComparisonTests(unittest.TestCase):
    def setUp(self):
        self.view = BeamComparison()

    def tearDown(self):
        self.view.close()

    def test_screen_profile_and_spreading_follow_gaussian_power_scaling(self):
        v = self.view
        assert_allclose(v.data['profile'], np.exp(-2*v.x**2/9))
        v.screen.value = 40
        width = v.data['screen_width']
        self.assertGreater(width, 3)
        assert_allclose(v.data['profile'], (3/width)**2*np.exp(-2*v.x**2/width**2))
        # Analytic transverse area integral of the displayed axisymmetric beam.
        assert_allclose(v.data['profile'].max()*np.pi*width**2/2, np.pi*9/2)
        assert_allclose(v.profiles[0].get_ydata(), 1)

    def test_updates_retain_artists_reset_once_and_close_detaches_callbacks(self):
        v = self.view
        ids = tuple(map(id, (*v.images, *v.profiles, *v.cursors)))
        for control in v.controls:
            before = v.updates
            control.value = control.max
            self.assertEqual(v.updates, before+1)
        self.assertTrue(bytes(v.picture.value).startswith(b'\x89PNG'))
        before = v.updates
        v.reset_button.click()
        self.assertEqual(v.updates, before+1)
        assert_allclose([c.value for c in v.controls], [.6, 3, 0])
        self.assertEqual(ids, tuple(map(id, (*v.images, *v.profiles, *v.cursors))))
        self.assertEqual([len(a.images) for a in v.axes[0]], [1, 1])
        before = v.updates
        v.close()
        v.close()
        v.wavelength.value = .8
        self.assertEqual(v.updates, before)

    def test_narrowing_and_longer_wavelength_increase_spreading(self):
        v = self.view
        v.screen.value = 40
        def factor(): return v.data['screen_width']/v.waist.value
        start = factor()
        v.waist.value = 2
        self.assertGreater(factor(), start)
        start = factor()
        v.wavelength.value = .8
        self.assertGreater(factor(), start)

if __name__ == '__main__': unittest.main()
