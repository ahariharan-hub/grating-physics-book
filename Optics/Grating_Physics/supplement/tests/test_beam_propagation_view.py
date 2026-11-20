"""Check propagation and linked observation planes in the beam intensity view."""

import gc
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.scalar_theory_view import ScalarTheoryExplorer


class BeamPropagationViewTests(unittest.TestCase):
    def setUp(self):
        self.lesson = ScalarTheoryExplorer('solutions')

    def tearDown(self):
        self.lesson.close()
        del self.lesson
        gc.collect()

    def check_linked_artists(self):
        lesson = self.lesson
        self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))
        self.assertEqual(len(lesson.figure.axes), 4)
        side, cross, profile = lesson.figure.axes[:3]
        self.assertEqual(side.get_aspect(), 'auto')
        self.assertEqual(cross.get_aspect(), 1)
        for axis, key in ((side, 'longitudinal_intensity'),
                          (cross, 'transverse_intensity')):
            self.assertEqual(axis.images[0].get_clim(), (0, 1))
            assert_allclose(axis.images[0].get_array(), lesson.data[key])
            self.assertTrue(np.isfinite(lesson.data[key]).all())
            self.assertGreaterEqual(lesson.data[key].min(), 0)
        assert_allclose(profile.get_ylim(), [0, 1.05])
        assert_allclose(profile.lines[0].get_xdata(), lesson.data['profile_points'][:, 0])
        assert_allclose(profile.lines[0].get_ydata(), lesson.data['profile_intensity'])
        assert_allclose(side.lines[0].get_ydata(), lesson.data['z_selected'])
        assert_allclose(lesson.data['longitudinal_points'][..., 1], 0)
        assert_allclose(lesson.data['transverse_points'][..., 2], lesson.data['z_selected'])
        assert_allclose(lesson.data['profile_points'][:, 2], lesson.data['z_selected'])

    def test_gaussian_spreads_and_conserves_transverse_power(self):
        lesson = self.lesson
        lesson.solution.value = 'gaussian'
        lesson.view.value = 'beam'
        waist = lesson.data['waist']
        rayleigh = lesson.data['rayleigh']
        initial_side = lesson.data['longitudinal_intensity'].copy()
        self.check_linked_artists()

        def transverse_power():
            points = lesson.data['transverse_points']
            return np.trapz(np.trapz(lesson.data['transverse_intensity'],
                                    x=points[0, :, 0], axis=1), x=points[:, 0, 1])

        waist_power = transverse_power()
        assert_allclose(waist_power, np.pi*waist**2/2, rtol=1e-7)
        assert_allclose(lesson.data['profile_intensity'].max(), 1)
        lesson.axial.value = 1
        assert_allclose(lesson.data['z_selected'], rayleigh)
        assert_allclose(lesson.data['width'], np.sqrt(2)*waist)
        assert_allclose(lesson.data['profile_intensity'].max(), .5)
        assert_allclose(lesson.data['transverse_intensity'].max(), .5)
        # This fixed transverse window contains essentially the whole beam at z_R.
        assert_allclose(transverse_power(), waist_power, rtol=6e-5)
        assert_allclose(lesson.data['longitudinal_intensity'], initial_side)
        # The width is defined at 1/e^2 of the local peak, not of the waist peak.
        x = lesson.data['profile_points'][:, 0]
        fraction = np.interp(lesson.data['width'], x, lesson.data['profile_intensity'])/.5
        assert_allclose(fraction, np.exp(-2), rtol=3e-4)
        self.check_linked_artists()
        assert_allclose(lesson.figure.axes[2].lines[1].get_ydata().max(), 1)

        lesson.axial.value = -1
        assert_allclose(lesson.data['z_selected'], -rayleigh)
        assert_allclose(lesson.data['profile_intensity'].max(), .5)
        self.check_linked_artists()

    def test_bessel_rings_persist_with_propagation_and_narrow_with_cone_angle(self):
        lesson = self.lesson
        lesson.solution.value = 'bessel'
        lesson.view.value = 'beam'
        initial_cross = lesson.data['transverse_intensity'].copy()
        initial_profile = lesson.data['profile_intensity'].copy()
        initial_radius = lesson.data['central_radius']
        side = lesson.data['longitudinal_intensity']
        assert_allclose(side, np.broadcast_to(side[0], side.shape), atol=1e-14)
        assert_allclose(initial_cross, initial_cross.T, atol=1e-14)
        assert_allclose(initial_profile.max(), 1)
        x = lesson.data['profile_points'][:, 0]
        first_zero = np.argmin(abs(x-initial_radius))
        self.assertLess(initial_profile[first_zero], 1e-20)
        ring = (x > initial_radius) & (x < 2*initial_radius)
        self.assertGreater(initial_profile[ring].max(), .15)
        self.check_linked_artists()

        lesson.beam_axial.value = 4
        assert_allclose(lesson.data['z_selected'], 4*lesson.wavelength.value)
        assert_allclose(lesson.data['transverse_intensity'], initial_cross, atol=1e-14)
        assert_allclose(lesson.data['profile_intensity'], initial_profile, atol=1e-14)
        self.check_linked_artists()
        lesson.transverse.value = .8
        assert_allclose(lesson.data['central_radius']/initial_radius, .6/.8)
        self.check_linked_artists()

    def test_intensity_has_fixed_scale_and_is_independent_of_phase_and_time(self):
        lesson = self.lesson
        lesson.view.value = 'beam'
        assert_allclose(lesson.data['longitudinal_intensity'], 1)
        assert_allclose(lesson.data['transverse_intensity'], 1)
        lesson.beam_axial.value = -3
        assert_allclose(lesson.data['z_selected'], -3*lesson.wavelength.value)
        self.check_linked_artists()

        lesson.solution.value = 'gaussian'
        initial_side = lesson.data['longitudinal_intensity'].copy()
        initial_cross = lesson.data['transverse_intensity'].copy()
        initial_profile = lesson.data['profile_intensity'].copy()
        lesson.phase.value = 75
        lesson.time.value = .25
        assert_allclose(lesson.data['longitudinal_intensity'], initial_side, atol=1e-14)
        assert_allclose(lesson.data['transverse_intensity'], initial_cross, atol=1e-14)
        assert_allclose(lesson.data['profile_intensity'], initial_profile, atol=1e-14)
        lesson.amplitude.value = .5
        assert_allclose(lesson.data['longitudinal_intensity'], initial_side/4, atol=1e-14)
        assert_allclose(lesson.data['transverse_intensity'], initial_cross/4, atol=1e-14)
        assert_allclose(lesson.data['profile_intensity'], initial_profile/4, atol=1e-14)
        self.check_linked_artists()
        lesson.amplitude.value = 0
        assert_allclose(lesson.data['longitudinal_intensity'], 0)
        assert_allclose(lesson.data['transverse_intensity'], 0)
        assert_allclose(lesson.data['profile_intensity'], 0)
        self.check_linked_artists()

    def test_view_switches_controls_and_resets_without_stale_beam_data(self):
        lesson = self.lesson
        self.assertEqual(lesson.view.value, 'field')
        self.assertEqual(lesson.beam_axial.layout.display, 'none')
        lesson.view.value = 'beam'
        self.assertEqual(lesson.phase.layout.display, 'none')
        self.assertEqual(lesson.time.layout.display, 'none')
        self.assertNotEqual(lesson.beam_axial.layout.display, 'none')
        lesson.solution.value = 'gaussian'
        self.assertNotEqual(lesson.axial.layout.display, 'none')
        self.assertNotEqual(lesson.waist.layout.display, 'none')
        self.assertEqual(lesson.beam_axial.layout.display, 'none')
        lesson.axial.value = 1.5
        lesson.view.value = 'field'
        self.assertNotEqual(lesson.phase.layout.display, 'none')
        self.assertNotEqual(lesson.time.layout.display, 'none')
        self.assertIn('signed_field', lesson.data)
        self.assertNotIn('longitudinal_intensity', lesson.data)

        lesson.view.value = 'beam'
        lesson.solution.value = 'standing'
        self.assertEqual(lesson.view.layout.display, 'none')
        self.assertEqual(lesson.beam_axial.layout.display, 'none')
        self.assertNotEqual(lesson.phase.layout.display, 'none')
        self.assertNotEqual(lesson.time.layout.display, 'none')
        self.assertIn('signed_field', lesson.data)
        lesson.reset_button.click()
        self.assertEqual((lesson.solution.value, lesson.view.value, lesson.axial.value,
                          lesson.beam_axial.value), ('plane', 'field', 0, 0))
        self.assertNotEqual(lesson.view.layout.display, 'none')
        self.assertIn('signed_field', lesson.data)
        lesson.close()
        lesson.close()
        lesson.view.value = 'beam'
        lesson.beam_axial.value = 2
        self.assertIsNone(lesson.figure)
        self.assertIsNone(lesson.data)
        self.assertEqual(bytes(lesson.picture.value), b'')


if __name__ == '__main__':
    unittest.main()
