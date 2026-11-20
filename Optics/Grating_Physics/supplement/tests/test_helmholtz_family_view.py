"""Check exact Helmholtz families in the shared scalar-solution explorer."""

import gc
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.scalar_theory_view import ScalarTheoryExplorer


class HelmholtzFamilyViewTests(unittest.TestCase):
    def check_artists(self, lesson):
        scene, profile = lesson.figure.axes[:2]
        self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))
        self.assertEqual(scene.get_aspect(), 1)
        self.assertEqual(scene.images[0].get_clim(), (-2, 2))
        shown = scene.images[0].get_array()
        np.testing.assert_array_equal(np.ma.getmaskarray(shown), lesson.data['mask'])
        self.assertTrue(np.all(np.isfinite(shown.compressed())))
        assert_allclose(shown.compressed(), lesson.data['signed_field'][~lesson.data['mask']])
        assert_allclose(profile.lines[0].get_xdata(), np.linalg.norm(lesson.data['profile_points'], axis=1))
        assert_allclose(profile.lines[0].get_ydata(), abs(lesson.data['profile']), equal_nan=True)
        assert_allclose(profile.get_ylim(), [0, 2.15])

    def close_and_check(self, lesson):
        lesson.close()
        lesson.close()
        self.assertIsNone(lesson.data)
        self.assertIsNone(lesson.figure)
        self.assertEqual(bytes(lesson.picture.value), b'')
        lesson.time.value = .5
        self.assertIsNone(lesson.figure)

    def test_cylindrical_slice_excludes_line_axis_and_preserves_amplitude_in_time(self):
        lesson = ScalarTheoryExplorer('solutions')
        try:
            lesson.solution.value = 'cylindrical'
            lesson.wavelength.value = 1.5
            lesson.amplitude.value = .7
            lesson.phase.value = 45
            points = lesson.data['points']
            assert_allclose(points[..., 1], 0)
            radius = np.hypot(points[..., 0], points[..., 2])
            np.testing.assert_array_equal(lesson.data['mask'], radius < .75)
            self.assertTrue(np.isnan(lesson.data['field'][radius == 0]).all())
            ray = lesson.data['profile_points'][:, 2]
            self.assertTrue(np.isnan(lesson.data['profile'][ray < .75]).all())
            at_wavelength = np.argmin(abs(ray-1.5))
            assert_allclose(abs(lesson.data['profile'][at_wavelength]), .7)
            self.check_artists(lesson)
            field, profile = lesson.data['field'].copy(), lesson.data['profile'].copy()
            lesson.time.value = .25
            assert_allclose(lesson.data['field'], field, equal_nan=True)
            assert_allclose(lesson.data['profile'], profile, equal_nan=True)
            assert_allclose(lesson.data['signed_field'], field.imag, atol=1e-14, equal_nan=True)
            self.check_artists(lesson)
            lesson.reset_button.click()
            self.assertEqual(lesson.solution.value, 'plane')
            self.assertEqual((lesson.wavelength.value, lesson.amplitude.value, lesson.phase.value,
                              lesson.time.value), (2, 1, 0, 0))
            self.assertTrue(all(control.layout.display == 'none'
                                for control in (lesson.transverse, lesson.degree, lesson.evanescent)))
        finally:
            self.close_and_check(lesson)
            del lesson
            gc.collect()

    def test_bessel_profile_is_transverse_and_ratio_changes_its_width(self):
        lesson = ScalarTheoryExplorer('solutions')
        try:
            lesson.solution.value = 'bessel'
            self.assertNotEqual(lesson.transverse.layout.display, 'none')
            self.assertEqual(lesson.degree.layout.display, 'none')
            self.assertEqual(lesson.evanescent.layout.display, 'none')
            ray = lesson.data['profile_points']
            assert_allclose(ray[:, 1:], 0)
            self.assertGreater(ray[-1, 0], 0)
            amplitude = abs(lesson.data['field'])
            # For an ideal Bessel beam, every axial row has the same transverse envelope.
            assert_allclose(amplitude, np.broadcast_to(amplitude[0], amplitude.shape), atol=1e-14)
            at_axis = np.argmin(abs(lesson.data['points'][0, :, 0]))
            assert_allclose(amplitude[:, at_axis], 1)
            profile_before = lesson.data['profile'].copy()
            lesson.transverse.value = .8
            self.assertGreater(np.max(abs(lesson.data['profile']-profile_before)), .4)
            # The first radial zero shrinks in inverse proportion to k_perp.
            def first_sign_change(profile):
                return np.flatnonzero(np.signbit(profile.real[1:]) != np.signbit(profile.real[:-1]))[0]
            before_zero = ray[first_sign_change(profile_before), 0]
            after_zero = ray[first_sign_change(lesson.data['profile']), 0]
            self.assertAlmostEqual(after_zero/before_zero, .6/.8, delta=.015)
            self.check_artists(lesson)
            lesson.amplitude.value = 0
            assert_allclose(lesson.data['field'], 0)
            assert_allclose(lesson.data['profile'], 0)
            self.check_artists(lesson)
            lesson.reset_button.click()
            self.assertEqual(lesson.solution.value, 'plane')
            self.assertEqual(lesson.transverse.value, .6)
            self.assertEqual(lesson.amplitude.value, 1)
        finally:
            self.close_and_check(lesson)
            del lesson
            gc.collect()

    def test_multipole_controls_show_regular_origin_and_angular_parity(self):
        lesson = ScalarTheoryExplorer('solutions')
        try:
            lesson.solution.value = 'multipole'
            self.assertNotEqual(lesson.degree.layout.display, 'none')
            points = lesson.data['points']
            origin = np.linalg.norm(points, axis=-1) == 0
            equator = points[..., 2] == 0
            for degree in (0, 1, 2):
                lesson.degree.value = degree
                field = lesson.data['field']
                self.assertTrue(np.all(np.isfinite(field)))
                assert_allclose(field[origin], 1 if degree == 0 else 0)
                assert_allclose(field[::-1], (-1)**degree*field, atol=1e-14)
                if degree == 1:
                    assert_allclose(field[equator], 0, atol=1e-14)
                self.check_artists(lesson)
            before = lesson.data['field'].copy()
            lesson.phase.value = 60
            assert_allclose(lesson.data['field'], before*np.exp(1j*np.pi/3), atol=1e-14)
            lesson.time.value = .25
            assert_allclose(lesson.data['signed_field'], lesson.data['field'].imag, atol=1e-14)
            self.check_artists(lesson)
            lesson.reset_button.click()
            self.assertEqual((lesson.solution.value, lesson.degree.value, lesson.phase.value,
                              lesson.time.value), ('plane', 1, 0, 0))
        finally:
            self.close_and_check(lesson)
            del lesson
            gc.collect()

    def test_evanescent_slice_stays_in_decaying_halfspace_and_tracks_decay_control(self):
        lesson = ScalarTheoryExplorer('solutions')
        try:
            lesson.solution.value = 'evanescent'
            self.assertNotEqual(lesson.evanescent.layout.display, 'none')
            self.assertEqual(lesson.transverse.layout.display, 'none')
            points = lesson.data['points']
            self.assertTrue(np.all(points[..., 2] >= 0))
            assert_allclose(lesson.figure.axes[0].images[0].get_extent(), [-4, 4, 0, 4])
            depth = lesson.data['profile_points'][:, 2]
            alpha = np.pi*np.sqrt(1.25**2-1)
            assert_allclose(abs(lesson.data['profile']), np.exp(-alpha*depth), atol=1e-14)
            assert_allclose(abs(lesson.data['field'][0]), 1)
            before = abs(lesson.data['profile']).copy()
            lesson.evanescent.value = 1.75
            after = abs(lesson.data['profile'])
            self.assertTrue(np.all(after[depth > 0] < before[depth > 0]))
            self.check_artists(lesson)
            profile = lesson.data['profile'].copy()
            field = lesson.data['field'].copy()
            lesson.time.value = .25
            assert_allclose(lesson.data['profile'], profile)
            assert_allclose(lesson.data['signed_field'], field.imag, atol=1e-14)
            self.check_artists(lesson)
            lesson.reset_button.click()
            self.assertEqual((lesson.solution.value, lesson.evanescent.value, lesson.time.value),
                             ('plane', 1.25, 0))
        finally:
            self.close_and_check(lesson)
            del lesson
            gc.collect()


if __name__ == '__main__':
    unittest.main()
