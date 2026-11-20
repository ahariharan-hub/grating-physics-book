"""Check that the scalar-theory explorers display the controlled physics."""

import gc
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.scalar_theory_view import ScalarTheoryExplorer


class ScalarTheoryViewTests(unittest.TestCase):
    def close_and_check(self, lesson):
        lesson.close()
        lesson.close()
        self.assertIsNone(lesson.figure)
        self.assertIsNone(lesson.data)
        self.assertEqual(bytes(lesson.picture.value), b'')
        # A disposed explorer no longer renders when an old control changes.
        lesson.wavelength.value = lesson.wavelength.min
        self.assertIsNone(lesson.figure)
        self.assertIsNone(lesson.data)

    def test_helmholtz_plot_exposes_trial_mismatch_and_restores_the_solution(self):
        lesson = ScalarTheoryExplorer('helmholtz')
        try:
            self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))
            limits = [(axis.get_xlim(), axis.get_ylim()) for axis in lesson.figure.axes]
            field_axis, residual_axis = lesson.figure.axes
            assert_allclose(field_axis.lines[0].get_ydata(), field_axis.lines[1].get_ydata())
            assert_allclose(residual_axis.lines[1].get_ydata(), 0)
            lesson.phase.value = 30
            lesson.amplitude.value = .6
            lesson.wavelength.value = 1.5
            lesson.ratio.value = 1.25
            field_axis, residual_axis = lesson.figure.axes
            field = lesson.data['field']
            assert_allclose(field_axis.lines[0].get_ydata(), field.real)
            assert_allclose(field_axis.lines[1].get_ydata(), 1.25**2*field.real)
            assert_allclose(residual_axis.lines[1].get_ydata(), (1-1.25**2)*field.real, atol=1e-14)
            self.assertGreater(np.max(abs(lesson.data['residual'])), 1)
            for axis, (xlim, ylim) in zip(lesson.figure.axes, limits):
                assert_allclose(axis.get_xlim(), xlim)
                assert_allclose(axis.get_ylim(), ylim)
            lesson.amplitude.value = 0
            assert_allclose(lesson.data['residual'], 0)
            lesson.reset_button.click()
            self.assertEqual((lesson.ratio.value, lesson.amplitude.value, lesson.phase.value,
                              lesson.wavelength.value), (1, 1, 0, 2))
            assert_allclose(lesson.data['residual'], 0)
        finally:
            self.close_and_check(lesson)
            del lesson
            gc.collect()

    def test_solution_slice_masks_the_source_and_time_changes_only_the_real_field(self):
        lesson = ScalarTheoryExplorer('solutions')
        try:
            self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))
            self.assertEqual(lesson.figure.axes[0].get_aspect(), 1)
            self.assertEqual(lesson.figure.axes[0].images[0].get_clim(), (-2, 2))
            assert_allclose(abs(lesson.data['field']), 1)
            lesson.solution.value = 'standing'
            # The profile includes z=lambda/4=0.5 and hence a fixed node.
            ray_z = lesson.data['profile_points'][:, 2]
            node = np.flatnonzero(np.isclose(ray_z, .5))[0]
            assert_allclose(lesson.data['profile'][node], 0, atol=1e-14)
            lesson.time.value = .25
            assert_allclose(lesson.data['profile'][node], 0, atol=1e-14)
            assert_allclose(lesson.data['signed_field'], 0, atol=1e-14)

            lesson.solution.value = 'spherical'
            lesson.wavelength.value = 1.5
            lesson.amplitude.value = .7
            lesson.phase.value = 45
            lesson.time.value = 0
            points = lesson.data['points']
            self.assertEqual(points.shape[-1], 3)
            assert_allclose(points[..., 1], 0)
            radius = np.linalg.norm(points, axis=-1)
            expected_mask = radius < .75
            np.testing.assert_array_equal(lesson.data['mask'], expected_mask)
            shown = lesson.figure.axes[0].images[0].get_array()
            np.testing.assert_array_equal(np.ma.getmaskarray(shown), expected_mask)
            self.assertTrue(np.all(np.isfinite(shown.compressed())))
            self.assertTrue(np.isnan(lesson.data['field'][radius == 0]).all())
            assert_allclose(abs(lesson.data['field'][~expected_mask]), .7*1.5/radius[~expected_mask])
            ray_z = lesson.data['profile_points'][:, 2]
            self.assertTrue(np.isnan(lesson.data['profile'][ray_z < .75]).all())
            assert_allclose(abs(lesson.data['profile'][ray_z >= .75]), .7*1.5/ray_z[ray_z >= .75])
            assert_allclose(lesson.figure.axes[1].lines[0].get_ydata(), abs(lesson.data['profile']),
                            equal_nan=True)

            field = lesson.data['field'].copy()
            profile = lesson.data['profile'].copy()
            real_before = lesson.data['signed_field'].copy()
            lesson.time.value = .25
            assert_allclose(lesson.data['field'], field, equal_nan=True)
            assert_allclose(lesson.data['profile'], profile, equal_nan=True)
            self.assertGreater(np.max(abs(lesson.data['signed_field'][~expected_mask]
                                            -real_before[~expected_mask])), .5)
            assert_allclose(lesson.figure.axes[0].images[0].get_array().compressed(),
                            field.imag[~expected_mask], atol=1e-14)
            lesson.reset_button.click()
            self.assertEqual(lesson.solution.value, 'plane')
            self.assertEqual((lesson.amplitude.value, lesson.phase.value, lesson.time.value,
                              lesson.wavelength.value), (1, 0, 0, 2))
            self.assertFalse(np.any(lesson.data['mask']))
            assert_allclose(abs(lesson.data['field']), 1)
        finally:
            self.close_and_check(lesson)
            del lesson
            gc.collect()

    def test_all_solution_families_share_controls_and_comparison_scales(self):
        lesson = ScalarTheoryExplorer('solutions')
        try:
            families = ('plane', 'standing', 'spherical', 'cylindrical',
                        'bessel', 'multipole', 'evanescent')
            self.assertEqual(tuple(value for _, value in lesson.solution.options), (*families, 'gaussian'))
            lesson.wavelength.value = 1.5
            lesson.amplitude.value = .6
            lesson.phase.value = 30
            lesson.time.value = .2
            specialized = {'bessel': lesson.transverse, 'multipole': lesson.degree,
                           'evanescent': lesson.evanescent}
            # Returning to plane also checks that the last family's control is hidden.
            for kind in (*families, 'plane'):
                with self.subTest(kind=kind):
                    lesson.solution.value = kind
                    self.assertEqual((lesson.wavelength.value, lesson.amplitude.value,
                                      lesson.phase.value, lesson.time.value), (1.5, .6, 30, .2))
                    for control_kind, control in specialized.items():
                        self.assertEqual(control.layout.display != 'none', control_kind == kind)
                    self.assertEqual(lesson.waist.layout.display, 'none')
                    self.assertEqual(lesson.axial.layout.display, 'none')
                    self.assertEqual(len(lesson.figure.axes), 3)
                    scene, profile = lesson.figure.axes[:2]
                    self.assertEqual(scene.get_aspect(), 1)
                    self.assertEqual(scene.images[0].get_clim(), (-2, 2))
                    assert_allclose(scene.get_xlim(), [-4, 4])
                    assert_allclose(scene.get_ylim(), [0, 4] if kind == 'evanescent' else [-4, 4])
                    assert_allclose(profile.get_xlim(), [0, 6])
                    assert_allclose(profile.get_ylim(), [0, 2.15])
                    assert_allclose(lesson.data['signed_field'],
                                    np.real(lesson.data['field']*np.exp(-2j*np.pi*.2)),
                                    equal_nan=True)
                    assert_allclose(profile.lines[0].get_ydata(), abs(lesson.data['profile']),
                                    equal_nan=True)
            lesson.reset_button.click()
            self.assertEqual(lesson.solution.value, 'plane')
            self.assertEqual((lesson.wavelength.value, lesson.amplitude.value,
                              lesson.phase.value, lesson.time.value), (2, 1, 0, 0))
        finally:
            self.close_and_check(lesson)
            del lesson
            gc.collect()

    def test_legacy_stage_opens_combined_explorer_with_gaussian_spreading(self):
        lesson = ScalarTheoryExplorer('more_solutions')
        try:
            self.assertEqual(lesson.stage, 'solutions')
            self.assertEqual(lesson.solution.value, 'plane')
            lesson.solution.value = 'gaussian'
            self.assertNotEqual(lesson.waist.layout.display, 'none')
            self.assertNotEqual(lesson.axial.layout.display, 'none')
            self.assertTrue(all(control.layout.display == 'none'
                                for control in (lesson.transverse, lesson.evanescent, lesson.degree)))
            self.assertIn('paraxial', lesson.figure.axes[0].get_title())
            waist = lesson.data['waist']
            points = lesson.data['points']
            assert_allclose(points[..., 2], 0)
            assert_allclose(lesson.data['field'], np.exp(-(points[..., 0]**2+points[..., 1]**2)/waist**2))
            radius = lesson.figure.axes[1].lines[0].get_xdata()
            assert_allclose(radius, lesson.data['profile_points'][:, 0]/waist)
            assert_allclose(abs(lesson.data['profile']), np.exp(-radius**2))

            lesson.axial.value = 1
            assert_allclose(lesson.data['z'], lesson.data['rayleigh'])
            assert_allclose(lesson.data['points'][..., 2], lesson.data['z'])
            assert_allclose(lesson.data['width']/waist, np.sqrt(2))
            assert_allclose(abs(lesson.data['profile']), np.exp(-radius**2/2)/np.sqrt(2))
            assert_allclose(lesson.figure.axes[1].lines[0].get_ydata(), abs(lesson.data['profile']))
            assert_allclose(lesson.figure.axes[1].lines[1].get_ydata(), np.exp(-radius**2))
            field, profile = lesson.data['field'].copy(), lesson.data['profile'].copy()
            lesson.time.value = .25
            assert_allclose(lesson.data['field'], field)
            assert_allclose(lesson.data['profile'], profile)
            assert_allclose(lesson.data['signed_field'], field.imag, atol=1e-14)

            rayleigh = lesson.data['rayleigh']
            lesson.waist.value = 4
            assert_allclose(lesson.data['waist'], 2*waist)
            assert_allclose(lesson.data['rayleigh'], 4*rayleigh)
            assert_allclose(abs(lesson.data['profile']), abs(profile))
            lesson.reset_button.click()
            self.assertEqual((lesson.solution.value, lesson.waist.value, lesson.axial.value,
                              lesson.time.value), ('plane', 2, 0, 0))
            self.assertEqual(lesson.waist.layout.display, 'none')
            self.assertEqual(lesson.axial.layout.display, 'none')
        finally:
            self.close_and_check(lesson)
            del lesson
            gc.collect()

    def test_boundary_plot_distinguishes_direction_despite_equal_boundary_values(self):
        lesson = ScalarTheoryExplorer('boundary')
        try:
            self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))
            forward_field = lesson.data['field'].copy()
            forward_history = lesson.data['history'].copy()
            forward_derivative = lesson.data['derivative'][0]
            assert_allclose(lesson.figure.axes[1].images[0].get_array(), forward_history)
            z = lesson.figure.axes[0].lines[0].get_xdata()
            times = lesson.data['times']
            quarter = np.argmin(abs(times-.25))
            forward_crest = np.argmin(abs(z-.5))
            assert_allclose(forward_history[quarter, forward_crest], 1)

            lesson.fraction.value = 1
            assert_allclose(lesson.data['field'][0], forward_field[0])
            assert_allclose(lesson.data['derivative'][0], -forward_derivative)
            assert_allclose(lesson.data['field'], np.conj(forward_field))
            assert_allclose(lesson.data['history'], forward_history[::-1], atol=1e-14)
            self.assertGreater(np.max(abs(lesson.data['history']-forward_history)), 1.9)
            backward_crest = np.argmin(abs(z-1.5))
            assert_allclose(lesson.data['history'][quarter, backward_crest], 1)

            history_before = lesson.data['history'].copy()
            lesson.time.value = .3
            assert_allclose(lesson.data['history'], history_before)
            assert_allclose(lesson.figure.axes[1].lines[0].get_ydata(), [.3, .3])
            assert_allclose(lesson.data['real_field'],
                            np.cos(-np.pi*z-2*np.pi*.3), atol=1e-14)
            lesson.fraction.value = .5
            assert_allclose(lesson.data['field'][0], 1)
            assert_allclose(lesson.data['derivative'][0], 0)
            assert_allclose(lesson.data['history'][quarter], 0, atol=1e-14)
            lesson.reset_button.click()
            self.assertEqual((lesson.fraction.value, lesson.time.value), (0, 0))
            assert_allclose(lesson.data['history'], forward_history)
        finally:
            self.close_and_check(lesson)
            del lesson
            gc.collect()

    def test_diffraction_controls_change_first_zeros_and_preserve_normalization(self):
        lesson = ScalarTheoryExplorer('diffraction')
        try:
            self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))

            def check_curve_and_markers():
                aperture, propagation, screen = lesson.figure.axes[:3]
                x = screen.lines[0].get_xdata()
                intensity = screen.lines[0].get_ydata()
                first_zero = lesson.wavelength.value*1000/lesson.width.value
                assert_allclose(lesson.data['first_zero_mm'], first_zero)
                assert_allclose(intensity, np.sinc(x/first_zero)**2, atol=1e-14)
                assert_allclose(intensity, intensity[::-1], atol=1e-14)
                assert_allclose(intensity[np.argmin(abs(x))], 1)
                assert_allclose(lesson.data['amplitude_scaled'][np.argmin(abs(x))], lesson.width.value)
                markers = [line for line in screen.lines if line.get_marker() == 'o']
                self.assertEqual(len(markers), 2)
                assert_allclose([line.get_xdata()[0] for line in markers], [-first_zero, first_zero])
                assert_allclose([line.get_ydata()[0] for line in markers], [0, 0])
                assert_allclose(aperture.lines[0].get_xdata()[[1, 4]],
                                [-lesson.width.value/2, lesson.width.value/2])

            check_curve_and_markers()
            initial_zero = lesson.data['first_zero_mm']
            initial_curve = lesson.data['relative_intensity'].copy()
            lesson.width.value = 40
            check_curve_and_markers()
            assert_allclose(lesson.data['first_zero_mm'], initial_zero/2)
            lesson.wavelength.value = .8
            check_curve_and_markers()
            assert_allclose(lesson.data['first_zero_mm'], 20)
            lesson.reset_button.click()
            self.assertEqual((lesson.width.value, lesson.wavelength.value), (20, .6))
            assert_allclose(lesson.data['relative_intensity'], initial_curve)
        finally:
            self.close_and_check(lesson)
            del lesson
            gc.collect()

    def test_superposition_adds_signed_fields_before_cycle_averaging(self):
        lesson = ScalarTheoryExplorer('superposition')
        try:
            self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))
            lesson.amplitude.value = 0
            assert_allclose(lesson.data['relative_intensity'], 1)
            lesson.amplitude.value = 1
            lesson.phase.value = 0
            z = lesson.data['z']
            assert_allclose(lesson.data['field'], 2*np.cos(np.pi*z), atol=1e-14)
            intensity = lesson.data['relative_intensity']
            assert_allclose(intensity, 4*np.cos(np.pi*z)**2, atol=1e-14)
            axes = tuple(lesson.figure.axes)
            lesson.time.value = .25
            self.assertIs(lesson.data['relative_intensity'], intensity)
            self.assertEqual(tuple(lesson.figure.axes), axes)
            assert_allclose(lesson._superposition_lines[2].get_ydata(), 0, atol=1e-14)
            lesson.wavelength.value = 3
            lesson.phase.value = 360
            self.assertTrue(np.isfinite(lesson.data['field']).all())
            lesson.reset_button.click()
            self.assertEqual((lesson.wavelength.value, lesson.amplitude.value,
                              lesson.phase.value, lesson.time.value), (2, .5, 60, 0))
        finally:
            self.close_and_check(lesson)

    def test_slit_map_shares_controls_and_retains_fixed_intensity_scale(self):
        from unittest.mock import patch
        from wave_optics import scalar_theory_view

        lesson = ScalarTheoryExplorer('diffraction')
        try:
            near = lesson.data['propagation']
            self.assertEqual(near['field'].shape, (241, 321))
            assert_allclose(near['intensity'], abs(near['field'])**2)
            image = lesson.figure.axes[1].images[0]
            assert_allclose(image.get_array(), near['intensity'])
            self.assertEqual(image.get_clim(), (0, 2))
            with patch.object(scalar_theory_view, 'slit_propagation',
                              wraps=scalar_theory_view.slit_propagation) as propagate:
                lesson._update()
                propagate.assert_not_called()
                self.assertIs(lesson.data['propagation'], near)
                lesson.width.value = 10
                self.assertEqual(propagate.call_count, 1)
            self.assertEqual(lesson.figure.axes[1].images[0].get_clim(), (0, 2))
            self.assertIn('1000 mm', lesson.readout.value)
            self.assertTrue(np.isfinite(lesson.data['propagation']['intensity']).all())
        finally:
            self.close_and_check(lesson)



if __name__ == '__main__':
    unittest.main()
