"""Check that the interactive controls expose the intended spectral physics."""

import gc
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.scalar_propagation_view import ScalarPropagationExplorer


class ScalarPropagationViewTests(unittest.TestCase):
    def dispose(self, lesson):
        lesson.close()
        lesson.close()
        self.assertIsNone(lesson.figure)
        self.assertIsNone(lesson.data)
        self.assertEqual(bytes(lesson.picture.value), b'')
        lesson.wavelength.value = .4
        self.assertIsNone(lesson.figure)

    def test_geometry_controls_cover_propagation_grazing_and_on_axis(self):
        lesson = ScalarPropagationExplorer('geometry')
        try:
            self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))
            self.assertTrue(all(not control.continuous_update for control in lesson._controls))
            self.assertAlmostEqual(float(lesson.data['geometry']['theta_deg']), 30)
            assert_allclose(abs(lesson.data['field']), 1)
            self.assertEqual(lesson.figure.axes[1].images[0].get_clim(), (-1, 1))
            self.assertEqual(lesson.figure.axes[1].get_aspect(), 1)
            lesson.beta.value = 0
            lesson.alpha.value = 1
            self.assertIn('Grazing', lesson.readout.value)
            self.assertAlmostEqual(float(lesson.data['geometry']['theta_deg']), 90)
            # A grazing x-wave does not vary with z.
            assert_allclose(lesson.data['field'][0], lesson.data['field'][-1])
            lesson.alpha.value = 0
            self.assertIn('undefined on axis', lesson.readout.value)
            self.assertTrue(np.isnan(lesson.data['geometry']['azimuth_deg']))
            # An axial wave is constant across each transverse row.
            assert_allclose(lesson.data['field'][:, 0], lesson.data['field'][:, -1])
            lesson.reset_button.click()
            self.assertEqual((lesson.alpha.value, lesson.beta.value, lesson.wavelength.value), (.3, .4, .6))
        finally:
            self.dispose(lesson)
            del lesson
            gc.collect()

    def test_evanescent_map_decays_and_has_no_real_direction_arrow(self):
        lesson = ScalarPropagationExplorer('geometry')
        try:
            lesson.beta.value = 0
            lesson.alpha.value = 1.25
            self.assertIn('Evanescent', lesson.readout.value)
            self.assertIn('No real polar', lesson.readout.value)
            self.assertTrue(np.isnan(lesson.data['geometry']['theta_deg']))
            z = lesson.data['z_over_wavelength']
            assert_allclose(abs(lesson.data['field'][:, 0]), np.exp(-2*np.pi*.75*z), atol=2e-15)
            # Matplotlib annotations carry the direction arrow only for real wavevectors.
            annotations = [text for text in lesson.figure.axes[1].texts
                           if getattr(text, 'arrow_patch', None) is not None]
            self.assertEqual(annotations, [])
            saved = lesson.data['field'].copy()
            old_frequency = float(lesson.data['geometry']['fx'])
            lesson.wavelength.value = .8
            assert_allclose(lesson.data['field'], saved)
            self.assertAlmostEqual(float(lesson.data['geometry']['fx']), old_frequency*.6/.8)
        finally:
            self.dispose(lesson)
            del lesson
            gc.collect()

    def test_transfer_distinguishes_decay_and_accumulated_error_then_resets(self):
        lesson = ScalarPropagationExplorer('transfer')
        try:
            self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))
            self.assertEqual(len(lesson.figure.axes), 3)
            self.assertIn('exact ASM phase minus Fresnel phase', lesson.readout.value)
            s = lesson.data['s']
            assert_allclose(abs(lesson.data['asm'][s <= 1]), 1, atol=2e-15)
            self.assertTrue(np.all(abs(lesson.data['asm'][s > 1]) < 1))
            self.assertTrue(np.all(lesson.data['error']['exact'][(s > 0) & (s <= 1)] < 0))
            self.assertTrue(np.all(np.isnan(lesson.data['error']['exact'][s > 1])))
            original = lesson.data['error']['exact'].copy()
            lesson.distance.value = 4
            assert_allclose(lesson.data['error']['exact'], 2*original, equal_nan=True)
            lesson.distance.value = 0
            assert_allclose(lesson.data['asm'], 1, atol=0, rtol=0)
            assert_allclose(lesson.data['fresnel'], 1, atol=0, rtol=0)
            assert_allclose(lesson.data['error']['exact'][s <= 1], 0, atol=0, rtol=0)
            lesson.wavelength.value = .8
            lesson.reset_button.click()
            self.assertEqual((lesson.distance.value, lesson.wavelength.value), (2, .6))
            assert_allclose(lesson.data['error']['exact'], original, equal_nan=True)
        finally:
            self.dispose(lesson)
            del lesson
            gc.collect()

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalarPropagationExplorer('unknown')


if __name__ == '__main__':
    unittest.main()
