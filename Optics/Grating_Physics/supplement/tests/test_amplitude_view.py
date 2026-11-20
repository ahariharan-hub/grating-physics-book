"""Physical identities and notebook-six view lifecycle."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.amplitude_view import (
    AmplitudeIntensityExplorer, STAGES, cycle_samples, example_order_vector,
)
from wave_optics.waves import field_from_phasor, plane_wave_phasor


class AmplitudeLessonTests(unittest.TestCase):
    def test_cycle_average_and_phasor_agree_for_arbitrary_phase(self):
        for amplitude in (0, .5, 1, 2):
            for phase in (-147, 0, 90, 180):
                data = cycle_samples(amplitude, phase, sample_count=397)
                assert_allclose(data['mean_field'], 0, atol=1e-14)
                assert_allclose(data['mean_square'], amplitude**2/2, atol=1e-14)
                assert_allclose(data['relative_intensity'], 2*data['mean_square'], atol=1e-14)
                assert_allclose(data['field'], amplitude*np.cos(
                    np.deg2rad(phase)-2*np.pi*data['tau']), atol=1e-14)

    def test_quarter_cycle_of_imaginary_phasor_is_not_zero_amplitude(self):
        data = cycle_samples(2, 90)
        assert_allclose(data['phasor'], 2j, atol=1e-14)
        assert_allclose(field_from_phasor(data['phasor'], time=.25, omega=2*np.pi), 2)
        self.assertEqual(data['relative_intensity'], 4)

    def test_assigned_strength_preserves_grating_phase_matching(self):
        kv = example_order_vector()
        assert_allclose(kv, [np.pi/2, 0, -np.sqrt(3)*np.pi/2])
        self.assertLess(kv[2], 0)
        assert_allclose(np.dot(kv, kv), np.pi**2)
        points = np.array([[.3, 0, -.7], [4.3, 0, -.7], [1.2, 0, -1.4]])
        base = plane_wave_phasor(points, kv, amplitude=.5, phase=.47)
        doubled = plane_wave_phasor(points, kv, amplitude=1, phase=.47)
        assert_allclose(doubled, 2*base)
        assert_allclose(abs(doubled)**2, 4*abs(base)**2)
        assert_allclose(base[1]/base[0], 1, atol=1e-14)
        assert_allclose(abs(base), .5)

    def test_all_views_update_keep_scales_and_release_widgets(self):
        for stage in STAGES:
            with self.subTest(stage=stage):
                lesson = AmplitudeIntensityExplorer(stage)
                initial_limits = [(axis.get_xlim(), axis.get_ylim()) for axis in lesson.figure.axes]
                before = bytes(lesson.picture.value)
                try:
                    lesson.amplitude.value = 2
                    if lesson.phase is not None:
                        lesson.phase.value = 90
                    if lesson.time is not None:
                        lesson.time.value = .25
                    self.assertTrue(bytes(lesson.picture.value).startswith(b'\x89PNG'))
                    self.assertNotEqual(before, bytes(lesson.picture.value))
                    self.assertAlmostEqual(lesson.data['relative_intensity'], 4)
                    for axis, (xlim, ylim) in zip(lesson.figure.axes, initial_limits):
                        assert_allclose(axis.get_xlim(), xlim)
                        assert_allclose(axis.get_ylim(), ylim)
                    lesson.amplitude.value = 0
                    self.assertEqual(lesson.data['relative_intensity'], 0)
                    if stage in ('phase', 'phasor'):
                        self.assertIn('undefined', lesson.readout.value)
                finally:
                    lesson.close()
                    lesson.close()
                self.assertIsNone(lesson.figure)
                self.assertEqual(bytes(lesson.picture.value), b'')
                self.assertIsNone(lesson.ui.comm)


if __name__ == '__main__':
    unittest.main()
