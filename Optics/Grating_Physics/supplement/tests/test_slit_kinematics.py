"""Continuous outgoing candidates and the independent kinematic control."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.slit_kinematics import slit_direction, SlitKinematicsExplorer


class SlitKinematicsTests(unittest.TestCase):
    def test_arbitrary_angles_preserve_length_and_outgoing_sign(self):
        for angle in (-90, -37.2, -14.5, 0, 14.5, 37.2, 90):
            with self.subTest(angle=angle):
                data = slit_direction(2, angle)
                self.assertAlmostEqual(np.hypot(data['kx'], data['kz']), np.pi)
                self.assertLessEqual(data['kz'], 0)
                self.assertAlmostEqual(np.rad2deg(np.arctan2(data['kx'], -data['kz'])), angle)
                longer = slit_direction(4, angle)
                assert_allclose([longer['kx'], longer['kz']], [.5*data['kx'], .5*data['kz']])

    def test_controls_update_direction_and_release_resources_on_rerun(self):
        from ipywidgets.widgets.widget import _instances
        before = set(_instances)
        for _ in range(2):
            explorer = SlitKinematicsExplorer()
            try:
                first = bytes(explorer.picture.value)
                explorer.angle.value = -14.5
                self.assertEqual(explorer.data['angle_deg'], -14.5)
                self.assertLess(explorer.data['kx'], 0)
                self.assertNotEqual(first, bytes(explorer.picture.value))
                explorer.wavelength.value = 4
                self.assertAlmostEqual(explorer.data['k'], np.pi/2)
                explorer.angle.value = 90
                self.assertIn('grazing', explorer.readout.value)
                self.assertAlmostEqual(explorer.data['kz'], 0)
            finally:
                explorer.close()
                explorer.close()
            self.assertIsNone(explorer.figure)
            self.assertEqual(set(_instances), before)


if __name__ == '__main__':
    unittest.main()
