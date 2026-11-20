"""Check that the lesson's path diagrams express far-field geometry."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.order_lesson import path_geometry, phase_connection, lesson_svg
from wave_optics.waves import plane_wave_phasor


class LessonGeometryTests(unittest.TestCase):
    def test_reference_line_is_perpendicular_and_paths_match_at_large_distance(self):
        for angle in (0, 15, 30, 60):
            data = path_geometry(period=2, angle_deg=angle)
            a, b, q, direction = (data[key] for key in ('a', 'b', 'q', 'direction'))
            self.assertAlmostEqual((b-q) @ direction, 0)
            self.assertAlmostEqual(np.linalg.norm(direction), 1)
            assert_allclose(q-a, data['extra']*direction)
            distant_point = 1e7*direction
            self.assertAlmostEqual(np.linalg.norm(distant_point-a)
                                   - np.linalg.norm(distant_point-b), data['extra'], places=5)
            self.assertAlmostEqual(np.linalg.norm(distant_point-q),
                                   np.linalg.norm(distant_point-b), places=5)

    def test_drawn_one_wavelength_example_repeats_at_each_pair(self):
        data = path_geometry(period=2, angle_deg=30)
        self.assertAlmostEqual(data['extra'], 1)  # lambda = 1 in the drawing
        points = np.column_stack([2*np.arange(4), np.zeros(4)])
        signed_path_differences = np.diff(points, axis=0) @ data['direction']
        assert_allclose(signed_path_differences, np.ones(3))

    def test_path_phase_and_plane_wave_translation_use_the_stated_signs(self):
        for angle in (0, 15, 30, 60):
            data = phase_connection(wavelength=1, period=2, angle_deg=angle)
            kx, kz = data['vector']
            spatial = plane_wave_phasor([[0, 0, 0], [2, 0, 0]], [kx, 0, kz])
            assert_allclose(spatial[1]/spatial[0], np.exp(1j*data['phase_from_path']))
            # Independent finite-path calculation of arrival phase A minus B.
            distance = 1e8
            r_b = np.linalg.norm(distance*data['direction'] - data['b'])
            path_difference = (2*distance*(data['b'] @ data['direction'])
                               - data['b'] @ data['b']) / (distance+r_b)
            assert_allclose(np.exp(1j*data['k']*path_difference),
                            spatial[1]/spatial[0], atol=2e-7)
            self.assertAlmostEqual(data['phase_from_path'], data['phase_from_step'])
        half_cycle = phase_connection(angle_deg=float(np.rad2deg(np.arcsin(.25))))
        self.assertAlmostEqual(half_cycle['phase_from_path'], np.pi)
        full_cycle = phase_connection()
        self.assertAlmostEqual(full_cycle['extra'], full_cycle['wavelength'])
        self.assertAlmostEqual(full_cycle['phase_from_path'], 2*np.pi)

    def test_all_lesson_scenes_render_without_registered_figures(self):
        from matplotlib._pylab_helpers import Gcf

        baseline = set(Gcf.figs)
        for scene in ('illumination', 'equal_paths', 'extra_path', 'repeated_paths',
                      'wavevector_connection', 'surface_phase'):
            svg = lesson_svg(scene)
            self.assertIn('<svg', svg)
            self.assertNotIn('<image', svg)  # drawings remain vector data
            self.assertEqual(set(Gcf.figs), baseline)


if __name__ == '__main__':
    unittest.main()
