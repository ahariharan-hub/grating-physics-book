"""Path geometry, phase signs, and the notebook's linked display."""

import gc
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.order_paths import path_difference_data
from wave_optics.order_waves import OrderWaveExplorer, cell_comparison_data, cell_comparison_limits


class PathGeometryTests(unittest.TestCase):
    def test_complete_routes_reproduce_phase_matching_for_signed_angles(self):
        for angle in (-60, -30, 0, 30, 60):
            lower, upper = cell_comparison_limits(1.7, 3.1, angle)
            for value in np.linspace(lower, upper, 7):
                wave = cell_comparison_data(1.7, 3.1, angle, value)
                data = path_difference_data(3.1, angle, wave['outgoing_angle'])
                # Each pair of endpoints lies on one plane perpendicular
                # to its own direction. Every ray segment travels forward.
                for points, direction in ((data['sources'], data['incident']),
                                          (data['ends'], data['outgoing'])):
                    projection = points @ direction
                    assert_allclose(projection[0], projection[1], atol=2e-14)
                incoming = np.linalg.norm(data['positions']-data['sources'], axis=1)
                outgoing = np.linalg.norm(data['ends']-data['positions'], axis=1)
                self.assertTrue(np.all(incoming > 0) and np.all(outgoing > 0))
                assert_allclose(incoming, data['incoming_lengths'], atol=2e-14)
                assert_allclose(outgoing, data['outgoing_lengths'], atol=2e-14)
                difference = (incoming+outgoing)[0]-(incoming+outgoing)[1]
                self.assertAlmostEqual(difference, data['delta_length'])
                self.assertAlmostEqual(wave['k']*difference, wave['difference'])
                # Arrival A/B phase is the inverse of the next/end seam ratio.
                assert_allclose(np.exp(1j*wave['k']*difference),
                                1/wave['seam_factor'], atol=3e-14)

    def test_half_and_whole_wavelength_examples_and_tilted_zero_order(self):
        for value, length in ((.5, 1), (1, 2), (-1, -2), (0, 0)):
            wave = cell_comparison_data(2, 4, 0, value)
            paths = path_difference_data(4, 0, wave['outgoing_angle'])
            self.assertAlmostEqual(paths['delta_length'], length)
            self.assertAlmostEqual(paths['incident_projection'], 0)
        tilted = path_difference_data(4, 30, 30)
        self.assertAlmostEqual(tilted['incident_projection'], 2)
        self.assertAlmostEqual(tilted['outgoing_projection'], 2)
        self.assertAlmostEqual(tilted['delta_length'], 0)

    def test_linked_controls_reuse_view_and_release_resources(self):
        from ipywidgets.widgets.widget import _instances
        before = set(_instances)
        explorer = OrderWaveExplorer(wavelength=2, period=4, angle_deg=0, selected_order=1)
        try:
            self.assertIsNone(explorer.path_figure)
            ui = explorer.paths_ui
            self.assertIs(explorer.paths_ui, ui)
            self.assertAlmostEqual(explorer.path_data['geometry']['delta_length'], 2)
            first = bytes(explorer.path_picture.value)
            explorer.compare.value = True
            explorer.comparison.value = .5
            self.assertAlmostEqual(explorer.path_data['geometry']['delta_length'], 1)
            self.assertIs(explorer.path_data['wave'], explorer.comparison_data)
            self.assertNotEqual(first, bytes(explorer.path_picture.value))
            explorer.period.value = 3
            self.assertEqual(explorer.path_data['geometry']['period'], 3)
            self.assertAlmostEqual(explorer.path_data['geometry']['delta_length'], 1)
            explorer.angle.value = -30
            self.assertAlmostEqual(explorer.path_data['geometry']['incident_projection'], -1.5)
            explorer.use_order.click()
            self.assertAlmostEqual(explorer.path_data['geometry']['delta_length'], 2)
            explorer.compare.value = False
            self.assertEqual(explorer.path_comparison_row.layout.display, 'none')
            self.assertEqual(explorer.path_data['wave']['comparison_order'], explorer.order.value)
            explorer.order.value = 0
            self.assertAlmostEqual(explorer.path_data['geometry']['delta_length'], 0)
            explorer.normal.click()
            self.assertAlmostEqual(explorer.path_data['geometry']['incident_projection'], 0)
            explorer.reset.click()
            self.assertEqual(explorer.path_data['wave']['comparison_order'], -1)
            self.assertEqual(explorer.path_data['geometry']['period'], 2)
        finally:
            explorer.close()
            explorer.close()
        self.assertIsNone(explorer.path_figure)
        self.assertIsNone(explorer.path_data)
        self.assertEqual(set(_instances), before)
        with self.assertRaises(RuntimeError):
            _ = explorer.paths_ui
        del explorer, ui
        gc.collect()


if __name__ == '__main__':
    unittest.main()
