"""Spreading-source phase, finite-array directions, and linked N controls."""

import gc
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from wave_optics.order_array import (
    opening_positions, spreading_field, far_field_amplitude, opening_array_data,
    OpeningArrayExplorer,
)
from wave_optics.order_waves import OrderWaveExplorer


class OpeningArrayPhysicsTests(unittest.TestCase):
    def test_single_source_is_circular_and_two_sources_cancel_at_half_cycle(self):
        radius = 8.
        angles = np.deg2rad([-70, -25, 0, 40, 85])
        wave = spreading_field(radius*np.sin(angles), -radius*np.cos(angles),
                               opening_positions(4, 1), 2, 0)
        assert_allclose(wave, np.exp(1j*np.pi*radius)/np.sqrt(radius), atol=2e-14)
        self.assertTrue(np.isnan(spreading_field(0, 0, np.array([0.]), 2, 0)))
        assert_allclose(abs(far_field_amplitude(np.arange(-90, 91), np.array([0.]), 2, 30)), 1)
        half_cycle_angle = np.rad2deg(np.arcsin(2/(2*4)))
        self.assertLess(abs(far_field_amplitude(half_cycle_angle, opening_positions(4, 2), 2, 0)), 1e-14)

    def test_spatial_sum_tends_to_same_far_field_with_signed_incidence(self):
        positions = opening_positions(2.5, 5)
        radius = 2e6
        angles = np.array([-55., -20., 0., 35., 65.])
        for incidence in (-30, 0, 30):
            field = spreading_field(radius*np.sin(np.deg2rad(angles)),
                                    -radius*np.cos(np.deg2rad(angles)), positions, 2, incidence)
            limit = field*np.sqrt(radius)*np.exp(-1j*np.pi*radius)/len(positions)
            expected = far_field_amplitude(angles, positions, 2, incidence)
            assert_allclose(limit, expected, atol=3e-5)

    def test_integer_peaks_stay_fixed_while_first_zeros_move_closer(self):
        previous_width = 90
        for count in (2, 5, 15, 21):
            positions = opening_positions(4, count)
            # Normal incidence, lambda=2, d=4: integer orders at -90,-30,0,30,90.
            assert_allclose(abs(far_field_amplitude([-90, -30, 0, 30, 90], positions, 2, 0)), 1, atol=1e-14)
            first_zero = np.rad2deg(np.arcsin(2/(count*4)))
            self.assertLess(abs(far_field_amplitude(first_zero, positions, 2, 0)), 1e-14)
            self.assertLess(first_zero, previous_width)
            previous_width = first_zero
        small = opening_array_data(2, 4, 0, 1)
        many = opening_array_data(2, 4, 0, 21)
        assert_array_equal(small['x'], many['x'])
        assert_array_equal(small['z'], many['z'])
        assert_array_equal(small['order_angles'], many['order_angles'])
        assert_allclose(small['intensity'], 1)
        self.assertTrue(np.all(np.isfinite(many['intensity'])))
        self.assertLessEqual(float(np.nanmax(abs(many['snapshot']))), 1+1e-14)
        self.assertTrue(np.any(np.isnan(many['snapshot'])))


class OpeningArrayControlTests(unittest.TestCase):
    def test_standalone_controls_update_without_an_order_explorer(self):
        from ipywidgets.widgets.widget import _instances
        before = set(_instances)
        explorer = OpeningArrayExplorer()
        try:
            initial_x = explorer.data['x'].copy()
            explorer.openings.value = 2
            assert_array_equal(explorer.data['x'], initial_x)
            explorer.period.value = 3
            assert_allclose(explorer.data['positions'], [0, 3])
            explorer.angle.value = 30
            assert_allclose(explorer.data['source_phases'], [0, 1.5*np.pi])
            explorer.wavelength.value = 3
            assert_allclose(explorer.data['source_phases'], [0, np.pi])
            explorer.normal.click()
            assert_allclose(explorer.data['source_phases'], 0)
            explorer.reset.click()
            self.assertEqual([explorer.data[key] for key in
                              ('wavelength', 'period', 'incident_angle', 'count')], [2, 4, 0, 1])
        finally:
            explorer.close()
            explorer.close()
        self.assertIsNone(explorer.figure)
        self.assertEqual(set(_instances), before)

    def test_lazy_shared_controls_cache_and_cleanup(self):
        from ipywidgets.widgets.widget import _instances
        before = set(_instances)
        explorer = OrderWaveExplorer(wavelength=2, period=4, angle_deg=0, selected_order=0)
        try:
            self.assertIsNone(explorer.array_figure)
            ui = explorer.array_ui
            self.assertIs(ui, explorer.array_ui)
            self.assertEqual(explorer.openings.value, 1)
            first = bytes(explorer.array_picture.value)
            explorer.openings.value = 5
            self.assertEqual(explorer.array_data['count'], 5)
            self.assertNotEqual(first, bytes(explorer.array_picture.value))
            snapshot = explorer.array_data
            explorer.order.value = 1
            self.assertIs(explorer.array_data, snapshot)  # selected order never filters the sum
            explorer.compare.value = True
            explorer.comparison.value = .5
            self.assertIs(explorer.array_data, snapshot)
            explorer.compare.value = False
            explorer.period.value = 2
            self.assertEqual(explorer.array_data['period'], explorer.data['period'])
            assert_allclose(np.diff(explorer.array_data['positions']), 2)
            explorer.wavelength.value = 3
            self.assertEqual(explorer.array_data['wavelength'], 3)
            explorer.angle.value = -30
            self.assertEqual(explorer.array_data['incident_angle'], -30)
            assert_allclose(np.diff(explorer.array_data['source_phases']), -2*np.pi/3)
            explorer.normal.click()
            assert_allclose(explorer.array_data['source_phases'], 0)
            explorer.reset.click()
            self.assertEqual(explorer.openings.value, 1)
            self.assertEqual(explorer.array_data['count'], 1)
        finally:
            explorer.close()
            explorer.close()
        self.assertIsNone(explorer.array_figure)
        self.assertIsNone(explorer.array_data)
        self.assertEqual(set(_instances), before)
        with self.assertRaises(RuntimeError):
            _ = explorer.array_ui
        del explorer, ui
        gc.collect()


if __name__ == '__main__':
    unittest.main()
