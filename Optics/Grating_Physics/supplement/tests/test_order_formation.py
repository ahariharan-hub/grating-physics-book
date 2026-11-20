"""Physical limits and interactive state transitions of the order lesson."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.order_formation import (
    OrderFormationExplorer, arrival_phasors, array_intensity, phase_step, wavefront_fields,
)


class PhaseMatchingTests(unittest.TestCase):
    def test_arrivals_agree_with_paths_to_the_same_distant_point(self):
        wavelength, period, incidence, angle, count = 1, 2.5, -30, 20, 8
        distance = 1e8
        x = period * np.arange(count)
        k = 2*np.pi/wavelength
        sine = np.sin(np.deg2rad(angle))
        paths = np.sqrt(distance**2 - 2*distance*x*sine + x*x)
        # Stable exact path difference relative to A, with incident phase added.
        path_differences = (x*x - 2*distance*x*sine) / (paths + distance)
        exact = np.exp(1j*k*(path_differences + x*np.sin(np.deg2rad(incidence))))
        assert_allclose(arrival_phasors(wavelength, period, incidence, angle, count),
                        exact, atol=1e-5)

    def test_two_opening_interference_and_finite_array_zeros(self):
        angles = np.linspace(-90, 90, 181)
        delta = phase_step(1, 2.5, 20, angles)
        assert_allclose(array_intensity(1, 2.5, 20, angles, 2),
                        np.cos(delta / 2)**2, atol=1e-14)
        # First zero next to m=0 occurs at delta=2*pi/N, not at an order.
        for count in (2, 8, 32):
            first_zero = np.rad2deg(np.arcsin(1 / (2.5 * count)))
            self.assertLess(array_intensity(1, 2.5, 0, first_zero, count), 1e-28)
            self.assertAlmostEqual(array_intensity(1, 2.5, 0, 0, count), 1)

    def test_oblique_order_fields_join_and_far_field_peaks_agree(self):
        x = np.linspace(-6.2, 7.4, 157)[None, :]
        s = np.linspace(0, 3, 31)[:, None]
        for incidence in (-30, 0, 20):
            for order in (-1, 0, 1):
                angle = np.rad2deg(np.arcsin(np.sin(np.deg2rad(incidence)) + order / 2.5))
                self.assertAlmostEqual(phase_step(1, 2.5, incidence, angle), 2*np.pi*order)
                continuous, tiled = wavefront_fields(x, s, 1, 2.5, incidence, angle)
                assert_allclose(tiled, continuous, atol=2e-14)
                for count in (2, 8, 32):
                    self.assertAlmostEqual(array_intensity(1, 2.5, incidence, angle, count), 1)

    def test_unmatched_tiling_obeys_floquet_but_has_a_complex_seam_jump(self):
        period, incidence, angle = 2.5, 20, 15
        x = np.array([.1, .3, .8])
        _, tiled = wavefront_fields(x, .2, 1, period, incidence, angle)
        _, shifted = wavefront_fields(x + period, .2, 1, period, incidence, angle)
        incident_factor = np.exp(2j * np.pi * np.sin(np.deg2rad(incidence)) * period)
        assert_allclose(shifted, tiled * incident_factor, atol=1e-14)
        self.assertFalse(np.allclose(shifted, tiled))
        _, seam = wavefront_fields([period - 1e-10, period], .2,
                                   1, period, incidence, angle)
        expected_jump = abs(np.expm1(1j * phase_step(1, period, incidence, angle)))
        self.assertAlmostEqual(abs(seam[1] - seam[0]), expected_jump, places=8)

    def test_grazing_orders_and_invalid_inputs(self):
        for angle in (-90, 90):
            self.assertAlmostEqual(array_intensity(1, 2, 0, angle, 8), 1)
            field, tiled = wavefront_fields(np.arange(5), 0, 1, 2, 0, angle)
            advanced, _ = wavefront_fields(np.arange(5), 3, 1, 2, 0, angle)
            assert_allclose(field, advanced, atol=2e-14)
            assert_allclose(field, tiled, atol=2e-14)
        for args in [(0, 2, 0, 0), (1, -1, 0, 0), (1, 2, 90, 0),
                     (1, 2, 0, 91), (1, 2, 0, np.nan)]:
            with self.assertRaises(ValueError):
                phase_step(*args)
        for count in (1, 2.5, np.inf):
            with self.assertRaises(ValueError):
                array_intensity(1, 2, 0, 0, count)


class ExplorerTests(unittest.TestCase):
    def test_follow_free_angle_disappearance_grazing_and_reset(self):
        explorer = OrderFormationExplorer()
        try:
            self.assertEqual(explorer.view, 'arrivals')
            self.assertEqual(explorer.angle.value, 0)
            self.assertTrue(explorer.data['matched'])
            explorer.order.value = 1
            explorer.incidence.value = 20
            self.assertTrue(explorer.data['matched'])
            self.assertAlmostEqual(explorer.data['cycles'], 1)
            explorer.angle.value = 15
            self.assertIsNone(explorer.order.value)
            self.assertFalse(explorer.data['matched'])
            before = explorer.data['arrival_traces'].copy()
            explorer.periods.value = 32
            assert_allclose(explorer.data['arrival_traces'][:2], before)
            self.assertEqual(explorer.data['arrival_traces'].shape[0], 32)
            explorer.order.value = 0
            self.assertAlmostEqual(explorer.angle.value, 20)
            explorer.reset.click()
            self.assertEqual(explorer.periods.value, 2)
            explorer.period.value = 2
            explorer.order.value = 2
            self.assertEqual(explorer.angle.value, 90)
            self.assertTrue(explorer.data['matched'])
            explorer.period.value = 1.9
            self.assertIsNone(explorer.order.value)
            self.assertIn('evanescent', explorer.readout.value)
            self.assertEqual(explorer.angle.value, 90)
            self.assertTrue(np.isfinite(explorer.data['intensity']).all())
            explorer.reset.click()
            self.assertEqual(explorer.order.value, 0)
            self.assertTrue(explorer.data['matched'])
        finally:
            explorer.close()

    def test_time_traces_show_half_cycle_cancellation_and_full_cycle_agreement(self):
        explorer = OrderFormationExplorer()
        try:
            explorer.angle.value = float(np.rad2deg(np.arcsin(.2)))
            traces = explorer.data['arrival_traces']
            assert_allclose(traces[0], -traces[1], atol=2e-14)
            self.assertLess(explorer.data['current'], 1e-28)
            explorer.order.value = 1
            explorer.periods.value = 8
            assert_allclose(explorer.data['arrival_traces'],
                            np.tile(explorer.data['arrival_traces'][0], (8, 1)), atol=2e-14)
            explorer.angle.value = 15
            mean_field = explorer.data['arrival_traces'].mean(axis=0)
            self.assertAlmostEqual(2*np.mean(mean_field[:-1]**2), explorer.data['current'])
            self.assertNotIn('tiled', explorer.data)
        finally:
            explorer.close()

    def test_optional_wavefront_explorer_remains_independent(self):
        arrival = OrderFormationExplorer()
        periodicity = OrderFormationExplorer(view='wavefronts')
        try:
            periodicity.order.value = 1
            periodicity.incidence.value = 20
            assert_allclose(periodicity.data['continuous'], periodicity.data['tiled'], atol=2e-14)
            self.assertEqual(arrival.angle.value, 0)
            self.assertEqual(arrival.incidence.value, 0)
            periodicity.angle.value = 15
            self.assertFalse(periodicity.data['matched'])
            periodicity.close()
            arrival.angle.value = 15
            self.assertTrue(arrival.picture.value)
        finally:
            arrival.close()
            periodicity.close()

    def test_rerun_cleanup_releases_only_owned_widgets_and_callbacks(self):
        from ipywidgets.widgets.widget import _instances
        from matplotlib._pylab_helpers import Gcf

        baseline = set(_instances)
        managers = set(Gcf.figs)
        explorer = OrderFormationExplorer()
        angle = explorer.angle
        try:
            population = set(_instances)
            explorer.angle.value = -15
            explorer.period.value = 4
            self.assertEqual(set(_instances), population)
        finally:
            explorer.close()
        explorer.close()
        self.assertEqual(set(_instances), baseline)
        self.assertEqual(set(Gcf.figs), managers)
        self.assertNotIn(explorer._update, angle._trait_notifiers['value']['change'])
        angle.value = 30
        self.assertIsNone(explorer.figure)
        self.assertEqual(bytes(explorer.picture.value), b'')


if __name__ == '__main__':
    unittest.main()
