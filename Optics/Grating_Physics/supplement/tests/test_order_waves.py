"""Order directions, translation phase, and selected-wave consistency."""

import gc
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from wave_optics.order_waves import (
    order_wave_data, OrderWaveExplorer, cell_comparison_data, cell_comparison_limits,
)
from wave_optics.waves import plane_wave_phasor


class OrderWavePhysicsTests(unittest.TestCase):
    def test_stated_example_and_zero_order_continuation(self):
        data = order_wave_data()
        assert_array_equal(data['orders'], [-1, 0])
        assert_allclose(data['angles'], [-30, 30])
        self.assertEqual(data['selected'], -1)
        self.assertAlmostEqual(data['incident_step'], np.pi)
        self.assertAlmostEqual(data['outgoing_step'], -np.pi)
        zero = order_wave_data(selected_order=0)
        assert_allclose(zero['vectors'][zero['index']], zero['incident'])
        xx, zz = np.meshgrid(zero['x'], zero['z_below'])
        assert_allclose(zero['outgoing_field'],
                        np.cos(zero['incident'][0]*xx+zero['incident'][2]*zz))

    def test_translated_complex_fields_match_incident_offset(self):
        for angle in (-60, -30, 0, 30, 60):
            data = order_wave_data(wavelength=1.3, period=3.2, angle_deg=angle)
            assert_allclose(np.linalg.norm(data['vectors'], axis=1), data['k'])
            self.assertTrue(np.all(data['vectors'][:, 2] <= 0))
            # Arbitrary starting point and phase, not just the origin/cosine.
            points = [[.31, 0, -.82], [.31+data['period'], 0, -.82]]
            incoming = plane_wave_phasor(points, data['incident'], phase=.73)
            for order, vector in zip(data['orders'], data['vectors']):
                wave = plane_wave_phasor(points, vector, phase=-.29)
                assert_allclose(wave[1]/wave[0], incoming[1]/incoming[0], atol=1e-12)
                self.assertAlmostEqual((vector[0]-data['incident'][0])*data['period'],
                                       2*np.pi*order)

    def test_signed_incidence_mirrors_orders_and_grazing_is_z_independent(self):
        positive = order_wave_data(angle_deg=35, period=4)
        negative = order_wave_data(angle_deg=-35, period=4)
        assert_array_equal(positive['orders'], -negative['orders'][::-1])
        assert_allclose(positive['angles'], -negative['angles'][::-1])
        grazing = order_wave_data(wavelength=1, period=4, angle_deg=0, selected_order=4)
        assert_array_equal(grazing['orders'], np.arange(-4, 5))
        self.assertTrue(grazing['grazing'][grazing['index']])
        self.assertEqual(grazing['angles'][grazing['index']], 90)
        assert_allclose(grazing['outgoing_field'][0], grazing['outgoing_field'][-1])
        missing = order_wave_data(wavelength=4, period=1, selected_order=-1)
        assert_array_equal(missing['orders'], [0])
        self.assertEqual(missing['selected'], 0)


class OrderWaveControlTests(unittest.TestCase):
    def test_selection_updates_fallback_reset_and_cleanup(self):
        from ipywidgets.widgets.widget import _instances
        from matplotlib._pylab_helpers import Gcf
        widgets_before, figures_before = set(_instances), set(Gcf.figs)
        explorer = OrderWaveExplorer()
        try:
            first = bytes(explorer.picture.value)
            explorer.order.value = 0
            self.assertEqual(explorer.data['selected'], 0)
            self.assertNotEqual(first, bytes(explorer.picture.value))
            explorer.period.value = 4
            self.assertIn(1, [value for _, value in explorer.order.options])
            explorer.order.value = 1
            self.assertTrue(explorer.data['grazing'][explorer.data['index']])
            explorer.period.value = 1
            self.assertEqual(explorer.order.value, 0)
            self.assertIn('previous order', explorer.readout.value)
            explorer.reset.click()
            self.assertEqual(explorer.order.value, -1)
            self.assertEqual(tuple(c.value for c in explorer._controls), (2, 2, 30))
            explorer.normal.click()
            self.assertEqual(explorer.angle.value, 0)
            self.assertEqual(len(explorer.figure.axes), 4)
        finally:
            explorer.close()
            explorer.close()
        self.assertEqual(set(_instances), widgets_before)
        self.assertEqual(set(Gcf.figs), figures_before)
        del explorer
        gc.collect()


class CellComparisonTests(unittest.TestCase):
    def test_phase_walk_and_seam_refer_to_the_same_wave(self):
        for angle, value in ((0, .5), (0, 1), (30, -.5), (-30, 0), (0, -2)):
            data = cell_comparison_data(2, 4, angle, value)
            vector = [data['outgoing_kx'], 0, data['outgoing_kz']]
            self.assertAlmostEqual(np.linalg.norm(vector), np.pi)
            # A and B share z and time. Their common phase cancels even away
            # from the origin; the graph must track this complex field ratio.
            points = np.column_stack([data['phase_x'], np.zeros(101), np.full(101, -3.65)])
            wave = plane_wave_phasor(points, vector, phase=.73)
            assert_allclose(wave/wave[0], np.exp(1j*data['phase_from_a']), atol=3e-14)
            self.assertAlmostEqual(data['phase_from_a'][-1], data['outgoing_step'])
            assert_allclose(np.exp(1j*data['incident_step'])/(wave[-1]/wave[0]),
                            data['seam_factor'], atol=3e-14)
        half = cell_comparison_data(2, 4, 0, .5)
        self.assertAlmostEqual(half['outgoing_kx'], np.pi/4)
        self.assertAlmostEqual(half['outgoing_step'], np.pi)
        self.assertEqual(half['incident_step'], 0)
        self.assertFalse(half['matched'])
        whole = cell_comparison_data(2, 4, 0, 1)
        self.assertAlmostEqual(whole['outgoing_step'], 2*np.pi)
        self.assertTrue(whole['matched'])

    def test_period_updates_in_same_physical_window_as_main_plot(self):
        explorer = OrderWaveExplorer(wavelength=1, period=2, angle_deg=0, selected_order=0)
        try:
            explorer.compare.value = True
            explorer.comparison.value = -.5
            previous_image = None
            for period in (2, 4, 1.5):
                explorer.period.value = period
                data = explorer.comparison_data
                self.assertEqual(data['period'], explorer.data['period'])
                assert_array_equal(data['x'], explorer.data['x'])
                assert_array_equal(data['z'], explorer.data['z_below'])
                assert_allclose(np.diff(data['cell_boundaries']), period)
                main_limits = explorer.figure.axes[0].get_xlim()
                for axis in explorer.comparison_figure.axes[:2]:
                    self.assertEqual(axis.get_xlim(), main_limits)
                    self.assertEqual(axis.get_ylim(), (-4.5, 0))
                    joins = [line.get_xdata()[0] for line in axis.lines
                             if line.get_linestyle() == ':']
                    assert_allclose(joins, data['cell_boundaries'])
                    self.assertTrue(any(t.get_text() == f'd = {period:g} µm' for t in axis.texts))
                self.assertIn(f'Shared period d = {period:g} µm', explorer.comparison_readout.value)
                image = bytes(explorer.comparison_picture.value)
                if previous_image is not None:
                    self.assertNotEqual(image, previous_image)
                previous_image = image
            # A parameter change while hidden must also appear when reopened.
            explorer.compare.value = False
            explorer.period.value = 3
            explorer.compare.value = True
            self.assertEqual(explorer.comparison_data['period'], 3)
            assert_allclose(np.diff(explorer.comparison_data['cell_boundaries']), 3)
        finally:
            explorer.close()
        del explorer
        gc.collect()

    def test_integer_fields_agree_including_tilted_incidence(self):
        for angle in (0, 30, -35):
            low, high = cell_comparison_limits(1, 2, angle)
            for order in range(int(np.ceil(low)), int(np.floor(high))+1):
                data = cell_comparison_data(1, 2, angle, order)
                self.assertTrue(data['matched'])
                self.assertEqual(data['order'], order)
                assert_allclose(data['copied'], data['continuous'], atol=3e-14)

    def test_noninteger_has_phase_jumps_but_the_continuous_wave_remains_smooth(self):
        for angle in (0, 30):
            data = cell_comparison_data(1, 4, angle, -1.5)
            self.assertFalse(data['matched'])
            self.assertIsNone(data['order'])
            self.assertAlmostEqual(abs(data['seam_jump']), np.pi)
            self.assertGreater(np.max(abs(data['copied']-data['continuous'])), 1.9)
            # Reference cell [0,d) is unchanged. Other cells acquire only the
            # incident offset, not the extra noninteger comparison phase.
            x = data['x']
            base = (x >= 0) & (x < 4)
            assert_allclose(data['copied'][:, base], data['continuous'][:, base], atol=1e-14)
            epsilon = 1e-8
            outgoing_kx = (2*np.pi*np.sin(np.deg2rad(angle)) + 2*np.pi*(-1.5)/4)
            outgoing_kz = -np.sqrt((2*np.pi)**2-outgoing_kx**2)
            points = [[4-epsilon, 0, -.37], [4+epsilon, 0, -.37]]
            smooth = plane_wave_phasor(points, [outgoing_kx, 0, outgoing_kz])
            self.assertLess(abs(smooth[1]-smooth[0]), 1e-6)
            # Analytic limits of the two cell expressions at x=d.
            left = np.exp(1j*(outgoing_kx*4 + outgoing_kz*(-.37)))
            right = np.exp(1j*(data['incident_step'] + outgoing_kz*(-.37)))
            assert_allclose(right/left, data['seam_factor'], atol=2e-14)

    def test_optional_controls_bounds_and_disposal(self):
        from ipywidgets.widgets.widget import _instances
        before = set(_instances)
        explorer = OrderWaveExplorer(wavelength=1, period=4, angle_deg=0)
        try:
            self.assertIsNone(explorer.comparison_data)
            explorer.compare.value = True
            self.assertTrue(explorer.comparison_data['matched'])
            explorer.comparison.value = -1.5
            self.assertFalse(explorer.comparison_data['matched'])
            self.assertEqual(explorer.order.value, -1)  # main order view is retained
            explorer.use_order.click()
            self.assertEqual(explorer.comparison.value, -1)
            self.assertTrue(explorer.comparison_data['matched'])
            explorer.comparison.value = -3.5
            explorer.wavelength.value = 4
            self.assertAlmostEqual(explorer.comparison.value, -1)
            self.assertIn('nearest real-direction limit', explorer.comparison_readout.value)
            explorer.angle.value = 30
            explorer.comparison.value = -.75
            self.assertFalse(explorer.comparison_data['matched'])
            explorer.order.value = 0
            self.assertEqual(explorer.comparison.value, 0)
            self.assertTrue(explorer.comparison_data['matched'])
            explorer.compare.value = False
            self.assertEqual(bytes(explorer.comparison_picture.value), b'')
            self.assertIsNone(explorer.comparison_data)
            explorer.compare.value = True
            self.assertTrue(explorer.comparison_data['matched'])
        finally:
            explorer.close()
        self.assertEqual(set(_instances), before)
        self.assertIsNone(explorer.comparison_figure)
        del explorer
        gc.collect()


if __name__ == '__main__':
    unittest.main()
