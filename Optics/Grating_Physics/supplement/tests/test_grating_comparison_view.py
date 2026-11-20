"""Computed plots, linked controls and disposal for the C.2 comparison view."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.scalar_grating import ScalarGratingConfiguration, solve_scalar_grating
from wave_optics.grating_comparison_view import GratingComparisonExplorer


class GratingComparisonViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solution = solve_scalar_grating(ScalarGratingConfiguration(M=20))

    def setUp(self):
        self.views = []

    def tearDown(self):
        for view in self.views:
            view.close()

    def explorer(self, solution=None):
        view = GratingComparisonExplorer(self.solution if solution is None else solution)
        self.views.append(view)
        return view

    def test_fields_match_both_backend_predictions_at_shared_plane(self):
        view = self.explorer()
        for z in (-.3, -.75, -1.8):
            view.z_ref.value = z
            assert_allclose(view.field_values['hbvp'], self.solution.reconstruct(view.x, z)[0], atol=2e-14)
            assert_allclose(view.field_values['thin'], view.thin_model.reconstruct(view.x, z)[0], atol=2e-14)
            assert_allclose(view._artists['field_hbvp'].get_ydata(), abs(view.field_values['hbvp']))
            assert_allclose(view._artists['field_thin'].get_ydata(), abs(view.field_values['thin']))
        for quantity in ('intensity', 'real', 'amplitude'):
            view.quantity.value = quantity
            expected = {'intensity': abs(view.field_values['thin'])**2,
                        'real': view.field_values['thin'].real, 'amplitude': abs(view.field_values['thin'])}[quantity]
            assert_allclose(view._artists['field_thin'].get_ydata(), expected)
        self.assertEqual(view.z_ref.max, -.3)
        self.assertFalse(view.z_ref.continuous_update)

    def test_coefficient_argand_difference_and_table_are_computed(self):
        view = self.explorer()
        view.tabs.selected_index = 1
        for order, z in ((2, -.3), (3, -.6), (12, -.35)):
            view.z_ref.value, view.order.value = z, order
            p, i = view.data, order+view.config.M
            assert_allclose(view._artists['magnitude_hbvp'].get_ydata(), abs(p.hbvp))
            assert_allclose(view._artists['phase_thin'].get_ydata(), p.thin_phase)
            assert_allclose(view._artists['difference'].get_ydata(), abs(p.hbvp-p.thin))
            assert_allclose(view._artists['argand_hbvp'].get_xdata(), [0, p.hbvp[i].real])
            assert_allclose(view._artists['argand_thin'].get_ydata(), [0, p.thin[i].imag])
            assert_allclose(view._artists['delta_arrow'].xy, [p.hbvp[i].real, p.hbvp[i].imag])
            assert_allclose(view._artists['delta_arrow'].get_position(), [p.thin[i].real, p.thin[i].imag])
            self.assertIn(f'Selected m = {order:+d}', view.table.value)
            self.assertIn(f'{p.thin_transmittance:.6f}', view.table.value)
            self.assertIn(f'{p.hbvp_transmittance:.6f}', view.table.value)
            self.assertIn(p.kinds[i], view.table.value)
        view.order.value = 2
        self.assertIn('Δφ = — rad', view.table.value)

    def test_observation_reuses_solution_artists_scales_and_hidden_tab_images(self):
        with patch('wave_optics.scalar_grating.solve_scalar_grating', side_effect=AssertionError('BVP rerun')):
            view = self.explorer()
            basis, axes, line = view.basis, tuple(view.field_figure.axes), view._artists['field_hbvp']
            limits = line.axes.get_ylim()
            view.z_ref.value = -1.
            self.assertEqual(view.coefficient_picture.value, b'')  # Not rendered before opening this tab.
            self.assertIs(view.basis, basis)
            self.assertIs(view._artists['field_hbvp'], line)
            self.assertEqual(tuple(view.field_figure.axes), axes)
            self.assertEqual(line.axes.get_ylim(), limits)
            field_png = view.field_picture.value
            view.tabs.selected_index = 1
            plane, fields = view.data, view.field_values
            view.order.value = 3
            self.assertIs(view.data, plane)
            self.assertIs(view.field_values, fields)
            view.z_ref.value = -1.1
            self.assertEqual(view.field_picture.value, field_png)
            self.assertGreater(len(view.coefficient_picture.value), 100)
            view.tabs.selected_index = 0
            self.assertNotEqual(view.field_picture.value, field_png)

    def test_reset_instances_and_complete_cleanup(self):
        from ipywidgets.widgets.widget import _instances
        baseline = set(_instances)
        first, second = self.explorer(), self.explorer()
        first.tabs.selected_index = 1
        first.z_ref.value, first.order.value = -1.5, 3
        second.z_ref.value = -.6
        first.reset_button.click()
        self.assertEqual(first.z_ref.value, -.3)
        self.assertEqual(first.order.value, 1)
        self.assertEqual(second.z_ref.value, -.6)
        first.close()
        first.close()
        first.z_ref.value = -1.
        self.assertIsNone(first.solution)
        self.assertIsNone(first.field_figure)
        second.z_ref.value = -.7
        self.assertGreater(len(second.field_picture.value), 100)
        second.close()
        self.assertEqual(set(_instances), baseline)

    def test_single_order_and_grazing_views_remain_defined(self):
        for config in (replace(self.solution.configuration, M=0),
                       replace(self.solution.configuration, wavelength=1., period=1., M=4)):
            view = self.explorer(solve_scalar_grating(config))
            view.tabs.selected_index = 1
            view.order.value = min(1, config.M)
            view.z_ref.value = -1.
            self.assertTrue(np.all(np.isfinite(view.data.thin)))
            self.assertGreater(len(view.coefficient_picture.value), 100)


if __name__ == '__main__':
    unittest.main()
