"""Ideal-mask visualization data, Fourier truncation and widget lifecycle."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.scalar_grating import ScalarGratingConfiguration
from wave_optics.binary_transmission_view import BinaryTransmissionExplorer
from wave_optics.thin_grating import thin_mask_coefficients


class BinaryTransmissionViewTests(unittest.TestCase):
    def setUp(self):
        self.views = []

    def tearDown(self):
        for view in self.views:
            view.close()

    def explorer(self, config=None):
        view = BinaryTransmissionExplorer(config)
        self.views.append(view)
        return view

    def test_exact_masks_phase_gaps_and_dielectric_coefficients(self):
        view = self.explorer()
        self.assertTrue(set(abs(view.exact['amplitude'])) == {0., 1.})
        assert_allclose(abs(view.exact['phase']), 1., atol=2e-16)
        assert_allclose(view.coefficients['phase'], thin_mask_coefficients(view.config), atol=0)
        phase = view._artists['ideal_phase_amplitude'].get_ydata()
        self.assertTrue(np.all(np.isnan(phase[abs(view.exact['amplitude']) == 0])))
        phase = view._artists['coefficient_phase_amplitude'].get_ydata()
        even_nonzero = (view.all_orders != 0) & (view.all_orders % 2 == 0)
        self.assertTrue(np.all(np.isnan(phase[even_nonzero])))
        view.order.value = 2
        self.assertIn('undefined / negligible', view._artists['normalization'].get_text())

    def test_truncation_changes_sum_without_changing_ideal_function_or_solving_bvp(self):
        with patch('wave_optics.scalar_grating.solve_scalar_grating', side_effect=AssertionError('BVP called')):
            view = self.explorer()
            exact = {key: values.copy() for key, values in view.exact.items()}
            limits = view._artists['ideal_amplitude_phase'].axes.get_ylim()
            view.show_sum.value = True
            view.order.value = 12
            for M in (4, 0, 20):
                view.truncation.value = M
                self.assertLessEqual(abs(view.order.value), M)
                keep = abs(view.all_orders) <= M
                for model in ('amplitude', 'phase'):
                    assert_allclose(view.exact[model], exact[model], atol=0)
                    expected = view.coefficients[model][keep] @ view.basis[keep]
                    assert_allclose(view.data['reconstruction'][model], expected, atol=0)
                    self.assertLessEqual(np.max(abs(expected)), view.amplitude_limit)
                    assert_allclose(view._artists['sum_amplitude_'+model].get_ydata(), abs(expected), atol=0)
                    self.assertEqual(view.data['retained_square'][model], np.sum(abs(view.coefficients[model][keep])**2))
                self.assertEqual(view._artists['ideal_amplitude_phase'].axes.get_ylim(), limits)

    def test_selection_argand_and_model_visibility_reuse_data(self):
        view = self.explorer()
        data, basis, axes = view.data, view.basis, tuple(view.figure.axes)
        view.order.value = -3
        self.assertIs(view.data, data)
        self.assertIs(view.basis, basis)
        self.assertEqual(tuple(view.figure.axes), axes)
        for model in ('amplitude', 'phase'):
            coefficient = view.data['coefficients'][model][-3+view.config.M]
            assert_allclose(view._artists['argand_'+model].get_xdata(), [0, coefficient.real])
            assert_allclose(view._artists['argand_'+model].get_ydata(), [0, coefficient.imag])
        view.model.value = 'phase'
        self.assertFalse(view._artists['ideal_amplitude_amplitude'].get_visible())
        self.assertTrue(view._artists['ideal_amplitude_phase'].get_visible())
        self.assertFalse(view._artists['sum_amplitude_phase'].get_visible())
        view.show_sum.value = True
        self.assertTrue(view._artists['sum_amplitude_phase'].get_visible())
        self.assertFalse(view._artists['sum_amplitude_amplitude'].get_visible())

    def test_reset_independence_and_full_cleanup(self):
        from ipywidgets.widgets.widget import _instances
        baseline = set(_instances)
        first, second = self.explorer(), self.explorer()
        first.truncation.value = 0
        first.model.value = 'amplitude'
        first.show_sum.value = True
        second.order.value = -3
        first.reset_button.click()
        self.assertEqual(first.truncation.value, first.config.M)
        self.assertEqual(first.order.value, 1)
        self.assertEqual(first.model.value, 'both')
        self.assertFalse(first.show_sum.value)
        self.assertEqual(second.order.value, -3)
        first.close()
        first.close()
        first.model.value = 'phase'
        self.assertIsNone(first.figure)
        second.order.value = 2
        self.assertGreater(len(second.picture.value), 100)
        second.close()
        self.assertEqual(set(_instances), baseline)

    def test_single_order_and_uniform_masks(self):
        for duty in (0., .5, 1.):
            view = self.explorer(replace(ScalarGratingConfiguration(), M=0, duty_cycle=duty))
            self.assertEqual(view.order.value, 0)
            view.show_sum.value = True
            assert_allclose(view.data['reconstruction']['amplitude'], duty, atol=0)
            self.assertEqual(view.mean_square['amplitude'], duty)
            self.assertAlmostEqual(view.mean_square['phase'], 1.)
            self.assertGreater(len(view.picture.value), 100)


if __name__ == '__main__':
    unittest.main()
