"""Sampling checks against known coefficients and analytical alias identities."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.field_sampling import DEFAULT_COEFFICIENTS, periodic_field, sampling_experiment
from wave_optics.amplitude_grating import field_fourier_coefficients


class FieldSamplingTests(unittest.TestCase):
    def test_resolved_coefficients_on_even_odd_and_shifted_grids(self):
        for count in (12, 13, 32):
            for start in (-.5, 0., .137):
                data = sampling_experiment(count, start_fraction=start)
                self.assertTrue(data['resolved'])
                self.assertLess(data['coefficient_error'], 1e-13)
                self.assertLess(data['field_error'], 1e-13)
                extracted = field_fourier_coefficients(data['field'], data['x_nm'], 1000., data['known_orders'])
                assert_allclose(extracted, DEFAULT_COEFFICIENTS, atol=1e-13)

    def test_more_periods_refine_bins_without_changing_physical_orders(self):
        for periods in (1, 2, 4):
            data = sampling_experiment(32, periods)
            self.assertEqual(data['dx_nm'], 1000/32)
            self.assertEqual(data['N'], 32*periods)
            indices = (data['known_orders']*periods) % data['N']
            assert_allclose(data['frequencies'][indices], data['known_orders']/1000)
            assert_allclose(data['coefficients'][indices], DEFAULT_COEFFICIENTS, atol=1e-13)
            self.assertLess(data['field_error'], 1e-13)

    def test_aliases_sum_coherently_at_same_bin(self):
        data = sampling_experiment(8)
        self.assertFalse(data['resolved'])
        self.assertIsNone(data['coefficient_error'])
        self.assertLess(data['roundtrip_error'], 1e-13)
        self.assertGreater(data['field_error'], .1)
        alias_index = int(np.flatnonzero(np.isclose(data['frequencies'], -.003))[0])
        assert_allclose(data['coefficients'][alias_index], DEFAULT_COEFFICIENTS[0]+DEFAULT_COEFFICIENTS[3], atol=1e-13)

    def test_general_origin_alias_phase(self):
        data = sampling_experiment(8, start_fraction=.137)
        alias_index = int(np.flatnonzero(np.isclose(data['frequencies'], -.003))[0])
        extra_phase = np.exp(2j*np.pi*8*data['x_start_nm']/1000)
        assert_allclose(data['coefficients'][alias_index],
                        DEFAULT_COEFFICIENTS[0]+DEFAULT_COEFFICIENTS[3]*extra_phase, atol=1e-13)

    def test_nyquist_equality_is_not_unique_and_field_is_periodic(self):
        data = sampling_experiment(10)
        self.assertFalse(data['resolved'])
        points = np.array([-.17, .21, .91])*1000
        assert_allclose(periodic_field(points), periodic_field(points+1000), atol=1e-13)
        self.assertTrue(np.isfinite(periodic_field(0.)))

    def test_invalid_inputs(self):
        for kwargs in ({'samples_per_period':0}, {'samples_per_period':4.5},
                       {'periods':True}, {'period_nm':-1}, {'start_fraction':np.inf},
                       {'orders':[0,0], 'coefficients':[1,1]}):
            with self.assertRaises(ValueError):
                sampling_experiment(**kwargs)


if __name__ == '__main__':
    unittest.main()
