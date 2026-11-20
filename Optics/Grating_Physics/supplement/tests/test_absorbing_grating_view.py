"""Linked physical/coefficient views and bounded computation, including disposal."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.absorbing_grating import AbsorbingGratingConfiguration
from wave_optics.amplitude_grating import AbsorbingComparisonData
from wave_optics.absorbing_grating_view import AbsorbingGratingExplorer
from wave_optics.scalar_grating_view import visible_phase
from wave_optics.amplitude_grating import FIELD_FFT_ATOL


class AbsorbingViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = AbsorbingComparisonData(AbsorbingGratingConfiguration(M=12), [.08, .3, .7])

    def setUp(self):
        self.view = AbsorbingGratingExplorer(self.data)

    def tearDown(self):
        self.view.close()

    def test_geometry_profiles_and_phases_follow_same_case_and_plane(self):
        v = self.view
        v.thickness.value, v.distance.value = 2, .2
        s, ideal = self.data.solutions[2], self.data.ideal_models[2]
        assert_allclose(v.fields['hbvp'], s.reconstruct(v.x, -.9)[0], atol=3e-14)
        assert_allclose(v.fields['thin'], ideal.reconstruct(v.x, -.9)[0], atol=3e-14)
        assert_allclose(v._artists['physical_field'].get_array(), v.maps[2])
        assert_allclose(v._artists['material'].get_array(), v.materials[2])
        for q in ('real', 'imag', 'intensity', 'amplitude', 'phase'):
            v.quantity.value = q
            u = v.fields['hbvp']
            expected = {'real': u.real, 'imag': u.imag, 'amplitude': abs(u), 'intensity': abs(u)**2,
                        'phase': visible_phase(u, break_wraps=True)}[q]
            assert_allclose(v._artists['profile_hbvp'].get_ydata(), expected)
        assert_allclose(v._artists['plane_lines'][0].get_ydata(), [-.9, -.9])
        assert_allclose(v._artists['exit_lines'][0].get_ydata(), [-.7, -.7])

    def test_complex_coefficients_flux_and_thickness_are_computed(self):
        v = self.view
        v.tabs.selected_index = 1
        v.order.value = 2
        self.assertTrue(np.isnan(v.plane.phase_difference[14]))
        v.order.value = 1
        v.distance.value = .13
        j, p = 13, v.plane
        assert_allclose(v._artists['efficiency_hbvp'].get_ydata(), p.hbvp_efficiency)
        assert_allclose(v._artists['difference'].get_ydata(), abs(p.difference))
        assert_allclose(v._artists['arrow'].xy, (p.hbvp[j].real, p.hbvp[j].imag))
        assert_allclose(v._artists['arrow'].get_position(), (p.thin[j].real, p.thin[j].imag))
        v.sweep.selected_index = 0
        assert_allclose(v._artists['power_A'].get_ydata(), [s.absorption for s in self.data.solutions])
        assert_allclose(v._artists['sweep_error'].get_ydata(), [abs(p.difference[j]) for p in v.planes])
        assert_allclose(v._artists['sweep_magnitude_hbvp'].get_ydata(), [abs(p.hbvp[j]) for p in v.planes])
        left, right = v._artists['sweep_error'].axes.get_xlim()
        self.assertLess(left, min(self.data.thicknesses))
        self.assertGreater(right, max(self.data.thicknesses))
        self.assertIn('Selected m = +1', v.table.value)
        self.assertIn('Δφ', v.table.value)

    def test_local_field_argand_and_fft_follow_position_plane_and_order(self):
        v = self.view
        with patch('wave_optics.amplitude_grating.solve_absorbing_grating', side_effect=AssertionError('unexpected solve')):
            original_field, original_fft = v.fields, v.fft_coefficients
            with patch('wave_optics.absorbing_grating_view.field_fourier_coefficients',
                       side_effect=AssertionError('representation controls must reuse the FFT')):
                v.x_position.value = float(v.x[93])
                v.quantity.value = 'imag'
                v.order.value = -2
                self.assertIs(v.fields, original_field)
                self.assertIs(v.fft_coefficients, original_fft)
            for distance in (0., .24, .83):
                v.distance.value = distance
                u, i, p = v.fields['hbvp'], v.point_index, v.plane
                assert_allclose(v._artists['field_argand'].get_offsets(), np.column_stack((u.real, u.imag)))
                assert_allclose(v._artists['field_argand'].get_array(), v.x)
                assert_allclose(v._artists['field_argand_point'].get_data(), [[u[i].real], [u[i].imag]])
                assert_allclose(v._artists['profile_point'].get_data(), [[v.x[i]], [u[i].imag]])
                self.assertLess(v.fft_error, FIELD_FFT_ATOL)
                assert_allclose(v.fft_coefficients, p.hbvp, atol=FIELD_FFT_ATOL)
                assert_allclose(v._artists['magnitude_fft'].get_ydata(), abs(p.hbvp), atol=FIELD_FFT_ATOL)
                assert_allclose(v._artists['phase_fft'].get_ydata(), visible_phase(v.fft_coefficients))
                j = v.order.value+self.data.configuration.M
                assert_allclose(v._artists['argand_fft'].get_data(), [[p.hbvp[j].real], [p.hbvp[j].imag]], atol=FIELD_FFT_ATOL)
                self.assertIn('absolute 1e-12 tolerance', v.fft_readout.value)
                self.assertIn('not an independent scattering', v.fft_readout.value)
            v.quantity.value = 'phase'
            assert_allclose(v._artists['profile_point'].get_ydata(), visible_phase([v.fields['hbvp'][v.point_index]]))
            v.reset_button.click()
            self.assertEqual(v.x_position.value, 0.)
            self.assertEqual(v.quantity.value, 'amplitude')

    def test_controls_reuse_solutions_images_and_artists_and_reset(self):
        v = self.view
        initial, basis, axis = v.pictures[0].value, v.basis, v.figures[0].axes[0]
        with patch('wave_optics.amplitude_grating.solve_absorbing_grating', side_effect=AssertionError('unexpected solve')):
            v.tabs.selected_index = 1
            v.distance.value, v.order.value, v.thickness.value = .25, 3, 0
            self.assertEqual(v.pictures[0].value, initial)
            self.assertEqual(v.pictures[2].value, b'')
            self.assertIs(v.basis, basis)
            self.assertIs(v.figures[0].axes[0], axis)
            v.tabs.selected_index = 0
            self.assertNotEqual(v.pictures[0].value, initial)
            v.reset_button.click()
            self.assertEqual(v.distance.value, 0.)
            self.assertEqual(v.order.value, 1)
            self.assertEqual(v.thickness.value, 1)

    def test_close_releases_all_widgets_and_callbacks(self):
        from ipywidgets.widgets.widget import _instances
        baseline = set(_instances)
        extra = AbsorbingGratingExplorer(self.data)
        extra.close()
        extra.close()
        extra.distance.value = .5
        self.assertIsNone(extra.data)
        self.assertEqual(extra.figures, [])
        self.assertEqual(set(_instances), baseline)

    def test_display_limit_preserves_full_field_fft_and_flux(self):
        with patch('wave_optics.amplitude_grating.solve_absorbing_grating', side_effect=AssertionError('unexpected solve')):
            v = AbsorbingGratingExplorer(self.data, display_order_limit=7)
            try:
                self.assertEqual(tuple(v.order.options), tuple(range(-7, 8)))
                assert_allclose(v.fields['hbvp'], self.view.fields['hbvp'])
                assert_allclose(v.fields['thin'], self.view.fields['thin'])
                self.assertEqual(v.basis.shape[0], 25)
                v.tabs.selected_index = 1
                for index, distance, order in ((0, .0, -7), (2, .23, 7)):
                    v.thickness.value, v.distance.value, v.order.value = index, distance, order
                    p, s = v.plane, self.data.solutions[index]
                    shown = abs(s.orders) <= 7
                    assert_allclose(v.fields['hbvp'], s.reconstruct(v.x, p.z_ref)[0], atol=3e-14)
                    assert_allclose(v.fft_coefficients, p.hbvp, atol=FIELD_FFT_ATOL)
                    self.assertEqual(len(v.fft_coefficients), 25)
                    for key in ('hbvp', 'thin'):
                        for name, expected in (('magnitude', abs(getattr(p, key))),
                                               ('phase', getattr(p, key+'_phase')),
                                               ('efficiency', getattr(p, key+'_efficiency'))):
                            artist = v._artists[name+'_'+key]
                            assert_allclose(artist.get_xdata(), s.orders[shown])
                            assert_allclose(artist.get_ydata(), expected[shown])
                            self.assertEqual(artist.axes.get_xlim(), (-7.75, 7.75))
                    assert_allclose(v._artists['magnitude_fft'].get_ydata(), abs(p.hbvp[shown]), atol=FIELD_FFT_ATOL)
                    assert_allclose(v._artists['difference'].get_ydata(), abs(p.difference[shown]))
                    assert_allclose(v._artists['delta_phase'].get_ydata(), p.phase_difference[shown])
                    value = p.hbvp[order+s.configuration.M]
                    assert_allclose(v._artists['argand_hbvp'].get_data(), [[0, value.real], [0, value.imag]])
                    self.assertIn(f'T = {s.transmittance:.5f}', v.readout.value)
                    self.assertIn('all 25 solved harmonics', v.readout.value)
            finally:
                v.close()


if __name__ == '__main__':
    unittest.main()
