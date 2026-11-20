"""Physics and ownership of the Lesson 2.2 thin-mask extension in 9.1."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from wave_optics.exit_field_view import ThinMaskExitFieldExplorer


class ThinMaskExitFieldTests(unittest.TestCase):
    def test_period_and_duty_control_spacing_and_open_fraction_independently(self):
        view = ThinMaskExitFieldExplorer()
        try:
            self.assertEqual(view.period.value, 2000)
            self.assertEqual(view.duty_cycle.value, .5)
            self.assertEqual(view.angle.value, 30)
            assert_allclose([view.data['opening_width_um'], view.data['closed_width_um']], [1, 1])
            assert_allclose(view.data['x'][[0, -1]], [-5, 5])
            assert_allclose([view.data['period_um'], view.data['periods_in_field'],
                             view.data['spatial_frequency_per_um'], view.data['G_rad_per_um']],
                            [2, 5, .5, np.pi])
            before = view.data['transmission'].copy()
            view.duty_cycle.value = .6
            self.assertEqual(view.period.value, 2000)
            self.assertFalse(np.array_equal(view.data['transmission'], before))
            assert_allclose([view.data['spatial_frequency_per_um'], view.data['G_rad_per_um']],
                            [.5, np.pi])
            assert_allclose([view.data['opening_width_um'], view.data['closed_width_um']], [1.2, .8])
            view.period.value = 650
            self.assertEqual(view.duty_cycle.value, .6)
            assert_allclose([view.data['opening_width_um'], view.data['closed_width_um']], [.39, .26])
            assert_allclose(view.data['periods_in_field'], 10/.65)
            assert_allclose(view.data['spatial_frequency_per_um'], 1/.65)
            assert_allclose(view.data['G_rad_per_um'], 2*np.pi/.65)
            for axis in view.figure.axes:
                assert_allclose(axis.get_xlim(), [-5, 5])
            for axis in view.surface.figure.axes[:2]:
                assert_allclose(axis.get_xlim(), [-5, 5])
        finally:
            view.close()

    def test_shared_controls_preserve_the_complex_incident_field_in_openings(self):
        view = ThinMaskExitFieldExplorer()
        try:
            self.assertIs(view.angle, view.surface.angle)
            self.assertIs(view.period, view.surface.period)
            self.assertIs(view.duty_cycle, view.surface.duty_cycle)
            view.surface.normal.click()
            assert_allclose(view.data['U_exit'], view.data['transmission'])
            for angle, period_nm, duty, wavelength in [
                    (30, 500, .4, 2), (-30, 1000, .2, 1.5), (60, 1000, .3, 4)]:
                with self.subTest(angle=angle, period_nm=period_nm, duty=duty, wavelength=wavelength):
                    view.angle.value = angle
                    view.period.value = period_nm
                    view.duty_cycle.value = duty
                    view.wavelength.value = wavelength
                    period = period_nm/1000
                    data = view.data
                    x = data['x']
                    kx = 2*np.pi/wavelength*np.sin(np.deg2rad(angle))
                    assert_allclose(data['U_exit'],
                                    data['transmission']*np.exp(1j*kx*x), atol=1e-14)
                    assert_allclose(data['amplitude'], data['transmission'], atol=1e-14)
                    assert_allclose(data['periodic_factor'], data['transmission'], atol=1e-14)
                    assert_allclose(data['phase_step'], kx*period)
                    self.assertTrue(np.all(np.isnan(data['phase'][data['transmission'] == 0])))
                    self.assertTrue(np.all(np.isnan(data['phase_ordinary_pbc'][data['transmission'] == 0])))
            view.surface.normal.click()
            assert_allclose(view.data['U_exit'], view.data['transmission'])
        finally:
            view.close()

    def test_ordinary_pbc_copy_omits_only_the_cell_to_cell_bloch_factor(self):
        view = ThinMaskExitFieldExplorer()
        try:
            data = view.data
            x = data['x']
            assert_allclose(data['phase_step'], np.pi)
            indices = [np.argmin(abs(x-centre)) for centre in (-4, -2, 0, 2, 4)]
            assert_allclose(data['U_exit'][indices], [1, -1, 1, -1, 1], atol=1e-14)
            assert_allclose(data['U_ordinary_pbc'][indices], np.ones(5), atol=1e-14)
            assert_allclose(abs(data['U_ordinary_pbc']), abs(data['U_exit']), atol=1e-14)
            reference = (x >= -1) & (x < 1)
            assert_allclose(data['U_ordinary_pbc'][reference], data['U_exit'][reference], atol=1e-14)
            # Check corresponding interior points, away from discontinuous edges.
            reference_indices = [np.argmin(abs(x-local)) for local in (-.2, .1, .3)]
            for j in (-2, -1, 1, 2):
                shifted = [np.argmin(abs(x-(x[index]+2*j))) for index in reference_indices]
                assert_allclose(data['U_ordinary_pbc'][shifted],
                                data['U_ordinary_pbc'][reference_indices], atol=1e-14)
                assert_allclose(data['U_exit'][shifted],
                                np.exp(1j*j*np.pi)*data['U_exit'][reference_indices], atol=1e-14)
            view.surface.normal.click()
            assert_allclose(view.data['U_ordinary_pbc'], view.data['U_exit'], atol=1e-14)
            # A commensurate inclined wave also satisfies ordinary PBC.
            view.period.value = 4000
            view.angle.value = 30
            assert_allclose(view.data['phase_step'], 2*np.pi)
            assert_allclose(view.data['U_ordinary_pbc'], view.data['U_exit'], atol=1e-13)
        finally:
            view.close()

    def test_centred_binary_order_coefficients_by_independent_projection(self):
        # Integrate only the open interval of [-p/2,p/2], where the field is
        # known directly. This also checks carrier removal for tilted incidence.
        period = .5
        for duty in (.4, .5):
            half_opening = duty*period/2
            for step in (0., np.pi/2, np.pi):
                for m in range(-7, 8):
                    kx = step/period
                    def integrand(x):
                        return np.exp(1j*kx*x)*np.exp(-1j*(kx+m*2*np.pi/period)*x)
                    coefficient = (quad(lambda x: integrand(x).real, -half_opening, half_opening)[0]
                                   + 1j*quad(lambda x: integrand(x).imag, -half_opening, half_opening)[0])/period
                    assert_allclose(coefficient, duty*np.sinc(m*duty), atol=1e-14)

    def test_close_disconnects_both_owned_views(self):
        view = ThinMaskExitFieldExplorer()
        controls = view._controls
        view.close()
        view.close()
        for control in controls:
            callbacks = control._trait_notifiers.get('value', {}).get('change', [])
            self.assertFalse(any(getattr(callback, '__self__', None) in (view, view.surface)
                                 for callback in callbacks))
        view.angle.value = 30
        self.assertIsNone(view.data)
        self.assertIsNone(view.figure)
        self.assertIsNone(view.surface.data)
        self.assertEqual(view.picture.value, b'')


if __name__ == '__main__':
    unittest.main()
