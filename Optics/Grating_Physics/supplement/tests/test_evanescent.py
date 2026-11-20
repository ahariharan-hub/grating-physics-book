"""Outgoing branch, decay, phase matching, and notebook-four controls."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from wave_optics.diffraction import diffraction_orders
from wave_optics.evanescent import evanescent_order, order_phasor, decay_envelope


class EvanescentPhysicsTests(unittest.TestCase):
    def test_excluded_notebook_three_order_and_decay_length(self):
        data = evanescent_order()
        self.assertEqual(data['status'], 'evanescent')
        self.assertIsNone(data['outgoing_angle'])
        self.assertAlmostEqual(data['alpha'], 1.5)
        self.assertAlmostEqual(data['kappa'], np.pi*np.sqrt(1.25))
        self.assertAlmostEqual(data['decay_length'], .2847050173668708)
        self.assertLess(data['kz'].imag, 0)
        assert_allclose(abs(order_phasor(.37, -np.array([0., .25, .5]), data)),
                        np.exp(-data['kappa']*np.array([0., .25, .5])))
        self.assertEqual(abs(order_phasor(0, -1000, data)), 0)

    def test_complex_dispersion_and_phase_repeat_on_both_sides(self):
        for angle in (-45, 0, 30):
            for m in range(-3, 4):
                data = evanescent_order(1.3, 2.1, angle, m)
                assert_allclose(data['kx']**2 + data['kz']**2, data['k']**2, atol=1e-12)
                # An arbitrary start, depth and complex phase ratio detect
                # sign errors that a cosine evaluated at the origin hides.
                a = order_phasor(.31, -.27, data)
                b = order_phasor(.31+data['period'], -.27, data)
                assert_allclose(b/a, np.exp(1j*data['kix']*data['period']), atol=1e-13)
                self.assertLessEqual(data['kz'].real, 0)
                self.assertLessEqual(data['kz'].imag, 0)

    def test_matches_existing_real_order_classification(self):
        labels = np.arange(-5, 6)
        for angle in (-60, -30, 0, 30, 60):
            for period in (1., 2., 4.):
                old = diffraction_orders(2., period, abs(angle), 180. if angle < 0 else 0.,
                                         orders=labels)
                for i, m in enumerate(labels):
                    data = evanescent_order(2., period, angle, m)
                    expected = ('grazing' if old['grazing'][i] else
                                'propagating' if old['valid'][i] else 'evanescent')
                    self.assertEqual(data['status'], expected)

    def test_grazing_limit_is_continuous_at_finite_depth(self):
        grazing = evanescent_order(2., 2., 0., 1)
        self.assertEqual(grazing['status'], 'grazing')
        self.assertEqual(grazing['kz'], 0)
        self.assertTrue(np.isinf(grazing['decay_length']))
        s = np.linspace(0, 2, 9)
        target = order_phasor(.31, -s, grazing)
        assert_allclose(target, np.full_like(s, target[0], dtype=complex))
        for period, state in ((2.+1e-7, 'propagating'), (2.-1e-7, 'evanescent')):
            data = evanescent_order(2., period, 0., 1)
            self.assertEqual(data['status'], state)
            assert_allclose(order_phasor(.31, -s, data), target, atol=.0021)

    def test_time_zero_does_not_erase_the_envelope(self):
        data = evanescent_order()
        s = data['decay_length']*np.arange(4)
        phasor = order_phasor(0., -s, data)
        assert_allclose(phasor.real, np.exp(-np.arange(4)))
        assert_allclose((phasor*np.exp(-2j*np.pi*.25)).real, 0, atol=1e-15)
        assert_allclose((phasor*np.exp(-2j*np.pi*.5)).real, -np.exp(-np.arange(4)))
        assert_allclose(abs(phasor), decay_envelope(s, data))

    def test_propagating_zero_order_continues_incident_wave(self):
        data = evanescent_order(order=0)
        self.assertEqual(data['status'], 'propagating')
        self.assertAlmostEqual(data['outgoing_angle'], 30.)
        self.assertAlmostEqual(data['kz'].real, -np.pi*np.cos(np.pi/6))
        x, z = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-2, 0, 9))
        expected = np.exp(1j*np.pi*(np.sin(np.pi/6)*x-np.cos(np.pi/6)*z))
        assert_allclose(order_phasor(x, z, data), expected)
        assert_allclose(decay_envelope(-z, data), 1.)

    def test_fields_satisfy_helmholtz_equation_numerically(self):
        # Independent finite differences of the actual complex fields.
        h, x, z = 1e-4, .17, -.31
        for period in (3., 2., 1.5):
            data = evanescent_order(2., period, 0., 1)
            centre = order_phasor(x, z, data)
            laplacian = (order_phasor(x+h, z, data) + order_phasor(x-h, z, data)
                         + order_phasor(x, z+h, data) + order_phasor(x, z-h, data)
                         - 4*centre)/h**2
            self.assertLess(abs(laplacian+data['k']**2*centre), 2e-6)

    def test_mirrored_incidence_and_order_preserve_decay(self):
        positive = evanescent_order(2., 1.8, 35., 2)
        negative = evanescent_order(2., 1.8, -35., -2)
        self.assertAlmostEqual(positive['kx'], -negative['kx'])
        self.assertEqual(positive['kz'], negative['kz'])
        x = np.linspace(-1, 1, 11)
        assert_allclose(order_phasor(x, -.2, positive), order_phasor(-x, -.2, negative))

    def test_rejects_bad_parameters_and_wrong_half_space(self):
        for kwargs in ({'wavelength': 0}, {'period': -1}, {'angle_deg': 90},
                       {'angle_deg': np.nan}, {'order': 1.5}, {'order': np.inf}):
            with self.assertRaises(ValueError):
                evanescent_order(**kwargs)
        data = evanescent_order()
        for x, z in ((0, .1), (np.inf, -.1), (0, np.nan)):
            with self.assertRaises(ValueError):
                order_phasor(x, z, data)
        with self.assertRaises(ValueError):
            decay_envelope(-.1, data)


class EvanescentExplorerTests(unittest.TestCase):
    def test_selection_survives_cutoff_presets_playback_and_disposal(self):
        from ipywidgets.widgets.widget import _instances
        from matplotlib._pylab_helpers import Gcf
        from wave_optics.evanescent_view import EvanescentOrderExplorer

        widgets_before, figures_before = set(_instances), set(Gcf.figs)
        explorer = EvanescentOrderExplorer()
        try:
            self.assertEqual(explorer.data['status'], 'evanescent')
            self.assertEqual(explorer.order.value, 1)
            for button, status in ((explorer.propagating, 'propagating'),
                                   (explorer.grazing, 'grazing'),
                                   (explorer.evanescent, 'evanescent')):
                button.click()
                self.assertEqual(explorer.data['status'], status)
                self.assertEqual(explorer.order.value, 1)
            explorer.reset.click()
            explorer.period.value = 4
            self.assertEqual(explorer.data['status'], 'grazing')
            explorer.period.value = 4.5
            self.assertEqual(explorer.data['status'], 'propagating')
            explorer.period.value = 2
            self.assertEqual(explorer.data['status'], 'evanescent')
            self.assertEqual(explorer.order.value, 1)
            prior_image = bytes(explorer.picture.value)
            explorer.play.value = 10
            self.assertEqual(explorer.time.value, .25)
            self.assertNotEqual(prior_image, bytes(explorer.picture.value))
            explorer.time.value = .5
            self.assertEqual(explorer.play.value, 20)
            profile = explorer.figure.axes[2]
            assert_allclose(profile.lines[2].get_ydata(), -profile.lines[0].get_ydata())
            explorer.depth.value = .25
            self.assertEqual(explorer.figure.axes[1].get_ylim(), (-.25, 0))
            self.assertEqual(explorer.figure.axes[1].images[0].get_clim(), (-1, 1))
            explorer.reset.click()
            self.assertEqual(explorer.data['order'], 1)
            self.assertEqual(explorer.time.value, 0)
        finally:
            explorer.close()
            explorer.close()
        self.assertEqual(set(_instances), widgets_before)
        self.assertEqual(set(Gcf.figs), figures_before)


if __name__ == '__main__':
    unittest.main()
