"""Presentation physics, observation caching and lifecycle of Notebook 0_5."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
from numpy.testing import assert_allclose

from wave_optics import scalar_grating_view as views
from wave_optics.scalar_grating import ScalarGratingConfiguration, solve_scalar_grating


class GratingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = views.ScalarGratingData()

    def test_default_model_solves_once_and_samples_same_backend_field(self):
        with patch.object(views, 'solve_scalar_grating', wraps=solve_scalar_grating) as solve:
            model = views.ScalarGratingData()
            self.assertEqual(solve.call_count, 1)
            for z in (0., -.15, -.3, -.6, -1.8):
                model.plane(z)
            model.evolution(3)
            self.assertEqual(solve.call_count, 1)
        assert_allclose(model.field[::71, ::53],
                        model.solution.reconstruct(model.x[::53], model.z[::71]), atol=2e-14)
        assert_allclose(model.index, model.config.refractive_index(model.x, model.z))
        self.assertEqual(model.config.M, 20)
        self.assertGreater(model.x_slice.size, 2*model.config.M+1)

    def test_projection_and_reconstruction_agree_in_every_region(self):
        model = self.model
        for z in (.4, 0., -.12, -.3, -1.8):
            plane = model.plane(z)
            assert_allclose(plane['projected'], plane['coefficients'], atol=2e-14)
            assert_allclose(plane['reconstructed'], plane['field'], atol=2e-14)
            assert_allclose(plane['coefficients'], model.solution.harmonics(z)[0])

    def test_outgoing_phase_and_decay_use_negative_z(self):
        model, s = self.model, self.model.solution
        for order in (0, 2, 3, -3):
            data = model.evolution(order)
            i = order+model.config.M
            expected = s.transmission[i]*np.exp(1j*s.beta_output[i]*model.depth)
            assert_allclose(data['coefficients'], expected, atol=2e-14)
            if data['kind'] == 'propagating':
                assert_allclose(data['ratio'], 1., atol=1e-14)
                assert_allclose(data['phase_advance'], s.beta_output[i].real*model.depth, atol=2e-14)
            else:
                assert_allclose(data['ratio'], np.exp(-s.beta_output[i].imag*model.depth), atol=2e-14)
                valid = np.isfinite(data['phase_advance'])
                assert_allclose(data['phase_advance'][valid], 0., atol=1e-14)
                self.assertTrue(np.all(np.isnan(data['phase_advance'][abs(expected) < views.PHASE_FLOOR])))

    def test_grazing_is_constant_but_has_zero_power(self):
        c = ScalarGratingConfiguration(wavelength=1., period=1., M=4)
        model = views.ScalarGratingData(solve_scalar_grating(c))
        data = model.evolution(1)
        self.assertEqual(data['kind'], 'grazing')
        assert_allclose(data['ratio'], 1.)
        assert_allclose(data['phase_advance'], 0., atol=1e-14)
        self.assertEqual(model.solution.transmission_efficiency[c.M+1], 0.)

    def test_phase_wrap_and_negligible_coefficients_are_hidden(self):
        field = np.array([np.exp(3j), np.exp(-3j), 1e-12j, 1+0j])
        phase = views.visible_phase(field, break_wraps=True)
        assert_allclose(phase[[0, 3]], [3, 0])
        self.assertTrue(np.isnan(phase[1:3]).all())
        self.assertTrue(np.isnan(views.visible_phase(field, relative_to=0)).all())
        c = replace(self.model.config, n_ridge=1., M=4)
        empty_order = views.ScalarGratingData(solve_scalar_grating(c)).evolution(1)
        self.assertTrue(np.isnan(empty_order['ratio']).all())
        self.assertTrue(np.isnan(empty_order['phase_advance']).all())

    def test_plane_validation_respects_transmitted_region(self):
        model = self.model
        model.plane(-model.config.thickness, output_only=True)
        for z in (-.29, 0., .4, -2., np.nan, .1j, [.1]):
            with self.subTest(z=z), self.assertRaises(ValueError):
                model.plane(z, output_only=True)


class GratingViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = views.ScalarGratingData()

    def setUp(self):
        self.explorers = []

    def tearDown(self):
        for view in self.explorers:
            view.close()
            view.close()
            self.assertIsNone(view.figure)
            self.assertIsNone(view.model)
            self.assertIsNone(view.data)
            self.assertEqual(bytes(view.picture.value), b'')
            self.assertIsNone(view.efficiency_figure)
            if view.efficiency_picture is not None:
                self.assertEqual(bytes(view.efficiency_picture.value), b'')
            self.assertEqual(view._artists, {})

    def explorer(self, stage):
        view = views.ScalarGratingExplorer(stage, self.model)
        self.explorers.append(view)
        self.assertTrue(bytes(view.picture.value).startswith(b'\x89PNG'))
        return view

    def test_material_and_field_selector_show_correct_arrays(self):
        geometry = self.explorer('geometry')
        assert_allclose(geometry.figure.axes[0].images[0].get_array(), self.model.index)
        view = self.explorer('field')
        for name in ('real', 'amplitude', 'intensity', 'phase'):
            view.quantity.value = name
            data, _, limits, _ = self.model.quantity(name)
            assert_allclose(view._artists['map'].get_array(), data)
            self.assertEqual(view._artists['map'].get_clim(), limits)
        self.assertEqual(view._artists['map_axis'].get_ylim(), (self.model.z_min, self.model.z_max))
        self.assertEqual(view._artists['map'].get_interpolation(), 'nearest')
        view.reset_button.click()
        self.assertEqual(view.quantity.value, 'intensity')

    def test_selected_harmonics_and_reconstruction_use_backend_values(self):
        view = self.explorer('harmonics')
        view.orders.value = (-3, 0, 2)
        axis = view._artists['harmonics_axis']
        for line, m in zip(axis.lines[:3], view.orders.value):
            assert_allclose(line.get_xdata(), abs(self.model.harmonics[:, m+self.model.config.M]))
            assert_allclose(line.get_ydata(), self.model.z)
        reconstruction = self.explorer('reconstruction')
        reconstruction.z_slice.value = -.123
        assert_allclose(reconstruction._artists['real'].get_ydata(), reconstruction.data['field'].real)
        assert_allclose(reconstruction._artists['sum_imag'].get_ydata(), reconstruction.data['field'].imag)
        self.assertLess(reconstruction.data['error'], 2e-14)

    def test_linked_reference_views_reuse_map_and_solver_and_keep_scales(self):
        view = self.explorer('reference')
        self.assertIsNone(view.efficiency_figure)
        view.evolution_ui
        view.efficiency_ui
        self.assertTrue(bytes(view.efficiency_picture.value).startswith(b'\x89PNG'))
        figure_axes = tuple(view.figure.axes)
        efficiency_axes = tuple(view.efficiency_figure.axes)
        efficiency_limits = [axis.get_ylim() for axis in efficiency_axes]
        image = view._artists['map']
        limits = image.get_clim()
        cached_field = self.model.field
        with patch.object(views, 'solve_scalar_grating', side_effect=AssertionError('BVP rerun')):
            for z in (-.3, -.61, -1.8):
                view.z_ref.value = z
                assert_allclose(view.data['field'], self.model.solution.reconstruct(self.model.x_slice, z)[0])
                assert_allclose(view._artists['plane'].get_ydata(), [z, z])
                assert_allclose(view._artists['evolution_markers'][0].get_xdata(), [-.3-z]*2)
                self.assertIn(f'z_ref = {z:.3f}', view.efficiency_readout.value)
                self.assertEqual(tuple(view.efficiency_figure.axes), efficiency_axes)
                self.assertEqual([axis.get_ylim() for axis in efficiency_axes], efficiency_limits)
                self.assertEqual(image.get_clim(), limits)
                self.assertIs(self.model.field, cached_field)
                self.assertEqual(tuple(view.figure.axes), figure_axes)
            view.order.value = 0
            assert_allclose(view.evolution_data['ratio'], 1., atol=1e-14)
        self.assertEqual(view.z_ref.max, -.3)
        self.assertEqual(view.z_ref.min, -1.8)
        self.assertFalse(view.z_ref.continuous_update)
        phase_limits = view._artists['coefficient_phase'].axes.get_xlim()
        self.assertLess(phase_limits[0], -self.model.config.M)
        self.assertGreater(phase_limits[1], self.model.config.M)
        view.z_ref.value = 0.  # Traitlets clamps assignments to the permitted half-space.
        self.assertEqual(view.z_ref.value, -.3)

    def test_independent_state_reset_and_full_widget_cleanup(self):
        from ipywidgets.widgets.widget import _instances
        baseline = set(_instances)
        a, b = self.explorer('reference'), self.explorer('reference')
        a.evolution_ui
        a.efficiency_ui
        a.z_ref.value, a.order.value = -1.2, 2
        b.z_ref.value = -.7
        a.reset_button.click()
        self.assertAlmostEqual(a.z_ref.value, -.35)
        self.assertEqual(a.order.value, 3)
        self.assertEqual(b.z_ref.value, -.7)
        a.close()
        a.z_ref.value = -1.
        self.assertIsNone(a.figure)
        b.z_ref.value = -1.1
        self.assertTrue(len(b.picture.value) > 0)
        b.close()
        self.assertEqual(set(_instances), baseline)

    def test_efficiency_display_uses_flux_not_squared_coefficients(self):
        view = self.explorer('reference')
        view.efficiency_ui
        s = self.model.solution
        i = self.model.config.M+2
        self.assertNotAlmostEqual(s.transmission_efficiency[i], abs(view.data['coefficients'][i])**2)
        for z in (-.3, -1.5):
            view.z_ref.value = z
            coefficients = s.at_reference_plane(z)
            assert_allclose([bar.get_height() for bar in view._artists['efficiency_flux']],
                            s.transmission_efficiency)
            assert_allclose([bar.get_height() for bar in view._artists['efficiency_magnitude']],
                            abs(coefficients))
            assert_allclose(view._artists['efficiency_phase'].get_offsets()[:, 1],
                            views.visible_phase(coefficients, relative_to=coefficients[self.model.config.M]))
        assert_allclose([bar.get_height() for bar in view._artists['efficiency_balance']],
                        [sum(s.reflection_efficiency), sum(s.transmission_efficiency)])
        self.assertIn('R + T =', view.efficiency_readout.value)
        flux_axis = view._artists['efficiency_flux'][0].axes
        zeros = flux_axis.collections[0].get_offsets()
        assert_allclose(zeros[:, 0], s.orders[s.transmission_efficiency == 0])
        assert_allclose(zeros[:, 1], 0)


class GratingNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_clean_and_connected_to_chapter(self):
        root = Path(__file__).resolve().parents[2]
        path = root/'0_5_From_Helmholtz_BVP_to_Diffracted_Field.ipynb'
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        self.assertNotIn('widgets', notebook.metadata)
        for cell in notebook.cells:
            self.assertNotIn('attachments', cell)
            if cell.cell_type == 'code':
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
                compile(cell.source, str(path), 'exec')
        for earlier in ('0_0_Foundations_of_Scalar_Optics.ipynb', '0_4_Propagation_of_Scalar_Optical_Fields.ipynb'):
            book = json.loads((root/earlier).read_text(encoding='utf-8'))
            self.assertTrue(any(path.name in ''.join(c['source']) for c in book['cells']))


if __name__ == '__main__':
    unittest.main()
