"""Guard the 0_5–0_7 absorbing lessons, shared solution and explorer data."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
from numpy.testing import assert_allclose

from wave_optics.absorbing_grating import (
    AbsorbingGratingConfiguration, solve_absorbing_grating,
)
from wave_optics.amplitude_grating import AbsorbingComparisonData, IdealAmplitudeGrating
from wave_optics.scalar_grating_view import ScalarGratingData, ScalarGratingExplorer


class AbsorbingLessonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solution = solve_absorbing_grating(AbsorbingGratingConfiguration(M=24))
        cls.data = ScalarGratingData(cls.solution)

    def setUp(self):
        self.views = []

    def tearDown(self):
        for view in self.views:
            view.close()

    def view(self, stage):
        result = ScalarGratingExplorer(stage, self.data, display_order_limit=7)
        self.views.append(result)
        self.assertIs(result.model.solution, self.solution)
        return result

    def test_complex_material_geometry_and_field_use_n_and_kappa(self):
        for stage in ('geometry', 'field'):
            v = self.view(stage)
            self.assertTrue(np.iscomplexobj(v.model.index))
            assert_allclose(v._artists['material_map'].get_array(), self.data.index.imag)
            v.material.value = 'n'
            assert_allclose(v._artists['material_map'].get_array(), self.data.index.real)
            self.assertIn('Real index n', v._artists['material_colorbar'].ax.get_ylabel())
            v.reset_button.click()
            self.assertEqual(v.material.value, 'kappa')
        self.assertIn('open fraction D_open = 0.5', self.views[0].readout.value)
        self.assertIn('1.5 + 0.8i', self.views[0].readout.value)
        v = self.views[1]
        for name in ('real', 'amplitude', 'intensity', 'phase'):
            v.quantity.value = name
            assert_allclose(v._artists['map'].get_array(), self.data.quantity(name)[0])
        assert_allclose(self.data.field[::80, ::31],
                        self.solution.reconstruct(self.data.x[::31], self.data.z[::80]), atol=3e-14)

    def test_harmonics_and_reconstruction_are_from_absorbing_solution(self):
        v = self.view('harmonics')
        self.assertEqual(tuple(v.orders.options), tuple(range(-7, 8)))
        v.orders.value = (-3, 0, 2)
        for line, m in zip(v._artists['harmonics_axis'].lines[:3], v.orders.value):
            assert_allclose(line.get_xdata(), abs(self.solution.harmonics(self.data.z)[:, m+self.solution.configuration.M]))
        v = self.view('reconstruction')
        for z in (.15, -.12, -.6):
            v.z_slice.value = z
            expected = self.solution.at_reference_plane(z)
            assert_allclose(v.data['coefficients'], expected)
            assert_allclose(v.data['projected'], expected, atol=3e-14)
            assert_allclose(v._artists['sum_real'].get_ydata(), v.data['field'].real, atol=3e-14)
            self.assertEqual(len(expected), 49)
            displayed = abs(self.solution.orders) <= 7
            assert_allclose([bar.get_height() for bar in v._artists['bars']], abs(expected[displayed]))
            assert_allclose(v._artists['coefficient_phase'].get_xdata(), np.arange(-7, 8))
            self.assertEqual(v._artists['coefficient_phase'].axes.get_xlim(), (-7.75, 7.75))
            self.assertIn('all 49 solved harmonics', v.readout.value)

    def test_reference_evolution_and_flux_accounting_preserve_absorption(self):
        v = self.view('reference')
        v.evolution_ui
        v.efficiency_ui
        self.assertEqual(tuple(v.order.options), tuple(range(-7, 8)))
        s = self.solution
        with (patch('wave_optics.scalar_grating_view.solve_scalar_grating', side_effect=AssertionError('lossless solve')),
              patch('wave_optics.absorbing_grating.solve_absorbing_grating', side_effect=AssertionError('unexpected solve'))):
            for z in (-.3, -.7, -1.8):
                v.z_ref.value = z
                assert_allclose(v.data['coefficients'], s.at_reference_plane(z))
                self.assertIn(f'A = {s.absorption:.6f}', v.efficiency_readout.value)
                self.assertIn(f'R + T + A = {s.energy_balance:.12f}', v.efficiency_readout.value)
                self.assertNotIn('R + T =', v.efficiency_readout.value)
                self.assertIn('independently integrated', v.efficiency_readout.value)
                displayed = abs(s.orders) <= 7
                assert_allclose([bar.get_height() for bar in v._artists['efficiency_magnitude']],
                                abs(s.at_reference_plane(z)[displayed]))
                assert_allclose([bar.get_height() for bar in v._artists['efficiency_flux']],
                                s.transmission_efficiency[displayed])
                assert_allclose([bar.get_height() for bar in v._artists['efficiency_balance']],
                                [sum(s.reflection_efficiency), sum(s.transmission_efficiency), s.absorption])
            v.order.value = 3
            assert_allclose(v.evolution_data['ratio'], np.exp(-s.beta_output[s.configuration.M+3].imag*self.data.depth), atol=3e-14)
            v.order.value = 1
            assert_allclose(v.evolution_data['ratio'], 1., atol=3e-14)

    def test_efficiency_plot_totals_include_orders_outside_display(self):
        v = ScalarGratingExplorer('reference', self.data, display_order_limit=1)
        self.views.append(v)
        v.efficiency_ui
        shown_transmission = sum(bar.get_height() for bar in v._artists['efficiency_flux'])
        total_transmission = v._artists['efficiency_balance'][1].get_height()
        self.assertGreater(total_transmission, shown_transmission)
        self.assertAlmostEqual(total_transmission, sum(self.solution.transmission_efficiency))
        self.assertEqual(len(v._artists['efficiency_magnitude']), 3)

    def test_comparison_reuses_initial_solution_and_only_solves_other_thicknesses(self):
        with patch('wave_optics.amplitude_grating.solve_absorbing_grating', wraps=solve_absorbing_grating) as solve:
            cases = AbsorbingComparisonData(solution=self.solution)
            self.assertEqual(solve.call_count, 6)
            index = list(cases.thicknesses).index(.3)
            self.assertIs(cases.solutions[index], self.solution)
            self.assertTrue(all(type(m) is IdealAmplitudeGrating for m in cases.ideal_models))
            plane = cases.planes(.2)[index]
            assert_allclose(plane.hbvp, self.solution.at_reference_plane(-.5))
            m = self.solution.orders
            expected = .5*np.sinc(.5*m)*np.exp(2j*np.pi*.3/.633+1j*self.solution.beta_output*.2)
            assert_allclose(plane.thin, expected, atol=3e-15)
            self.assertEqual(solve.call_count, 6)
        with self.assertRaises(ValueError):
            AbsorbingComparisonData(replace(self.solution.configuration, thickness=.5), solution=self.solution)
        with self.assertRaises(TypeError):
            AbsorbingComparisonData(solution=object())

    def test_absorbing_controls_dispose_without_leaking_widgets(self):
        from ipywidgets.widgets.widget import _instances
        baseline = set(_instances)
        for stage in ('geometry', 'field'):
            view = self.view(stage)
            view.material.value = 'n'
            view.close()
            view.material.value = 'kappa'
        self.assertEqual(set(_instances), baseline)

    def test_main_narrative_has_one_model_and_phase_lesson_is_separate(self):
        root = Path(__file__).resolve().parents[2]
        lessons = [nbformat.read(root/name, as_version=4) for name in (
            '0_5_From_Helmholtz_BVP_to_Diffracted_Field.ipynb',
            '0_6_From_Complex_Fields_to_Diffraction_Orders.ipynb',
            '0_7_Ideal_Binary_Masks_vs_Physical_Gratings.ipynb')]
        for lesson in lessons:
            nbformat.validate(lesson)
        source = '\n'.join(c.source for lesson in lessons for c in lesson.cells if c.cell_type == 'code')
        narrative = '\n'.join(c.source for lesson in lessons for c in lesson.cells if c.cell_type == 'markdown')
        for prohibited in ('solve_scalar_grating(', 'ScalarGratingConfiguration(', 'ThinPhaseGrating(',
                           'BinaryTransmissionExplorer(', 'GratingComparisonExplorer('):
            self.assertNotIn(prohibited, source)
        self.assertIn('ScalarGratingData(grating_solution)', source)
        self.assertIn('AbsorbingComparisonData(solution=grating_solution)', source)
        self.assertNotIn('phase mask', narrative.lower())
        self.assertNotIn('phase grating', narrative.lower())
        for required in ('Scalar Fourier Modal Method (FMM)', 'RCWA', 'Chrome-on-glass', 'Pistor', 'R+T+A=1'):
            self.assertIn(required, narrative)
        for cell in [c for lesson in lessons for c in lesson.cells]:
            if cell.cell_type == 'code':
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
        archive = nbformat.read(root/'supplement/Binary_Amplitude_and_Dielectric_Phase_Masks.ipynb', as_version=4)
        nbformat.validate(archive)
        self.assertTrue(any('GratingComparisonExplorer(' in c.source for c in archive.cells))


class NotebookReloadTests(unittest.TestCase):
    def test_setup_recovers_missing_cached_solver_and_refreshes_dependents(self):
        # Reloading classes must not invalidate other test modules' imports.
        script = textwrap.dedent('''
            import gc
            import importlib
            import nbformat
            import numpy as np
            from ipywidgets.widgets.widget import _instances
            import wave_optics.scalar_grating as scalar

            notebook = nbformat.read('0_5_From_Helmholtz_BVP_to_Diffracted_Field.ipynb', as_version=4)
            setup = next(c.source for c in notebook.cells if c.id == 'grating-setup')
            # Reproduce a pre-absorber kernel: the shared module is cached,
            # but its new helper is unavailable to the first absorber import.
            del scalar._solve_layer
            try:
                importlib.import_module('wave_optics.absorbing_grating')
            except ImportError as error:
                assert '_solve_layer' in str(error)
            else:
                raise AssertionError('The stale-module reproducer did not fail.')

            namespace = {}
            def verify():
                import wave_optics.absorbing_grating as absorbing
                import wave_optics.binary_transmission as binary
                import wave_optics.thin_grating as thin
                import wave_optics.amplitude_grating as amplitude
                assert absorbing._solve_layer is scalar._solve_layer
                assert absorbing.ScalarGratingSolution is scalar.ScalarGratingSolution
                assert thin.ScalarGratingSolution is scalar.ScalarGratingSolution
                assert binary.ScalarGratingConfiguration is scalar.ScalarGratingConfiguration
                assert amplitude.ThinPhaseGrating is thin.ThinPhaseGrating
                solution = namespace['grating_solution']
                assert isinstance(solution, scalar.ScalarGratingSolution)
                assert solution.configuration.M == 24
                assert namespace['display_order_limit'] == 7
                np.testing.assert_array_equal(solution.orders, np.arange(-24, 25))
                assert abs(solution.energy_balance - 1) < 1e-10
                cases = amplitude.AbsorbingComparisonData(solution=solution, thicknesses=(.3,))
                plane = cases.planes(.2)[0]
                np.testing.assert_allclose(plane.hbvp, solution.at_reference_plane(-.5))
                return cases

            exec(setup, namespace)
            cases = verify()
            baseline_widgets = set(_instances)
            old_view = namespace['grating_view'].ScalarGratingExplorer('geometry', namespace['grating_data'])
            namespace['grating_views']['geometry'] = old_view
            old_comparison = namespace['absorbing_view'].AbsorbingGratingExplorer(cases)
            namespace['absorber_comparison'] = old_comparison
            old_solution_class = scalar.ScalarGratingSolution
            del scalar._solve_layer
            # Now all dependents and live explorers are cached as well.
            exec(setup, namespace)
            verify()
            assert scalar.ScalarGratingSolution is not old_solution_class
            assert old_view._closed and old_comparison._closed
            assert set(_instances) == baseline_widgets
            del old_view, old_comparison
            namespace.clear()
            gc.collect()
            print('Stale first import and cached-dependent rerun both recovered.')
        ''')
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', script],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)


if __name__ == '__main__':
    unittest.main()
