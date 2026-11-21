"""Check 0.6 coefficient extraction, propagation, linked controls and flux."""
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import unittest
from validation_tests import private_test_directory
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = '0_6_From_Complex_Fields_to_Diffraction_Orders.ipynb'


def verify():
    sys.path.insert(0, str(ROOT))
    namespace = {'__name__': '__orders_validation__'}
    try:
        with redirect_stdout(StringIO()):
            notebook = json.loads((ROOT/NOTEBOOK).read_text(encoding='utf-8'))
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    exec(compile(''.join(cell['source']), f'{NOTEBOOK}:cell{index}', 'exec'), namespace)
        solution, config = namespace['grating_solution'], namespace['grating_config']
        projection_error = float(abs(namespace['try_projected']-namespace['try_direct']))
        assert projection_error < 1e-13
        np.testing.assert_array_equal(solution.orders[solution.output_kinds == 'propagating'], np.arange(-2, 3))
        figures = ROOT/'_build/workflow/figures06'
        figures.mkdir(parents=True, exist_ok=True)
        reconstruction, reference = (namespace['grating_views'][name] for name in ('reconstruction', 'reference'))
        for name, picture in (('reconstruction', reconstruction.picture), ('reference', reference.picture),
                              ('evolution', reference.evolution_picture), ('efficiency', reference.efficiency_picture)):
            data = bytes(picture.value)
            assert data.startswith(b'\x89PNG\r\n\x1a\n') and len(data) > 10000
            (figures/(name+'.png')).write_bytes(data)
        reconstruction_error = parseval_error = 0.
        propagating = solution.output_kinds == 'propagating'
        flux = solution.transmission_efficiency.copy()
        with (patch('wave_optics.scalar_grating_view.solve_scalar_grating', side_effect=AssertionError('unexpected solve')),
              patch('wave_optics.absorbing_grating.solve_absorbing_grating', side_effect=AssertionError('unexpected solve'))):
            for z in (.4, -.12, -.6):
                reconstruction.z_slice.value = z
                plane = reconstruction.data
                expected = solution.at_reference_plane(z)
                np.testing.assert_allclose(plane['projected'], expected, atol=3e-14)
                np.testing.assert_allclose(plane['reconstructed'], plane['field'], atol=3e-14)
                reconstruction_error = max(reconstruction_error, plane['error'])
                parseval_error = max(parseval_error, float(abs(np.mean(abs(plane['field'])**2)-np.sum(abs(expected)**2))))
            assert parseval_error < 1e-13
            for quantity in ('real', 'amplitude', 'intensity', 'phase'):
                reference.quantity.value = quantity
                np.testing.assert_allclose(reference._artists['map'].get_array(), reference.model.quantity(quantity)[0])
            limits = [axis.get_ylim() for axis in reference.efficiency_figure.axes]
            for z in (-.3, -.5, -1., -1.8):
                reference.z_ref.value = z
                assert reconstruction.z_slice.value == -.6  # Independent slice control.
                coeff = solution.at_reference_plane(z)
                np.testing.assert_allclose(reference.data['coefficients'], coeff)
                np.testing.assert_allclose(abs(coeff[propagating]), abs(solution.transmission[propagating]), atol=2e-14)
                np.testing.assert_allclose(reference._artists['plane'].get_ydata(), [z, z])
                shown = abs(solution.orders) <= 7
                np.testing.assert_allclose([b.get_height() for b in reference._artists['efficiency_flux']], flux[shown])
                np.testing.assert_allclose([b.get_height() for b in reference._artists['efficiency_balance']],
                                          [solution.reflectance, solution.transmittance, solution.absorption])
                assert [axis.get_ylim() for axis in reference.efficiency_figure.axes] == limits
                for order in (0, 2, 3, -3):
                    reference.order.value = order
                    i = order+config.M
                    evolution = reference.evolution_data
                    expected = solution.transmission[i]*np.exp(1j*solution.beta_output[i]*reference.model.depth)
                    np.testing.assert_allclose(evolution['coefficients'], expected, atol=3e-14)
                    if order in (0, 2):
                        np.testing.assert_allclose(evolution['ratio'], 1, atol=3e-14)
                    else:
                        np.testing.assert_allclose(evolution['ratio'], np.exp(-solution.beta_output[i].imag*reference.model.depth), atol=3e-14)
                        assert flux[i] == 0
                assert reference.model.solution is solution
            for view in (reconstruction, reference):
                view.reset_button.click()
                assert all(c.value == v for c, v in view._defaults)
        from wave_optics.absorbing_grating import solve_absorbing_grating
        finer = solve_absorbing_grating(replace(config, M=72))
        orders = np.arange(-4, 5)
        convergence_error = float(np.max(abs(solution.transmission[orders+24]-finer.transmission[orders+72])))
        assert convergence_error < 2.55e-5
        record = dict(notebook=NOTEBOOK, code_cells=7, projection_error=projection_error,
                      reconstruction_error=reconstruction_error, parseval_error=parseval_error,
                      coefficient_convergence_error=convergence_error, energy_balance_error=float(abs(solution.energy_balance-1)),
                      propagating_orders=list(range(-2, 3)), linked_planes=4, followed_orders=[0, 2, 3, -3],
                      controls_and_resets='passed; same solution reused')
    finally:
        for view in namespace.get('grating_views', {}).values():
            view.close()

    tests = private_test_directory(ROOT)
    result = unittest.TestResult()
    if tests is not None:
        suite = unittest.TestSuite()
        for filename in ('test_absorbing_lesson.py', 'test_scalar_grating_view.py'):
            spec = importlib.util.spec_from_file_location(filename[:-3], tests/filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if filename == 'test_absorbing_lesson.py':
                for name in ('test_reference_evolution_and_flux_accounting_preserve_absorption',
                             'test_efficiency_plot_totals_include_orders_outside_display'):
                    suite.addTest(module.AbsorbingLessonTests(name))
            else:
                for name in ('test_grazing_is_constant_but_has_zero_power',
                             'test_phase_wrap_and_negligible_coefficients_are_hidden',
                             'test_plane_validation_respects_transmitted_region'):
                    suite.addTest(module.GratingDataTests(name))
                for name in ('test_linked_reference_views_reuse_map_and_solver_and_keep_scales',
                             'test_independent_state_reset_and_full_widget_cleanup'):
                    suite.addTest(module.GratingViewTests(name))
        stream = StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
        print(stream.getvalue())
        if not result.wasSuccessful():
            raise RuntimeError('Diffraction-order numerical/widget checks failed.')
    scripts = []
    with redirect_stdout(StringIO()):
        for script in sorted((ROOT/'Wave_solver_py_scripts/scripts/chapter_0').glob('0_6_*.py')):
            runpy.run_path(str(script), run_name='__validation__')
            scripts.append(script.name)
    assert len(scripts) == 4
    record.update(physics_and_widget_tests=result.testsRun, tests='passed', companion_scripts=scripts,
                  browser_ui='not covered by backend checks')
    (ROOT/'book/orders-validation.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    verify()
