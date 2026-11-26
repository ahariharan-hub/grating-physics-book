"""Check 0.7 common-plane mask comparisons, cached controls and thickness sweep."""
from contextlib import redirect_stdout
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
NOTEBOOK = '0_7_Ideal_Binary_Masks_vs_Physical_Gratings.ipynb'


def verify():
    sys.path.insert(0, str(ROOT))
    namespace = {'__name__': '__mask_validation__'}
    try:
        with redirect_stdout(StringIO()):
            notebook = json.loads((ROOT/NOTEBOOK).read_text(encoding='utf-8'))
            solves = None
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                if 'AbsorbingComparisonData(solution=' in source:
                    with patch('wave_optics.amplitude_grating.solve_absorbing_grating',
                               wraps=namespace['amplitude_model'].solve_absorbing_grating) as solve:
                        exec(compile(source, f'{NOTEBOOK}:cell{index}', 'exec'), namespace)
                        solves = solve.call_count
                else:
                    exec(compile(source, f'{NOTEBOOK}:cell{index}', 'exec'), namespace)
        cases, view = namespace['absorber_cases'], namespace['absorber_comparison']
        assert solves == 6 and len(cases.solutions) == 7
        assert cases.solutions[3] is namespace['grating_solution']
        np.testing.assert_allclose(namespace['try_coefficients'][(namespace['try_orders'] != 0) & (namespace['try_orders'] % 2 == 0)], 0, atol=1e-15)
        j = cases.configuration.M+1
        errors = [float(abs(p.difference[j])) for p in cases.planes(0)]
        np.testing.assert_allclose(np.array(errors)[[0, 3, 6]], [.2266, .02078, .08598], atol=.001)
        assert np.argmin(errors) == 3
        fft_error = 0.
        with patch('wave_optics.amplitude_grating.solve_absorbing_grating', side_effect=AssertionError('unexpected solve')):
            for quantity in ('real', 'imag', 'amplitude', 'intensity', 'phase'):
                view.quantity.value = quantity
            view.x_position.value = float(view.x[93])
            point = view.fields['hbvp'][view.point_index]
            np.testing.assert_allclose(view._artists['field_argand_point'].get_data(), [[point.real], [point.imag]])
            view.compare_ideal.value = True
            assert view._artists['profile_thin'].get_visible()
            view.tabs.selected_index = 1
            for index, distance, order in ((0, 0., 1), (3, .25, 2), (6, .5, 3)):
                view.thickness.value, view.distance.value, view.order.value = index, distance, order
                s, ideal = cases.solutions[index], cases.ideal_models[index]
                z = -s.configuration.thickness-distance
                np.testing.assert_allclose(view.plane.hbvp, s.at_reference_plane(z))
                np.testing.assert_allclose(view.plane.thin, ideal.at_reference_plane(z))
                fft_error = max(fft_error, view.fft_error)
                assert view.fft_error < 1e-12
                np.testing.assert_allclose(view.fft_coefficients, view.plane.hbvp, atol=1e-12)
                assert len(view.fft_coefficients) == 49
            view.order.value = 1
            view.sweep.selected_index = 0
            np.testing.assert_allclose(view._artists['sweep_error'].get_ydata(),
                                       [abs(p.difference[j]) for p in view.planes])
            assert all(abs(s.energy_balance-1) < 1e-10 for s in cases.solutions)
            view.reset_button.click()
            assert all(c.value == v for c, v in zip(view._controls, view._defaults))
            figures = ROOT/'_build/workflow/figures07'
            figures.mkdir(parents=True, exist_ok=True)
            for name, tab in (('field', 0), ('coefficients', 1)):
                view.tabs.selected_index = tab
                picture = bytes(view.pictures[tab].value)
                assert picture.startswith(b'\x89PNG\r\n\x1a\n') and len(picture) > 10000
                (figures/(name+'.png')).write_bytes(picture)
            view.sweep.selected_index = 0
            (figures/'thickness.png').write_bytes(bytes(view.pictures[2].value))
        record = dict(notebook=NOTEBOOK, code_cells=4, additional_thickness_solves=solves,
                      thickness_cases=7, initial_solution_reused=True, max_fft_error=fft_error,
                      order1_exit_differences=errors, minimum_sampled_difference_thickness=float(cases.thicknesses[3]),
                      controls_and_reset='passed; precomputed solutions reused',
                      max_energy_balance_error=float(max(abs(s.energy_balance-1) for s in cases.solutions)))
    finally:
        if 'absorber_comparison' in namespace:
            namespace['absorber_comparison'].close()

    tests = private_test_directory(ROOT)
    result = unittest.TestResult()
    if tests is not None:
        suite = unittest.TestSuite()
        for filename in ('test_amplitude_grating.py', 'test_absorbing_grating_view.py'):
            spec = importlib.util.spec_from_file_location(filename[:-3], tests/filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        stream = StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
        print(stream.getvalue())
        if not result.wasSuccessful():
            raise RuntimeError('Ideal-mask comparison numerical/widget checks failed.')
    scripts = []
    with redirect_stdout(StringIO()):
        for script in sorted((ROOT/'Wave_solver_py_scripts/scripts/chapter_0').glob('0_7_*.py')):
            runpy.run_path(str(script), run_name='__validation__')
            scripts.append(script.name)
    assert len(scripts) == 4
    record.update(physics_and_widget_tests=result.testsRun, tests='passed', companion_scripts=scripts,
                  browser_ui='not covered by backend checks')
    (ROOT/'book/masks-validation.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    verify()
