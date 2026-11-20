"""Verify notebook 0.5 scattering physics and real explorer callbacks."""
from contextlib import redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = '0_5_From_Helmholtz_BVP_to_Diffracted_Field.ipynb'


def verify():
    sys.path.insert(0, str(ROOT))
    namespace = {'__name__': '__scattering_validation__'}
    try:
        with redirect_stdout(StringIO()):
            notebook = json.loads((ROOT/NOTEBOOK).read_text(encoding='utf-8'))
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    exec(compile(''.join(cell['source']), f'{NOTEBOOK}:cell{index}', 'exec'), namespace)
        solution = namespace['grating_solution']
        assert solution.configuration.M == 24 and len(solution.orders) == 49
        balance_error = float(abs(solution.energy_balance-1))
        assert balance_error < 1e-10 and solution.absorption > 0
        examples = namespace['basis_examples']
        reference = examples[48].transmission[50]
        differences = {m: float(abs(s.transmission[m+2]-reference)/abs(reference)) for m, s in examples.items()}
        assert np.isclose(differences[2], .339, atol=.001) and differences[24] < .0003
        figures = ROOT/'_build/workflow/figures05'
        figures.mkdir(parents=True, exist_ok=True)
        controls = {}
        # Inspecting the fixed solution must not invoke either scattering solver.
        with (patch('wave_optics.scalar_grating_view.solve_scalar_grating', side_effect=AssertionError('unexpected solve')),
              patch('wave_optics.absorbing_grating.solve_absorbing_grating', side_effect=AssertionError('unexpected solve'))):
            for stage, view in namespace['grating_views'].items():
                picture = bytes(view.picture.value)
                assert picture.startswith(b'\x89PNG\r\n\x1a\n') and len(picture) > 10000
                (figures/(stage+'.png')).write_bytes(picture)
                for control in view._controls:
                    view.reset_button.click()
                    before = bytes(view.picture.value)
                    if control is view.material:
                        control.value = 'n'
                        np.testing.assert_allclose(view._artists['material_map'].get_array(), view.model.index.real)
                    elif control is view.quantity:
                        for quantity in ('real', 'amplitude', 'intensity', 'phase'):
                            control.value = quantity
                            np.testing.assert_allclose(view._artists['map'].get_array(), view.model.quantity(quantity)[0])
                        control.value = 'real'
                    elif control is view.orders:
                        assert tuple(control.options) == tuple(range(-7, 8))
                        control.value = (-3, 0, 2)
                        for line, m in zip(view._artists['harmonics_axis'].lines[:3], control.value):
                            np.testing.assert_allclose(line.get_xdata(), abs(view.model.harmonics[:, m+24]))
                    assert bytes(view.picture.value) != before, (stage, control.description)
                view.reset_button.click()
                assert all(c.value == v for c, v in view._defaults)
                assert view.model.solution is solution
                controls[stage] = len(view._controls)
        record = dict(notebook=NOTEBOOK, code_cells=7, reflected_power=solution.reflectance,
                      transmitted_power=solution.transmittance, volume_absorption=solution.absorption,
                      energy_balance_error=balance_error, eigen_residual=solution.eigen_residual,
                      relative_C2_difference=differences, control_callbacks=controls, reset_buttons='passed')
    finally:
        for view in namespace.get('grating_views', {}).values():
            view.close()

    suite = unittest.TestSuite()
    for filename in ('test_absorbing_grating.py', 'test_absorbing_lesson.py'):
        spec = importlib.util.spec_from_file_location(filename[:-3], ROOT/'supplement/tests'/filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if filename == 'test_absorbing_grating.py':
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        else:
            for name in ('test_complex_material_geometry_and_field_use_n_and_kappa',
                         'test_harmonics_and_reconstruction_are_from_absorbing_solution',
                         'test_absorbing_controls_dispose_without_leaking_widgets'):
                suite.addTest(module.AbsorbingLessonTests(name))
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(module.NotebookReloadTests))
    stream = StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise RuntimeError('Scattering numerical/widget checks failed.')
    scripts = []
    with redirect_stdout(StringIO()):
        for script in sorted((ROOT/'Wave_solver_py_scripts/scripts/chapter_0').glob('0_5_*.py')):
            runpy.run_path(str(script), run_name='__validation__')
            scripts.append(script.name)
    assert len(scripts) == 4
    record.update(physics_and_widget_tests=result.testsRun, tests='passed', companion_scripts=scripts,
                  browser_ui='not covered by backend checks')
    (ROOT/'book/scattering-validation.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    verify()
