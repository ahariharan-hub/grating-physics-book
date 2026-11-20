"""Execute 0.4 in memory and run its numerical and widget regression checks."""
from contextlib import redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = '0_4_Propagation_of_Scalar_Optical_Fields.ipynb'


def verify():
    sys.path.insert(0, str(ROOT))
    namespace = {'__name__': '__propagation_validation__'}
    try:
        with redirect_stdout(StringIO()):
            notebook = json.loads((ROOT/NOTEBOOK).read_text(encoding='utf-8'))
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    exec(compile(''.join(cell['source']), f'{NOTEBOOK}:cell{index}', 'exec'), namespace)
        error = float(np.max(abs(namespace['try_rebuilt']-namespace['try_u0'])))
        assert error < 1e-14
        np.testing.assert_allclose(namespace['try_H'], np.exp(2j*np.pi*.8), atol=1e-14)
        for lesson in namespace['propagation_lessons'].values():
            assert bytes(lesson.picture.value).startswith(b'\x89PNG\r\n\x1a\n')
        # Release the notebook's explorers before constructing test instances.
    finally:
        for lesson in namespace.get('propagation_lessons', {}).values():
            lesson.close()
        for panel in namespace.get('propagation_panels', {}).values():
            namespace['_close_widget_tree'](panel)

    suite = unittest.TestSuite()
    for filename in ('test_scalar_propagation.py', 'test_scalar_propagation_view.py',
                     'test_scalar_propagation_lesson.py'):
        spec = importlib.util.spec_from_file_location(filename[:-3], ROOT/'supplement/tests'/filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    stream = StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise RuntimeError('Propagation numerical/widget checks failed.')

    from wave_optics.scalar_theory import slit_propagation
    padding_error = sampling_error = 0.
    for width, wavelength in ((10, .4), (10, .8), (40, .4), (40, .8)):
        options = dict(width_um=width, wavelength_um=wavelength, distances_um=[0, 2, 20, 200, 1000])
        base = slit_propagation(**options)['intensity']
        padding_error = max(padding_error, float(np.max(abs(base-slit_propagation(**options, window_um=8192)['intensity']))))
        sampling_error = max(sampling_error, float(np.max(abs(base-slit_propagation(**options, dx_um=.03125)['intensity']))))
    assert padding_error < .006 and sampling_error < .004
    scripts = []
    with redirect_stdout(StringIO()):
        for script in sorted((ROOT/'Wave_solver_py_scripts/scripts/chapter_0').glob('0_4_*.py')):
            runpy.run_path(str(script), run_name='__validation__')
            scripts.append(script.name)
    assert len(scripts) == 4
    record = dict(notebook=NOTEBOOK, code_cells=8, fft_roundtrip_error=error,
                  physics_and_widget_tests=result.testsRun, tests='passed',
                  padding_intensity_error=padding_error, sampling_intensity_error=sampling_error,
                  companion_scripts=scripts,
                  browser_ui='not covered by backend checks')
    (ROOT/'book/propagation-validation.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    verify()
