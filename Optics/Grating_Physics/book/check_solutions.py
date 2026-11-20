"""Check notebook 0.2 calculations and explorer callbacks outside the browser."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = '0_2_Solutions_of_the_Helmholtz_Equation.ipynb'


def verify():
    notebook = json.loads((ROOT / NOTEBOOK).read_text(encoding='utf-8'))
    namespace = {'__name__': '__solutions_validation__'}
    lesson = None
    try:
        with redirect_stdout(StringIO()):
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    exec(compile(''.join(cell['source']), f'{NOTEBOOK}:cell{index}', 'exec'), namespace)
        radial = float(np.max(np.abs(namespace['normalized_residual'])))
        cone = float(np.max(np.abs(namespace['laplacian_sum'] + namespace['k']**2 * namespace['cone_sum'])) / namespace['k']**2)
        amplitude = float(np.max(np.abs(np.abs(namespace['propagated']) - np.abs(namespace['cone_sum']))))
        assert max(radial, cone, amplitude) < 1e-12

        lesson = namespace['scalar_theory_lessons']['solutions']
        families = []
        for label, value in lesson.solution.options:
            lesson.solution.value = value
            picture = bytes(lesson.picture.value)
            assert picture.startswith(b'\x89PNG\r\n\x1a\n') and len(picture) > 10000
            families.append(value)
        assert len(families) == 8

        for family in ('plane', 'bessel', 'gaussian'):
            lesson.solution.value = family
            lesson.view.value = 'beam'
            before = bytes(lesson.picture.value)
            lesson.beam_axial.value = 2
            lesson.view.value = 'field'
            assert bytes(lesson.picture.value) != before
        lesson.solution.value = 'gaussian'
        before = bytes(lesson.picture.value)
        lesson.waist.value = 3
        lesson.axial.value = 1
        assert bytes(lesson.picture.value) != before
        lesson._reset()
        assert lesson.solution.value == 'plane' and lesson.view.value == 'field'
        assert lesson.waist.value == 2 and lesson.axial.value == 0

        record = dict(notebook=NOTEBOOK, radial_residual=radial, cone_residual=cone,
                      propagation_amplitude_change=amplitude, solution_families=families,
                      beam_views='passed: plane, bessel, gaussian',
                      gaussian_controls='passed', reset='passed',
                      browser_ui='not covered by this backend callback check')
        (ROOT / 'book/solutions-validation.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(record, indent=2))
    finally:
        if lesson is not None:
            lesson.close()


if __name__ == '__main__':
    verify()
