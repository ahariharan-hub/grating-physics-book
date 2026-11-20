"""Verify 0.3 numerical results and real widget callbacks without a browser."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = '0_3_Boundary_Conditions_and_Diffraction.ipynb'


def verify():
    namespace = {'__name__': '__boundary_validation__'}
    try:
        with redirect_stdout(StringIO()):
            notebook = json.loads((ROOT / NOTEBOOK).read_text(encoding='utf-8'))
            # Execute source strings in memory; never write back to the notebook.
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    exec(compile(''.join(cell['source']), f'{NOTEBOOK}:cell{index}', 'exec'), namespace)
        # The slit exercise reuses k, so evaluate the superposition residual with pi.
        residual = float(np.max(abs(namespace['d2U'] + np.pi**2*namespace['U'])))
        mismatch = float(np.max(abs(namespace['amplitude_sum']-namespace['analytic_amplitude'])))
        assert residual < 1e-12 and mismatch < 1e-8
        np.testing.assert_allclose(abs(namespace['amplitude_sum'])**2,
                                   [1, 4/np.pi**2, 0, 4/(9*np.pi**2)], atol=1e-8)
        lessons = namespace['scalar_theory_lessons']
        changes = {}
        for stage, lesson in lessons.items():
            count = 0
            for control in lesson._controls:
                lesson.reset_button.click()
                before = bytes(lesson.picture.value)
                control.value = .25 if control is lesson.time else (control.min if control.value != control.min else control.max)
                after = bytes(lesson.picture.value)
                assert after.startswith(b'\x89PNG\r\n\x1a\n') and len(after) > 10000
                assert after != before, (stage, control.description)
                count += 1
            lesson.reset_button.click()
            assert all(c.value == v for c, v in lesson._defaults)
            changes[stage] = count
        lesson = lessons['superposition']
        lesson.amplitude.value = 0
        np.testing.assert_allclose(lesson.data['relative_intensity'], 1, atol=1e-14)
        lesson.amplitude.value = .5
        intensity = lesson.data['relative_intensity'].copy()
        signed = lesson.data['real_fields'].copy()
        lesson.time.value = .25
        np.testing.assert_array_equal(lesson.data['relative_intensity'], intensity)
        assert not np.allclose(lesson.data['real_fields'], signed)
        lesson = lessons['boundary']
        for phase in (0, 60):
            lesson.phase.value = phase
            for fraction in (0, .5, 1):
                lesson.fraction.value = fraction
                np.testing.assert_allclose(lesson.data['field'][0], np.exp(1j*np.deg2rad(phase)), atol=1e-14)
                np.testing.assert_allclose(lesson.data['derivative'][0],
                    1j*lesson.data['k']*(1-2*fraction)*np.exp(1j*np.deg2rad(phase)), atol=1e-14)
        lesson = lessons['diffraction']
        zeros = {}
        for width in (10, 20, 40):
            lesson.width.value = width
            zeros[width] = lesson.data['first_zero_mm']
            assert np.isclose(zeros[width], lesson.wavelength.value*1000/width)
            near = lesson.data['propagation']
            assert np.isfinite(near['field']).all()
            np.testing.assert_array_equal(near['field'][0], near['boundary_field'])
            np.testing.assert_allclose(near['intensity'], abs(near['field'])**2)
            assert np.isclose(lesson.data['relative_intensity'][600], 1)
        for lesson in lessons.values():
            lesson.reset_button.click()
            assert all(c.value == v for c, v in lesson._defaults)
        record = dict(notebook=NOTEBOOK, superposition_residual=residual,
                      slit_quadrature_error=mismatch, control_callbacks=changes,
                      first_zero_mm=zeros, boundary_trace_and_derivative='passed',
                      time_independent_intensity='passed', aperture_reconstruction='passed',
                      reset_buttons='passed', browser_ui='not covered by backend checks')
        (ROOT/'book/boundary-validation.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(record, indent=2))
    finally:
        for lesson in namespace.get('scalar_theory_lessons', {}).values():
            lesson.close()
        for name in ('superposition_check_panel', 'slit_check_panel'):
            if name in namespace:
                namespace['_close_widget_tree'](namespace[name])


if __name__ == '__main__':
    verify()
