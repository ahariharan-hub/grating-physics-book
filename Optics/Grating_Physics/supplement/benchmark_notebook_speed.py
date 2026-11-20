"""Benchmark numbered optics lessons without modifying the notebooks."""
from pathlib import Path
import os
import sys
import json
import time
import statistics
import argparse
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('notebooks', nargs='*', help='Notebook paths relative to the project root')
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--output-dir', default='notebook_speed')
args = parser.parse_args()
if args.rounds < 1:
    parser.error('--rounds must be positive')
OUT = ROOT / 'output' / args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
# Keep temporary Jupyter state within the workspace and use this interpreter.
for key, folder in [('JUPYTER_CONFIG_DIR', 'config'), ('JUPYTER_DATA_DIR', 'data'), ('JUPYTER_RUNTIME_DIR', 'runtime'), ('IPYTHONDIR', 'ipython')]:
    destination = OUT / folder
    destination.mkdir(exist_ok=True)
    os.environ[key] = str(destination)
kernel_dir = OUT / 'data' / 'kernels' / 'optics-benchmark'
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Optics benchmark', 'language': 'python'
}), encoding='utf-8')
os.environ['MPLBACKEND'] = 'Agg'
import nbformat
from nbclient import NotebookClient

FILES = [
    'supplement/0_Foundations_of_Scalar_Optics_Original.ipynb',
    '0_1_Scalar_Fields_and_Helmholtz.ipynb',
    '0_2_Solutions_of_the_Helmholtz_Equation.ipynb',
    '0_3_Boundary_Conditions_and_Diffraction.ipynb',
    'supplement/1_Plane_Waves_and_Wavevectors_Original.ipynb',
    '1_1_Oscillation_and_Wavelength.ipynb',
    '1_2_Ideal_Plane_Waves_versus_Finite_Optical_Beams.ipynb',
    '1_3_Wavefronts_and_Wavevectors.ipynb',
    '1_4_Plane_Waves_in_Three_Dimensions.ipynb',
]

FILES = args.notebooks or FILES

def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))

results = {'interpreter': sys.executable, 'method': f'{args.rounds} fresh kernel(s) per notebook; default demonstrations; headless widget execution, no browser rendering', 'runs': []}
for repetition in range(args.rounds):
    # Reverse order on alternating rounds to reduce ordering bias.
    for name in (FILES if repetition % 2 == 0 else list(reversed(FILES))):
        notebook_path = ROOT / name
        nb = nbformat.read(notebook_path, as_version=4)
        saved_outputs = sum(len(json.dumps(c.get('outputs', []))) for c in nb.cells if c.cell_type == 'code')
        saved_widget_bytes = len(json.dumps(nb.metadata.get('widgets', {})))
        nbformat.validate(nb)
        client = NotebookClient(nb, timeout=180, kernel_name='optics-benchmark', resources={'metadata': {'path': str(ROOT)}}, record_timing=True, store_widget_state=True)
        start = time.perf_counter()
        try:
            client.execute()
            total = time.perf_counter() - start
            cells = []
            for number, cell in enumerate(nb.cells, 1):
                if cell.cell_type != 'code':
                    continue
                timings = cell.metadata.get('execution', {})
                begin, end = timings.get('iopub.status.busy'), timings.get('iopub.status.idle')
                seconds = (stamp(end)-stamp(begin)).total_seconds() if begin and end else None
                cells.append({'cell': number, 'seconds': seconds, 'source_first_line': cell.source.splitlines()[0] if cell.source else ''})
            result = {'file': name, 'round': repetition+1, 'wall_seconds': total, 'cell_seconds': sum(c['seconds'] or 0 for c in cells), 'cells': cells, 'file_bytes': notebook_path.stat().st_size, 'saved_output_bytes': saved_outputs, 'saved_widget_bytes': saved_widget_bytes, 'executed_output_bytes': sum(len(json.dumps(c.get('outputs', []))) for c in nb.cells if c.cell_type == 'code'), 'executed_widget_bytes': len(json.dumps(nb.metadata.get('widgets', {}))), 'ok': True}
        except Exception as error:
            result = {'file': name, 'round': repetition+1, 'wall_seconds': time.perf_counter()-start, 'ok': False, 'error': str(error)}
        results['runs'].append(result)
        (OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps({key: result[key] for key in ['file', 'round', 'wall_seconds', 'ok']}), flush=True)
summary = []
for name in FILES:
    runs = [r for r in results['runs'] if r['file'] == name and r['ok']]
    if runs:
        summary.append({'file': name, 'median_wall_seconds': statistics.median(r['wall_seconds'] for r in runs), 'median_cell_seconds': statistics.median(r['cell_seconds'] for r in runs), 'runs': len(runs), 'file_bytes': runs[0]['file_bytes'], 'saved_output_bytes': runs[0]['saved_output_bytes'], 'saved_widget_bytes': runs[0]['saved_widget_bytes'], 'executed_widget_bytes': runs[0]['executed_widget_bytes']})
results['summary'] = summary
(OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('DONE', flush=True)
