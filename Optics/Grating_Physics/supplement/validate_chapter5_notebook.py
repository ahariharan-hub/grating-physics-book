"""Execute one notebook's cells in a clean process; save temporary QA images.

Usage: python supplement/validate_chapter5_notebook.py notebook.ipynb
No saved notebook outputs are changed. This is headless Python/widget execution,
not a browser or Jupyter-kernel rendering test.
"""
from pathlib import Path
import json
import sys
import time
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import IPython.display as ipd
out=root/'supplement'/'chapter5_qa'/Path(sys.argv[1]).stem
out.mkdir(parents=True,exist_ok=True)
count=[0]
def save_image(data):
    count[0]+=1
    (out/f'figure_{count[0]:02}.png').write_bytes(bytes(data))
def display(*objects,**kwargs):
    for obj in objects:
        if isinstance(obj,ipd.Image):save_image(obj.data)
        elif getattr(obj,'format',None)=='png' and getattr(obj,'value',None):save_image(obj.value)
        elif hasattr(obj,'children'):display(*obj.children)
def show():
    for num in plt.get_fignums():
        count[0]+=1
        plt.figure(num).savefig(out/f'figure_{count[0]:02}.png',dpi=110)
    plt.close('all')
plt.show=show
ipd.display=display
nb=json.loads((root/sys.argv[1]).read_text(encoding='utf-8'))
ns={'__name__':'__main__'}
start=time.perf_counter()
cells=0
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        cells+=1
        exec(compile(''.join(c['source']),f'{sys.argv[1]}:cell-{i}','exec'),ns)
# Keep standalone interactive Plotly figures available for browser inspection.
for obj in list(ns.values()):
    figures=getattr(obj,'_figures',None)
    if isinstance(figures,dict):
        for key,fig in figures.items():
            if '--render-plotly' in sys.argv and hasattr(fig,'write_image'):
                fig.write_image(out/f'{key}.png',width=1400,height=850,scale=1)
    if type(obj).__name__.endswith('Explorer') and hasattr(obj,'close'):obj.close()
result=dict(notebook=sys.argv[1],code_cells=cells,figures=count[0],seconds=round(time.perf_counter()-start,2),passed=True)
(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
