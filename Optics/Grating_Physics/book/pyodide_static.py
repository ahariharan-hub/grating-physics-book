"""Render saved widget figures as ordinary static notebook outputs.

Only generated notebook copies are changed. Live cell execution subsequently
produces the original widgets, using the unchanged scientific code.
"""
import base64
import copy
import math
import json
from pathlib import Path
from urllib.parse import urlsplit

import nbformat
import numpy as np


WIDGET_VIEW = 'application/vnd.jupyter.widget-view+json'
WIDGET_STATE = 'application/vnd.jupyter.widget-state+json'
PLOTLY_MIME = 'application/vnd.plotly.v1+json'


def arrays(value):
    if isinstance(value, dict):
        if isinstance(value.get('buffer'), bytes) and 'dtype' in value:
            return arrays(np.frombuffer(value['buffer'], dtype=value['dtype']).reshape(value['shape']).tolist())
        if isinstance(value.get('bdata'), str) and 'dtype' in value:
            data = np.frombuffer(base64.b64decode(value['bdata']), dtype=value['dtype'])
            shape = value.get('shape')
            if shape:
                data = data.reshape(tuple(map(int, shape.split(','))) if isinstance(shape, str) else shape)
            return arrays(data.tolist())
        return {key: arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [arrays(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def static_outputs(notebook):
    models = notebook.metadata.get('widgets', {}).get(WIDGET_STATE, {}).get('state', {})
    figures = notebook.metadata.get('browser_plotly_figures', {})
    snapshots = 0

    def render(model_id, seen):
        nonlocal snapshots
        if model_id in seen or model_id not in models:
            return []
        seen.add(model_id)
        model = models[model_id]
        state = copy.deepcopy(model.get('state', {}))
        for buffer in model.get('buffers', []):
            # Plotly also saves transient transport messages. Only the image
            # bytes and figure data/layout belong to the static snapshot.
            if buffer['path'][0] not in ('value', '_widget_data', '_widget_layout'):
                continue
            target = state
            for part in buffer['path'][:-1]:
                target = target[part]
            target[buffer['path'][-1]] = base64.b64decode(buffer['data'])
        kind = model.get('model_name')
        data = None
        if kind == 'ImageModel' and state.get('value'):
            image = state['value']
            format = state.get('format', 'png')
            data = {'image/svg+xml': image.decode('utf-8')} if format == 'svg' else {
                'image/' + ('jpeg' if format == 'jpg' else format): base64.b64encode(image).decode()}
            snapshots += 1
        elif kind == 'AnyModel' and '_widget_data' in state:
            if model_id not in figures:
                raise RuntimeError('Regenerate native outputs to capture the final Plotly figure: ' + model_id)
            figure = arrays(copy.deepcopy(figures[model_id]))
            figure['config'] = {'responsive': True}
            data = {PLOTLY_MIME: figure}
            snapshots += 1
        elif kind in ('HTMLModel', 'HTMLMathModel') and state.get('value'):
            data = {'text/html': state['value']}
        if data:
            return [nbformat.v4.new_output('display_data', data=data, metadata={})]
        if kind == 'OutputModel':
            return copy.deepcopy(state.get('outputs', []))
        children = state.get('children', [])
        if kind in ('TabModel', 'AccordionModel', 'StackModel'):
            index = state.get('selected_index')
            children = children[index:index+1] if isinstance(index, int) else []
        result = []
        for child in children:
            if isinstance(child, str) and child.startswith('IPY_MODEL_'):
                result.extend(render(child[len('IPY_MODEL_'):], seen))
        return result

    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        outputs = []
        for output in cell.get('outputs', []):
            view = output.get('data', {}).get(WIDGET_VIEW)
            if view:
                outputs.extend(render(view['model_id'], set()))
            else:
                outputs.append(output)
        cell.outputs = outputs
    notebook.metadata.pop('widgets', None)
    notebook.metadata.pop('browser_plotly_figures', None)
    return snapshots


def check_exported_figures(stage, html, notebooks, base_url=''):
    """Check the actual referenced MIME assets, including cached export mistakes."""
    stage, html = Path(stage), Path(html).resolve()
    exported = {}

    def visit(value):
        if isinstance(value, dict):
            asset = value.get(PLOTLY_MIME)
            if isinstance(asset, dict) and 'path' in asset:
                path = urlsplit(asset['path']).path
                if base_url and path.startswith(base_url + '/'):
                    path = path[len(base_url):]
                file = (html / path.lstrip('/')).resolve()
                if html not in file.parents:
                    raise RuntimeError('Unexpected Plotly output asset: ' + str(file))
                exported[file] = json.loads(file.read_text(encoding='utf-8'))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for page in html.glob('*.json'):
        visit(json.loads(page.read_text(encoding='utf-8')))
    checked = 0
    for name in notebooks:
        notebook = nbformat.read(stage / name, as_version=4)
        for cell in notebook.cells:
            for output in cell.get('outputs', []):
                figure = output.get('data', {}).get(PLOTLY_MIME)
                if figure is not None:
                    if not any(figure == candidate for candidate in exported.values()):
                        raise RuntimeError('Exported Plotly data is missing or stale: ' + name)
                    checked += 1
    return checked
