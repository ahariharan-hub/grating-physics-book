"""Verify the local preview and widget comm traffic without exposing its token.

This is a protocol/backend check, not a replacement for a browser UI test.
"""
import asyncio
import json
from pathlib import Path
import uuid
import urllib.request

import nbformat
from tornado.httpclient import HTTPRequest
from tornado.websocket import websocket_connect

ROOT = Path(__file__).resolve().parents[1]
STATE = json.loads((ROOT / '_build/local-runtime.json').read_text())
HEADERS = {'Authorization': 'token ' + STATE['token'], 'Origin': 'http://localhost:3010'}


def api(path, method='GET', body=None):
    headers = dict(HEADERS)
    if body is not None:
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request('http://127.0.0.1:8897' + path,
                data=json.dumps(body).encode() if body is not None else None, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.headers.get('Access-Control-Allow-Origin') == HEADERS['Origin']
        data = response.read()
        return json.loads(data) if data else None


async def verify():
    with urllib.request.urlopen('http://localhost:3010/', timeout=10) as response:
        assert response.status == 200 and b'Grating Physics' in response.read()
    specifications = api('/api/kernelspecs')
    assert 'grating-book' in specifications['kernelspecs']['python3']['spec']['argv'][0]
    session = api('/api/sessions', 'POST', {'name': 'pilot-validation',
                  'path': '0_1_Scalar_Fields_and_Helmholtz.ipynb', 'type': 'notebook', 'kernel': {'name': 'python3'}})
    kernel_id = session['kernel']['id']
    socket = None
    try:
        socket = await websocket_connect(HTTPRequest(
            f'ws://127.0.0.1:8897/api/kernels/{kernel_id}/channels?session_id={uuid.uuid4().hex}', headers=HEADERS))
        notebook = nbformat.read(ROOT / '0_1_Scalar_Fields_and_Helmholtz.ipynb', as_version=4)
        source = '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'code')
        source += '\nlesson = scalar_theory_lessons["helmholtz"]\nlesson.ratio.value = 1.25\nprint("PILOT_RATIO", lesson.ratio.value)\n'
        message_id = uuid.uuid4().hex
        socket.write_message(json.dumps(dict(header=dict(msg_id=message_id, username='pilot', session=uuid.uuid4().hex,
                             msg_type='execute_request', version='5.3'), parent_header={}, metadata={}, channel='shell',
                             content=dict(code=source, silent=False, store_history=False, user_expressions={}, allow_stdin=False, stop_on_error=True))))
        streams = []
        comms = 0
        replied = False
        for _ in range(1000):
            frame = await asyncio.wait_for(socket.read_message(), timeout=60)
            if frame is None:
                raise RuntimeError('Kernel websocket closed before validation completed.')
            if isinstance(frame, bytes):
                comms += 1  # Widget image-buffer messages use binary websocket frames.
                continue
            message = json.loads(frame)
            kind = message.get('msg_type', message.get('header', {}).get('msg_type'))
            if kind in ('comm_open', 'comm_msg'):
                comms += 1
            if kind == 'stream':
                streams.append(message['content']['text'])
            if kind == 'error':
                raise RuntimeError(message['content']['ename'] + ': ' + message['content']['evalue'])
            if kind == 'execute_reply' and message['parent_header'].get('msg_id') == message_id:
                assert message['content']['status'] == 'ok'
                replied = True
            if replied and kind == 'status' and message['content'].get('execution_state') == 'idle':
                break
        assert replied and comms > 0 and 'PILOT_RATIO 1.25' in ''.join(streams)
        record = dict(preview_http='passed', authenticated_server='passed', cors='passed',
                      kernel_environment='grating-book', notebook_execution='passed', widget_trait_change=1.25,
                      widget_comm_frames=comms, browser_ui='unverified: computer-use URL policy block')
        (ROOT / 'book/local-validation.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(record, indent=2))
    finally:
        if socket is not None:
            socket.close()
        api('/api/sessions/' + session['id'], 'DELETE')


if __name__ == '__main__':
    asyncio.run(verify())
