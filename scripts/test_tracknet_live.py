"""Check wire decoding without loading the model or opening a camera."""
import asyncio
import struct
import sys
import types
import numpy as np
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

seen = []
def detect(model, windows):
    seen.extend(int(frame[0, 0]) for frame in windows[0])
    return [(0.4, 0.6, 0.9, True)]

# The original service keeps its own CORS rule, here one naming only port 5173.
original = FastAPI()
original.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:5173'],
                        allow_methods=['GET', 'POST', 'DELETE'])
@original.post('/analyze')
async def analyze():
    return {'job_id': 'a' * 32}

sys.modules['tracknet_service'] = types.SimpleNamespace(
    app=original, _load_model=lambda: object(), _detect_batch=detect,
    _active_backend='test', _active_device='test')
import tracknet_live

class Socket:
    def __init__(self, packet):
        self.packet = packet
        self.messages = []
    async def accept(self): pass
    async def close(self): pass
    async def receive_bytes(self):
        if self.packet is None: raise WebSocketDisconnect()
        result, self.packet = self.packet, None
        return result
    async def send_json(self, message): self.messages.append(message)

async def test():
    packet = struct.pack('<dd', 42, 123456) + b''.join(bytes([i])*(512*288) for i in [8,9,10])
    socket = Socket(packet)
    await tracknet_live.live(socket)
    assert seen == [8,9,10]
    assert socket.messages[-1]['frame'] == 42
    assert socket.messages[-1]['timestamp'] == 123456
    assert socket.messages[-1]['visible'] is True
    invalid = Socket(b'bad')
    await tracknet_live.live(invalid)
    assert invalid.messages[-1]['type'] == 'error'
    print('PASS: service binary decoding, frame order, timestamps, invalid packet')

asyncio.run(test())

def test_local_pages():
    client = TestClient(tracknet_live.app)
    # Vite moves to 5174, 5175, ... when 5173 is taken.
    for origin in ['http://127.0.0.1:5173', 'http://127.0.0.1:5175', 'http://localhost:3000']:
        response = client.post('/analyze', headers={'origin': origin})
        assert response.status_code == 200
        assert response.headers['access-control-allow-origin'] == origin, origin
    preflight = client.options('/analyze', headers={
        'origin': 'http://127.0.0.1:5175', 'access-control-request-method': 'DELETE'})
    assert preflight.status_code == 200
    assert preflight.headers['access-control-allow-methods'] == 'DELETE'
    # Any other page is still the original service's call.
    foreign = client.post('/analyze', headers={'origin': 'https://example.com'})
    assert 'access-control-allow-origin' not in foreign.headers
    assert client.options('/analyze', headers={
        'origin': 'https://example.com', 'access-control-request-method': 'POST'}).status_code == 400
    with client.websocket_connect('/live/tracknet-fast', headers={'origin': 'http://127.0.0.1:5175'}) as socket:
        assert socket.receive_json()['type'] == 'ready'
        socket.send_bytes(b'bad')
        assert socket.receive_json()['type'] == 'error'
    print('PASS: local pages on any port, other origins left to the service, live socket')

test_local_pages()
