"""ShuttleBoard live transport; uploaded-video routes stay in the original service."""
import asyncio
import os
import sys
import struct
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("TRACKNET_DIR", str(Path(__file__).resolve().parents[2] / "TrackNet-Badminton-Tracking-tensorflow2-main")))
import tracknet_service as service
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

from local_pages import AllowLocalPages

app = service.app
# The original service's CORS list may name only port 5173; the app's dev
# server may be on another port.
app.add_middleware(AllowLocalPages)
live_lock = asyncio.Lock()


@app.websocket("/live/tracknet-fast")
async def live(websocket: WebSocket):
    await websocket.accept()
    try:
        model = service._load_model()
        await websocket.send_json(dict(type="ready", backend=service._active_backend,
                                      device=service._active_device, frame_stack=3))
        while True:
            packet = await websocket.receive_bytes()
            if len(packet) != 16 + 3 * 512 * 288:
                raise ValueError("Invalid live frame group")
            frame, timestamp = struct.unpack_from("<dd", packet)
            if not np.isfinite(frame) or not np.isfinite(timestamp):
                raise ValueError("Invalid frame time")
            window = np.frombuffer(packet, dtype=np.uint8, offset=16).reshape(3, 288, 512)
            start = time.perf_counter()
            async with live_lock:
                detection = await asyncio.to_thread(service._detect_batch, model, [list(window)])
            x, y, confidence, visible = detection[0]
            await websocket.send_json(dict(type="detection", frame=frame, timestamp=timestamp,
                x=x, y=y, confidence=confidence, visible=visible,
                processing_ms=(time.perf_counter()-start)*1000,
                backend=service._active_backend, device=service._active_device))
    except WebSocketDisconnect:
        pass
    except Exception as error:
        try:
            await websocket.send_json(dict(type="error", error=str(error)))
            await websocket.close()
        except RuntimeError:
            pass
