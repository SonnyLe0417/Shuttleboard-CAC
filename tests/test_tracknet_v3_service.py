"""V3 HTTP contract/lifecycle tests; no models, weights, GPU, or FFmpeg needed."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tracknet_v3_service as service


POINTS = [
    {"frame": 0, "time": 0.0, "x": 0.25, "y": 0.75, "confidence": 0.9, "visible": True},
    {"frame": 1, "time": 0.04, "x": None, "y": None, "confidence": 0.0, "visible": False},
]


class FakeEngine:
    backend = "mock-v3"
    device = "CPU"
    seq_len = 8

    def __init__(self):
        self.calls = []

    def analyze(self, video_path, output_dir, progress, cancelled, output_video=True):
        self.calls.append(video_path)
        assert output_video is True
        assert Path(video_path).read_bytes() == b"video-data"
        progress(0.5, 1, 2)
        return self.result(output_dir)

    @staticmethod
    def result(output_dir):
        target = Path(output_dir) / "prediction.mp4"
        target.write_bytes(b"encoded-video")
        return {"frames": 2, "coordinates": POINTS, "video_path": str(target), "metrics": {"fps": 20}}


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    pytest.fail("Timed out waiting for background service work")


def ready(client):
    return wait_for(lambda: client.get("/health").json()["model_loaded"])


def submit(client):
    response = client.post("/analyze", files={"video": ("rally.mp4", b"video-data", "video/mp4")})
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def terminal(client, job_id):
    def poll():
        job = client.get(f"/jobs/{job_id}").json()
        return job if job["status"] in service.TERMINAL else None
    return wait_for(poll)


@pytest.fixture
def running(tmp_path):
    engine = FakeEngine()
    app = service.create_app(data_root=tmp_path, engine_factory=lambda: engine)
    with TestClient(app) as client:
        ready(client)
        yield client, app, engine


def test_contract_and_completed_jobs_survive_restart(tmp_path):
    engine = FakeEngine()
    app = service.create_app(data_root=tmp_path, engine_factory=lambda: engine)
    with TestClient(app) as client:
        ready(client)
        health = client.get("/health").json()
        assert health | {"backend": "mock-v3", "device": "CPU", "frame_stack": 8} == health
        job_id = submit(client)
        job = terminal(client, job_id)
        assert job["status"] == "completed"
        assert job["progress"] == 1
        assert "input_file" not in job and "video_file" not in job
        result = client.get(f"/jobs/{job_id}/result").json()
        assert result == {
            "job_id": job_id, "frames": 2, "coordinates": POINTS,
            "video_url": f"/jobs/{job_id}/video", "metrics": {"fps": 20},
        }
        response = client.get(result["video_url"])
        assert response.content == b"encoded-video"
        assert response.headers["content-type"] == "video/mp4"
        assert client.delete(f"/jobs/{job_id}").json()["status"] == "completed"
        assert (tmp_path / job_id / "manifest.json").exists()
    # Even a model load failure must not prevent access to stored results.
    def broken_factory():
        raise RuntimeError("Weights unavailable")
    with TestClient(service.create_app(data_root=tmp_path, engine_factory=broken_factory)) as client:
        assert client.get(f"/jobs/{job_id}").json()["status"] == "completed"
        assert client.get(f"/jobs/{job_id}/result").json() == result
        assert client.get(result["video_url"]).content == b"encoded-video"


def test_uploaded_settings_are_passed_to_a_cached_engine_variant(tmp_path):
    engines = []

    class SettingsEngine(FakeEngine):
        def __init__(self, settings=None):
            super().__init__()
            self.settings = settings or {}
            engines.append(self)

    app = service.create_app(data_root=tmp_path, engine_factory=SettingsEngine)
    with TestClient(app) as client:
        ready(client)
        settings = {
            "backend": "openvino",
            "device": "CPU",
            "evalMode": "weight",
            "inpaint": False,
            "intraThreads": 8,
        }
        response = client.post(
            "/analyze",
            files={
                "video": ("rally.mp4", b"video-data", "video/mp4"),
                "settings": (None, json.dumps(settings), "application/json"),
            },
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert terminal(client, job_id)["status"] == "completed"
        assert engines[0].settings == {}
        assert engines[1].settings == settings
        second = client.post(
            "/analyze",
            files={
                "video": ("rally.mp4", b"video-data", "video/mp4"),
                "settings": (None, json.dumps(settings), "application/json"),
            },
        ).json()["job_id"]
        assert terminal(client, second)["status"] == "completed"
        assert len(engines) == 2


def test_job_result_keeps_direct_and_repaired_provenance_for_playback(tmp_path):
    points = [
        dict(POINTS[0], detected=True, repaired=False),
        dict(POINTS[1], x=0.4, y=0.6, visible=True, detected=False, repaired=True),
    ]
    class ProvenanceEngine(FakeEngine):
        @staticmethod
        def result(output_dir):
            return {**FakeEngine.result(output_dir), "coordinates": points}
    with TestClient(service.create_app(data_root=tmp_path, engine_factory=ProvenanceEngine)) as client:
        ready(client)
        job_id = submit(client)
        assert terminal(client, job_id)["status"] == "completed"
        result = client.get(f"/jobs/{job_id}/result").json()
        assert result["coordinates"] == points
        assert result["coordinates"][1]["confidence"] == 0
        assert client.get(result["video_url"]).content == b"encoded-video"


def test_background_loading_keeps_health_responsive_and_loads_once(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def factory():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return FakeEngine()
    app = service.create_app(data_root=tmp_path, engine_factory=factory)
    with TestClient(app) as client:
        try:
            assert entered.wait(2)
            start = time.monotonic()
            for _ in range(3):
                response = client.get("/health")
                assert response.status_code == 200
                assert response.json()["status"] == "loading"
            assert time.monotonic() - start < 2
            assert client.post("/analyze").status_code == 503
        finally:
            release.set()
        ready(client)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: app.state.service.load_model(), range(8)))
        assert calls == [1]
        terminal(client, submit(client))
        terminal(client, submit(client))
        assert calls == [1]


def test_model_load_error_is_reported_without_retries(tmp_path):
    calls = []
    def factory():
        calls.append(1)
        raise ImportError("Mock missing model dependency")
    app = service.create_app(data_root=tmp_path, engine_factory=factory)
    with TestClient(app) as client:
        wait_for(lambda: client.get("/health").json()["status"] == "error")
        for _ in range(3):
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["model_loaded"] is False
            assert "Mock missing model dependency" in health.json()["error"]
            assert client.post("/analyze").status_code == 503
            app.state.service.load_model()
        assert calls == [1]


@pytest.mark.parametrize("phase", ["inference", "encoding"])
def test_single_worker_and_cancellation_through_encoding(tmp_path, phase):
    entered, observed, release = threading.Event(), threading.Event(), threading.Event()
    class BlockingEngine(FakeEngine):
        def analyze(self, video_path, output_dir, progress, cancelled, output_video=True):
            self.calls.append(video_path)
            progress(0.99 if phase == "encoding" else 0.25, 2 if phase == "encoding" else 1, 2)
            entered.set()
            try:
                wait_for(cancelled)
                observed.set()
                assert release.wait(5)
                # Simulate an engine returning after cancellation raced with encoding.
                progress(1.0, 2, 2)
                return self.result(output_dir)
            finally:
                observed.set()
    engine = BlockingEngine()
    app = service.create_app(data_root=tmp_path, engine_factory=lambda: engine)
    with TestClient(app) as client:
        ready(client)
        first = submit(client)
        try:
            assert entered.wait(2)
            second = submit(client)
            assert client.get(f"/jobs/{first}").json()["status"] == "processing"
            assert client.get(f"/jobs/{second}").json()["status"] == "queued"
            assert len(engine.calls) == 1
            assert client.get("/health").json()["model_loaded"]
            assert client.get(f"/jobs/{first}/result").status_code == 409
            assert client.get(f"/jobs/{first}/video").status_code == 404
            for job_id in [second, first]:
                response = client.delete(f"/jobs/{job_id}")
                assert response.status_code == 200
                assert response.json()["status"] == "failed"
                assert response.json()["cancelled"] is True
                assert client.delete(f"/jobs/{job_id}").json() == response.json()
            assert observed.wait(2)
        finally:
            release.set()
        client.portal.call(app.state.service.queue.join)
        assert len(engine.calls) == 1
        for job_id in [first, second]:
            assert client.get(f"/jobs/{job_id}/result").status_code == 409
            assert client.get(f"/jobs/{job_id}/video").status_code == 404
            saved = json.loads((tmp_path / job_id / "manifest.json").read_text())
            assert saved["status"] == "failed" and saved["cancelled"] is True


def test_shutdown_requests_cancellation_and_preserves_failed_manifest(tmp_path):
    entered, observed = threading.Event(), threading.Event()
    class Engine(FakeEngine):
        def analyze(self, video_path, output_dir, progress, cancelled, output_video=True):
            entered.set()
            wait_for(cancelled)
            observed.set()
            raise RuntimeError("cancelled while encoding")
    app = service.create_app(data_root=tmp_path, engine_factory=Engine)
    with TestClient(app) as client:
        ready(client)
        job_id = submit(client)
        assert entered.wait(2)
    assert observed.is_set()
    saved = json.loads((tmp_path / job_id / "manifest.json").read_text())
    assert saved["status"] == "failed"
    assert "shutdown" in saved["error"]


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_restart_fails_interrupted_jobs(tmp_path, status):
    job_id = "a" * 32
    directory = tmp_path / job_id
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({
        "job_id": job_id, "status": status, "progress": 0.3, "frames": 3,
    }))
    with TestClient(service.create_app(data_root=tmp_path, engine_factory=FakeEngine)) as client:
        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "failed" and "restart" in job["error"]
        assert client.get(f"/jobs/{job_id}/result").status_code == 409
    assert json.loads((directory / "manifest.json").read_text())["status"] == "failed"


def test_corrupt_manifest_does_not_break_health(tmp_path):
    directory = tmp_path / ("a" * 32)
    directory.mkdir()
    (directory / "manifest.json").write_text("{broken json")
    with TestClient(service.create_app(data_root=tmp_path, engine_factory=FakeEngine)) as client:
        assert client.get("/health").status_code == 200
        assert client.get(f"/jobs/{directory.name}").json()["status"] == "failed"


@pytest.mark.parametrize("job_id", ["bad", "A" * 32, "a" * 31, "a" * 33, "a" * 32 + ".json", "..%5Csecret"])
@pytest.mark.parametrize("method,suffix", [("GET", ""), ("DELETE", ""), ("GET", "/result"), ("GET", "/video")])
def test_strict_job_ids(running, job_id, method, suffix):
    client, _, _ = running
    assert client.request(method, f"/jobs/{job_id}{suffix}").status_code == 400


def test_unknown_job_ids(running):
    client, _, _ = running
    for method, suffix in [("GET", ""), ("DELETE", ""), ("GET", "/result"), ("GET", "/video")]:
        assert client.request(method, f"/jobs/{'f' * 32}{suffix}").status_code == 404


@pytest.mark.parametrize("endpoint", ["/analyze", "/preview"])
def test_upload_limit_and_cleanup(tmp_path, endpoint):
    app = service.create_app(data_root=tmp_path, engine_factory=FakeEngine, max_upload_bytes=8)
    with TestClient(app) as client:
        ready(client)
        response = client.post(endpoint, files={"video": ("x.mp4", b"123456789", "video/mp4")})
        assert response.status_code == 413
        assert list(tmp_path.iterdir()) == []
        assert not app.state.service.jobs


def multipart(data=b"video-data", close=True):
    return (b'--boundary\r\nContent-Disposition: form-data; name="video"; filename="../../evil.mp4"\r\n'
            b'Content-Type: video/mp4\r\n\r\n' + data + (b"\r\n--boundary--\r\n" if close else b""))


def test_streamed_upload_without_content_length_stops_reading_at_limit(tmp_path):
    async def scenario():
        app = service.create_app(data_root=tmp_path, engine_factory=FakeEngine, max_upload_bytes=16)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                while not (await client.get("/health")).json()["model_loaded"]:
                    await asyncio.sleep(0.01)
                consumed = []
                async def body():
                    yield multipart(b"", close=False)
                    for _ in range(100):
                        consumed.append(1)
                        yield b"x" * 10
                    yield b"\r\n--boundary--\r\n"
                response = await client.post("/analyze", content=body(), headers={
                    "content-type": "multipart/form-data; boundary=boundary",
                })
                assert response.status_code == 413
                assert len(consumed) < 100
                assert not list(tmp_path.iterdir())
    asyncio.run(scenario())


def test_split_multipart_headers_and_filename_sanitization(running):
    client, app, _ = running
    body = multipart()
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            async def chunks():
                for byte in body:
                    yield bytes([byte])
            return await http.post("/analyze", content=chunks(), headers={
                "content-type": "multipart/form-data; boundary=boundary",
            })
    # Use the application's event loop for its asyncio queue.
    response = client.portal.call(scenario)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert terminal(client, job_id)["status"] == "completed"
    assert (app.state.service.root / job_id / "input.mp4").is_file()
    assert not (app.state.service.root / "evil.mp4").exists()


@pytest.mark.parametrize("kind", ["empty", "truncated", "wrong-field", "wrong-type", "duplicate", "no-boundary", "oversized-header"])
def test_malformed_uploads_leave_no_jobs(running, kind):
    client, app, _ = running
    body = multipart()
    headers = {"content-type": "multipart/form-data; boundary=boundary"}
    if kind == "empty":
        body = multipart(b"")
    elif kind == "truncated":
        body = multipart(close=False)
    elif kind == "wrong-field":
        body = body.replace(b'name="video"', b'name="other"')
    elif kind == "wrong-type":
        body = body.replace(b"video/mp4", b"text/plain")
    elif kind == "duplicate":
        body = body.replace(b"--boundary--\r\n", multipart())
    elif kind == "no-boundary":
        headers["content-type"] = "multipart/form-data"
    elif kind == "oversized-header":
        body = body.replace(b"../../evil.mp4", b"x" * 17000)
    assert client.post("/analyze", content=body, headers=headers).status_code in {400, 413, 415}
    assert not app.state.service.jobs
    assert not list(app.state.service.root.iterdir())


def test_engine_failure_does_not_stop_worker(tmp_path):
    class FlakyEngine(FakeEngine):
        def analyze(self, *args, **kwargs):
            if not self.calls:
                self.calls.append("failed")
                raise RuntimeError("Bad input video")
            return super().analyze(*args, **kwargs)
    with TestClient(service.create_app(data_root=tmp_path, engine_factory=FlakyEngine)) as client:
        ready(client)
        failed = terminal(client, submit(client))
        assert failed["status"] == "failed" and "Bad input" in failed["error"]
        assert terminal(client, submit(client))["status"] == "completed"


def test_output_path_cannot_escape_job_directory(tmp_path):
    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"private")
    class EscapingEngine(FakeEngine):
        def analyze(self, *args, **kwargs):
            return {"video_path": str(outside), "frames": 0, "coordinates": []}
    with TestClient(service.create_app(data_root=tmp_path, engine_factory=EscapingEngine)) as client:
        ready(client)
        job_id = submit(client)
        assert terminal(client, job_id)["status"] == "failed"
        assert client.get(f"/jobs/{job_id}/video").status_code == 404


def test_preview_works_without_model_and_cleans_files(tmp_path, monkeypatch):
    def broken_factory():
        raise RuntimeError("No model")
    def convert(source, target, cancelled, **kwargs):
        assert source.read_bytes() == b"video-data"
        assert not cancelled()
        target.write_bytes(b"browser-video")
    monkeypatch.setattr(service, "_make_browser_compatible_video", convert)
    with TestClient(service.create_app(data_root=tmp_path, engine_factory=broken_factory)) as client:
        response = client.post("/preview", files={"video": ("x.avi", b"video-data", "video/avi")})
        assert response.status_code == 200 and response.content == b"browser-video"
        assert response.headers["content-type"] == "video/mp4"
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("error,status", [(FileNotFoundError(), 503), (RuntimeError("Invalid video"), 422), (TimeoutError("Timeout"), 504)])
def test_preview_errors_are_explicit_and_cleaned(running, monkeypatch, error, status):
    client, app, _ = running
    def convert(*args, **kwargs):
        raise error
    monkeypatch.setattr(service, "_make_browser_compatible_video", convert)
    response = client.post("/preview", files={"video": ("x.avi", b"video-data", "video/avi")})
    assert response.status_code == status
    assert not list(app.state.service.root.iterdir())


@pytest.mark.parametrize("reason", ["cancel", "timeout"])
def test_ffmpeg_is_terminated_during_encoding(tmp_path, monkeypatch, reason):
    cancel = threading.Event()
    class Process:
        returncode = None
        terminated = False
        killed = False
        waits = 0
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            self.waits += 1
            if self.killed:
                self.returncode = -9
                return -9
            cancel.set()
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        def terminate(self):
            self.terminated = True
        def kill(self):
            self.killed = True
    process = Process()
    monkeypatch.setattr(service.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(service.ConversionCancelled if reason == "cancel" else TimeoutError):
        service._make_browser_compatible_video(
            tmp_path / "input.avi", tmp_path / "output.mp4", cancel.is_set,
            timeout=600 if reason == "cancel" else 0,
        )
    assert process.terminated and process.killed
    assert process.returncode == -9


def test_live_is_explicitly_unsupported(running):
    client, _, _ = running
    with client.websocket_connect("/live/tracknet-fast") as websocket:
        message = websocket.receive_json()
        assert message["type"] == "error"
        assert "current engine" in message["error"]
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()
        assert closed.value.code == 1008


def test_cors_supports_cancel(running):
    client, _, _ = running
    response = client.options("/jobs/" + "a" * 32, headers={
        "origin": "http://localhost:5173", "access-control-request-method": "DELETE",
    })
    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]


def upload_from(client, origin, path="/analyze"):
    return client.post(path, headers={"origin": origin}, files={"video": ("rally.mp4", b"video-data", "video/mp4")})


def test_pages_from_this_machine_may_use_any_port(running):
    # Vite moves to 5174, 5175, ... when 5173 is taken.
    client, _, _ = running
    for origin in ["http://127.0.0.1:5175", "http://localhost:3000", "http://[::1]:5173", "http://localhost"]:
        response = upload_from(client, origin)
        assert response.status_code == 202, origin
        assert response.headers["access-control-allow-origin"] == origin
    preflight = client.options("/jobs/" + "a" * 32, headers={
        "origin": "http://127.0.0.1:5175", "access-control-request-method": "DELETE",
    })
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://127.0.0.1:5175"


def test_a_page_from_elsewhere_cannot_start_work_it_cannot_see(running):
    # The browser would hide the answer, so an accepted upload would analyse
    # a whole video for nobody, again on every retry.
    client, app, _ = running
    for origin in ["https://example.com", "http://localhost.example.com", "http://127.0.0.1.example.com:5173"]:
        response = upload_from(client, origin)
        assert response.status_code == 403, origin
        assert "access-control-allow-origin" not in response.headers
        assert upload_from(client, origin, "/preview").status_code == 403
    assert not app.state.service.jobs
    assert submit(client)  # scripts and the launcher send no origin


def test_configured_origins_are_allowed_too(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACKNET_ALLOW_ORIGINS", "http://192.168.1.20:5173, https://court.example")
    app = service.create_app(data_root=tmp_path, engine_factory=FakeEngine)
    with TestClient(app) as client:
        ready(client)
        for origin in ["http://192.168.1.20:5173", "https://court.example", "http://127.0.0.1:5175"]:
            response = upload_from(client, origin)
            assert response.status_code == 202, origin
            assert response.headers["access-control-allow-origin"] == origin
        assert upload_from(client, "http://192.168.1.21:5173").status_code == 403
    monkeypatch.setenv("TRACKNET_ALLOW_ORIGINS", "*")
    with TestClient(service.create_app(data_root=tmp_path, engine_factory=FakeEngine)) as client:
        ready(client)
        assert upload_from(client, "http://192.168.1.21:5173").status_code == 202
