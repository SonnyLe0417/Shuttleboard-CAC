"""Independent local V3 API. Run with one uvicorn process/worker.

Model imports and initialization happen in the lifespan's background worker.
TRACKNET_DATA_DIR, TRACKNET_MAX_UPLOAD_BYTES (default 1 GiB),
TRACKNET_ALLOW_ORIGINS (pages beyond this machine's own, which may always use
it), TRACKNET_FFMPEG and TRACKNET_PREVIEW_TIMEOUT configure the service;
V3Engine owns the remaining TRACKNET_* model settings. Live V3
inference is deliberately unsupported: use the current engine for live play.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from contextlib import asynccontextmanager
import importlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable
import uuid

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from starlette.background import BackgroundTask

from local_pages import LOCAL_ORIGIN


LOG = logging.getLogger(__name__)
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
TERMINAL = {"completed", "failed"}
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / ".tracknet-v3" / "service_data"


def _engine_factory(settings=None):
    name = f"{__package__}.tracknet_v3" if __package__ else "tracknet_v3"
    return importlib.import_module(name).V3Engine(settings)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _job_id(value: str) -> str:
    if not JOB_ID.fullmatch(value):
        raise HTTPException(400, "Invalid job id; expected 32 lowercase hexadecimal characters.")
    return value


def _contained_file(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Job output is missing or outside its job directory.")
    return path


class Service:
    def __init__(self, data_root: Path, engine_factory: Callable[[], Any]):
        self.root = data_root.resolve()
        self.engine_factory = engine_factory
        self.engine = None
        self.engine_variants: OrderedDict[str, Any] = OrderedDict()
        self.model_error: str | None = None
        self.model_attempted = False
        self.model_lock = threading.Lock()
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.cancellations: dict[str, threading.Event] = {}
        self.stopping = threading.Event()
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracknet-v3")
        self.preview_gate = asyncio.Semaphore(1)

    def persist(self, job_id: str) -> None:
        # All callers hold lock; cancellation and completion cannot race on disk.
        _atomic_json(self.root / job_id / "manifest.json", self.jobs[job_id])

    def restore(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self.root.iterdir():
            if not JOB_ID.fullmatch(directory.name) or not directory.is_dir() or directory.is_symlink():
                continue
            manifest = directory / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                job = json.loads(manifest.read_text(encoding="utf-8"))
                if job.get("job_id") != directory.name or job.get("status") not in {
                    "queued", "processing", "completed", "failed"
                }:
                    raise ValueError("Invalid job manifest")
                if job["status"] not in TERMINAL:
                    job.update(status="failed", error="Analysis interrupted by service restart.", eta_seconds=None)
                elif job["status"] == "completed":
                    _contained_file(directory, "result.json")
                    _contained_file(directory, job["video_file"])
                with self.lock:
                    self.jobs[directory.name] = job
                    self.cancellations[directory.name] = threading.Event()
                    self.persist(directory.name)
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                LOG.exception("Cannot restore job %s", directory.name)
                with self.lock:
                    self.jobs[directory.name] = {
                        "job_id": directory.name, "status": "failed", "progress": 0.0,
                        "frames": 0, "total_frames": 0, "eta_seconds": None,
                        "error": "Stored job manifest or output is damaged or missing.",
                    }
                    self.cancellations[directory.name] = threading.Event()
                    self.persist(directory.name)

    def load_model(self) -> None:
        with self.model_lock:
            if self.model_attempted:
                return
            self.model_attempted = True
            try:
                engine = self.engine_factory()
                with self.lock:
                    self.engine = engine
            except Exception as exc:
                LOG.exception("V3 model failed to load")
                with self.lock:
                    self.model_error = str(exc) or type(exc).__name__

    def engine_for(self, settings: dict[str, object]) -> Any:
        if not settings:
            with self.lock:
                return self.engine
        key = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        with self.lock:
            if key in self.engine_variants:
                self.engine_variants.move_to_end(key)
                return self.engine_variants[key]
            candidates = [self.engine, *reversed(self.engine_variants.values())]
        try:
            reusable = next((candidate for candidate in candidates
                             if callable(getattr(candidate, "can_reuse_models", None))
                             and candidate.can_reuse_models(settings)), None)
            engine = reusable.with_settings(settings) if reusable is not None else self.engine_factory(settings)
        except TypeError as exc:
            raise ValueError(
                "This TrackNet service cannot apply per-video settings; restart it with environment variables."
            ) from exc
        with self.lock:
            self.engine_variants[key] = engine
            # Model variants can consume hundreds of MB each. Keep recent choices
            # warm without retaining every settings combination for the service's lifetime.
            while len(self.engine_variants) > 3:
                self.engine_variants.popitem(last=False)
        return engine

    def snapshot(self, job_id: str) -> dict[str, Any]:
        _job_id(job_id)
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, "Job not found.")
            return {key: value for key, value in self.jobs[job_id].items()
                    if key not in {"input_file", "video_file"}}

    def cancel(self, job_id: str, error: str = "Analysis cancelled.") -> dict[str, Any]:
        self.snapshot(job_id)
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] not in TERMINAL:
                self.cancellations[job_id].set()
                job.update(status="failed", cancelled=True, error=error, eta_seconds=None)
                self.persist(job_id)
            return self.snapshot(job_id)

    def process(self, job_id: str) -> None:
        directory = self.root / job_id
        event = self.cancellations[job_id]
        cancelled = lambda: event.is_set() or self.stopping.is_set()
        started = time.monotonic()
        last_saved = 0.0

        def progress(fraction, frames, total):
            nonlocal last_saved
            if cancelled():
                return
            fraction = float(fraction)
            if not math.isfinite(fraction):
                return
            fraction = min(0.99, max(0.0, fraction))
            now = time.monotonic()
            with self.lock:
                job = self.jobs[job_id]
                if cancelled() or job["status"] != "processing":
                    return
                fraction = max(job["progress"], fraction)
                job.update(progress=fraction, frames=max(0, int(frames)), total_frames=max(0, int(total)),
                           eta_seconds=(now - started) * (1 - fraction) / fraction if fraction else None)
                if now - last_saved >= 2.0:
                    self.persist(job_id)
                    last_saved = now

        try:
            with self.lock:
                if cancelled() or self.jobs[job_id]["status"] in TERMINAL:
                    return
                self.jobs[job_id].update(status="processing")
                self.persist(job_id)
                source = _contained_file(directory, self.jobs[job_id]["input_file"])
            engine = self.engine_for(self.jobs[job_id].get("settings", {}))
            result = engine.analyze(
                str(source), str(directory), progress=progress, cancelled=cancelled, output_video=True,
            )
            if cancelled():
                return
            # Encoding belongs to V3Engine and receives the same cancellation
            # callback. Check again before publishing any completed artifacts.
            video = Path(result["video_path"]).resolve()
            video = _contained_file(directory, str(video))
            payload = {
                "job_id": job_id, "frames": int(result["frames"]),
                "coordinates": result["coordinates"], "video_url": f"/jobs/{job_id}/video",
                "metrics": result.get("metrics", {}),
            }
            if not isinstance(payload["coordinates"], list):
                raise ValueError("Engine returned invalid coordinates.")
            _atomic_json(directory / "result.json", payload)
            with self.lock:
                if cancelled() or self.jobs[job_id]["status"] != "processing":
                    return
                self.jobs[job_id].update(
                    status="completed", progress=1.0, frames=payload["frames"], eta_seconds=0,
                    video_file=video.relative_to(directory).as_posix(),
                )
                self.persist(job_id)
        except Exception as exc:
            LOG.exception("V3 analysis failed for %s", job_id)
            with self.lock:
                if self.jobs[job_id]["status"] not in TERMINAL:
                    self.jobs[job_id].update(status="failed", error=str(exc) or type(exc).__name__, eta_seconds=None)
                    self.persist(job_id)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self.load_model)
        while True:
            job_id = await self.queue.get()
            try:
                if job_id is None:
                    return
                await loop.run_in_executor(self.executor, self.process, job_id)
            finally:
                self.queue.task_done()


class _VideoUpload:
    """Multipart callbacks write directly to disk, never spooling an entire body."""

    def __init__(self, directory: Path, limit: int):
        self.directory, self.limit = directory, limit
        self.path: Path | None = None
        self.part_name: bytes | None = None
        self.settings = bytearray()
        self.file = None
        self.size = 0
        self.header_size = 0
        self.headers: dict[bytes, bytes] = {}
        self.field = bytearray()
        self.value = bytearray()
        self.ended = False
        self.part_ended = False

    def on_part_begin(self):
        self.part_name = None
        self.header_size = 0
        self.headers = {}

    def _header(self, target, data, start, end):
        self.header_size += end - start
        if self.header_size > 16384:
            raise HTTPException(413, "Upload headers are too large.")
        target.extend(data[start:end])

    def on_header_field(self, data, start, end):
        self._header(self.field, data, start, end)

    def on_header_value(self, data, start, end):
        self._header(self.value, data, start, end)

    def on_header_end(self):
        self.headers[bytes(self.field).lower()] = bytes(self.value)
        self.field.clear()
        self.value.clear()

    def on_headers_finished(self):
        disposition, options = parse_options_header(self.headers.get(b"content-disposition", b""))
        name = options.get(b"name")
        if disposition != b"form-data" or name not in {b"video", b"settings"}:
            raise HTTPException(400, "Upload a video and optional settings field.")
        self.part_name = name
        if name == b"settings":
            if self.headers.get(b"content-type", b"text/plain").lower() not in {
                b"application/json", b"text/plain", b"application/octet-stream"
            }:
                raise HTTPException(415, "TrackNet settings must be JSON.")
            return
        if self.path is not None:
            raise HTTPException(400, "Upload exactly one video field.")
        filename = options.get(b"filename", b"")
        if not filename:
            raise HTTPException(400, "Upload exactly one file using the video field.")
        content_type = self.headers.get(b"content-type", b"").lower()
        if not (content_type.startswith(b"video/") or content_type == b"application/octet-stream"):
            raise HTTPException(415, "Please upload a video file.")
        suffix = Path(filename.decode("utf-8", errors="replace")).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
            suffix = ".video"
        self.path = self.directory / f"input{suffix}"
        self.file = self.path.open("xb")

    def on_part_data(self, data, start, end):
        if self.part_name == b"settings":
            if len(self.settings) + end - start > 16384:
                raise HTTPException(413, "TrackNet settings are too large.")
            self.settings.extend(data[start:end])
            return
        self.size += end - start
        if self.size > self.limit:
            raise HTTPException(413, "Video exceeds TRACKNET_MAX_UPLOAD_BYTES.")
        self.file.write(data[start:end])

    def on_part_end(self):
        self.part_ended = True

    def on_end(self):
        self.ended = True

    def close(self):
        if self.file is not None:
            self.file.close()


async def _receive_video(request: Request, directory: Path, limit: int) -> tuple[Path, dict[str, object]]:
    # Parsing UploadFile parameters would spool the body before our size check.
    # Bound both the whole multipart stream and the actual file while receiving.
    content_type, options = parse_options_header(request.headers.get("content-type", ""))
    boundary = options.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary or len(boundary) > 200:
        raise HTTPException(400, "Expected a multipart video upload with a valid boundary.")
    body_limit = limit + 65536
    if "content-length" in request.headers:
        try:
            length = int(request.headers["content-length"])
        except ValueError:
            raise HTTPException(400, "Invalid Content-Length.") from None
        if length < 0:
            raise HTTPException(400, "Invalid Content-Length.")
        if length > body_limit:
            raise HTTPException(413, "Video upload is too large.")
    upload = _VideoUpload(directory, limit)
    callbacks = {name: getattr(upload, name) for name in dir(upload) if name.startswith("on_")}
    parser = MultipartParser(boundary, callbacks)
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > body_limit:
                raise HTTPException(413, "Video upload is too large.")
            for offset in range(0, len(chunk), 1024 * 1024):
                await asyncio.to_thread(parser.write, chunk[offset:offset + 1024 * 1024])
        parser.finalize()
        if not upload.ended or not upload.part_ended or upload.path is None or not upload.size:
            raise HTTPException(400, "Video upload is empty or incomplete.")
        settings: dict[str, object] = {}
        if upload.settings:
            try:
                parsed = json.loads(upload.settings.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HTTPException(400, "TrackNet settings must be valid JSON.") from exc
            if not isinstance(parsed, dict):
                raise HTTPException(400, "TrackNet settings must be a JSON object.")
            settings = parsed
        return upload.path, settings
    except ValueError as exc:
        raise HTTPException(400, "Malformed multipart upload.") from exc
    finally:
        upload.close()


class ConversionCancelled(Exception):
    pass


def _make_browser_compatible_video(
    source: Path, target: Path, cancelled: Callable[[], bool] = lambda: False,
    *, ffmpeg: str = "ffmpeg", timeout: float = 600,
) -> None:
    """Bounded-memory FFmpeg conversion with cancellation during encoding."""
    if cancelled():
        raise ConversionCancelled("Preview cancelled.")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source), "-map", "0:v:0", "-map", "0:a?", "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(target),
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if cancelled():
                    raise ConversionCancelled("Preview cancelled.")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Preview conversion timed out.")
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            if cancelled():
                raise ConversionCancelled("Preview cancelled.")
            if process.returncode:
                errors.seek(0, os.SEEK_END)
                errors.seek(max(0, errors.tell() - 4096))
                detail = errors.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"FFmpeg could not convert this video: {detail}")
            if not target.is_file() or not target.stat().st_size:
                raise RuntimeError("FFmpeg produced no preview video.")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def create_app(
    *, data_root: Path | str | None = None, engine_factory: Callable[[], Any] | None = None,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    root = Path(data_root if data_root is not None else os.getenv("TRACKNET_DATA_DIR", str(DEFAULT_DATA_DIR)))
    upload_limit = max_upload_bytes if max_upload_bytes is not None else int(os.getenv("TRACKNET_MAX_UPLOAD_BYTES", str(1024 ** 3)))
    if upload_limit <= 0:
        raise ValueError("TRACKNET_MAX_UPLOAD_BYTES must be positive.")
    ffmpeg = os.getenv("TRACKNET_FFMPEG", "ffmpeg")
    preview_timeout = float(os.getenv("TRACKNET_PREVIEW_TIMEOUT", "600"))
    if not math.isfinite(preview_timeout) or preview_timeout <= 0:
        raise ValueError("TRACKNET_PREVIEW_TIMEOUT must be positive and finite.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = Service(root, engine_factory or _engine_factory)
        app.state.service = state
        try:
            await asyncio.to_thread(state.restore)
            runner = asyncio.create_task(state.run())
            try:
                yield
            finally:
                state.stopping.set()
                for job_id in list(state.jobs):
                    state.cancel(job_id, "Analysis interrupted by service shutdown.")
                state.queue.put_nowait(None)
                await runner
        finally:
            await asyncio.to_thread(state.executor.shutdown, wait=True)

    app = FastAPI(title="TrackNet V3 Service", version="3.0.0", lifespan=lifespan)
    allowed_origins = [origin.strip() for origin in os.getenv("TRACKNET_ALLOW_ORIGINS", "").split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins, allow_origin_regex=LOCAL_ORIGIN.pattern,
        allow_credentials=True, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
    )

    def refuse_foreign_page(request: Request) -> None:
        # The browser hides the answer from a page CORS does not allow, but an
        # upload is a simple request, so it would still start work nobody sees.
        # The same rule as the CORS middleware's, "*" included.
        origin = request.headers.get("origin")
        if origin and "*" not in allowed_origins and origin not in allowed_origins and not LOCAL_ORIGIN.fullmatch(origin):
            raise HTTPException(403, f"{origin} may not use this service; add it to TRACKNET_ALLOW_ORIGINS.")

    @app.get("/health")
    async def health():
        state = app.state.service
        with state.lock:
            engine = state.engine
            return {
                "ok": True, "model_loaded": engine is not None, "engine": "v3",
                "status": "ready" if engine is not None else "error" if state.model_error else "loading",
                "backend": getattr(engine, "backend", "uninitialized"),
                "device": getattr(engine, "device", "unknown"),
                "frame_stack": getattr(engine, "seq_len", None),
                "error": state.model_error, "live_supported": False,
            }

    @app.post("/analyze", status_code=202)
    async def analyze(request: Request):
        refuse_foreign_page(request)
        state = app.state.service
        if state.engine is None or state.stopping.is_set():
            raise HTTPException(503, state.model_error or "V3 model is loading; retry shortly.")
        job_id = uuid.uuid4().hex
        directory = state.root / job_id
        directory.mkdir()
        try:
            source, settings = await _receive_video(request, directory, upload_limit)
            with state.lock:
                state.jobs[job_id] = {
                    "job_id": job_id, "status": "queued", "progress": 0.0,
                    "frames": 0, "total_frames": 0, "eta_seconds": None,
                    "input_file": source.name, "settings": settings,
                }
                state.cancellations[job_id] = threading.Event()
                state.persist(job_id)
            state.queue.put_nowait(job_id)
        except BaseException:
            with state.lock:
                state.jobs.pop(job_id, None)
                state.cancellations.pop(job_id, None)
            shutil.rmtree(directory)
            raise
        return {"job_id": job_id, "status_url": f"/jobs/{job_id}", "result_url": f"/jobs/{job_id}/result"}

    @app.get("/jobs/{job_id}")
    async def job_status(job_id: str):
        return app.state.service.snapshot(job_id)

    @app.delete("/jobs/{job_id}")
    async def cancel_job(job_id: str):
        return app.state.service.cancel(job_id)

    @app.get("/jobs/{job_id}/result")
    def job_result(job_id: str):
        state = app.state.service
        job = state.snapshot(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, job.get("error", "Job is not complete."))
        try:
            path = _contained_file(state.root / job_id, "result.json")
            content = path.read_bytes()
            json.loads(content)  # Preserve the damaged-result check for restored jobs.
            # Already serialized by the worker. Avoid FastAPI walking and
            # re-encoding every coordinate on each browser visit.
            return Response(content=content, media_type="application/json")
        except (OSError, ValueError):
            raise HTTPException(404, "Stored job result is missing or damaged.") from None

    @app.get("/jobs/{job_id}/video")
    def job_video(job_id: str):
        state = app.state.service
        job = state.snapshot(job_id)
        if job["status"] != "completed":
            raise HTTPException(404, "Processed video not available.")
        try:
            path = _contained_file(state.root / job_id, state.jobs[job_id]["video_file"])
        except (OSError, ValueError, KeyError):
            raise HTTPException(404, "Processed video file is missing.") from None
        return FileResponse(path, media_type="video/mp4", filename="tracknet_predict.mp4")

    @app.post("/preview")
    async def preview(request: Request):
        refuse_foreign_page(request)
        state = app.state.service
        directory = Path(tempfile.mkdtemp(prefix="preview-", dir=state.root))
        cancelled = threading.Event()
        conversion = None
        delivered = False
        try:
            source, _ = await _receive_video(request, directory, upload_limit)
            target = directory / "preview.mp4"
            async with state.preview_gate:
                if await request.is_disconnected() or state.stopping.is_set():
                    raise HTTPException(499, "Preview cancelled.")
                conversion = asyncio.create_task(asyncio.to_thread(
                    _make_browser_compatible_video, source, target,
                    lambda: cancelled.is_set() or state.stopping.is_set(),
                    ffmpeg=ffmpeg, timeout=preview_timeout,
                ))
                while not conversion.done():
                    if await request.is_disconnected():
                        cancelled.set()
                    await asyncio.wait({conversion}, timeout=0.1)
                await conversion
            response = FileResponse(
                target, media_type="video/mp4", filename="shuttleboard_preview.mp4",
                background=BackgroundTask(shutil.rmtree, directory),
            )
            delivered = True
            return response
        except FileNotFoundError:
            raise HTTPException(503, "FFmpeg is required to create a browser-compatible preview.") from None
        except ConversionCancelled:
            raise HTTPException(499, "Preview cancelled.") from None
        except TimeoutError as exc:
            raise HTTPException(504, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            cancelled.set()
            if conversion is not None and not conversion.done():
                # Keep files alive until the encoder has stopped using them.
                await asyncio.gather(conversion, return_exceptions=True)
            if not delivered:
                shutil.rmtree(directory)

    @app.websocket("/live/tracknet-fast")
    async def live(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({
            "type": "error",
            "error": "TrackNet V3 supports uploaded videos only. Use the current engine (TRACKNET_ENGINE=current) for live tracking.",
        })
        await websocket.close(code=1008)

    return app


app = create_app()
