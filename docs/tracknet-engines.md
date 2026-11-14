# TrackNet engine commands

Run these commands in PowerShell from the ShuttleBoard project directory.

```powershell
# Existing engine (also the default when TRACKNET_ENGINE is unset)
$env:TRACKNET_ENGINE = 'current'
npm run shuttleboard

# V3 engine, after its repository, environment, and models are installed
$env:TRACKNET_ENGINE = 'v3'
$env:TRACKNET_DEVICE = 'CPU'
$env:TRACKNET_EVAL_MODE = 'nonoverlap'
npm run shuttleboard
```

Stop the running launcher with `Ctrl+C` before switching engines. Both engines
use the same service address, `http://127.0.0.1:8000` by default, and launch child
processes with Windows console windows hidden.

| Setting | `current` (default) | `v3` |
| --- | --- | --- |
| Service module | `scripts/tracknet_live.py` | `scripts/tracknet_v3_service.py` |
| Repository override | `TRACKNET_DIR` | `TRACKNET_V3_DIR` |
| Default repository | `../TrackNet-Badminton-Tracking-tensorflow2-main` | `.tracknet-v3/repo` |
| Default Python | Repository's `.venv311/Scripts/python.exe` | `.tracknet-v3/.venv/Scripts/python.exe` |
| Default `TRACKNET_BACKEND` | `openvino` (unchanged) | `openvino` |
| Default sequence stride | 4 | 1 |

Default paths are relative to the ShuttleBoard project. Relative overrides are
resolved from the directory where the launcher is invoked. `TRACKNET_PYTHON`
overrides the executable for either engine; `TRACKNET_BACKEND` overrides its
backend. A custom `TRACKNET_V3_DIR` relocates only the V3 repository, not its
Python environment. For example:

```powershell
$env:TRACKNET_ENGINE = 'v3'
$env:TRACKNET_V3_DIR = 'D:\Models\TrackNetV3'
$env:TRACKNET_PYTHON = 'D:\Python environments\tracknet-v3\Scripts\python.exe'
$env:TRACKNET_BACKEND = 'openvino'
npm run shuttleboard
```

V3 runtime assets belong under `.tracknet-v3/{repo,.venv,ckpts,openvino,data}`.
The launcher selects an installed runtime; it does not download repositories,
create Python environments, or prepare checkpoints. V3 service implementation
and runtime setup are separate from engine selection.

The launcher checks `/health` for `model_loaded`, `backend`, and `device` using
the existing readiness contract. `TRACKNET_PORT`, `SHUTTLEBOARD_PORT`,
`TRACKNET_DEVICE`, and the existing inference/thread settings remain available.
If changing `TRACKNET_PORT`, also configure `VITE_TRACKNET_URL` for the frontend.
Both services accept requests from pages this computer serves (`localhost`,
`127.0.0.1` or `[::1]`) on any port, so an app that `npm run dev` moved to 5174
or 5175 can still analyze videos. To use the V3 service from another address,
such as a phone on the same network, list it in `TRACKNET_ALLOW_ORIGINS`
(comma-separated); V3 refuses uploads from any other page rather than analyzing
a video whose result the browser would hide. With the current engine, other
addresses follow the original `tracknet_service.py`'s own CORS settings.

Run the focused launcher checks without installing either model runtime:

```powershell
node --test tests/tracknet-launch-config.test.mjs
```

## V3 settings

V3 is opt-in. The current engine stays the default. V3 currently handles uploaded
videos; select `current` for the existing live camera connection. Its grayscale
three-frame messages cannot be fed to the V3 color model without changing the
camera transport. The V3 live endpoint returns a clear error explaining this.

When TrackNet AI is selected in the video analyzer, the processing panel lets you
choose V3's backend, device, overlap mode, InpaintNet, and CPU thread count. Those
choices are sent with that upload and applied by the V3 service. The engine choice
itself still happens when the app starts because `current` and `v3` are different
services. The panel shows which service is currently running.

| Setting | V3 default | Meaning |
| --- | --- | --- |
| `TRACKNET_DEVICE` | `AUTO` | OpenVINO chooses hardware; `CPU` and `GPU` are also supported. |
| `TRACKNET_EVAL_MODE` | `weight` | Combines overlapping predictions. `average` uses equal weights; `nonoverlap` runs each group once for speed. |
| `TRACKNET_FRAME_STRIDE` | `1` | Moves the start of the next group. Frames inside each group remain consecutive. Maximum is the checkpoint's sequence length (8). `nonoverlap` sets this to 8. |
| `TRACKNET_BATCH_SIZE` | `1` | Groups processed together. Larger values need more memory. |
| `TRACKNET_WIDTH`, `TRACKNET_HEIGHT` | `512`, `288` | Model input size. Must be multiples of 8; changing them requires re-exporting the models at that size. |
| `TRACKNET_INPAINT` | `1` | Set to `0` to disable repair of missing positions. |
| `TRACKNET_BACKGROUND_SAMPLES` | `120` | Approximate maximum frames sampled across a video for its background. |
| `TRACKNET_PRECISION` | `f32` | Full-precision calculations. Optional `f16` needs its own accuracy and hardware checks. Weights are exported without compression. |
| `TRACKNET_INTRA_THREADS` | `2` | CPU threads used by V3. |
| `TRACKNET_V3_CKPTS` | `.tracknet-v3/ckpts` | Official checkpoint directory. |
| `TRACKNET_V3_OPENVINO_DIR` | `.tracknet-v3/openvino` | Exported models, metadata, conversion check, and compiled-model cache. |

V3 reads checkpoint settings (8 RGB frames plus background; 16 positions for
repair), uses matching preprocessing for both backends, and rejects failed
conversion checks. A final partial batch is padded internally and trimmed back
to the original frame count. No INT8 quantization is enabled.

For a faster V3 mode on the CPU:

```powershell
$env:TRACKNET_ENGINE = 'v3'
$env:TRACKNET_BACKEND = 'openvino'
$env:TRACKNET_DEVICE = 'CPU'
$env:TRACKNET_EVAL_MODE = 'nonoverlap'
$env:TRACKNET_INTRA_THREADS = '8'
npm run shuttleboard
```

This reduces repeated analysis of the same frames. Its accuracy relative to
`weight` mode has not been established. See the measured comparison in
[tracknet-v3.md](tracknet-v3.md) before choosing a speed setting.

For the full overlapping calculation, set `TRACKNET_EVAL_MODE` to `weight`.
Stop the app with Ctrl+C before changing settings. Environment settings last
for the current PowerShell session.

### API and cancellation

V3 preserves the upload, preview, job status, result, and video endpoints.
`DELETE /jobs/{job_id}` cancels a job; the response uses `status: failed` and
`cancelled: true` so existing status polling understands the terminal state.
The frontend calls this when clearing or replacing a job. Older services can
return 405 and still clear their UI as before.

The service loads models once and processes one upload job at a time. Health
requests remain responsive. Jobs survive restarts; interrupted jobs are marked
failed. Run one service worker. Uploads default to a 1 GiB limit, configurable
with `TRACKNET_MAX_UPLOAD_BYTES`. Data stays in `.tracknet-v3/service_data`
(override with `TRACKNET_DATA_DIR`). This is a local service without account
authentication; the launcher binds to 127.0.0.1.

Directly detected points retain their heatmap score. Repaired points have
`repaired: true`, `detected: false`, and confidence 0: they are estimated
positions, not a new confident detection. The output video shows repairs in
orange and direct detections in red. These are review aids, not verified calls.
