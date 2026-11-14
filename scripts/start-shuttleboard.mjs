import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { getTrackNetLaunchConfig } from "./tracknet-launch-config.mjs";

const appDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
let tracknet;
try {
  tracknet = getTrackNetLaunchConfig(appDir);
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
const { tracknetDir, pythonPath } = tracknet;
const tracknetPort = process.env.TRACKNET_PORT ?? "8000";
const appPort = process.env.SHUTTLEBOARD_PORT ?? "5173";
const ffmpegCandidates = [
  process.env.TRACKNET_FFMPEG,
  "C:\\Program Files\\obs-studio\\bin\\64bit\\ffmpeg.exe",
  "C:\\Program Files\\Streamlabs OBS\\resources\\app.asar.unpacked\\node_modules\\obs-studio-node\\ffmpeg.exe",
].filter(Boolean);
const ffmpegPath = ffmpegCandidates.find((candidate) => existsSync(candidate));
const appUrl = `http://127.0.0.1:${appPort}`;
const tracknetUrl = `http://127.0.0.1:${tracknetPort}`;
const prewarmRoutes = ["/", "/auth", "/dashboard", "/match-lobby", "/scoreboard"];

if (!existsSync(tracknetDir)) {
  console.error(`TrackNet folder not found: ${tracknetDir}`);
  console.error(tracknet.directoryHint);
  process.exit(1);
}

if (!existsSync(pythonPath)) {
  console.error(`TrackNet Python environment not found: ${pythonPath}`);
  console.error(tracknet.pythonHint);
  process.exit(1);
}

const children = [];
let shuttingDown = false;

function start(label, command, args, cwd, env = {}) {
  const child = spawn(command, args, {
    cwd,
    env: { ...process.env, ...env },
    stdio: ["inherit", "pipe", "pipe"],
    windowsHide: true,
  });

  child.stdout.on("data", (data) => process.stdout.write(`[${label}] ${data}`));
  child.stderr.on("data", (data) => process.stderr.write(`[${label}] ${data}`));
  child.on("error", (error) => {
    console.error(`[${label}] ${error.message}`);
    shutdown(1);
  });
  child.on("exit", (code, signal) => {
    if (!shuttingDown && (code ?? 0) !== 0) {
      console.error(`[${label}] exited with ${signal ?? `code ${code}`}.`);
      shutdown(code ?? 1);
    }
  });

  children.push(child);
  return child;
}

function shutdown(code = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) child.kill();
  }
  setTimeout(() => process.exit(code), 250);
}

async function waitForHttp(url, label, timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { redirect: "manual" });
      if (response.status < 500) {
        console.log(`[startup] ${label} is responding`);
        return true;
      }
    } catch {
      // The service may still be compiling or loading its model.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  console.warn(`[startup] ${label} did not become ready within ${timeoutMs / 1000}s`);
  return false;
}

async function prewarmShuttleBoard() {
  if (!(await waitForHttp(`${appUrl}/`, "ShuttleBoard"))) return;
  console.log(`[startup] Prewarming ${prewarmRoutes.length} ShuttleBoard routes`);
  for (const route of prewarmRoutes) {
    try {
      await fetch(`${appUrl}${route}`, { redirect: "manual" });
      console.log(`[startup] Warmed ${route}`);
    } catch (error) {
      console.warn(`[startup] Could not warm ${route}: ${error.message}`);
    }
  }
}

async function waitForTrackNetModel() {
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${tracknetUrl}/health`);
      if (response.ok) {
        const status = await response.json();
        if (status.model_loaded) {
          console.log(
            `[startup] TrackNet ready (${status.backend ?? "unknown"}/${status.device ?? "unknown"})`,
          );
          return;
        }
      }
    } catch {
      // TrackNet can take time to start its Python environment and model.
    }
    await new Promise((resolve) => setTimeout(resolve, 1_000));
  }
  console.warn(
    "[startup] TrackNet is still loading; the AI page will remain unavailable until ready.",
  );
}

// Another server on the app's port would answer the readiness check and the
// prewarm below before Vite gave up on the port.
const appPortFree = await new Promise((resolvePort) => {
  const probe = createServer();
  probe.once("error", () => resolvePort(false));
  probe.listen(Number(appPort), "127.0.0.1", () => probe.close(() => resolvePort(true)));
});
if (!appPortFree) {
  console.error(
    `Port ${appPort} is already in use. Close the other dev server, or set SHUTTLEBOARD_PORT.`,
  );
  process.exit(1);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

console.log(`Starting ShuttleBoard first on ${appUrl}`);
console.log(`TrackNet (${tracknet.engine}) will start in the background on ${tracknetUrl}`);

start(
  "shuttleboard",
  process.execPath,
  [
    resolve(appDir, "node_modules", "vite", "bin", "vite.js"),
    "dev",
    "--host",
    "127.0.0.1",
    "--port",
    appPort,
    // Never move to another port: the address printed and prewarmed is this one.
    "--strictPort",
  ],
  appDir,
);

void (async () => {
  await waitForHttp(`${appUrl}/`, "ShuttleBoard");
  if (shuttingDown) return;
  console.log(`Starting TrackNet in the background on ${tracknetUrl}`);
  start(
    "tracknet",
    pythonPath,
    [
      "-m",
      "uvicorn",
      tracknet.serviceModule,
      "--app-dir",
      resolve(appDir, "scripts"),
      "--host",
      "127.0.0.1",
      "--port",
      tracknetPort,
    ],
    tracknetDir,
    {
      PYTHONUNBUFFERED: "1",
      ...tracknet.env,
      TRACKNET_DEVICE: process.env.TRACKNET_DEVICE ?? "AUTO",
      TRACKNET_BATCH_SIZE: process.env.TRACKNET_BATCH_SIZE ?? "1",
      TRACKNET_INTRA_THREADS: process.env.TRACKNET_INTRA_THREADS ?? "2",
      TRACKNET_INTER_THREADS: process.env.TRACKNET_INTER_THREADS ?? "1",
      TF_NUM_INTRAOP_THREADS: process.env.TF_NUM_INTRAOP_THREADS ?? "2",
      TF_NUM_INTEROP_THREADS: process.env.TF_NUM_INTEROP_THREADS ?? "1",
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? "2",
      MKL_NUM_THREADS: process.env.MKL_NUM_THREADS ?? "2",
      ...(ffmpegPath ? { TRACKNET_FFMPEG: ffmpegPath } : {}),
    },
  );

  await Promise.all([prewarmShuttleBoard(), waitForTrackNetModel()]);
})();
