import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { TrackNetResult, TrackNetSettings } from "@/lib/tracknet-client";

const STORAGE_KEY = "shuttleboard.tracknet-analysis";

export type TrackNetAnalysisPhase =
  "idle" | "uploading" | "queued" | "processing" | "completed" | "failed";

export interface TrackNetAnalysisJob {
  jobId: string;
  fileName: string;
  startedAt: number;
}

interface TrackNetAnalysisContextValue {
  job: TrackNetAnalysisJob | null;
  phase: TrackNetAnalysisPhase;
  progress: number;
  etaSeconds: number | null;
  result: TrackNetResult | null;
  error: string | null;
  start: (file: File, settings?: TrackNetSettings) => Promise<void>;
  clear: () => void;
}

const TrackNetAnalysisContext = createContext<TrackNetAnalysisContextValue | null>(null);

function readStoredJob(): TrackNetAnalysisJob | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    if (!value) return null;
    const parsed = JSON.parse(value) as Partial<TrackNetAnalysisJob>;
    if (
      typeof parsed.jobId !== "string" ||
      typeof parsed.fileName !== "string" ||
      typeof parsed.startedAt !== "number"
    ) {
      return null;
    }
    return { jobId: parsed.jobId, fileName: parsed.fileName, startedAt: parsed.startedAt };
  } catch {
    return null;
  }
}

function storeJob(job: TrackNetAnalysisJob | null) {
  try {
    if (job) window.localStorage.setItem(STORAGE_KEY, JSON.stringify(job));
    else window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Local storage can be unavailable in private browsing; the in-memory job still works.
  }
}

export function TrackNetAnalysisProvider({ children }: { children: ReactNode }) {
  const [job, setJob] = useState<TrackNetAnalysisJob | null>(null);
  const [phase, setPhase] = useState<TrackNetAnalysisPhase>("idle");
  const [progress, setProgress] = useState(0);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);
  const [result, setResult] = useState<TrackNetResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const runTokenRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const jobRef = useRef<TrackNetAnalysisJob | null>(null);

  const cancelActiveJob = useCallback(() => {
    const activeJob = jobRef.current;
    jobRef.current = null;
    if (activeJob) {
      void import("@/lib/tracknet-client")
        .then(({ cancelTrackNetJob }) => cancelTrackNetJob(activeJob.jobId))
        .catch(() => undefined);
    }
  }, []);

  const pollJob = useCallback(
    (nextJob: TrackNetAnalysisJob, options?: { clearMissingJob?: boolean }) => {
      const token = ++runTokenRef.current;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setPhase("processing");
      setError(null);

      void (async () => {
        try {
          const { getTrackNetResult, waitForTrackNet } = await import("@/lib/tracknet-client");
          await waitForTrackNet(
            nextJob.jobId,
            (nextProgress, nextEtaSeconds) => {
              if (runTokenRef.current !== token) return;
              setPhase(nextProgress > 0 ? "processing" : "queued");
              setProgress(nextProgress);
              setEtaSeconds(nextEtaSeconds ?? null);
            },
            controller.signal,
          );
          const nextResult = await getTrackNetResult(nextJob.jobId, controller.signal);
          if (runTokenRef.current !== token) return;
          setProgress(1);
          setEtaSeconds(null);
          setResult(nextResult);
          setPhase("completed");
        } catch (pollError) {
          if (runTokenRef.current !== token) return;
          // By status, not message text: messages can quote addresses like :4040.
          const missingJob =
            options?.clearMissingJob &&
            pollError instanceof Error &&
            "status" in pollError &&
            pollError.status === 404;
          if (missingJob) {
            jobRef.current = null;
            setJob(null);
            setPhase("idle");
            setProgress(0);
            setEtaSeconds(null);
            setResult(null);
            setError(null);
            storeJob(null);
            return;
          }
          setPhase("failed");
          setEtaSeconds(null);
          setError(pollError instanceof Error ? pollError.message : "TrackNet analysis failed.");
        }
      })();
    },
    [],
  );

  useEffect(() => {
    const storedJob = readStoredJob();
    if (!storedJob) return;
    jobRef.current = storedJob;
    setJob(storedJob);
    pollJob(storedJob, { clearMissingJob: true });
  }, [pollJob]);

  useEffect(
    () => () => {
      ++runTokenRef.current;
      abortRef.current?.abort();
    },
    [],
  );

  const start = useCallback(
    async (file: File, settings?: TrackNetSettings) => {
      cancelActiveJob();
      const token = ++runTokenRef.current;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setPhase("uploading");
      setProgress(0);
      setEtaSeconds(null);
      setResult(null);
      setError(null);
      try {
        const { submitTrackNetVideo, cancelTrackNetJob } = await import("@/lib/tracknet-client");
        const { jobId } = await submitTrackNetVideo(file, controller.signal, settings);
        if (runTokenRef.current !== token) {
          void cancelTrackNetJob(jobId);
          return;
        }
        const nextJob = { jobId, fileName: file.name, startedAt: Date.now() };
        jobRef.current = nextJob;
        setJob(nextJob);
        storeJob(nextJob);
        setPhase("queued");
        pollJob(nextJob);
      } catch (startError) {
        if (runTokenRef.current !== token) return;
        setPhase("failed");
        setError(startError instanceof Error ? startError.message : "TrackNet upload failed.");
      }
    },
    [pollJob, cancelActiveJob],
  );

  const clear = useCallback(() => {
    cancelActiveJob();
    ++runTokenRef.current;
    abortRef.current?.abort();
    setJob(null);
    setPhase("idle");
    setProgress(0);
    setEtaSeconds(null);
    setResult(null);
    setError(null);
    storeJob(null);
  }, [cancelActiveJob]);

  const value = useMemo(
    () => ({ job, phase, progress, etaSeconds, result, error, start, clear }),
    [job, phase, progress, etaSeconds, result, error, start, clear],
  );

  return (
    <TrackNetAnalysisContext.Provider value={value}>{children}</TrackNetAnalysisContext.Provider>
  );
}

export function useTrackNetAnalysis() {
  const context = useContext(TrackNetAnalysisContext);
  if (!context) {
    throw new Error("useTrackNetAnalysis must be used inside TrackNetAnalysisProvider");
  }
  return context;
}
