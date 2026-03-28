"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

const POLL_MS = 2000;

/** Default API origin (env wins). LAN auto-detection runs in the browser after mount. */
function getDefaultApiBase(): string {
  return (
    process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000"
  );
}

type StatusResponse = { status: string; message?: string };

type ReportDone = {
  status: "done";
  video_url: string | null;
  summary_image_url?: string | null;
  timeline_image_url?: string | null;
  danger_clips?: string[];
  report: {
    summary?: Record<string, unknown>;
    meta?: Record<string, unknown>;
  };
  meta: {
    frames?: number;
    model?: string;
    features?: number;
    training_data?: string;
  };
};

type ReportProcessing = { status: "processing" };
type ReportError = { status: "error"; message: string };
type ReportIdle = { status: "idle" };
type ModelInfoResponse = {
  timestamp?: string | null;
  n_samples?: number;
  class_distribution?: Record<string, number>;
  feature_importances?: Record<string, number>;
  sklearn_version?: string | null;
};

function resolveMediaUrl(
  base: string,
  url: string | null | undefined,
): string | null {
  if (!url) return null;
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  return `${base}${url.startsWith("/") ? "" : "/"}${url}`;
}

function riskBarClass(label: string): string {
  if (label === "HIGH") return "bg-red-500";
  if (label === "MEDIUM") return "bg-amber-400";
  return "bg-emerald-500";
}

function humanize(s: string): string {
  return s.replace(/_/g, " ").trim();
}

function clamp(n: number, a: number, b: number): number {
  return Math.max(a, Math.min(b, n));
}

type Phase = "idle" | "uploading" | "processing" | "done" | "error";

function statusLayer(
  phase: Phase,
  layer: Exclude<Phase, "idle">,
  base: string,
  stackAlign: "center" | "start" = "center",
): string {
  const active = phase === layer;
  const align = stackAlign === "start" ? "items-start" : "items-center";
  return [
    base,
    "status-layer",
    active
      ? `relative z-20 flex min-h-[5.5rem] ${align} opacity-100`
      : `pointer-events-none absolute inset-0 z-0 flex ${align} opacity-0`,
  ].join(" ");
}

function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <h2 className="mb-4 text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">
      {children}
    </h2>
  );
}

function BackgroundFX() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const prefersReduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
    if (prefersReduced) return;

    let w = 0;
    let h = 0;
    let dpr = 1;

    const resize = () => {
      dpr = window.devicePixelRatio || 1;
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const mouse = { x: w / 2, y: h / 2, tx: w / 2, ty: h / 2 };
    const onMove = (e: MouseEvent) => {
      mouse.tx = e.clientX;
      mouse.ty = e.clientY;
    };
    window.addEventListener("mousemove", onMove, { passive: true });

    const starCount = 90;
    const stars = Array.from({ length: starCount }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      z: Math.random(), // 0..1 depth
      r: 0.3 + Math.random() * 1.4,
      v: 0.12 + Math.random() * 0.5,
    }));

    let raf = 0;
    let last = performance.now();

    const tick = (now: number) => {
      const dt = Math.min(32, now - last);
      last = now;

      mouse.x += (mouse.tx - mouse.x) * 0.06;
      mouse.y += (mouse.ty - mouse.y) * 0.06;

      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = "lighter";

      const ox = (mouse.x - w / 2) / w;
      const oy = (mouse.y - h / 2) / h;

      for (const s of stars) {
        // parallax drift
        s.x += ox * (0.9 - s.z) * dt * 0.02;
        s.y += (s.v * (0.2 + (1.0 - s.z))) * dt * 0.12;

        if (s.y > h + 10) {
          s.y = -10;
          s.x = Math.random() * w;
        }
        if (s.x < -10) s.x = w + 10;
        if (s.x > w + 10) s.x = -10;

        const a = 0.08 + (1.0 - s.z) * 0.28;
        ctx.fillStyle = `rgba(56, 189, 248, ${a})`;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
      }

      raf = window.requestAnimationFrame(tick);
    };

    raf = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none fixed inset-0 -z-10 opacity-30"
    />
  );
}

function Tilt({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement | null>(null);

  const onMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (!el) return;

    const reduced =
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) return;

    const rect = el.getBoundingClientRect();
    const x = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    const y = clamp((e.clientY - rect.top) / rect.height, 0, 1);

    const ry = (x - 0.5) * 12; // left/right
    const rx = (0.5 - y) * 10; // up/down

    el.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg) translateZ(0)`;
  }, []);

  const onLeave = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.transform = `perspective(900px) rotateX(0deg) rotateY(0deg) translateZ(0)`;
  }, []);

  return (
    <div
      ref={ref}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      className={`rg-tilt relative ${className ?? ""}`}
    >
      {children}
    </div>
  );
}

function RiskBadge({ level }: { level: string }) {
  const isHigh = level === "HIGH";
  const isMed = level === "MEDIUM";
  const cls = isHigh
    ? "bg-red-600 text-white ring-2 ring-red-400/50 shadow-red-900/30"
    : isMed
      ? "bg-amber-400 text-amber-950 ring-2 ring-amber-300/60 shadow-amber-900/20"
      : "bg-emerald-600 text-white ring-2 ring-emerald-400/45 shadow-emerald-900/25";

  return (
    <div
      className={`relative flex w-full overflow-hidden items-center justify-center rounded-xl px-4 py-4 text-center text-3xl font-bold uppercase tracking-wide shadow-lg transition-all duration-300 sm:py-5 sm:text-4xl ${cls}`}
      aria-label={`Risk level ${level}`}
    >
      <div className="rg-holo-ring" aria-hidden />
      <div className="rg-holo-scan" aria-hidden />
      {level}
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div>
      <div className="mb-2 flex justify-between text-xs text-slate-500">
        <span>Confidence</span>
        <span className="font-mono text-slate-300">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-slate-800/90 ring-1 ring-slate-700/80">
        <div
          className="h-full max-w-full rounded-full bg-gradient-to-r from-blue-600 to-cyan-400 transition-[width] duration-700 ease-out motion-reduce:transition-none"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function ExplanationPanel({
  reasons,
  contributions,
}: {
  reasons: string[] | undefined;
  contributions: Record<string, number> | undefined;
}) {
  const topFactors = contributions
    ? Object.entries(contributions)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 8)
    : [];

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-slate-800/90 bg-slate-950/25 p-4 sm:p-5">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Why this risk?
        </h3>
        {reasons && reasons.length > 0 ? (
          <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-relaxed text-slate-200 marker:text-slate-500">
            {reasons.slice(0, 12).map((r) => (
              <li key={r} className="pl-0.5">
                {humanize(r)}
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-slate-500">
            No rule-based triggers; model confidence is primary.
          </p>
        )}
      </div>

      {topFactors.length > 0 ? (
        <div className="rounded-xl border border-slate-800/90 bg-slate-950/25 p-4 sm:p-5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            Top Factors
          </h3>
          <ul className="mt-3 list-disc space-y-2 pl-5 font-mono text-sm text-slate-300 marker:text-slate-500">
            {topFactors.map(([k]) => (
              <li key={k} className="pl-0.5">
                {k}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

export default function Home() {
  const [apiBase, setApiBase] = useState(getDefaultApiBase);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [reportPayload, setReportPayload] = useState<ReportDone | null>(null);
  const [drag, setDrag] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [videoReady, setVideoReady] = useState(false);
  const [videoLoadError, setVideoLoadError] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const env = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
    if (env) {
      setApiBase(env);
      return;
    }
    const { protocol, hostname } = window.location;
    if (hostname !== "localhost" && hostname !== "127.0.0.1") {
      setApiBase(`${protocol}//${hostname}:8000`);
    }
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const fetchReport = useCallback(async () => {
    const res = await fetch(`${apiBase}/report`);
    const data = (await res.json()) as
      | ReportDone
      | ReportProcessing
      | ReportError
      | ReportIdle;

    if (data.status === "processing") {
      return;
    }
    if (data.status === "error") {
      stopPolling();
      setPhase("error");
      setError((data as ReportError).message || "Analysis failed");
      return;
    }
    if (data.status === "idle") {
      return;
    }
    if (data.status === "done") {
      stopPolling();
      setReportPayload(data as ReportDone);
      setPhase("done");
      setError(null);
    }
  }, [apiBase, stopPolling]);

  const pollStatus = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/status`);
      const data = (await res.json()) as StatusResponse;
      if (data.status === "processing") return;
      if (data.status === "error") {
        stopPolling();
        setPhase("error");
        setError(data.message || "Analysis failed");
        return;
      }
      if (data.status === "done") {
        await fetchReport();
      }
    } catch {
      stopPolling();
      setPhase("error");
      setError("Lost connection to API");
    }
  }, [apiBase, fetchReport, stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    pollRef.current = setInterval(() => {
      void pollStatus();
    }, POLL_MS);
  }, [pollStatus, stopPolling]);

  useEffect(() => () => stopPolling(), [stopPolling]);
  useEffect(() => {
    let ignore = false;
    const loadModelInfo = async () => {
      try {
        const res = await fetch(`${apiBase}/model-info`);
        if (!res.ok) return;
        const data = (await res.json()) as ModelInfoResponse;
        if (!ignore) setModelInfo(data);
      } catch {
        // Optional panel: keep UI running if metadata endpoint is unavailable.
      }
    };
    void loadModelInfo();
    return () => {
      ignore = true;
    };
  }, [apiBase]);

  const onAnalyze = useCallback(
    async (file: File) => {
      setError(null);
      setReportPayload(null);
      setPhase("uploading");

      const fd = new FormData();
      fd.append("file", file);

      try {
        const res = await fetch(`${apiBase}/analyze`, {
          method: "POST",
          body: fd,
        });

        if (res.status === 409) {
          setPhase("error");
          setError("Another analysis is already running. Try again shortly.");
          return;
        }

        const j = await res.json().catch(() => ({}));

        if (!res.ok) {
          const msg =
            (j as { detail?: string; message?: string }).detail ||
            (j as { message?: string }).message ||
            res.statusText;
          setPhase("error");
          setError(typeof msg === "string" ? msg : "Request failed");
          return;
        }

        if ((j as { status?: string }).status === "processing") {
          if (fileInputRef.current) fileInputRef.current.value = "";
          setPhase("processing");
          startPolling();
          void pollStatus();
          return;
        }
      } catch (e) {
        setPhase("error");
        setError(e instanceof Error ? e.message : "Request failed");
      }
    },
    [apiBase, pollStatus, startPolling],
  );

  const handleFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) {
        setSelectedFileName(f.name);
        void onAnalyze(f);
      }
    },
    [onAnalyze],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDrag(false);
      if (phase === "uploading" || phase === "processing") return;
      const f = e.dataTransfer.files?.[0];
      if (f) {
        setSelectedFileName(f.name);
        void onAnalyze(f);
      }
    },
    [onAnalyze, phase],
  );

  const summary = reportPayload?.report?.summary as
    | {
        risk_distribution?: Record<string, number>;
        total_frames?: number;
        scene_stability?: number;
        risk_trend?: string;
        avg_scene_complexity?: number;
      }
    | undefined;

  const innerMeta = reportPayload?.report?.meta as
    | {
        last_frame?: {
          risk?: string;
          confidence?: number;
          primary_cause?: string;
          reasons?: string[];
          feature_contributions?: Record<string, number>;
        };
      }
    | undefined;

  const apiMeta = reportPayload?.meta;
  const last = innerMeta?.last_frame;
  const dist = summary?.risk_distribution || {};
  const maxCount = Math.max(1, ...Object.values(dist));
  const videoSrc = resolveMediaUrl(apiBase, reportPayload?.video_url ?? null);
  const summaryImageSrc = resolveMediaUrl(
    apiBase,
    reportPayload?.summary_image_url ?? null,
  );
  const timelineImageSrc = resolveMediaUrl(
    apiBase,
    reportPayload?.timeline_image_url ?? null,
  );
  const dangerClips = (reportPayload?.danger_clips ?? [])
    .map((u) => resolveMediaUrl(apiBase, u))
    .filter((u): u is string => Boolean(u));
  const conf = last?.confidence ?? 0;
  const featureImportanceRows = Object.entries(modelInfo?.feature_importances ?? {}).sort(
    (a, b) => b[1] - a[1],
  );
  const maxImportance = Math.max(
    0.0001,
    ...featureImportanceRows.map(([, v]) => Number(v) || 0),
  );

  useEffect(() => {
    setVideoReady(false);
    setVideoLoadError(null);
  }, [videoSrc]);

  const busy = phase === "processing" || phase === "uploading";

  return (
    <>
      <div className="rg-bg" aria-hidden />
      <BackgroundFX />
      <main className="relative z-10 mx-auto min-h-screen max-w-6xl px-4 py-6 sm:px-5 sm:py-8 lg:px-6">
      <header className="mb-8 border-b border-slate-800/80 pb-6 transition-opacity duration-300">
        <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
          RoadGuard-X
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
          Upload a driving clip. Analysis runs offline on your machine (same
          pipeline as the CLI). Results appear when processing finishes.
        </p>
      </header>

      {/* Upload */}
      <section className="transition-all duration-300">
        <div
          className={`rounded-2xl border-2 border-dashed p-8 text-center transition-all duration-200 sm:p-12 ${
            drag
              ? "border-blue-500/80 bg-slate-800/60 shadow-lg shadow-blue-900/20"
              : "border-slate-700 bg-slate-900/40 hover:border-slate-600"
          } ${busy ? "opacity-60" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            if (!busy) setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
        >
          <p className="text-slate-200">
            Drop a video here, or use the button below
          </p>
          <div className="mt-5 flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
            <input
              ref={fileInputRef}
              id="fileInput"
              type="file"
              accept="video/mp4"
              className="sr-only"
              disabled={busy}
              aria-label="Choose a video file to analyze"
              onChange={handleFile}
            />
            <label
              htmlFor="fileInput"
              className={`inline-flex rounded-lg px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors ${
                busy
                  ? "cursor-not-allowed bg-slate-600"
                  : "cursor-pointer bg-blue-600 hover:bg-blue-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950"
              }`}
            >
              Choose Video
            </label>
          </div>
          {selectedFileName ? (
            <p
              className="mt-4 truncate text-sm text-slate-300"
              title={selectedFileName}
            >
              Selected: <span className="font-mono text-slate-200">{selectedFileName}</span>
            </p>
          ) : null}
          <p className="mt-3 text-xs text-slate-500">
            MP4 · local processing · no cloud
          </p>
        </div>
      </section>

      {/* Status strip: stacked layers — only one visible; opacity crossfade */}
      <div
        className={`relative mt-8 ${phase !== "idle" ? "min-h-[5.5rem]" : ""}`}
        aria-live="polite"
        aria-busy={phase === "uploading" || phase === "processing"}
      >
        {phase !== "idle" ? (
          <div className="relative isolate min-h-[5.5rem]">
            <div
              aria-hidden={phase !== "uploading"}
              className={statusLayer(
                phase,
                "uploading",
                "rounded-xl border border-slate-800 bg-slate-900/50 px-4 py-3 text-sm text-slate-300",
              )}
            >
              Uploading video...
            </div>
            <div
              aria-hidden={phase !== "processing"}
              className={statusLayer(
                phase,
                "processing",
                "rounded-xl border border-blue-900/40 bg-blue-950/20 px-4 py-4",
              )}
            >
              <div className="flex items-center gap-3">
                <div
                  className="h-8 w-8 shrink-0 animate-spin rounded-full border-2 border-slate-600 border-t-blue-500"
                  aria-hidden
                />
                <p className="font-medium text-slate-200">Analyzing...</p>
              </div>
            </div>
            <div
              aria-hidden={phase !== "done"}
              className={statusLayer(
                phase,
                "done",
                "rounded-xl border border-emerald-900/40 bg-emerald-950/25 px-4 py-3 text-sm font-medium text-emerald-200/95",
              )}
            >
              Analysis Complete
            </div>
            <div
              aria-hidden={phase !== "error"}
              className={statusLayer(
                phase,
                "error",
                "rounded-xl border border-red-800/60 bg-red-950/35 px-4 py-3 text-sm text-red-100",
                "start",
              )}
              role="alert"
            >
              {error ?? "Something went wrong."}
            </div>
          </div>
        ) : null}
      </div>

      {/* Results */}
      {phase === "done" && reportPayload && (
        <div className="animate-fade-in-up mt-10 space-y-8 sm:space-y-10">
          <div className="grid min-w-0 gap-8 lg:grid-cols-5 lg:gap-10">
            {/* Processed Output */}
            <section className="min-w-0 lg:col-span-3">
              <SectionTitle>Processed Output</SectionTitle>
              <div className="overflow-hidden rounded-2xl border border-slate-800 bg-black shadow-xl shadow-black/40">
                <div className="p-4 sm:p-6">
                  <div className="relative aspect-video overflow-hidden rounded-xl bg-slate-950">
                    {videoSrc ? (
                      <>
                        {!videoReady && (
                          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-slate-950/95">
                            <div className="h-9 w-9 animate-spin rounded-full border-2 border-slate-700 border-t-cyan-500" />
                            <p className="text-xs text-slate-500">Loading video…</p>
                          </div>
                        )}
                        {videoLoadError ? (
                          <div className="absolute bottom-0 left-0 right-0 z-20 border-t border-amber-900/60 bg-amber-950/90 px-3 py-2 text-left text-[11px] leading-snug text-amber-100">
                            {videoLoadError}
                          </div>
                        ) : null}
                        <video
                          key={videoSrc}
                          className="relative z-0 h-full w-full object-contain"
                          controls
                          autoPlay
                          loop
                          muted
                          playsInline
                          preload="auto"
                          src={videoSrc}
                          onLoadedMetadata={() => setVideoReady(true)}
                          onError={() => {
                            setVideoReady(true);
                            setVideoLoadError(
                              "Could not decode this video in the browser. On Windows run `where ffmpeg` in a new terminal; if empty, fix PATH and restart the terminal, then re-run analysis. Check GET /health → ffmpeg_available. Use the same host in the URL as the API (or set NEXT_PUBLIC_API_URL).",
                            );
                          }}
                        />
                      </>
                    ) : (
                      <div className="flex h-full items-center justify-center p-6 text-slate-600">
                        No video file
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </section>

            <aside className="flex min-w-0 flex-col gap-8 lg:col-span-2">
              {/* Analysis Metrics */}
              <section>
                <SectionTitle>Analysis Metrics</SectionTitle>
                <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4 shadow-lg sm:p-6">
                  <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
                    Last frame risk
                  </p>
                  {last?.risk ? (
                    <div className="mt-4">
                      <RiskBadge level={last.risk} />
                    </div>
                  ) : null}
                  {last ? (
                    <div className="mt-6 space-y-6">
                      <ConfidenceBar value={conf} />
                      <div>
                        <h3 className="text-xs font-medium text-slate-500">
                          Primary cause
                        </h3>
                        <p className="mt-2 text-sm text-slate-200">
                          {humanize(last.primary_cause || "") || "—"}
                        </p>
                      </div>
                    </div>
                  ) : null}

                  <div className="mt-8 border-t border-slate-800/90 pt-8">
                    <ExplanationPanel
                      reasons={last?.reasons}
                      contributions={last?.feature_contributions}
                    />
                  </div>

                  {summaryImageSrc ? (
                    <div className="mt-8 border-t border-slate-800/90 pt-8">
                      <h3 className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
                        Session summary
                      </h3>
                      <div className="mt-4 rounded-xl border border-slate-800/80 bg-slate-950/20 p-4">
                        <img
                          src={summaryImageSrc}
                          alt="Session summary"
                          className="w-full rounded-lg border border-slate-800/70"
                        />
                        <p className="mt-2 text-xs text-slate-500">
                          Top risk frames from this session
                        </p>
                      </div>
                    </div>
                  ) : null}

                  {timelineImageSrc ? (
                    <div className="mt-8 border-t border-slate-800/90 pt-8">
                      <h3 className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
                        Risk timeline
                      </h3>
                      <div className="mt-4 rounded-xl border border-slate-800/80 bg-slate-950/20 p-4">
                        <img
                          src={timelineImageSrc}
                          alt="Risk timeline"
                          className="w-full rounded-lg border border-slate-800/70"
                        />
                        <p className="mt-2 text-xs text-slate-500">
                          Confidence and complexity across all frames. Vertical dashed lines = lane departure events.
                        </p>
                      </div>
                    </div>
                  ) : null}

                  {dangerClips.length > 0 ? (
                    <div className="mt-8 border-t border-slate-800/90 pt-8">
                      <h3 className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
                        Flagged danger moments
                      </h3>
                      <div className="mt-4 space-y-4">
                        {dangerClips.map((clipUrl, idx) => (
                          <div
                            key={clipUrl}
                            className="rounded-xl border border-slate-800/80 bg-slate-950/20 p-4"
                          >
                            <video
                              key={clipUrl}
                              src={clipUrl}
                              controls
                              muted
                              preload="metadata"
                              playsInline
                              className="max-w-full rounded-lg"
                            />
                            <p className="mt-2 text-xs text-slate-500">Clip {idx + 1}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  <div className="mt-8 border-t border-slate-800/90 pt-8">
                    <h3 className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
                      Session overview
                    </h3>
                    <div className="mt-4 grid grid-cols-2 gap-3 sm:gap-4">
                      <MetricCard
                        label="Frames processed"
                        value={String(
                          summary?.total_frames ?? apiMeta?.frames ?? "—",
                        )}
                      />
                      <MetricCard
                        label="Scene stability"
                        value={
                          summary?.scene_stability != null
                            ? Number(summary.scene_stability).toFixed(3)
                            : "—"
                        }
                      />
                      <MetricCard
                        label="Risk trend"
                        value={
                          summary?.risk_trend != null
                            ? String(summary.risk_trend)
                            : "—"
                        }
                      />
                      <MetricCard
                        label="Complexity"
                        value={
                          summary?.avg_scene_complexity != null
                            ? Number(summary.avg_scene_complexity).toFixed(3)
                            : "—"
                        }
                      />
                    </div>
                  </div>

                  <div className="mt-8 border-t border-slate-800/90 pt-8">
                    <h3 className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
                      Risk distribution (session)
                    </h3>
                    <div className="mt-4 grid gap-4 sm:grid-cols-3">
                      {Object.entries(dist).map(([label, count], idx) => (
                        <div
                          key={label}
                          className="rounded-xl border border-slate-800/80 bg-slate-950/20 p-4"
                        >
                          <div className="mb-2 flex justify-between text-xs text-slate-400">
                            <span className="font-medium">{label}</span>
                            <span>{count}</span>
                          </div>
                          <div className="h-3 overflow-hidden rounded-full bg-slate-800">
                            <div
                              className={`h-full ${riskBarClass(label)} rg-fill-rect transition-all duration-700 ease-out motion-reduce:transition-none`}
                              style={{
                                width: `${(Number(count) / maxCount) * 100}%`,
                                animationDelay: `${idx * 70}ms`,
                              }}
                            />
                          </div>
                        </div>
                      ))}
                      {Object.keys(dist).length === 0 && (
                        <p className="text-sm text-slate-500">
                          No distribution data
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </section>

              {/* Model Info */}
              <section>
                <SectionTitle>Model Info</SectionTitle>
                <Tilt className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4 shadow-lg sm:p-6">
                  <dl className="space-y-4 text-sm">
                    <div className="flex flex-col gap-0.5 sm:flex-row sm:justify-between sm:gap-4">
                      <dt className="text-slate-500">Model</dt>
                      <dd className="text-slate-100">
                        {apiMeta?.model ?? "Random Forest v2"}
                      </dd>
                    </div>
                    <div className="flex flex-col gap-0.5 sm:flex-row sm:justify-between sm:gap-4">
                      <dt className="text-slate-500">Features</dt>
                      <dd className="font-mono text-slate-200">
                        {apiMeta?.features ?? 9}
                      </dd>
                    </div>
                    <div className="border-t border-slate-800 pt-4">
                      <dt className="text-slate-500">Training</dt>
                      <dd className="mt-2 text-slate-200">
                        Real + Synthetic
                        {apiMeta?.training_data ? (
                          <span className="mt-2 block text-xs leading-snug text-slate-500">
                            {apiMeta.training_data}
                          </span>
                        ) : null}
                      </dd>
                    </div>
                  </dl>
                  {featureImportanceRows.length > 0 ? (
                    <div className="mt-8 border-t border-slate-800 pt-6">
                      <h3 className="text-sm font-semibold text-slate-200">
                        What the model learned.
                      </h3>
                      <div className="mt-4 rounded-xl border border-slate-800/80 bg-slate-950/25 p-4">
                        <svg viewBox="0 0 760 320" className="w-full">
                          {featureImportanceRows.map(([name, raw], idx) => {
                            const value = Number(raw) || 0;
                            const y = 24 + idx * 30;
                            const barW = (value / maxImportance) * 360;
                            return (
                              <g key={name} transform={`translate(0, ${y})`}>
                                <text
                                  x="8"
                                  y="14"
                                  fill="#94a3b8"
                                  fontSize="12"
                                  fontFamily="monospace"
                                >
                                  {name}
                                </text>
                                <rect
                                  x="320"
                                  y="2"
                                  width="380"
                                  height="16"
                                  rx="6"
                                  fill="#1e293b"
                                />
                                <rect
                                  x="320"
                                  y="2"
                                  width={Math.max(4, barW)}
                                  height="16"
                                  rx="6"
                                  fill="#38bdf8"
                                  className="rg-bar-rect"
                                  style={{
                                    animationDelay: `${idx * 80}ms`,
                                  }}
                                />
                                <text
                                  x="710"
                                  y="14"
                                  textAnchor="end"
                                  fill="#e2e8f0"
                                  fontSize="12"
                                  fontFamily="monospace"
                                >
                                  {(value * 100).toFixed(1)}%
                                </text>
                              </g>
                            );
                          })}
                        </svg>
                        <p className="mt-3 text-xs text-slate-500">
                          Trained: {modelInfo?.timestamp ?? "unknown"} · Samples:{" "}
                          {modelInfo?.n_samples ?? 0}
                        </p>
                      </div>
                    </div>
                  ) : null}
                </Tilt>
              </section>
            </aside>
          </div>
        </div>
      )}

      <footer className="mt-12 border-t border-slate-800 pt-6 text-center text-xs text-slate-600 sm:mt-16">
        API: <code className="text-slate-500">{apiBase}</code>
        <span className="mx-2">·</span>
        CLI:{" "}
        <code className="text-slate-500">python main.py --source sample</code>
      </footer>
      </main>
    </>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-800/90 bg-slate-950/30 p-4 transition-colors duration-300 hover:border-slate-700/90 sm:p-5">
      <div className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div className="mt-2 break-words font-mono text-sm text-slate-200">
        {value}
      </div>
    </div>
  );
}
