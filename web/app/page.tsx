"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  FeatureImportanceChart,
  HeroIllustration,
  HeroMetricCard,
  IconBrain,
  IconChart,
  IconFilm,
  IconLayers,
  IconTarget,
  SampleOutputGallery,
  UploadWaves,
} from "@/components/dashboard-ui";

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
  hyperparameters?: Record<string, number | string>;
  train_test_split?: {
    test_size?: number;
    random_state?: number;
    stratify?: boolean;
    n_train?: number;
    n_test?: number;
  };
  evaluation?: {
    accuracy?: number;
    precision_macro?: number;
    recall_macro?: number;
    f1_macro?: number;
    per_class?: Record<
      string,
      { precision?: number; recall?: number; f1?: number }
    >;
    confusion_matrix?: number[][];
  };
  confusion_matrix_url?: string | null;
  classification_report_url?: string | null;
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
  if (label === "HIGH") return "bg-[#b91c1c]";
  if (label === "MEDIUM") return "bg-[#c2410c]";
  return "bg-accent-blue";
}

function humanize(s: string): string {
  return s.replace(/_/g, " ").trim();
}

function SubSection({
  title,
  children,
  className = "",
}: {
  title: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`mt-8 border-t border-[var(--rg-border)] pt-8 ${className}`}>
      <h3 className="rg-stat-label">{title}</h3>
      {children}
    </div>
  );
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
  return <h2 className="rg-section-title mb-4">{children}</h2>;
}

function UploadIcon() {
  return (
    <div className="rg-upload-icon-wrap mx-auto flex h-20 w-20 items-center justify-center rounded-2xl">
      <svg
        className="h-9 w-9 text-accent-purple"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.5}
        aria-hidden
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
        />
      </svg>
    </div>
  );
}

function RiskBadge({ level }: { level: string }) {
  const cls =
    level === "HIGH"
      ? "rg-risk-high"
      : level === "MEDIUM"
        ? "rg-risk-medium"
        : "rg-risk-low";

  return (
    <div
      className={`flex w-full items-center justify-center rounded-lg px-4 py-5 text-center font-display text-3xl uppercase tracking-[0.05em] sm:text-4xl ${cls}`}
      aria-label={`Risk level ${level}`}
    >
      {level}
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div>
      <div className="mb-2 flex justify-between text-xs text-zinc-500">
        <span>Inference confidence</span>
        <span className="font-mono text-zinc-100">{pct.toFixed(0)}%</span>
      </div>
      <div className="rg-bar-track">
        <div className="rg-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function Panel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`rg-card ${className}`}>{children}</div>;
}

function ExplanationPanel({
  risk,
  confidence,
  primaryCause,
  summary,
  reasons,
  contributions,
}: {
  risk?: string;
  confidence?: number;
  primaryCause?: string;
  summary?: string;
  reasons: string[] | undefined;
  contributions: Record<string, number> | undefined;
}) {
  const topFactors = contributions
    ? Object.entries(contributions)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 8)
    : [];
  const maxContrib = Math.max(
    0.0001,
    ...topFactors.map(([, v]) => Math.abs(Number(v) || 0)),
  );

  return (
    <div className="space-y-4">
      {summary ? (
        <div className="rg-card-inset p-4 sm:p-5">
          <h3 className="rg-eyebrow">Session explanation</h3>
          <p className="mt-3 text-sm leading-relaxed text-zinc-300">{summary}</p>
        </div>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rg-card-inset p-4">
          <h3 className="rg-stat-label">Predicted risk</h3>
          <p className="rg-stat-value mt-2">{risk ?? "—"}</p>
        </div>
        <div className="rg-card-inset p-4">
          <h3 className="rg-stat-label">Confidence</h3>
          <p className="rg-stat-value mt-2">
            {confidence != null ? `${(confidence * 100).toFixed(1)}%` : "—"}
          </p>
        </div>
      </div>

      <div className="rg-card-inset p-4 sm:p-5">
        <h3 className="rg-stat-label">Primary cause</h3>
        <p className="mt-2 text-sm font-medium text-zinc-200">
          {humanize(primaryCause || "") || "—"}
        </p>
      </div>

      <div className="rg-card-inset p-4 sm:p-5">
        <h3 className="rg-stat-label">Why this risk?</h3>
        {reasons && reasons.length > 0 ? (
          <ul className="mt-3 space-y-2 text-sm leading-relaxed text-zinc-300">
            {reasons.slice(0, 12).map((r) => (
              <li key={r} className="flex gap-2">
                <span className="text-accent-purple">—</span>
                <span>{humanize(r)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-zinc-500">
            No rule-based triggers; model confidence drives the label.
          </p>
        )}
      </div>

      {topFactors.length > 0 ? (
        <div className="rg-card-inset p-4 sm:p-5">
          <h3 className="rg-stat-label">Feature contributions</h3>
          <ul className="mt-4 space-y-3">
            {topFactors.map(([k, v]) => {
              const value = Number(v) || 0;
              const pct = (Math.abs(value) / maxContrib) * 100;
              return (
                <li key={k}>
                  <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="font-mono text-zinc-300">{humanize(k)}</span>
                    <span className="font-mono text-zinc-500">
                      {(value * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="rg-bar-track mt-1.5 h-1.5">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-accent-purple to-accent-blue"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function ModelEvaluationPanel({
  modelInfo,
  apiBase,
}: {
  modelInfo: ModelInfoResponse | null;
  apiBase: string;
}) {
  const evaluation = modelInfo?.evaluation;
  const split = modelInfo?.train_test_split;
  const hasMetrics =
    evaluation &&
    (evaluation.accuracy != null ||
      evaluation.f1_macro != null ||
      evaluation.precision_macro != null);
  const confusionSrc = resolveMediaUrl(
    apiBase,
    modelInfo?.confusion_matrix_url ?? null,
  );

  if (!hasMetrics && !confusionSrc) {
    return (
      <p className="mt-6 text-xs leading-relaxed text-zinc-500">
        Run{" "}
        <code className="font-mono text-accent-purple/90">python train_model.py</code> from{" "}
        <code className="font-mono text-accent-purple/90">roadguard_x/</code> to generate held-out
        test metrics and a confusion matrix.
      </p>
    );
  }

  const fmtPct = (v: number | undefined) =>
    v != null ? `${(v * 100).toFixed(2)}%` : "—";
  const fmtScore = (v: number | undefined) =>
    v != null ? v.toFixed(4) : "—";

  return (
    <div className="mt-6">
      <div className="flex items-center gap-2">
        <div className="rg-icon-blue flex h-8 w-8 items-center justify-center rounded-lg">
          <IconChart />
        </div>
        <h3 className="font-display text-base font-semibold text-zinc-100">Model evaluation</h3>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-zinc-500">
        Held-out synthetic test set — not per-upload accuracy.
      </p>
      <p className="mt-1 text-xs text-zinc-600">
        {split?.n_train ?? "—"} train / {split?.n_test ?? "—"} test
        {split?.test_size != null
          ? ` · ${Math.round((1 - split.test_size) * 100)}/${Math.round(split.test_size * 100)} split`
          : ""}
      </p>

      <div className="mt-4 grid grid-cols-2 gap-2.5">
        <MetricCard label="Test accuracy" value={fmtPct(evaluation?.accuracy)} accent="purple" />
        <MetricCard label="F1 (macro)" value={fmtScore(evaluation?.f1_macro)} accent="blue" />
        <MetricCard label="Precision" value={fmtScore(evaluation?.precision_macro)} accent="blue" />
        <MetricCard label="Recall" value={fmtScore(evaluation?.recall_macro)} accent="purple" />
        <MetricCard label="Train samples" value={String(modelInfo?.n_samples ?? "—")} />
        <MetricCard label="Test samples" value={String(split?.n_test ?? "—")} />
      </div>

      {evaluation?.per_class ? (
        <div className="rg-glass-inset mt-4 overflow-x-auto rounded-xl">
          <table className="w-full min-w-[260px] text-left text-xs">
            <thead>
              <tr className="border-b border-white/[0.06] text-zinc-500">
                <th className="px-3 py-2.5 font-medium">Class</th>
                <th className="px-3 py-2.5 font-medium">Prec</th>
                <th className="px-3 py-2.5 font-medium">Rec</th>
                <th className="px-3 py-2.5 font-medium">F1</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(evaluation.per_class).map(([label, scores]) => (
                <tr key={label} className="border-b border-white/[0.04] text-zinc-300">
                  <td className="px-3 py-2 font-mono text-accent-purple">{label}</td>
                  <td className="px-3 py-2 font-mono">{fmtScore(scores.precision)}</td>
                  <td className="px-3 py-2 font-mono">{fmtScore(scores.recall)}</td>
                  <td className="px-3 py-2 font-mono">{fmtScore(scores.f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {confusionSrc ? (
        <div className="rg-matrix-frame mt-4">
          <p className="rg-stat-label relative z-10">Confusion matrix</p>
          <img
            src={confusionSrc}
            alt="Confusion matrix"
            className="relative z-10 mt-3 w-full rounded-lg border border-white/[0.08] bg-white shadow-lg"
          />
        </div>
      ) : null}

      {modelInfo?.hyperparameters &&
      Object.keys(modelInfo.hyperparameters).length > 0 ? (
        <div className="rg-glass-inset mt-4 rounded-xl p-4">
          <p className="rg-stat-label">Hyperparameters</p>
          <dl className="mt-2 space-y-1.5 font-mono text-xs">
            {Object.entries(modelInfo.hyperparameters).map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4 text-zinc-500">
                <dt>{k}</dt>
                <dd className="text-zinc-200">{String(v)}</dd>
              </div>
            ))}
          </dl>
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
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
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

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const setFilePreview = useCallback((file: File) => {
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
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
      setError(
        `Lost connection to API at ${apiBase}. Render free tier may be waking up — wait 30s and retry.`,
      );
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
        const msg = e instanceof Error ? e.message : "Request failed";
        setError(
          msg === "Failed to fetch"
            ? `Cannot reach API at ${apiBase}. If this is a deployed site, redeploy the API with CORS for your frontend (Vercel → Render). On Render free tier, wait ~30s and retry (cold start).`
            : msg,
        );
      }
    },
    [apiBase, pollStatus, startPolling],
  );

  const handleFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) {
        setSelectedFileName(f.name);
        setFilePreview(f);
        void onAnalyze(f);
      }
    },
    [onAnalyze, setFilePreview],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDrag(false);
      if (phase === "uploading" || phase === "processing") return;
      const f = e.dataTransfer.files?.[0];
      if (f) {
        setSelectedFileName(f.name);
        setFilePreview(f);
        void onAnalyze(f);
      }
    },
    [onAnalyze, phase, setFilePreview],
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
        explanation?: {
          risk?: string;
          confidence?: number;
          primary_cause?: string;
          reasons?: string[];
          feature_contributions?: Record<string, number>;
          summary?: string;
          top_features?: string[];
        };
      }
    | undefined;

  const apiMeta = reportPayload?.meta;
  const explanation = innerMeta?.explanation ?? innerMeta?.last_frame;
  const last = innerMeta?.last_frame ?? innerMeta?.explanation;
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
      <div className="rg-bg" aria-hidden>
        <div className="rg-bg-orb rg-bg-orb-1" />
        <div className="rg-bg-orb rg-bg-orb-2" />
      </div>

      <main className="relative z-10 mx-auto min-h-screen max-w-7xl px-4 py-8 sm:px-6 sm:py-10 lg:px-8">
        {/* ── Hero ── */}
        <header className="mb-10">
          <div className="grid items-center gap-8 lg:grid-cols-2 lg:gap-12">
            <div>
              <h1 className="rg-display text-5xl sm:text-6xl lg:text-7xl">
                RoadGuard-X
              </h1>
              <p className="mt-5 max-w-lg text-base leading-relaxed text-zinc-400">
                Explainable driving-scene intelligence — OpenCV feature extraction,
                Random Forest classification, and rule-based reasoning. Fully offline.
              </p>
              <div className="mt-7 flex flex-wrap gap-2">
                <span className="rg-pill">
                  <span className="rg-pill-live" /> Local Processing
                </span>
                <span className="rg-pill">OpenCV</span>
                <span className="rg-pill">Random Forest</span>
                <span className="rg-pill">Explainable AI</span>
              </div>
            </div>
            <HeroIllustration />
          </div>

          {modelInfo?.evaluation?.accuracy != null ? (
            <div className="mt-8 grid grid-cols-2 gap-3 lg:grid-cols-4">
              <HeroMetricCard
                label="Model test accuracy"
                value={`${(modelInfo.evaluation.accuracy * 100).toFixed(1)}%`}
                icon={<IconTarget />}
                accent="purple"
                sparkPoints={[0.94, 0.96, 0.97, 0.98, 0.99, 0.996]}
              />
              <HeroMetricCard
                label="F1 (macro)"
                value={modelInfo.evaluation.f1_macro?.toFixed(3) ?? "—"}
                icon={<IconChart />}
                accent="blue"
                sparkPoints={[0.97, 0.98, 0.985, 0.99, 0.992, 0.995]}
              />
              <HeroMetricCard
                label="Features"
                value="9"
                icon={<IconLayers />}
                accent="blue"
                sparkPoints={[6, 7, 8, 8, 9, 9]}
              />
              <HeroMetricCard
                label="Demo clip"
                value="demo.mp4"
                icon={<IconFilm />}
                accent="purple"
                sparkPoints={[1, 1, 1, 1, 1, 1]}
              />
            </div>
          ) : null}
        </header>

        {/* ── Main + Sidebar ── */}
        <div className="grid gap-6 xl:grid-cols-[1fr_340px] xl:gap-8">
          <div className="min-w-0 space-y-6">
            {/* Upload */}
            <section>
              <div
                className={`rg-upload ${drag ? "rg-upload-drag" : ""} ${busy ? "pointer-events-none opacity-60" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  if (!busy) setDrag(true);
                }}
                onDragLeave={() => setDrag(false)}
                onDrop={onDrop}
              >
                <div className="rg-upload-border" />
                <UploadWaves />
                <div className="rg-upload-inner px-6 py-10 text-center sm:px-10 sm:py-14">
                  {previewUrl ? (
                    <div className="relative z-10 mx-auto mb-5 max-w-sm overflow-hidden rounded-lg border border-white/[0.08]">
                      <video
                        src={previewUrl}
                        className="aspect-video w-full object-cover"
                        muted
                        playsInline
                        preload="metadata"
                      />
                      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-[#05070A] to-transparent px-3 py-2">
                        <p className="truncate font-mono text-xs text-zinc-300">{selectedFileName}</p>
                      </div>
                    </div>
                  ) : (
                    <UploadIcon />
                  )}
                  <p className="relative z-10 mt-5 font-display text-xl font-semibold text-white sm:text-2xl">
                    Analyse a driving clip
                  </p>
                  <p className="relative z-10 mx-auto mt-2 max-w-md text-sm text-zinc-500">
                    Drag & drop or browse — try the bundled{" "}
                    <span className="font-mono text-accent-blue">demo.mp4</span>
                  </p>
                  <div className="relative z-10 mt-8 flex flex-col items-center gap-3">
                    <input
                      ref={fileInputRef}
                      id="fileInput"
                      type="file"
                      accept="video/mp4"
                      className="sr-only"
                      disabled={busy}
                      aria-label="Choose a video file to analyse"
                      onChange={handleFile}
                    />
                    <label
                      htmlFor="fileInput"
                      className={`rg-btn relative z-10 ${busy ? "cursor-not-allowed opacity-50" : "cursor-pointer"}`}
                    >
                      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
                      </svg>
                      Choose video
                    </label>
                  </div>
                  {selectedFileName && !previewUrl ? (
                    <p
                      className="relative z-10 mx-auto mt-5 max-w-md truncate rounded-lg border border-white/[0.08] bg-black/40 px-3 py-2 text-sm text-zinc-300 backdrop-blur-sm"
                      title={selectedFileName}
                    >
                      <span className="text-zinc-500">Selected · </span>
                      <span className="font-mono">{selectedFileName}</span>
                    </p>
                  ) : null}
                  <div className="relative z-10 mt-5 flex flex-wrap items-center justify-center gap-2">
                    <span className="rg-tag">MP4</span>
                    <span className="rg-tag">Local inference</span>
                    <span className="rg-tag rg-tag-accent">No cloud</span>
                  </div>
                </div>
              </div>
            </section>

            {/* Status */}
            <div
              className={`relative ${phase !== "idle" ? "min-h-[5.5rem]" : ""}`}
              aria-live="polite"
              aria-busy={phase === "uploading" || phase === "processing"}
            >
              {phase !== "idle" ? (
                <div className="relative isolate min-h-[5.5rem]">
                  <div
                    aria-hidden={phase !== "uploading"}
                    className={statusLayer(phase, "uploading", "rg-card px-4 py-3 text-sm text-zinc-400")}
                  >
                    <span className="text-accent-purple">↑</span> Uploading video...
                  </div>
                  <div
                    aria-hidden={phase !== "processing"}
                    className={statusLayer(
                      phase,
                      "processing",
                      "rg-card border border-accent-purple/15 px-4 py-4",
                    )}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="h-8 w-8 shrink-0 animate-spin rounded-full border-2 border-zinc-700 border-t-accent-purple"
                        aria-hidden
                      />
                      <div>
                        <p className="font-medium text-zinc-100">Analysing frames</p>
                        <p className="text-xs text-zinc-500">OpenCV · Random Forest · explainability</p>
                      </div>
                    </div>
                  </div>
                  <div
                    aria-hidden={phase !== "done"}
                    className={statusLayer(
                      phase,
                      "done",
                      "rg-card border border-accent-purple/20 px-4 py-3 text-sm font-medium text-accent-purple",
                    )}
                  >
                    Analysis complete
                  </div>
                  <div
                    aria-hidden={phase !== "error"}
                    className={statusLayer(
                      phase,
                      "error",
                      "rg-card-inset border border-red-500/30 px-4 py-3 text-sm text-red-200",
                      "start",
                    )}
                    role="alert"
                  >
                    {error ?? "Something went wrong."}
                  </div>
                </div>
              ) : null}
            </div>

            <SampleOutputGallery />

            {/* Results */}
            {phase === "done" && reportPayload && (
              <div className="animate-fade-in-up space-y-10">
                <section>
                  <SectionTitle>Processed output</SectionTitle>
                  <Panel className="overflow-hidden shadow-lift">
                    <div className="p-4 sm:p-6">
                      <div className="relative aspect-video overflow-hidden rounded-lg bg-[#05070A]">
                        {videoSrc ? (
                          <>
                            {!videoReady && (
                              <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-[#05070A]/95">
                                <div className="h-9 w-9 animate-spin rounded-full border-2 border-zinc-700 border-t-accent-purple" />
                                <p className="text-xs text-zinc-500">Loading video…</p>
                              </div>
                            )}
                            {videoLoadError ? (
                              <div className="absolute bottom-0 left-0 right-0 z-20 border-t border-red-900/60 bg-red-950/90 px-3 py-2 text-left text-[11px] leading-snug text-red-100">
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
                          <div className="flex h-full items-center justify-center p-6 text-zinc-500">
                            No video file
                          </div>
                        )}
                      </div>
                    </div>
                  </Panel>
                </section>

                <section>
                  <SectionTitle>Analysis metrics</SectionTitle>
                  <Panel className="p-5 sm:p-6">
                    <div className="grid gap-6 lg:grid-cols-2">
                      <div>
                        <p className="rg-stat-label">Last frame risk</p>
                        {last?.risk ? (
                          <div className="mt-4">
                            <RiskBadge level={last.risk} />
                          </div>
                        ) : null}
                        {last ? (
                          <div className="mt-6">
                            <ConfidenceBar value={conf} />
                          </div>
                        ) : null}
                      </div>
                      <SubSection title="Session overview" className="mt-0 border-0 pt-0">
                        <div className="mt-4 grid grid-cols-2 gap-3">
                          <MetricCard
                            label="Frames processed"
                            value={String(summary?.total_frames ?? apiMeta?.frames ?? "—")}
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
                            value={summary?.risk_trend != null ? String(summary.risk_trend) : "—"}
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
                      </SubSection>
                    </div>

                    <div className="mt-8 border-t border-white/[0.06] pt-8">
                      <ExplanationPanel
                        risk={explanation?.risk ?? last?.risk}
                        confidence={explanation?.confidence ?? last?.confidence}
                        primaryCause={explanation?.primary_cause ?? last?.primary_cause}
                        summary={innerMeta?.explanation?.summary ?? undefined}
                        reasons={explanation?.reasons ?? last?.reasons}
                        contributions={
                          explanation?.feature_contributions ?? last?.feature_contributions
                        }
                      />
                    </div>

                    <div className="mt-8 grid gap-6 lg:grid-cols-2">
                      {summaryImageSrc ? (
                        <SubSection title="Session summary" className="mt-0 border-0 pt-0">
                          <div className="rg-card-inset mt-4 p-4">
                            <img
                              src={summaryImageSrc}
                              alt="Session summary"
                              className="w-full rounded-lg border border-white/[0.06]"
                            />
                            <p className="mt-2 text-xs text-zinc-500">Top risk frames from this session</p>
                          </div>
                        </SubSection>
                      ) : null}
                      {timelineImageSrc ? (
                        <SubSection title="Risk timeline" className="mt-0 border-0 pt-0">
                          <div className="rg-card-inset mt-4 p-4">
                            <img
                              src={timelineImageSrc}
                              alt="Risk timeline"
                              className="w-full rounded-lg border border-white/[0.06]"
                            />
                            <p className="mt-2 text-xs text-zinc-500">
                              Confidence and complexity across all frames.
                            </p>
                          </div>
                        </SubSection>
                      ) : null}
                    </div>

                    {dangerClips.length > 0 ? (
                      <SubSection title="Flagged danger moments">
                        <div className="mt-4 grid gap-4 sm:grid-cols-2">
                          {dangerClips.map((clipUrl, idx) => (
                            <div key={clipUrl} className="rg-card-inset p-4">
                              <video
                                key={clipUrl}
                                src={clipUrl}
                                controls
                                muted
                                preload="metadata"
                                playsInline
                                className="max-w-full rounded-lg"
                              />
                              <p className="mt-2 text-xs text-zinc-500">Clip {idx + 1}</p>
                            </div>
                          ))}
                        </div>
                      </SubSection>
                    ) : null}

                    <SubSection title="Risk distribution (session)">
                      <div className="mt-4 grid gap-4 sm:grid-cols-3">
                        {Object.entries(dist).map(([label, count], idx) => (
                          <div key={label} className="rg-card-inset p-4">
                            <div className="mb-2 flex justify-between text-xs text-zinc-500">
                              <span className="font-medium text-zinc-200">{label}</span>
                              <span className="font-mono">{count}</span>
                            </div>
                            <div className="rg-bar-track h-3">
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
                          <p className="text-sm text-zinc-500">No distribution data</p>
                        )}
                      </div>
                    </SubSection>
                  </Panel>
                </section>
              </div>
            )}
          </div>

          {/* ── Sticky sidebar ── */}
          <aside className="rg-sidebar space-y-6">
            <Panel className="p-5">
              <div className="flex items-center gap-2">
                <div className="rg-icon-purple flex h-8 w-8 items-center justify-center rounded-lg">
                  <IconBrain />
                </div>
                <h2 className="font-display text-sm font-semibold text-zinc-100">Model info</h2>
              </div>
              <dl className="mt-5 space-y-4 text-sm">
                <div className="flex justify-between gap-4">
                  <dt className="rg-stat-label">Model</dt>
                  <dd className="text-right text-zinc-200">
                    {apiMeta?.model ?? "Random Forest v2"}
                  </dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="rg-stat-label">Features</dt>
                  <dd className="font-mono text-zinc-200">{apiMeta?.features ?? 9}</dd>
                </div>
                <div className="border-t border-white/[0.06] pt-4">
                  <dt className="rg-stat-label">Training</dt>
                  <dd className="mt-2 text-zinc-300">
                    Real + Synthetic
                    {(apiMeta?.training_data || modelInfo?.n_samples) ? (
                      <span className="mt-2 block text-xs leading-snug text-zinc-500">
                        {apiMeta?.training_data ??
                          `${modelInfo?.n_samples ?? 0} synthetic samples`}
                      </span>
                    ) : null}
                  </dd>
                </div>
              </dl>
              {featureImportanceRows.length > 0 ? (
                <div className="mt-6 border-t border-white/[0.06] pt-5">
                  <h3 className="text-sm font-semibold text-zinc-200">What the model learned</h3>
                  <FeatureImportanceChart
                    rows={featureImportanceRows}
                    maxImportance={maxImportance}
                    timestamp={modelInfo?.timestamp}
                    nSamples={modelInfo?.n_samples}
                  />
                </div>
              ) : null}
              <ModelEvaluationPanel modelInfo={modelInfo} apiBase={apiBase} />
            </Panel>
          </aside>
        </div>

        <footer className="mt-16 border-t border-white/[0.06] pt-8 text-center text-xs text-zinc-600">
          <span>API </span>
          <code className="font-mono text-accent-blue/90">{apiBase}</code>
          <span className="mx-2 opacity-40">·</span>
          <span>CLI </span>
          <code className="font-mono text-zinc-400">python main.py --source sample</code>
        </footer>
      </main>
    </>
  );
}

function MetricCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: "purple" | "blue";
}) {
  const accentBorder =
    accent === "purple"
      ? "border-accent-purple/15"
      : accent === "blue"
        ? "border-accent-blue/15"
        : "";

  return (
    <div className={`rg-stat-tile ${accentBorder}`}>
      <div className="rg-stat-label">{label}</div>
      <div className="rg-stat-value text-sm">{value}</div>
    </div>
  );
}
