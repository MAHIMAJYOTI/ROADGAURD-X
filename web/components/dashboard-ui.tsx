"use client";

import type { ReactNode } from "react";

const PURPLE = "#6D5EF7";
const BLUE = "#4F8CFF";

/* ─── Sparkline ─── */
export function Sparkline({
  points,
  color = PURPLE,
  className = "",
}: {
  points: number[];
  color?: string;
  className?: string;
}) {
  const w = 72;
  const h = 24;
  const max = Math.max(...points, 0.001);
  const min = Math.min(...points);
  const range = max - min || 1;
  const id = `spark-${color.replace("#", "")}`;
  const coords = points
    .map((p, i) => {
      const x = (i / (points.length - 1)) * w;
      const y = h - ((p - min) / range) * (h - 4) - 2;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className={`h-6 w-[4.5rem] opacity-70 ${className}`} aria-hidden>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor={color} stopOpacity="0.15" />
          <stop offset="100%" stopColor={color} stopOpacity="0.85" />
        </linearGradient>
      </defs>
      <polyline
        fill="none"
        stroke={`url(#${id})`}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={coords}
      />
    </svg>
  );
}

/* ─── Hero metric card ─── */
export function HeroMetricCard({
  label,
  value,
  icon,
  accent,
  sparkPoints,
}: {
  label: string;
  value: string;
  icon: ReactNode;
  accent: "purple" | "blue";
  sparkPoints: number[];
}) {
  const isPurple = accent === "purple";
  const spark = isPurple ? PURPLE : BLUE;

  return (
    <div className="rg-metric-card group relative overflow-hidden rounded-xl p-4 sm:p-5">
      <div className="flex items-start justify-between gap-2">
        <div
          className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${isPurple ? "rg-icon-purple" : "rg-icon-blue"}`}
        >
          {icon}
        </div>
        <Sparkline points={sparkPoints} color={spark} />
      </div>
      <p className="rg-stat-label mt-3.5">{label}</p>
      <p className="mt-1 truncate font-mono text-xl font-semibold tracking-tight text-white sm:text-2xl">
        {value}
      </p>
    </div>
  );
}

/* ─── Hero CV illustration ─── */
export function HeroIllustration() {
  return (
    <div className="rg-hero-visual relative mx-auto w-full max-w-md lg:max-w-none">
      <div className="rg-hero-glow pointer-events-none absolute -inset-6 rounded-full" aria-hidden />
      <div className="rg-glass relative overflow-hidden rounded-xl">
        <svg viewBox="0 0 480 300" className="w-full" aria-hidden>
          <defs>
            <linearGradient id="road-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0D1117" />
              <stop offset="100%" stopColor="#05070A" />
            </linearGradient>
            <linearGradient id="lane-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={PURPLE} stopOpacity="0.35" />
              <stop offset="100%" stopColor={BLUE} stopOpacity="0.1" />
            </linearGradient>
          </defs>
          <rect width="480" height="300" fill="url(#road-grad)" />
          <path d="M120 300 L200 120 L280 120 L360 300 Z" fill="#161b22" opacity="0.95" />
          <path d="M200 120 L240 120 L240 300 L200 300 Z" fill="url(#lane-grad)" opacity="0.4" />
          <line x1="220" y1="130" x2="220" y2="290" stroke={PURPLE} strokeWidth="1" strokeDasharray="10 8" opacity="0.35" />
          {[0, 1, 2, 3, 4].map((i) => (
            <line key={i} x1="0" y1={60 + i * 48} x2="480" y2={60 + i * 48} stroke="#ffffff" strokeWidth="0.5" opacity="0.03" />
          ))}
          <rect x="155" y="175" width="52" height="32" rx="3" fill="none" stroke={PURPLE} strokeWidth="1.2" />
          <rect x="268" y="158" width="48" height="28" rx="3" fill="none" stroke={BLUE} strokeWidth="1.2" opacity="0.85" />
          <rect x="310" y="195" width="44" height="26" rx="3" fill="none" stroke="#f97316" strokeWidth="1.2" opacity="0.65" />
          <circle cx="400" cy="72" r="44" fill="none" stroke={PURPLE} strokeWidth="0.7" opacity="0.2" />
          <circle cx="400" cy="72" r="28" fill="none" stroke={BLUE} strokeWidth="0.7" opacity="0.25" />
          <circle cx="400" cy="72" r="12" fill="none" stroke={PURPLE} strokeWidth="0.8" opacity="0.35" />
          <line x1="400" y1="72" x2="428" y2="52" stroke={BLUE} strokeWidth="1.2" opacity="0.6" className="rg-radar-sweep" />
          <circle cx="416" cy="56" r="2.5" fill={PURPLE} opacity="0.8" />
        </svg>
        <div className="absolute bottom-3 left-3 rounded-lg border border-white/[0.08] bg-[#05070A]/80 px-3 py-2 backdrop-blur-sm">
          <p className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">Risk score</p>
          <p className="font-mono text-base font-semibold text-white">0.18</p>
          <p className="text-[10px] text-accent-blue">Low risk</p>
        </div>
        <div className="absolute right-3 top-3 flex items-center gap-1.5 rounded-full border border-white/[0.08] bg-[#0D1117]/90 px-2.5 py-1">
          <span className="rg-pulse-dot h-1.5 w-1.5 rounded-full" />
          <span className="text-[10px] font-medium text-zinc-400">Live CV</span>
        </div>
      </div>
    </div>
  );
}

/* ─── Sample output gallery ─── */
const SAMPLE_FRAMES = [
  { risk: "LOW", label: "Low risk", boxes: [{ x: 30, y: 55, w: 40, h: 22, color: BLUE }] },
  {
    risk: "MEDIUM",
    label: "Medium risk",
    boxes: [
      { x: 20, y: 50, w: 35, h: 20, color: "#f97316" },
      { x: 55, y: 58, w: 30, h: 18, color: "#fb923c" },
    ],
  },
  {
    risk: "HIGH",
    label: "High risk",
    boxes: [
      { x: 15, y: 48, w: 38, h: 22, color: "#ef4444" },
      { x: 50, y: 52, w: 42, h: 24, color: "#f87171" },
    ],
  },
  { risk: "LOW", label: "Clear lane", boxes: [{ x: 35, y: 58, w: 36, h: 20, color: PURPLE }] },
  { risk: "HIGH", label: "Lane departure", boxes: [{ x: 25, y: 45, w: 50, h: 28, color: "#ef4444" }] },
] as const;

function SampleFrame({
  risk,
  label,
  boxes,
}: {
  risk: string;
  label: string;
  boxes: readonly { x: number; y: number; w: number; h: number; color: string }[];
}) {
  const badgeCls =
    risk === "HIGH"
      ? "bg-red-500/15 text-red-300 border-red-500/25"
      : risk === "MEDIUM"
        ? "bg-orange-500/15 text-orange-300 border-orange-500/25"
        : "bg-accent-blue/10 text-[#93b4ff] border-accent-blue/20";

  return (
    <div className="rg-sample-card group shrink-0 snap-center">
      <div className="relative aspect-[4/3] overflow-hidden rounded-lg border border-white/[0.06] bg-[#0D1117]">
        <svg viewBox="0 0 100 80" className="h-full w-full" aria-hidden>
          <rect width="100" height="80" fill="#05070A" />
          <path d="M20 80 L35 25 L65 25 L80 80 Z" fill="#161b22" />
          <line x1="50" y1="28" x2="50" y2="78" stroke={PURPLE} strokeWidth="0.5" strokeDasharray="3 3" opacity="0.25" />
          {boxes.map((b, i) => (
            <rect key={i} x={b.x} y={b.y} width={b.w} height={b.h} fill="none" stroke={b.color} strokeWidth="1.2" rx="1" />
          ))}
        </svg>
        <span className={`absolute left-2 top-2 rounded border px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide ${badgeCls}`}>
          {risk}
        </span>
      </div>
      <p className="mt-1.5 text-center text-xs text-zinc-500">{label}</p>
    </div>
  );
}

export function SampleOutputGallery() {
  return (
    <section>
      <div className="mb-4">
        <h2 className="rg-section-title">Sample output</h2>
        <p className="mt-1 text-sm text-zinc-500">Risk-labelled detections from the analysis pipeline</p>
      </div>
      <div className="rg-gallery-scroll flex gap-3 overflow-x-auto pb-1">
        {SAMPLE_FRAMES.map((frame, i) => (
          <SampleFrame key={i} {...frame} />
        ))}
      </div>
    </section>
  );
}

/* ─── Feature importance chart ─── */
const BAR_COLORS = [
  "#6D5EF7",
  "#7B6FF8",
  "#4F8CFF",
  "#5B96FF",
  "#8B7CF9",
  "#6A9AFF",
  "#9D8FFA",
  "#7AA8FF",
  "#A89BFB",
];

export function FeatureImportanceChart({
  rows,
  maxImportance,
  timestamp,
  nSamples,
}: {
  rows: [string, number][];
  maxImportance: number;
  timestamp?: string | null;
  nSamples?: number;
}) {
  if (rows.length === 0) return null;

  return (
    <div className="rg-glass-inset mt-3 rounded-lg p-3">
      <svg viewBox="0 0 760 320" className="w-full">
        {rows.map(([name, raw], idx) => {
          const value = Number(raw) || 0;
          const y = 24 + idx * 30;
          const barW = (value / maxImportance) * 360;
          const color = BAR_COLORS[idx % BAR_COLORS.length];
          return (
            <g key={name} transform={`translate(0, ${y})`}>
              <text x="8" y="14" fill="#71717a" fontSize="10" fontFamily="var(--font-mono)">
                {name}
              </text>
              <rect x="320" y="2" width="380" height="14" rx="7" fill="rgba(255,255,255,0.04)" />
              <rect
                x="320"
                y="2"
                width={Math.max(4, barW)}
                height="14"
                rx="7"
                fill={color}
                className="rg-bar-rect"
                style={{ animationDelay: `${idx * 70}ms` }}
              />
              <text x="710" y="13" textAnchor="end" fill="#d4d4d8" fontSize="10" fontFamily="var(--font-mono)">
                {(value * 100).toFixed(1)}%
              </text>
            </g>
          );
        })}
      </svg>
      <p className="mt-2 text-[11px] text-zinc-500">
        Trained: {timestamp ?? "unknown"} · Samples: {nSamples ?? 0}
      </p>
    </div>
  );
}

/* ─── Icons ─── */
export function IconTarget() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="4" />
      <path d="M12 3v2M12 19v2M3 12h2M19 12h2" />
    </svg>
  );
}

export function IconChart() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" d="M4 18V8M9 18V4M14 18v-6M19 18v-10" />
    </svg>
  );
}

export function IconLayers() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
    </svg>
  );
}

export function IconFilm() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <rect x="2" y="4" width="20" height="16" rx="2" />
      <path d="M2 8h4M2 12h4M2 16h4M18 8h4M18 12h4M18 16h4" />
    </svg>
  );
}

export function IconBrain() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" d="M9.5 4A5.5 5.5 0 0115 9.5M9.5 20A5.5 5.5 0 014 14.5M14.5 20A5.5 5.5 0 0120 14.5M9.5 4A5.5 5.5 0 014 9.5" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  );
}

export function UploadWaves() {
  return (
    <div className="rg-upload-waves pointer-events-none absolute inset-0 overflow-hidden rounded-xl" aria-hidden>
      <div className="rg-wave rg-wave-1" />
      <div className="rg-wave rg-wave-2" />
    </div>
  );
}
