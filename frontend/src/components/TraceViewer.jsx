import { useState } from "react";

export default function TraceViewer({ trace }) {
  if (!trace) {
    return (
      <div className="rounded-lg border border-dashed border-slate-700 p-4 text-sm text-slate-500">
        Auditable trace will appear here after an analysis run.
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-slate-800 bg-panel p-4">
      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-accent">
        Auditable execution trace
      </h2>
      <p className="mb-3 text-xs text-slate-400">
        {trace.task} · confidence {(trace.confidence_score * 100).toFixed(1)}%
      </p>
      <Node label="trace" value={trace} />
    </div>
  );
}

function Node({ label, value, depth = 0 }) {
  const [open, setOpen] = useState(depth < 1);
  const isObject = value !== null && typeof value === "object";

  if (!isObject) {
    return (
      <div className="font-mono text-xs">
        <span className="text-slate-500">{label}: </span>
        <span className="text-slate-200">{JSON.stringify(value)}</span>
      </div>
    );
  }

  const keys = Array.isArray(value) ? value.map((_, i) => i) : Object.keys(value);
  return (
    <div className="ml-1">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="font-mono text-xs text-slate-300 hover:text-accent"
      >
        {open ? "▾" : "▸"} {label}
        {Array.isArray(value) ? ` [${value.length}]` : ""}
      </button>
      {open ? (
        <div className="ml-3 border-l border-slate-700 pl-2">
          {keys.map((key) => (
            <Node key={String(key)} label={String(key)} value={value[key]} depth={depth + 1} />
          ))}
        </div>
      ) : null}
    </div>
  );
}
