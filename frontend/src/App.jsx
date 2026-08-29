import { useMemo, useState } from "react";
import MapViewer from "./components/MapViewer.jsx";
import QueryInput from "./components/QueryInput.jsx";
import TraceViewer from "./components/TraceViewer.jsx";
import ReportDownloader from "./components/ReportDownloader.jsx";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export default function App() {
  const [query, setQuery] = useState(
    "Highlight built-up area change between T1 and T2 over this Cartosat scene."
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [trace, setTrace] = useState(null);
  const [geojson, setGeojson] = useState(null);
  const [health, setHealth] = useState(null);

  const bounds = useMemo(() => {
    if (!trace?.input_metadata?.bounds) return null;
    return trace.input_metadata.bounds;
  }, [trace]);

  async function refreshHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/v1/health`);
      setHealth(await res.json());
    } catch (err) {
      setHealth({ status: "unreachable", error: String(err) });
    }
  }

  async function onAnalyze({ files }) {
    setBusy(true);
    setError(null);
    const body = new FormData();
    body.append("query", query);
    if (files.optical) body.append("optical", files.optical);
    if (files.opticalT2) body.append("optical_t2", files.opticalT2);
    if (files.sar) body.append("sar", files.sar);
    try {
      const res = await fetch(`${API_BASE}/api/v1/satquery/analyze`, {
        method: "POST",
        body,
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      const payload = await res.json();
      setTrace(payload.trace);
      setGeojson(payload.geojson);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen bg-ink text-slate-100">
      <header className="flex items-center justify-between border-b border-slate-800 px-6 py-4">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-accent">SIH26167</p>
          <h1 className="text-2xl font-semibold">SatQuery AI</h1>
          <p className="text-sm text-slate-400">
            Agentic EO assistant — air-gapped, on-premise GPU
          </p>
        </div>
        <button
          type="button"
          onClick={refreshHealth}
          className="rounded-md border border-slate-700 px-3 py-1.5 text-sm hover:border-accent"
        >
          Health {health?.gpu_available ? "· GPU" : ""}
        </button>
      </header>

      <main className="grid gap-4 p-4 lg:grid-cols-12">
        <section className="space-y-4 lg:col-span-8">
          <QueryInput
            query={query}
            onQueryChange={setQuery}
            busy={busy}
            onSubmit={onAnalyze}
          />
          {error ? (
            <p className="rounded-md border border-red-500/40 bg-red-950/40 px-3 py-2 text-sm text-red-200">
              {error}
            </p>
          ) : null}
          <MapViewer geojson={geojson} bounds={bounds} />
        </section>
        <aside className="space-y-4 lg:col-span-4">
          <TraceViewer trace={trace} />
          <ReportDownloader trace={trace} geojson={geojson} />
          {health ? (
            <pre className="overflow-auto rounded-md bg-panel p-3 text-xs text-slate-400">
              {JSON.stringify(health, null, 2)}
            </pre>
          ) : null}
        </aside>
      </main>
    </div>
  );
}
