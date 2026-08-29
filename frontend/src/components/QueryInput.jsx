import { useRef, useState } from "react";

export default function QueryInput({ query, onQueryChange, busy, onSubmit }) {
  const opticalRef = useRef(null);
  const t2Ref = useRef(null);
  const sarRef = useRef(null);
  const [names, setNames] = useState({ optical: "", t2: "", sar: "" });

  function handleSubmit(event) {
    event.preventDefault();
    onSubmit({
      files: {
        optical: opticalRef.current?.files?.[0],
        opticalT2: t2Ref.current?.files?.[0],
        sar: sarRef.current?.files?.[0],
      },
    });
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-3 rounded-lg border border-slate-800 bg-panel p-4"
    >
      <label className="block text-sm text-slate-400">
        Natural-language EO query
        <textarea
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          rows={3}
          className="mt-1 w-full rounded-md border border-slate-700 bg-ink px-3 py-2 text-slate-100 outline-none focus:border-accent"
        />
      </label>
      <div className="grid gap-3 md:grid-cols-3">
        <FileSlot
          label="Optical GeoTIFF (required)"
          inputRef={opticalRef}
          required
          onChange={(name) => setNames((s) => ({ ...s, optical: name }))}
          hint={names.optical}
        />
        <FileSlot
          label="T2 optical (change)"
          inputRef={t2Ref}
          onChange={(name) => setNames((s) => ({ ...s, t2: name }))}
          hint={names.t2}
        />
        <FileSlot
          label="SAR / RISAT"
          inputRef={sarRef}
          onChange={(name) => setNames((s) => ({ ...s, sar: name }))}
          hint={names.sar}
        />
      </div>
      <button
        type="submit"
        disabled={busy}
        className="rounded-md bg-accent px-4 py-2 font-medium text-ink disabled:opacity-50"
      >
        {busy ? "Running agent…" : "Analyze"}
      </button>
    </form>
  );
}

function FileSlot({ label, inputRef, required, onChange, hint }) {
  return (
    <label className="block text-xs text-slate-400">
      {label}
      <input
        ref={inputRef}
        type="file"
        accept=".tif,.tiff,.gtiff"
        required={required}
        onChange={(e) => onChange(e.target.files?.[0]?.name || "")}
        className="mt-1 block w-full text-xs"
      />
      {hint ? <span className="text-accent">{hint}</span> : null}
    </label>
  );
}
