import { jsPDF } from "jspdf";

export default function ReportDownloader({ trace, geojson }) {
  function download() {
    if (!trace) return;
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    doc.setFontSize(16);
    doc.text("SatQuery AI — Analysis Report", 40, 48);
    doc.setFontSize(10);
    doc.text(`Problem statement: SIH26167`, 40, 68);
    doc.text(`Trace ID: ${trace.trace_id}`, 40, 84);
    doc.text(`Task: ${trace.task}`, 40, 100);
    doc.text(`Confidence: ${trace.confidence_score}`, 40, 116);
    doc.text("Query:", 40, 140);
    const queryLines = doc.splitTextToSize(trace.query, 515);
    doc.text(queryLines, 40, 156);
    let y = 156 + queryLines.length * 12 + 16;
    doc.text("Output:", 40, y);
    const outLines = doc.splitTextToSize(trace.output || "", 515);
    doc.text(outLines, 40, y + 16);
    y = y + 16 + outLines.length * 12 + 16;
    doc.text("Registry execution:", 40, y);
    trace.registry_execution?.forEach((step, idx) => {
      y += 14;
      doc.text(`${idx + 1}. ${step.model}`, 48, y);
    });
    if (geojson?.features) {
      y += 20;
      doc.text(`GeoJSON features: ${geojson.features.length}`, 40, y);
    }
    doc.save(`satquery-${trace.trace_id}.pdf`);
  }

  return (
    <button
      type="button"
      disabled={!trace}
      onClick={download}
      className="w-full rounded-md border border-slate-700 px-4 py-2 text-sm hover:border-accent disabled:opacity-40"
    >
      Download PDF report
    </button>
  );
}
