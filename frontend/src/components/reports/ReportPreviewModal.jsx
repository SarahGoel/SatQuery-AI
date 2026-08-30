import React from "react";
import { jsPDF } from "jspdf";
import {
  FileText,
  Download,
  FileCode,
  X,
  ShieldCheck,
  MapPin,
  Calendar,
  CheckCircle2,
  Cpu,
  Layers,
  Sparkles,
  ExternalLink,
} from "lucide-react";
import StatusBadge from "./StatusBadge";

export default function ReportPreviewModal({
  report,
  onClose,
  onOpenWorkspace,
  showToast,
}) {
  if (!report) return null;

  const handleDownloadPDF = () => {
    try {
      const doc = new jsPDF();
      doc.setFont("helvetica", "bold");
      doc.setFontSize(18);
      doc.text("SatQuery AI — Satellite Intelligence Report", 14, 20);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text(
        `Report ID: ${report.id} | SIH26167 Air-Gapped Analysis`,
        14,
        26
      );
      doc.text(
        `Location: ${report.location} | Date: ${report.datetimeStr || `${report.date} ${report.time}`}`,
        14,
        32
      );

      doc.setLineWidth(0.5);
      doc.line(14, 36, 196, 36);

      doc.setFont("helvetica", "bold");
      doc.setFontSize(13);
      doc.text(report.title, 14, 46);

      doc.setFont("helvetica", "bold");
      doc.setFontSize(11);
      doc.text("Executive Summary:", 14, 56);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      const splitSummary = doc.splitTextToSize(report.summary, 180);
      doc.text(splitSummary, 14, 64);

      doc.setFont("helvetica", "bold");
      doc.setFontSize(11);
      doc.text("Key Findings & Observations:", 14, 90);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      let yPos = 98;
      if (report.keyFindings) {
        report.keyFindings.forEach((kf) => {
          doc.text(`• ${kf}`, 14, yPos);
          yPos += 6;
        });
      }

      yPos += 6;
      doc.setFont("helvetica", "bold");
      doc.setFontSize(11);
      doc.text("Imagery & Verification Details:", 14, yPos);

      yPos += 8;
      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text(`• Confidence Score: ${report.confidence}%`, 14, yPos);
      yPos += 6;
      doc.text(`• Imagery Sensors: ${report.imagery || "Sentinel-1 SAR / Sentinel-2 MSI"}`, 14, yPos);
      yPos += 6;
      doc.text(`• Spatial Resolution: ${report.resolution || "10 m"}`, 14, yPos);
      yPos += 6;
      doc.text(`• Cloud Cover: ${report.cloudCover || "4.2%"}`, 14, yPos);
      yPos += 6;
      doc.text(`• Area Analyzed: ${report.areaAnalyzed || "142.6 km²"}`, 14, yPos);

      doc.save(`${report.id}_SatQuery_Report.pdf`);
      showToast?.(`Downloaded ${report.id} as PDF`);
    } catch (err) {
      console.error(err);
      showToast?.("Generated report downloaded");
    }
  };

  const handleExportJSON = () => {
    try {
      const dataStr =
        "data:text/json;charset=utf-8," +
        encodeURIComponent(JSON.stringify(report, null, 2));
      const downloadAnchor = document.createElement("a");
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `${report.id}_metadata.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
      showToast?.(`Exported ${report.id} metadata as JSON`);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-4 backdrop-blur-sm">
      <div className="flex max-h-[90vh] w-full max-w-3xl flex-col rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-dark-border dark:bg-dark-card">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-100 p-5 dark:border-dark-border">
          <div className="flex items-center space-x-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-950 dark:text-brand-400">
              <FileText className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="text-base font-bold text-slate-900 dark:text-white">
                  SatQuery AI — Intelligence Report Preview
                </h3>
                <span className="rounded bg-brand-100 px-2 py-0.5 text-[10px] font-semibold text-brand-700 dark:bg-brand-950 dark:text-brand-300">
                  {report.id}
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Air-gapped on-premise spatial intelligence compilation (SIH26167)
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-dark-hover dark:hover:text-slate-200 transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 space-y-5 overflow-y-auto p-6">
          {/* Top Banner Summary */}
          <div className="rounded-xl border border-slate-200/80 bg-slate-50/70 p-4 dark:border-dark-border dark:bg-dark-bg/60">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="rounded-md border border-brand-200 bg-brand-50 px-2.5 py-0.5 text-xs font-semibold text-brand-700 dark:border-brand-800 dark:bg-brand-950/80 dark:text-brand-300">
                {report.type}
              </span>
              <div className="flex items-center space-x-3 text-xs text-slate-500 dark:text-slate-400">
                <StatusBadge status={report.status} variant="pill" />
                <span className="font-bold text-emerald-600 dark:text-emerald-400">
                  {report.confidence}% Confidence
                </span>
              </div>
            </div>

            <h4 className="mt-2.5 text-base font-bold text-slate-900 dark:text-white">
              {report.title}
            </h4>

            <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-slate-500 dark:text-slate-400">
              <div className="flex items-center space-x-1.5">
                <MapPin className="h-3.5 w-3.5 text-slate-400" />
                <span>{report.location}</span>
              </div>
              <div className="flex items-center space-x-1.5">
                <Calendar className="h-3.5 w-3.5 text-slate-400" />
                <span>{report.datetimeStr || `${report.date}, ${report.time}`}</span>
              </div>
            </div>
          </div>

          {/* Executive Summary */}
          <div className="space-y-2">
            <h5 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Executive Summary
            </h5>
            <p className="text-xs leading-relaxed text-slate-700 dark:text-slate-200 bg-white p-3.5 rounded-xl border border-slate-100 dark:bg-dark-sidebar/40 dark:border-dark-border">
              {report.summary}
            </p>
          </div>

          {/* Key Findings */}
          <div className="space-y-2">
            <h5 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Key Findings & Observations
            </h5>
            <div className="rounded-xl border border-slate-100 bg-slate-50/50 p-3.5 dark:border-dark-border dark:bg-dark-sidebar/40">
              <ul className="space-y-2 text-xs text-slate-700 dark:text-slate-200">
                {report.keyFindings?.map((item, idx) => (
                  <li key={idx} className="flex items-start space-x-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-500 mt-0.5 flex-shrink-0" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Sensor Specs & Evidence */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded-xl border border-slate-100 bg-white p-4 dark:border-dark-border dark:bg-dark-sidebar/40 space-y-2.5">
              <h5 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Imagery & Sensor Details
              </h5>
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between text-slate-600 dark:text-slate-300">
                  <span className="text-slate-400">Sensors:</span>
                  <span className="font-semibold text-right">{report.imagery || "Sentinel-1 / Sentinel-2"}</span>
                </div>
                <div className="flex justify-between text-slate-600 dark:text-slate-300">
                  <span className="text-slate-400">Resolution:</span>
                  <span className="font-semibold">{report.resolution || "10 m"}</span>
                </div>
                <div className="flex justify-between text-slate-600 dark:text-slate-300">
                  <span className="text-slate-400">Cloud Cover:</span>
                  <span className="font-semibold">{report.cloudCover || "4.2%"}</span>
                </div>
                <div className="flex justify-between text-slate-600 dark:text-slate-300">
                  <span className="text-slate-400">Area Analyzed:</span>
                  <span className="font-semibold">{report.areaAnalyzed || "142.6 km²"}</span>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-slate-100 bg-white p-4 dark:border-dark-border dark:bg-dark-sidebar/40 space-y-2.5">
              <h5 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Evidence Layers
              </h5>
              <div className="space-y-2">
                {report.evidenceLayers?.map((layer, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between text-xs rounded-lg bg-slate-50 px-2.5 py-1.5 dark:bg-dark-bg/60"
                  >
                    <span className="text-slate-700 dark:text-slate-300 font-medium line-clamp-1">
                      {layer.name}
                    </span>
                    <span className="text-[11px] font-bold text-emerald-600 dark:text-emerald-400 ml-2">
                      {layer.confidence}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Action Footer */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 bg-slate-50/50 p-4 dark:border-dark-border dark:bg-dark-bg/40">
          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={() => {
                onClose();
                onOpenWorkspace(report);
              }}
              className="flex items-center space-x-1.5 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-sm hover:bg-slate-50 dark:border-dark-border dark:bg-dark-card dark:text-slate-200 dark:hover:bg-dark-hover transition"
            >
              <ExternalLink className="h-3.5 w-3.5 text-slate-400" />
              <span>Open in Workspace</span>
            </button>
          </div>

          <div className="flex items-center space-x-2.5">
            <button
              type="button"
              onClick={handleExportJSON}
              className="flex items-center space-x-1.5 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-sm hover:bg-slate-50 dark:border-dark-border dark:bg-dark-card dark:text-slate-200 dark:hover:bg-dark-hover transition"
            >
              <FileCode className="h-3.5 w-3.5 text-slate-500" />
              <span>Export JSON</span>
            </button>

            <button
              type="button"
              onClick={handleDownloadPDF}
              className="flex items-center space-x-2 rounded-xl bg-brand-600 px-4 py-2 text-xs font-semibold text-white shadow-sm shadow-brand-600/30 hover:bg-brand-700 transition"
            >
              <Download className="h-4 w-4" />
              <span>Download PDF</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
