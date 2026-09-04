import React from "react";
import {
  Sparkles,
  X,
  Waves,
  Leaf,
  RefreshCw,
  Droplets,
  AlertTriangle,
  Building,
  Building2,
  FileText,
  MapPin,
  Trash2,
} from "lucide-react";
import StatusBadge from "./StatusBadge";

export default function AnalysisDetailsPanel({
  analysis,
  onClose,
  onViewReport,
  onOpenWorkspace,
  onDelete,
}) {
  if (!analysis) return null;

  const getTypeIcon = () => {
    switch (analysis.type) {
      case "Flood Risk":
        return (
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-100 text-blue-600 dark:bg-blue-950 dark:text-blue-400">
            <Waves className="h-6 w-6" />
          </div>
        );
      case "NDVI":
        return (
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-emerald-100 text-emerald-600 dark:bg-emerald-950 dark:text-emerald-400">
            <Leaf className="h-6 w-6" />
          </div>
        );
      case "Change Detection":
        return (
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-amber-100 text-amber-600 dark:bg-amber-950 dark:text-amber-400">
            <RefreshCw className="h-6 w-6" />
          </div>
        );
      case "Water Detection":
        return (
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-cyan-100 text-cyan-600 dark:bg-cyan-950 dark:text-cyan-400">
            <Droplets className="h-6 w-6" />
          </div>
        );
      case "Disaster Assessment":
        return (
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-red-100 text-red-600 dark:bg-red-950 dark:text-red-400">
            <AlertTriangle className="h-6 w-6" />
          </div>
        );
      case "Risk Assessment":
        return (
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-amber-100 text-amber-600 dark:bg-amber-950 dark:text-amber-400">
            <Building className="h-6 w-6" />
          </div>
        );
      case "Urban Analysis":
      default:
        return (
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-purple-100 text-purple-600 dark:bg-purple-950 dark:text-purple-400">
            <Building2 className="h-6 w-6" />
          </div>
        );
    }
  };

  return (
    <div className="flex h-full flex-col rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm dark:border-dark-border dark:bg-dark-card">
      {/* Panel Top Bar */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-3 dark:border-dark-border">
        <div className="flex items-center space-x-2">
          <Sparkles className="h-4 w-4 text-brand-600 dark:text-brand-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 dark:text-slate-200">
            Analysis Details
          </h3>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-lg p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-dark-hover dark:hover:text-slate-200 transition"
          title="Close details"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Main Content Scrollable Area */}
      <div className="mt-4 flex-1 space-y-4 overflow-y-auto pr-1">
        {/* Hero Item Badge */}
        <div className="flex items-start space-x-3.5">
          {getTypeIcon()}
          <div className="flex-1">
            <div className="flex items-center justify-between gap-2">
              <h4 className="text-sm font-bold text-slate-900 dark:text-white">
                {analysis.title}
              </h4>
              <StatusBadge status={analysis.status} variant="pill" />
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              {analysis.location}
            </p>
            <p className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
              {analysis.datetimeStr || `${analysis.date}, ${analysis.time}`}
            </p>
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 gap-3 border-y border-slate-100 py-3.5 dark:border-dark-border">
          <div>
            <span className="text-[11px] text-slate-400 dark:text-slate-500">
              Analysis Type
            </span>
            <p className="text-xs font-semibold text-slate-800 dark:text-slate-200 mt-0.5">
              {analysis.type}
            </p>
          </div>

          <div>
            <span className="text-[11px] text-slate-400 dark:text-slate-500">
              Confidence
            </span>
            <p className="text-xs font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">
              {analysis.confidence}%
            </p>
          </div>

          <div>
            <span className="text-[11px] text-slate-400 dark:text-slate-500">
              Imagery
            </span>
            <p className="text-xs font-medium text-slate-800 dark:text-slate-200 mt-0.5">
              {analysis.imagery || "Sentinel-1 SAR, Sentinel-2 MSI"}
            </p>
          </div>

          <div>
            <span className="text-[11px] text-slate-400 dark:text-slate-500">
              Resolution
            </span>
            <p className="text-xs font-medium text-slate-800 dark:text-slate-200 mt-0.5">
              {analysis.resolution || "10 m"}
            </p>
          </div>

          <div>
            <span className="text-[11px] text-slate-400 dark:text-slate-500">
              Cloud Cover
            </span>
            <p className="text-xs font-medium text-slate-800 dark:text-slate-200 mt-0.5">
              {analysis.cloudCover || "4.2%"}
            </p>
          </div>

          <div>
            <span className="text-[11px] text-slate-400 dark:text-slate-500">
              Area Analyzed
            </span>
            <p className="text-xs font-medium text-slate-800 dark:text-slate-200 mt-0.5">
              {analysis.areaAnalyzed || "142.6 km²"}
            </p>
          </div>
        </div>

        {/* Summary */}
        <div className="space-y-1.5">
          <h5 className="text-xs font-bold text-slate-900 dark:text-white">
            Summary
          </h5>
          <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
            {analysis.summary}
          </p>
        </div>

        {/* Key Findings */}
        <div className="space-y-1.5">
          <h5 className="text-xs font-bold text-slate-900 dark:text-white">
            Key Findings
          </h5>
          <ul className="space-y-1.5 text-xs text-slate-600 dark:text-slate-300">
            {analysis.keyFindings?.map((finding, idx) => (
              <li key={idx} className="flex items-start space-x-2">
                <span className="text-brand-500 mt-0.5 font-bold">•</span>
                <span className="leading-snug">{finding}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="mt-5 space-y-2 pt-3 border-t border-slate-100 dark:border-dark-border">
        <button
          type="button"
          onClick={() => onViewReport(analysis)}
          className="flex w-full items-center justify-center space-x-2 rounded-xl bg-brand-600 py-2.5 text-xs font-semibold text-white shadow-sm shadow-brand-600/30 transition hover:bg-brand-700"
        >
          <FileText className="h-4 w-4" />
          <span>View Report</span>
        </button>

        <button
          type="button"
          onClick={() => onOpenWorkspace(analysis)}
          className="flex w-full items-center justify-center space-x-2 rounded-xl border border-slate-200 bg-slate-50/60 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-100 dark:border-dark-border dark:bg-dark-sidebar/60 dark:text-slate-200 dark:hover:bg-dark-hover"
        >
          <MapPin className="h-4 w-4 text-slate-400" />
          <span>Open in Geospatial Workspace</span>
        </button>

        <button
          type="button"
          onClick={() => onDelete(analysis)}
          className="flex w-full items-center justify-center space-x-2 py-1.5 text-xs font-medium text-red-600 transition hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
        >
          <Trash2 className="h-3.5 w-3.5" />
          <span>Delete Analysis</span>
        </button>
      </div>
    </div>
  );
}
