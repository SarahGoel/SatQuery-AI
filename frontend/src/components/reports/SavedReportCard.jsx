import React from "react";
import {
  FileText,
  Download,
  BookmarkCheck,
  MapPin,
  Calendar,
  Layers,
  ExternalLink,
} from "lucide-react";
import StatusBadge from "./StatusBadge";

export default function SavedReportCard({
  report,
  onViewReport,
  onDownloadReport,
  onToggleSave,
  onOpenWorkspace,
}) {
  return (
    <div className="flex flex-col justify-between rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm transition-all duration-200 hover:shadow-md dark:border-dark-border dark:bg-dark-card">
      <div>
        {/* Top Badges */}
        <div className="flex items-center justify-between gap-2">
          <span className="inline-flex items-center rounded-md border border-brand-200/60 bg-brand-50 px-2.5 py-0.5 text-[11px] font-semibold text-brand-700 dark:border-brand-800/40 dark:bg-brand-950/60 dark:text-brand-300">
            {report.type}
          </span>
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400">
              {report.confidence}% Conf.
            </span>
            <StatusBadge status={report.status} />
          </div>
        </div>

        {/* Title */}
        <h4 className="mt-3 text-sm font-bold text-slate-900 dark:text-white line-clamp-2">
          {report.title}
        </h4>

        {/* Location & Date */}
        <div className="mt-2 space-y-1 text-xs text-slate-500 dark:text-slate-400">
          <div className="flex items-center space-x-1.5">
            <MapPin className="h-3.5 w-3.5 text-slate-400 flex-shrink-0" />
            <span className="line-clamp-1">{report.location}</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <Calendar className="h-3.5 w-3.5 text-slate-400 flex-shrink-0" />
            <span>{report.datetimeStr || `${report.date}, ${report.time}`}</span>
          </div>
        </div>

        {/* Summary Snippet */}
        <p className="mt-3 text-xs leading-relaxed text-slate-600 dark:text-slate-300 line-clamp-3">
          {report.summary}
        </p>
      </div>

      {/* Action Footer */}
      <div className="mt-5 flex items-center justify-between border-t border-slate-100 pt-3 dark:border-dark-border">
        <div className="flex items-center space-x-1">
          <button
            type="button"
            onClick={() => onToggleSave(report)}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 text-purple-600 hover:bg-purple-50 dark:border-dark-border dark:text-purple-400 dark:hover:bg-purple-950/40 transition"
            title="Remove from saved"
          >
            <BookmarkCheck className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => onDownloadReport(report)}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-dark-border dark:text-slate-300 dark:hover:bg-dark-hover transition"
            title="Download PDF report"
          >
            <Download className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => onOpenWorkspace(report)}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 dark:border-dark-border dark:text-slate-300 dark:hover:bg-dark-hover transition"
            title="Open in Workspace"
          >
            <ExternalLink className="h-4 w-4" />
          </button>
        </div>

        <button
          type="button"
          onClick={() => onViewReport(report)}
          className="flex items-center space-x-1.5 rounded-xl bg-brand-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow-sm shadow-brand-600/30 transition hover:bg-brand-700"
        >
          <FileText className="h-3.5 w-3.5" />
          <span>View Report</span>
        </button>
      </div>
    </div>
  );
}
