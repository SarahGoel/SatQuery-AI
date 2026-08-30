import React from "react";
import { ArrowUpDown } from "lucide-react";
import HistoryRow from "./HistoryRow";

export default function AnalysisHistoryTable({
  analyses,
  selectedAnalysis,
  onSelectAnalysis,
  onViewReport,
  onOpenWorkspace,
  onDownloadReport,
  onToggleSave,
  onDeleteAnalysis,
  sortOrder,
  onToggleSort,
}) {
  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200/80 bg-white shadow-sm dark:border-dark-border dark:bg-dark-card">
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-200/80 bg-slate-50/50 text-[11px] font-semibold text-slate-500 dark:border-dark-border dark:bg-dark-sidebar/50 dark:text-slate-400">
              <th scope="col" className="py-3 pl-4 pr-3">
                Analysis
              </th>
              <th scope="col" className="px-3 py-3">
                Location
              </th>
              <th scope="col" className="px-3 py-3">
                Analysis Type
              </th>
              <th scope="col" className="px-3 py-3">
                <button
                  type="button"
                  onClick={onToggleSort}
                  className="flex items-center space-x-1 hover:text-slate-800 dark:hover:text-slate-200 transition"
                  title="Sort by date"
                >
                  <span>Date & Time</span>
                  <ArrowUpDown className="h-3 w-3 text-brand-600 dark:text-brand-400" />
                </button>
              </th>
              <th scope="col" className="px-3 py-3">
                Confidence
              </th>
              <th scope="col" className="px-3 py-3">
                Status
              </th>
              <th scope="col" className="py-3 pl-3 pr-4 text-right">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-dark-border">
            {analyses.map((analysis) => (
              <HistoryRow
                key={analysis.id}
                analysis={analysis}
                isSelected={selectedAnalysis?.id === analysis.id}
                onSelect={onSelectAnalysis}
                onViewReport={onViewReport}
                onOpenWorkspace={onOpenWorkspace}
                onDownloadReport={onDownloadReport}
                onToggleSave={onToggleSave}
                onDelete={onDeleteAnalysis}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
