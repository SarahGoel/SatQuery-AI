import React from "react";
import { FileSearch, Plus, RotateCcw } from "lucide-react";

export default function EmptyHistoryState({
  hasFilters,
  onResetFilters,
  onStartAnalysis,
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-white/60 p-12 text-center dark:border-dark-border dark:bg-dark-card/50">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-50 text-brand-600 dark:bg-brand-950/60 dark:text-brand-400">
        <FileSearch className="h-7 w-7" />
      </div>

      <h3 className="mt-4 text-base font-bold text-slate-900 dark:text-white">
        {hasFilters ? "No matching analyses found" : "No analyses yet"}
      </h3>

      <p className="mt-1 max-w-sm text-xs text-slate-500 dark:text-slate-400">
        {hasFilters
          ? "No satellite analyses match your active search filters. Try clearing your filters or changing keywords."
          : "Your completed satellite analyses will appear here."}
      </p>

      <div className="mt-6 flex items-center space-x-3">
        {hasFilters ? (
          <button
            type="button"
            onClick={onResetFilters}
            className="flex items-center space-x-1.5 rounded-xl border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-700 shadow-sm hover:bg-slate-50 dark:border-dark-border dark:bg-dark-card dark:text-slate-200 dark:hover:bg-dark-hover transition"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Reset Filters</span>
          </button>
        ) : (
          <button
            type="button"
            onClick={onStartAnalysis}
            className="flex items-center space-x-1.5 rounded-xl bg-brand-600 px-4 py-2 text-xs font-semibold text-white shadow-sm shadow-brand-600/30 hover:bg-brand-700 transition"
          >
            <Plus className="h-4 w-4" />
            <span>Start an Analysis</span>
          </button>
        )}
      </div>
    </div>
  );
}
