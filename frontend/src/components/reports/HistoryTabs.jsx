import React from "react";

export default function HistoryTabs({ activeTab, onTabChange }) {
  return (
    <div className="flex border-b border-slate-200 dark:border-dark-border">
      <button
        type="button"
        onClick={() => onTabChange("all")}
        className={`relative pb-3 pt-1 text-sm font-semibold transition-colors ${
          activeTab === "all"
            ? "text-brand-600 dark:text-brand-400"
            : "text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200"
        }`}
      >
        <span>All Analyses</span>
        {activeTab === "all" && (
          <span className="absolute bottom-0 left-0 right-0 h-0.5 rounded-t-full bg-brand-600 dark:bg-brand-500" />
        )}
      </button>

      <button
        type="button"
        onClick={() => onTabChange("saved")}
        className={`relative ml-6 pb-3 pt-1 text-sm font-semibold transition-colors ${
          activeTab === "saved"
            ? "text-brand-600 dark:text-brand-400"
            : "text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200"
        }`}
      >
        <span>Saved Reports</span>
        {activeTab === "saved" && (
          <span className="absolute bottom-0 left-0 right-0 h-0.5 rounded-t-full bg-brand-600 dark:bg-brand-500" />
        )}
      </button>
    </div>
  );
}
