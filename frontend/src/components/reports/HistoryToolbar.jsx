import React from "react";
import { Search, Calendar, RotateCcw } from "lucide-react";
import FilterDropdown from "./FilterDropdown";
import {
  ANALYSIS_TYPES,
  LOCATIONS,
  STATUS_OPTIONS,
} from "../../mock/reportsData";

export default function HistoryToolbar({
  searchQuery,
  onSearchChange,
  dateRange,
  onDateRangeChange,
  analysisType,
  onAnalysisTypeChange,
  location,
  onLocationChange,
  status,
  onStatusChange,
  onClearFilters,
  hasActiveFilters,
}) {
  const dateRangeOptions = [
    "All Time",
    "Last 24 Hours",
    "Last 7 Days",
    "Last 30 Days",
    "May 2025",
    "Custom Range...",
  ];

  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      {/* Search Input with correct left padding */}
      <div className="relative flex-1 min-w-[240px] max-w-md">
        <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3.5">
          <Search className="h-4 w-4 text-slate-400 dark:text-slate-500" />
        </div>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search analyses..."
          className="w-full rounded-xl border border-slate-200 bg-white py-2 pl-10 pr-4 text-xs text-slate-900 placeholder-slate-400 transition focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500 dark:border-dark-border dark:bg-dark-card dark:text-white dark:placeholder-slate-500"
        />
      </div>

      {/* Filter Dropdowns + Clear Filters */}
      <div className="flex flex-wrap items-center gap-2.5">
        <FilterDropdown
          label="Date Range"
          value={dateRange}
          options={dateRangeOptions}
          onChange={onDateRangeChange}
          icon={Calendar}
          rightIcon={Calendar}
        />

        <FilterDropdown
          label="Analysis Type"
          value={analysisType}
          options={ANALYSIS_TYPES}
          onChange={onAnalysisTypeChange}
        />

        <FilterDropdown
          label="Location"
          value={location}
          options={LOCATIONS}
          onChange={onLocationChange}
        />

        <FilterDropdown
          label="Status"
          value={status}
          options={STATUS_OPTIONS}
          onChange={onStatusChange}
        />

        <button
          type="button"
          onClick={onClearFilters}
          className={`flex items-center space-x-1.5 rounded-xl border px-3 py-2 text-xs font-medium transition ${
            hasActiveFilters
              ? "border-brand-300 bg-brand-50/60 text-brand-700 hover:bg-brand-100/70 dark:border-brand-800 dark:bg-brand-950/40 dark:text-brand-300"
              : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900 dark:border-dark-border dark:bg-dark-card dark:text-slate-300 dark:hover:bg-dark-hover dark:hover:text-white"
          }`}
          title="Reset all filters"
        >
          <RotateCcw className="h-3.5 w-3.5 text-slate-400 dark:text-slate-500" />
          <span>Clear Filters</span>
        </button>
      </div>
    </div>
  );
}
