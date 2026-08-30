import React, { useState, useRef, useEffect } from "react";
import {
  Waves,
  Leaf,
  RefreshCw,
  Droplets,
  AlertTriangle,
  Building,
  Building2,
  MoreHorizontal,
  Eye,
  FileText,
  Download,
  Bookmark,
  BookmarkCheck,
  Trash2,
  ExternalLink,
} from "lucide-react";
import StatusBadge from "./StatusBadge";

export default function HistoryRow({
  analysis,
  isSelected,
  onSelect,
  onViewReport,
  onOpenWorkspace,
  onDownloadReport,
  onToggleSave,
  onDelete,
}) {
  const [showMenu, setShowMenu] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setShowMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const getTypeIcon = () => {
    switch (analysis.type) {
      case "Flood Risk":
        return (
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-blue-600 dark:bg-blue-950 dark:text-blue-400">
            <Waves className="h-4 w-4" />
          </div>
        );
      case "NDVI":
        return (
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-emerald-100 text-emerald-600 dark:bg-emerald-950 dark:text-emerald-400">
            <Leaf className="h-4 w-4" />
          </div>
        );
      case "Change Detection":
        return (
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-amber-100 text-amber-600 dark:bg-amber-950 dark:text-amber-400">
            <RefreshCw className="h-4 w-4" />
          </div>
        );
      case "Water Detection":
        return (
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-cyan-100 text-cyan-600 dark:bg-cyan-950 dark:text-cyan-400">
            <Droplets className="h-4 w-4" />
          </div>
        );
      case "Disaster Assessment":
        return (
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-red-100 text-red-600 dark:bg-red-950 dark:text-red-400">
            <AlertTriangle className="h-4 w-4" />
          </div>
        );
      case "Risk Assessment":
        return (
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-amber-100 text-amber-600 dark:bg-amber-950 dark:text-amber-400">
            <Building className="h-4 w-4" />
          </div>
        );
      case "Urban Analysis":
      default:
        return (
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-purple-100 text-purple-600 dark:bg-purple-950 dark:text-purple-400">
            <Building2 className="h-4 w-4" />
          </div>
        );
    }
  };

  const getTypeBadgeStyle = () => {
    switch (analysis.type) {
      case "Flood Risk":
        return "bg-blue-50 text-blue-700 border-blue-200/70 dark:bg-blue-950/60 dark:text-blue-300 dark:border-blue-800/40";
      case "NDVI":
        return "bg-emerald-50 text-emerald-700 border-emerald-200/70 dark:bg-emerald-950/60 dark:text-emerald-300 dark:border-emerald-800/40";
      case "Change Detection":
        return "bg-amber-50 text-amber-700 border-amber-200/70 dark:bg-amber-950/60 dark:text-amber-300 dark:border-amber-800/40";
      case "Water Detection":
        return "bg-cyan-50 text-cyan-700 border-cyan-200/70 dark:bg-cyan-950/60 dark:text-cyan-300 dark:border-cyan-800/40";
      case "Disaster Assessment":
        return "bg-red-50 text-red-700 border-red-200/70 dark:bg-red-950/60 dark:text-red-300 dark:border-red-800/40";
      case "Risk Assessment":
        return "bg-amber-50 text-amber-700 border-amber-200/70 dark:bg-amber-950/60 dark:text-amber-300 dark:border-amber-800/40";
      case "Urban Analysis":
      default:
        return "bg-purple-50 text-purple-700 border-purple-200/70 dark:bg-purple-950/60 dark:text-purple-300 dark:border-purple-800/40";
    }
  };

  return (
    <tr
      onClick={() => onSelect(analysis)}
      className={`cursor-pointer transition-colors ${
        isSelected
          ? "bg-brand-50/60 dark:bg-brand-950/30"
          : "hover:bg-slate-50/80 dark:hover:bg-dark-hover/70"
      }`}
    >
      {/* Analysis Column */}
      <td className="py-3.5 pl-4 pr-3 whitespace-nowrap">
        <div className="flex items-center space-x-3">
          {getTypeIcon()}
          <span className="font-semibold text-xs text-slate-900 dark:text-white">
            {analysis.title}
          </span>
        </div>
      </td>

      {/* Location Column */}
      <td className="px-3 py-3.5 text-xs text-slate-600 dark:text-slate-300 whitespace-nowrap">
        <span>{analysis.location}</span>
      </td>

      {/* Analysis Type Badge */}
      <td className="px-3 py-3.5 text-xs whitespace-nowrap">
        <span
          className={`inline-flex items-center rounded-md border px-2.5 py-0.5 text-[11px] font-semibold ${getTypeBadgeStyle()}`}
        >
          {analysis.type}
        </span>
      </td>

      {/* Date & Time */}
      <td className="px-3 py-3.5 text-xs text-slate-600 dark:text-slate-400 whitespace-nowrap">
        {analysis.datetimeStr || `${analysis.date}, ${analysis.time}`}
      </td>

      {/* Confidence */}
      <td className="px-3 py-3.5 text-xs font-bold text-emerald-600 dark:text-emerald-400 whitespace-nowrap">
        {analysis.confidence}%
      </td>

      {/* Status */}
      <td className="px-3 py-3.5 text-xs whitespace-nowrap">
        <StatusBadge status={analysis.status} />
      </td>

      {/* Actions */}
      <td className="py-3.5 pl-3 pr-4 text-right whitespace-nowrap">
        <div className="relative inline-block text-left" ref={menuRef}>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setShowMenu(!showMenu);
            }}
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-transparent text-slate-400 transition hover:border-slate-200 hover:bg-white hover:text-slate-700 dark:hover:border-dark-border dark:hover:bg-dark-card dark:hover:text-slate-200"
            title="Options"
          >
            <MoreHorizontal className="h-4 w-4" />
          </button>

          {showMenu && (
            <div
              onClick={(e) => e.stopPropagation()}
              className="absolute right-0 z-50 mt-1 w-48 rounded-xl border border-slate-200 bg-white p-1 shadow-xl ring-1 ring-black/5 dark:border-dark-border dark:bg-dark-card"
            >
              <button
                type="button"
                onClick={() => {
                  setShowMenu(false);
                  onSelect(analysis);
                }}
                className="flex w-full items-center space-x-2.5 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 transition hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-dark-hover"
              >
                <Eye className="h-3.5 w-3.5 text-brand-600" />
                <span>View Details</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setShowMenu(false);
                  onViewReport(analysis);
                }}
                className="flex w-full items-center space-x-2.5 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 transition hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-dark-hover"
              >
                <FileText className="h-3.5 w-3.5 text-brand-600" />
                <span>View Report</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setShowMenu(false);
                  onDownloadReport(analysis);
                }}
                className="flex w-full items-center space-x-2.5 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 transition hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-dark-hover"
              >
                <Download className="h-3.5 w-3.5 text-slate-500" />
                <span>Download Report</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setShowMenu(false);
                  onOpenWorkspace(analysis);
                }}
                className="flex w-full items-center space-x-2.5 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 transition hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-dark-hover"
              >
                <ExternalLink className="h-3.5 w-3.5 text-slate-500" />
                <span>Open in Workspace</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setShowMenu(false);
                  onToggleSave(analysis);
                }}
                className="flex w-full items-center space-x-2.5 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 transition hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-dark-hover"
              >
                {analysis.isSaved ? (
                  <>
                    <BookmarkCheck className="h-3.5 w-3.5 text-purple-600" />
                    <span>Remove Bookmark</span>
                  </>
                ) : (
                  <>
                    <Bookmark className="h-3.5 w-3.5 text-slate-500" />
                    <span>Save / Bookmark</span>
                  </>
                )}
              </button>

              <div className="my-1 border-t border-slate-100 dark:border-dark-border" />

              <button
                type="button"
                onClick={() => {
                  setShowMenu(false);
                  onDelete(analysis);
                }}
                className="flex w-full items-center space-x-2.5 rounded-lg px-2.5 py-1.5 text-xs text-red-600 transition hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/40"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span>Delete Analysis</span>
              </button>
            </div>
          )}
        </div>
      </td>
    </tr>
  );
}
