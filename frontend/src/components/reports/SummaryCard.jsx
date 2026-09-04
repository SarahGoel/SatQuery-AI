import React from "react";
import { FileText, CheckCircle2, Bookmark, Clock } from "lucide-react";

export default function SummaryCard({
  type,
  title,
  value,
}) {
  const getIcon = () => {
    switch (type) {
      case "total":
        return (
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-brand-600 border border-blue-100 dark:bg-blue-950/50 dark:text-blue-400 dark:border-blue-900/50">
            <FileText className="h-5 w-5" />
          </div>
        );
      case "completed":
        return (
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600 border border-emerald-100 dark:bg-emerald-950/50 dark:text-emerald-400 dark:border-emerald-900/50">
            <CheckCircle2 className="h-5 w-5" />
          </div>
        );
      case "saved":
        return (
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-50 text-purple-600 border border-purple-100 dark:bg-purple-950/50 dark:text-purple-400 dark:border-purple-900/50">
            <Bookmark className="h-5 w-5" />
          </div>
        );
      case "last":
      default:
        return (
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-50 text-amber-600 border border-amber-100 dark:bg-amber-950/50 dark:text-amber-400 dark:border-amber-900/50">
            <Clock className="h-5 w-5" />
          </div>
        );
    }
  };

  return (
    <div className="relative flex flex-col justify-between rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm transition-all duration-200 hover:shadow-md dark:border-dark-border dark:bg-dark-card">
      <div className="flex items-center space-x-3.5">
        {getIcon()}
        <div>
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
            {title}
          </p>
          <h3 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white mt-0.5">
            {value}
          </h3>
        </div>
      </div>
    </div>
  );
}
