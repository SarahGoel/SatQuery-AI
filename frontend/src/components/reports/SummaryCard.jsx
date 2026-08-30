import React from "react";
import { FileText, CheckCircle2, Bookmark, Clock } from "lucide-react";

export default function SummaryCard({
  type,
  title,
  value,
  subtitle,
  sparklineColor = "blue",
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

  const getSparklinePath = () => {
    switch (sparklineColor) {
      case "emerald":
      case "green":
        return "M2 18 Q 15 16, 25 10 T 45 14 T 65 6 T 85 12 T 98 4";
      case "purple":
        return "M2 16 Q 18 19, 32 12 T 52 14 T 72 8 T 88 10 T 98 6";
      case "amber":
        return "M2 18 Q 16 15, 30 18 T 50 12 T 70 8 T 86 14 T 98 6";
      case "blue":
      default:
        return "M2 16 Q 16 12, 30 15 T 50 8 T 70 14 T 86 6 T 98 4";
    }
  };

  const getSparklineStroke = () => {
    switch (sparklineColor) {
      case "emerald":
      case "green":
        return "#10b981";
      case "purple":
        return "#a855f7";
      case "amber":
        return "#f59e0b";
      case "blue":
      default:
        return "#3b82f6";
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

      <div className="mt-4 flex items-center justify-between pt-2 border-t border-slate-100/80 dark:border-dark-border/60">
        <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
          {subtitle}
        </span>
        <div className="w-24 h-6">
          <svg className="w-full h-full overflow-visible" viewBox="0 0 100 24">
            <path
              d={getSparklinePath()}
              fill="none"
              stroke={getSparklineStroke()}
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
      </div>
    </div>
  );
}
