import React from "react";

export default function StatusBadge({ status, variant = "dot" }) {
  const isCompleted = status === "Completed";

  if (variant === "pill") {
    return (
      <span
        className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${
          isCompleted
            ? "bg-emerald-50 text-emerald-700 border border-emerald-200/80 dark:bg-emerald-950/60 dark:text-emerald-400 dark:border-emerald-800/50"
            : "bg-amber-50 text-amber-700 border border-amber-200/80 dark:bg-amber-950/60 dark:text-amber-400 dark:border-amber-800/50"
        }`}
      >
        {status}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center space-x-1.5 text-xs font-medium">
      <span
        className={`h-2 w-2 rounded-full ${
          isCompleted
            ? "bg-emerald-500 shadow-sm shadow-emerald-500/50"
            : "bg-amber-500 shadow-sm shadow-amber-500/50 animate-pulse"
        }`}
      />
      <span
        className={
          isCompleted
            ? "text-emerald-600 dark:text-emerald-400 font-medium"
            : "text-amber-600 dark:text-amber-400 font-medium"
        }
      >
        {status}
      </span>
    </span>
  );
}
