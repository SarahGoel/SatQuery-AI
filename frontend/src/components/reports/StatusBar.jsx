import React from "react";
import { Crosshair, Info } from "lucide-react";

export default function StatusBar({
  area = "No area selected",
  coordinates = "--",
  imagery = "--",
  resolution = "--",
  cloudCover = "--",
  tip = "Select an area and ask a question to begin analysis.",
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200/80 bg-white/90 px-4 py-2.5 text-xs text-slate-600 shadow-sm backdrop-blur-sm dark:border-dark-border dark:bg-dark-card/90 dark:text-slate-400">
      {/* Left: Metadata indicators */}
      <div className="flex flex-wrap items-center space-x-3 divide-x divide-slate-200 dark:divide-dark-border">
        <div className="flex items-center space-x-1.5 font-medium text-slate-800 dark:text-slate-200">
          <Crosshair className="h-3.5 w-3.5 text-slate-400" />
          <span>{area}</span>
        </div>
        <div className="pl-3">
          <span className="text-slate-400 dark:text-slate-500">Coordinates:</span>{" "}
          <span className="font-medium text-slate-700 dark:text-slate-300">{coordinates}</span>
        </div>
        <div className="pl-3">
          <span className="text-slate-400 dark:text-slate-500">Imagery:</span>{" "}
          <span className="font-medium text-slate-700 dark:text-slate-300">{imagery}</span>
        </div>
        <div className="pl-3">
          <span className="text-slate-400 dark:text-slate-500">Resolution:</span>{" "}
          <span className="font-medium text-slate-700 dark:text-slate-300">{resolution}</span>
        </div>
        <div className="pl-3">
          <span className="text-slate-400 dark:text-slate-500">Cloud Cover:</span>{" "}
          <span className="font-medium text-slate-700 dark:text-slate-300">{cloudCover}</span>
        </div>
      </div>

      {/* Right: Helpful Tip */}
      <div className="flex items-center space-x-1.5 text-xs text-slate-500 dark:text-slate-400">
        <span>Tip:</span>
        <span className="font-medium text-brand-600 dark:text-brand-400 cursor-pointer hover:underline">
          {tip}
        </span>
        <Info className="h-3.5 w-3.5 text-slate-400" />
      </div>
    </div>
  );
}
