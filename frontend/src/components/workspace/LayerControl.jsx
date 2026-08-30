import React, { useState } from "react";
import {
  ChevronUp,
  ChevronDown,
  AlertTriangle,
  Landmark,
  Route,
  Waves,
} from "lucide-react";

export default function LayerControl({
  baseImagery = "optical",
  onBaseImageryChange = () => {},
  indexLayers = { ndvi: false, ndwi: false, ndbi: false, ndmi: false },
  onToggleIndexLayer = () => {},
  vectorLayers = {
    floodRisk: false,
    adminBoundary: false,
    roads: false,
    waterBodies: false,
  },
  onToggleVectorLayer = () => {},
  otherLayers = { cloudMask: false },
  onToggleOtherLayer = () => {},
}) {
  const [isMainCollapsed, setIsMainCollapsed] = useState(false);
  const [isIndexCollapsed, setIsIndexCollapsed] = useState(false);

  return (
    <div className="flex h-full w-full flex-col overflow-y-auto rounded-2xl border border-slate-200 bg-white p-4 shadow-sm transition-colors dark:border-dark-border dark:bg-dark-card space-y-4">
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-3 dark:border-dark-border">
        <h2 className="text-sm font-bold text-slate-900 dark:text-white">
          Data & Layers
        </h2>
        <button
          type="button"
          onClick={() => setIsMainCollapsed(!isMainCollapsed)}
          className="rounded-lg p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-dark-hover dark:hover:text-slate-200"
          title="Toggle Layers Panel"
        >
          {isMainCollapsed ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronUp className="h-4 w-4" />
          )}
        </button>
      </div>

      {!isMainCollapsed && (
        <div className="space-y-4 text-xs">
          {/* Base Imagery */}
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-700 dark:text-slate-200">
              Base Imagery
            </h3>
            <div className="space-y-2">
              {/* Optical (Sentinel-2) */}
              <label className="flex cursor-pointer items-center space-x-2.5 rounded-lg p-1 transition hover:bg-slate-50 dark:hover:bg-dark-hover/60">
                <input
                  type="radio"
                  name="baseImagery"
                  value="optical"
                  checked={baseImagery === "optical"}
                  onChange={() => onBaseImageryChange("optical")}
                  className="h-3.5 w-3.5 accent-brand-600 focus:ring-0"
                />
                <span className="font-medium text-slate-800 dark:text-slate-200">
                  Optical (Sentinel-2)
                </span>
              </label>

              {/* SAR (Sentinel-1) */}
              <label className="flex cursor-pointer items-center space-x-2.5 rounded-lg p-1 transition hover:bg-slate-50 dark:hover:bg-dark-hover/60">
                <input
                  type="radio"
                  name="baseImagery"
                  value="sar"
                  checked={baseImagery === "sar"}
                  onChange={() => onBaseImageryChange("sar")}
                  className="h-3.5 w-3.5 accent-brand-600 focus:ring-0"
                />
                <span className="font-medium text-slate-800 dark:text-slate-200">
                  SAR (Sentinel-1)
                </span>
              </label>
            </div>
          </div>

          {/* Index Layers */}
          <div className="space-y-2 border-t border-slate-100 pt-3 dark:border-dark-border">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-slate-700 dark:text-slate-200">
                Index Layers
              </h3>
              <button
                type="button"
                onClick={() => setIsIndexCollapsed(!isIndexCollapsed)}
                className="rounded p-0.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
              >
                {isIndexCollapsed ? (
                  <ChevronDown className="h-3.5 w-3.5" />
                ) : (
                  <ChevronUp className="h-3.5 w-3.5" />
                )}
              </button>
            </div>

            {!isIndexCollapsed && (
              <div className="space-y-2 pl-0.5">
                {[
                  { key: "ndvi", label: "NDVI" },
                  { key: "ndwi", label: "NDWI" },
                  { key: "ndbi", label: "NDBI" },
                  { key: "ndmi", label: "NDMI" },
                ].map(({ key, label }) => (
                  <label
                    key={key}
                    className="flex cursor-pointer items-center space-x-2.5 rounded-lg p-1 transition hover:bg-slate-50 dark:hover:bg-dark-hover/60"
                  >
                    <input
                      type="checkbox"
                      checked={!!indexLayers[key]}
                      onChange={() => onToggleIndexLayer(key)}
                      className="h-3.5 w-3.5 rounded border-slate-300 text-brand-600 focus:ring-0 dark:border-slate-700 dark:bg-dark-bg"
                    />
                    <span className="font-medium text-slate-800 dark:text-slate-200">
                      {label}
                    </span>
                  </label>
                ))}
              </div>
            )}
          </div>

          {/* Vector Layers */}
          <div className="space-y-2 border-t border-slate-100 pt-3 dark:border-dark-border">
            <h3 className="font-semibold text-slate-700 dark:text-slate-200">
              Vector Layers
            </h3>
            <div className="space-y-2 pl-0.5">
              {/* Flood Risk Zones */}
              <label className="flex cursor-pointer items-center space-x-2.5 rounded-lg p-1 transition hover:bg-slate-50 dark:hover:bg-dark-hover/60">
                <input
                  type="checkbox"
                  checked={!!vectorLayers.floodRisk}
                  onChange={() => onToggleVectorLayer("floodRisk")}
                  className="h-3.5 w-3.5 rounded border-slate-300 text-brand-600 focus:ring-0 dark:border-slate-700 dark:bg-dark-bg"
                />
                <AlertTriangle className="h-3.5 w-3.5 text-red-500" />
                <span className="font-medium text-slate-800 dark:text-slate-200">
                  Flood Risk Zones
                </span>
              </label>

              {/* Administrative Boundary */}
              <label className="flex cursor-pointer items-center space-x-2.5 rounded-lg p-1 transition hover:bg-slate-50 dark:hover:bg-dark-hover/60">
                <input
                  type="checkbox"
                  checked={!!vectorLayers.adminBoundary}
                  onChange={() => onToggleVectorLayer("adminBoundary")}
                  className="h-3.5 w-3.5 rounded border-slate-300 text-brand-600 focus:ring-0 dark:border-slate-700 dark:bg-dark-bg"
                />
                <Landmark className="h-3.5 w-3.5 text-purple-500" />
                <span className="font-medium text-slate-800 dark:text-slate-200">
                  Administrative Boundary
                </span>
              </label>

              {/* Roads */}
              <label className="flex cursor-pointer items-center space-x-2.5 rounded-lg p-1 transition hover:bg-slate-50 dark:hover:bg-dark-hover/60">
                <input
                  type="checkbox"
                  checked={!!vectorLayers.roads}
                  onChange={() => onToggleVectorLayer("roads")}
                  className="h-3.5 w-3.5 rounded border-slate-300 text-brand-600 focus:ring-0 dark:border-slate-700 dark:bg-dark-bg"
                />
                <Route className="h-3.5 w-3.5 text-amber-500" />
                <span className="font-medium text-slate-800 dark:text-slate-200">
                  Roads
                </span>
              </label>

              {/* Water Bodies */}
              <label className="flex cursor-pointer items-center space-x-2.5 rounded-lg p-1 transition hover:bg-slate-50 dark:hover:bg-dark-hover/60">
                <input
                  type="checkbox"
                  checked={!!vectorLayers.waterBodies}
                  onChange={() => onToggleVectorLayer("waterBodies")}
                  className="h-3.5 w-3.5 rounded border-slate-300 text-brand-600 focus:ring-0 dark:border-slate-700 dark:bg-dark-bg"
                />
                <Waves className="h-3.5 w-3.5 text-cyan-500" />
                <span className="font-medium text-slate-800 dark:text-slate-200">
                  Water Bodies
                </span>
              </label>
            </div>
          </div>

          {/* Other Layers */}
          <div className="space-y-2 border-t border-slate-100 pt-3 dark:border-dark-border">
            <h3 className="font-semibold text-slate-700 dark:text-slate-200">
              Other Layers
            </h3>
            <div className="space-y-2 pl-0.5">
              <label className="flex cursor-pointer items-center space-x-2.5 rounded-lg p-1 transition hover:bg-slate-50 dark:hover:bg-dark-hover/60">
                <input
                  type="checkbox"
                  checked={!!otherLayers.cloudMask}
                  onChange={() => onToggleOtherLayer("cloudMask")}
                  className="h-3.5 w-3.5 rounded border-slate-300 text-brand-600 focus:ring-0 dark:border-slate-700 dark:bg-dark-bg"
                />
                <span className="font-medium text-slate-800 dark:text-slate-200">
                  Cloud Mask
                </span>
              </label>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
