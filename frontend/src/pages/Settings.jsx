import React, { useState } from "react";
import {
  Settings as SettingsIcon,
  Sun,
  Moon,
  Monitor,
  Map,
  Bell,
  HardDrive,
  ShieldCheck,
  CheckCircle2,
  Save,
  RotateCcw,
  Sliders
} from "lucide-react";
import { useTheme } from "../context/ThemeContext";

export default function Settings() {
  const { theme, setTheme } = useTheme();

  const [activeTab, setActiveTab] = useState("appearance"); // 'appearance', 'map', 'notifications', 'airgap'
  const [toastMessage, setToastMessage] = useState(null);

  // Settings states
  const [mapSettings, setMapSettings] = useState({
    defaultBasemap: "satellite",
    projection: "EPSG:3857",
    unitSystem: "metric",
    coordFormat: "dd",
    autoCenterROI: true,
  });

  const [notifSettings, setNotifSettings] = useState({
    floodAlerts: true,
    rainfallAlerts: true,
    environmentalAlerts: true,
    minSeverity: "warning",
    soundAlerts: false,
  });

  const [aiSettings, setAiSettings] = useState({
    confidenceThreshold: 80,
    cloudMaskThreshold: 15,
    autoVectorize: true,
    modelFlavor: "UNet-SAR-EO Multi-band (On-premise GPU)",
  });

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const handleSaveSettings = () => {
    showToast("Settings successfully saved to local station profile!");
  };

  const handleResetDefaults = () => {
    setTheme("light");
    setMapSettings({
      defaultBasemap: "satellite",
      projection: "EPSG:3857",
      unitSystem: "metric",
      coordFormat: "dd",
      autoCenterROI: true,
    });
    setNotifSettings({
      floodAlerts: true,
      rainfallAlerts: true,
      environmentalAlerts: true,
      minSeverity: "warning",
      soundAlerts: false,
    });
    showToast("Settings restored to factory defaults.");
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Toast */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center space-x-2 rounded-xl bg-slate-900 px-4 py-2.5 text-xs font-semibold text-white shadow-2xl">
          <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white flex items-center gap-2.5">
            <SettingsIcon className="h-6 w-6 text-brand-600" />
            <span>Application & Station Settings</span>
          </h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Configure analyst preferences, display themes, geospatial projections, and air-gap cache limits.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={handleResetDefaults}
            className="flex items-center space-x-1.5 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50 dark:border-dark-border dark:bg-dark-card dark:text-slate-300 dark:hover:bg-dark-hover"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Reset Defaults</span>
          </button>
          <button
            type="button"
            onClick={handleSaveSettings}
            className="flex items-center space-x-1.5 rounded-xl bg-brand-600 px-4 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-brand-700"
          >
            <Save className="h-3.5 w-3.5" />
            <span>Save Preferences</span>
          </button>
        </div>
      </div>

      {/* Settings Navigation Tabs */}
      <div className="flex space-x-1 rounded-2xl border border-slate-200/90 bg-white p-1.5 shadow-sm dark:border-dark-border dark:bg-dark-card">
        <button
          type="button"
          onClick={() => setActiveTab("appearance")}
          className={`flex-1 rounded-xl py-2 text-xs font-semibold transition ${
            activeTab === "appearance"
              ? "bg-brand-600 text-white shadow-sm"
              : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
          }`}
        >
          Theme & Display
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("map")}
          className={`flex-1 rounded-xl py-2 text-xs font-semibold transition ${
            activeTab === "map"
              ? "bg-brand-600 text-white shadow-sm"
              : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
          }`}
        >
          Map & Projections
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("notifications")}
          className={`flex-1 rounded-xl py-2 text-xs font-semibold transition ${
            activeTab === "notifications"
              ? "bg-brand-600 text-white shadow-sm"
              : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
          }`}
        >
          Alerts & Subscriptions
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("airgap")}
          className={`flex-1 rounded-xl py-2 text-xs font-semibold transition ${
            activeTab === "airgap"
              ? "bg-brand-600 text-white shadow-sm"
              : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
          }`}
        >
          Air-Gap & Hardware
        </button>
      </div>

      {/* Tab 1: Appearance */}
      {activeTab === "appearance" && (
        <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm dark:border-dark-border dark:bg-dark-card space-y-6">
          <h2 className="text-base font-bold text-slate-900 dark:text-white">
            Display & Theme Preferences
          </h2>

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                Interface Color Mode
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg">
                <button
                  type="button"
                  onClick={() => setTheme("light")}
                  className={`flex items-center space-x-3 rounded-xl border p-3.5 text-left transition ${
                    theme === "light"
                      ? "border-brand-600 bg-brand-50/70 text-brand-900 dark:border-brand-500 shadow-sm"
                      : "border-slate-200 hover:bg-slate-50 dark:border-dark-border dark:hover:bg-dark-hover"
                  }`}
                >
                  <Sun className="h-5 w-5 text-amber-500" />
                  <div>
                    <p className="text-xs font-bold text-slate-900 dark:text-white">
                      Light Mode (Default)
                    </p>
                    <p className="text-[11px] text-slate-400">
                      Optimized for daylight lab & ops rooms
                    </p>
                  </div>
                </button>

                <button
                  type="button"
                  onClick={() => setTheme("dark")}
                  className={`flex items-center space-x-3 rounded-xl border p-3.5 text-left transition ${
                    theme === "dark"
                      ? "border-brand-600 bg-brand-50/70 text-brand-900 dark:border-brand-500 dark:bg-brand-950/40 shadow-sm"
                      : "border-slate-200 hover:bg-slate-50 dark:border-dark-border dark:hover:bg-dark-hover"
                  }`}
                >
                  <Moon className="h-5 w-5 text-indigo-400" />
                  <div>
                    <p className="text-xs font-bold text-slate-900 dark:text-white">
                      Dark Mode
                    </p>
                    <p className="text-[11px] text-slate-400">
                      High contrast for nocturnal tracking
                    </p>
                  </div>
                </button>
              </div>
            </div>

            <div className="border-t border-slate-100 pt-4 dark:border-dark-border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-semibold text-slate-800 dark:text-slate-200">
                    High-Density Compact Layout
                  </p>
                  <p className="text-[11px] text-slate-400">
                    Reduce table padding for maximum satellite raster screen estate
                  </p>
                </div>
                <input
                  type="checkbox"
                  defaultChecked
                  className="h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Map & Projections */}
      {activeTab === "map" && (
        <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm dark:border-dark-border dark:bg-dark-card space-y-6">
          <h2 className="text-base font-bold text-slate-900 dark:text-white">
            Geospatial & Map Engine Preferences
          </h2>

          <div className="space-y-4 max-w-xl">
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                Default Map Projection
              </label>
              <select
                value={mapSettings.projection}
                onChange={(e) =>
                  setMapSettings({ ...mapSettings, projection: e.target.value })
                }
                className="w-full rounded-xl border border-slate-200 bg-slate-50/70 p-2.5 text-xs text-slate-900 dark:border-dark-border dark:bg-dark-bg/60 dark:text-white"
              >
                <option value="EPSG:3857">EPSG:3857 (WGS 84 / Pseudo-Mercator - Standard Web)</option>
                <option value="EPSG:4326">EPSG:4326 (WGS 84 Geographic Lat/Lon)</option>
                <option value="EPSG:32643">EPSG:32643 (UTM Zone 43N - India Central)</option>
              </select>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                Unit of Distance & Area
              </label>
              <select
                value={mapSettings.unitSystem}
                onChange={(e) =>
                  setMapSettings({ ...mapSettings, unitSystem: e.target.value })
                }
                className="w-full rounded-xl border border-slate-200 bg-slate-50/70 p-2.5 text-xs text-slate-900 dark:border-dark-border dark:bg-dark-bg/60 dark:text-white"
              >
                <option value="metric">Metric (Kilometers, Hectares, Meters)</option>
                <option value="nautical">Nautical (Nautical Miles, Knots, Square NM)</option>
              </select>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                Coordinate Display Format
              </label>
              <select
                value={mapSettings.coordFormat}
                onChange={(e) =>
                  setMapSettings({ ...mapSettings, coordFormat: e.target.value })
                }
                className="w-full rounded-xl border border-slate-200 bg-slate-50/70 p-2.5 text-xs text-slate-900 dark:border-dark-border dark:bg-dark-bg/60 dark:text-white"
              >
                <option value="dd">Decimal Degrees (e.g. 26.2006° N, 92.9376° E)</option>
                <option value="dms">Degrees Minutes Seconds (e.g. 26°12'02" N, 92°56'15" E)</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Notifications */}
      {activeTab === "notifications" && (
        <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm dark:border-dark-border dark:bg-dark-card space-y-6">
          <h2 className="text-base font-bold text-slate-900 dark:text-white">
            Alert Triggers & Thresholds
          </h2>

          <div className="space-y-4 max-w-xl">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3 dark:border-dark-border">
              <div>
                <p className="text-xs font-semibold text-slate-800 dark:text-slate-200">
                  Flood Inundation & Embankment Alerts
                </p>
                <p className="text-[11px] text-slate-400">
                  Notify on SAR backscatter drop exceeding threshold (-5.0 dB)
                </p>
              </div>
              <input
                type="checkbox"
                checked={notifSettings.floodAlerts}
                onChange={(e) =>
                  setNotifSettings({ ...notifSettings, floodAlerts: e.target.checked })
                }
                className="h-4 w-4 rounded border-slate-300 text-brand-600"
              />
            </div>

            <div className="flex items-center justify-between border-b border-slate-100 pb-3 dark:border-dark-border">
              <div>
                <p className="text-xs font-semibold text-slate-800 dark:text-slate-200">
                  Convective Heavy Rainfall Advisories
                </p>
                <p className="text-[11px] text-slate-400">
                  Notify when INSAT-3DR cloud top temperature drops below -60°C
                </p>
              </div>
              <input
                type="checkbox"
                checked={notifSettings.rainfallAlerts}
                onChange={(e) =>
                  setNotifSettings({ ...notifSettings, rainfallAlerts: e.target.checked })
                }
                className="h-4 w-4 rounded border-slate-300 text-brand-600"
              />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-slate-800 dark:text-slate-200">
                  Audible Critical Alert Tone
                </p>
                <p className="text-[11px] text-slate-400">
                  Play station sound chime upon High Risk event trigger
                </p>
              </div>
              <input
                type="checkbox"
                checked={notifSettings.soundAlerts}
                onChange={(e) =>
                  setNotifSettings({ ...notifSettings, soundAlerts: e.target.checked })
                }
                className="h-4 w-4 rounded border-slate-300 text-brand-600"
              />
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Air-Gap & Hardware */}
      {activeTab === "airgap" && (
        <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm dark:border-dark-border dark:bg-dark-card space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-slate-900 dark:text-white">
              Air-Gapped Node Architecture
            </h2>
            <span className="flex items-center space-x-1.5 rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-800 dark:bg-emerald-950 dark:text-emerald-400">
              <ShieldCheck className="h-4 w-4" />
              <span>ISRO SIH26167 Certified</span>
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-4 dark:border-dark-border dark:bg-dark-bg/60 space-y-2">
              <div className="flex items-center space-x-2 text-brand-600">
                <HardDrive className="h-4 w-4" />
                <h3 className="text-xs font-bold text-slate-900 dark:text-white">
                  Local Cache & Storage
                </h3>
              </div>
              <p className="text-xs text-slate-500">
                Local NVMe GeoTIFF & Vector Tile storage quota.
              </p>
              <div className="space-y-1 pt-2">
                <div className="flex justify-between text-[11px] font-semibold text-slate-600 dark:text-slate-300">
                  <span>14.8 GB Used</span>
                  <span>100 GB Allocated</span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-200 dark:bg-dark-hover">
                  <div className="h-full w-[15%] rounded-full bg-brand-600" />
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-slate-100 bg-slate-50 p-4 dark:border-dark-border dark:bg-dark-bg/60 space-y-2">
              <div className="flex items-center space-x-2 text-emerald-600">
                <ShieldCheck className="h-4 w-4" />
                <h3 className="text-xs font-bold text-slate-900 dark:text-white">
                  Air-Gap Isolation Status
                </h3>
              </div>
              <p className="text-xs text-slate-500">
                Zero external telemetry or unauthorized outbound requests.
              </p>
              <div className="pt-2 text-xs font-mono text-emerald-600 dark:text-emerald-400 font-semibold">
                ✓ ALL PORTS LOCKED (STATION ON-PREM)
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
