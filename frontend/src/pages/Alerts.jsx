import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Bell,
  AlertTriangle,
  CloudRain,
  ShieldAlert,
  Search,
  Filter,
  CheckCircle2,
  ExternalLink,
  MapPin,
  Clock,
  Radio,
  FileDown,
  Info
} from "lucide-react";
import { MOCK_ALERTS } from "../mock/mockData";

export default function Alerts() {
  const navigate = useNavigate();
  const [alerts, setAlerts] = useState(MOCK_ALERTS);
  const [filterType, setFilterType] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [toastMessage, setToastMessage] = useState(null);

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const handleAcknowledge = (id) => {
    setAlerts((prev) =>
      prev.map((a) => (a.id === id ? { ...a, status: "Acknowledged" } : a))
    );
    showToast("Alert marked as Acknowledged by Analyst");
  };

  const filteredAlerts = alerts.filter((alert) => {
    const matchesSearch =
      alert.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      alert.location.toLowerCase().includes(searchQuery.toLowerCase()) ||
      alert.description.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesType =
      filterType === "all" ||
      (filterType === "flood" && alert.type.toLowerCase().includes("flood")) ||
      (filterType === "rainfall" && alert.type.toLowerCase().includes("rain")) ||
      (filterType === "weather" && alert.type.toLowerCase().includes("weather")) ||
      (filterType === "environmental" &&
        (alert.type.toLowerCase().includes("environmental") ||
          alert.type.toLowerCase().includes("disaster")));

    const matchesSeverity =
      severityFilter === "all" || alert.severity === severityFilter;

    return matchesSearch && matchesType && matchesSeverity;
  });

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Toast */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center space-x-2 rounded-xl bg-slate-900 px-4 py-2.5 text-xs font-semibold text-white shadow-2xl">
          <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Page Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white flex items-center gap-2.5">
            <Bell className="h-6 w-6 text-red-500" />
            <span>Geospatial Alerts & Early Warning</span>
          </h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Real-time radar backscatter, optical anomaly, and meteorological trigger feeds.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={() => showToast("Alert feeds refreshed from local ground station cache")}
            className="flex items-center space-x-2 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50 dark:border-dark-border dark:bg-dark-card dark:text-slate-200 dark:hover:bg-dark-hover"
          >
            <Radio className="h-4 w-4 text-brand-600" />
            <span>Sync Feeds</span>
          </button>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div className="rounded-2xl border border-slate-200/90 bg-white p-4 shadow-sm dark:border-dark-border dark:bg-dark-card">
          <p className="text-xs text-slate-400">Total Active Alerts</p>
          <p className="text-xl font-bold text-slate-900 dark:text-white">
            {alerts.length}
          </p>
        </div>
        <div className="rounded-2xl border border-red-200/60 bg-red-50/50 p-4 shadow-sm dark:border-red-900/40 dark:bg-red-950/20">
          <p className="text-xs font-semibold text-red-700 dark:text-red-400">High Risk (Critical)</p>
          <p className="text-xl font-bold text-red-700 dark:text-red-300">
            {alerts.filter((a) => a.severity === "critical").length}
          </p>
        </div>
        <div className="rounded-2xl border border-amber-200/60 bg-amber-50/50 p-4 shadow-sm dark:border-amber-900/40 dark:bg-amber-950/20">
          <p className="text-xs font-semibold text-amber-800 dark:text-amber-400">Moderate Risk</p>
          <p className="text-xl font-bold text-amber-800 dark:text-amber-300">
            {alerts.filter((a) => a.severity === "warning").length}
          </p>
        </div>
        <div className="rounded-2xl border border-blue-200/60 bg-blue-50/50 p-4 shadow-sm dark:border-blue-900/40 dark:bg-blue-950/20">
          <p className="text-xs font-semibold text-blue-700 dark:text-blue-400">Informational</p>
          <p className="text-xl font-bold text-blue-700 dark:text-blue-300">
            {alerts.filter((a) => a.severity === "info").length}
          </p>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col gap-3 rounded-2xl border border-slate-200/90 bg-white p-4 shadow-sm dark:border-dark-border dark:bg-dark-card md:flex-row md:items-center md:justify-between">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search alerts by area, description, or trigger..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-xl border border-slate-200 bg-slate-50/70 py-2 pl-9 pr-4 text-xs text-slate-900 placeholder-slate-400 focus:border-brand-500 focus:outline-none dark:border-dark-border dark:bg-dark-bg/60 dark:text-white"
          />
        </div>

        {/* Filter Pills */}
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => setFilterType("all")}
            className={`rounded-xl px-3 py-1.5 text-xs font-semibold transition ${
              filterType === "all"
                ? "bg-brand-600 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-dark-hover dark:text-slate-300"
            }`}
          >
            All Types
          </button>
          <button
            type="button"
            onClick={() => setFilterType("flood")}
            className={`rounded-xl px-3 py-1.5 text-xs font-semibold transition ${
              filterType === "flood"
                ? "bg-brand-600 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-dark-hover dark:text-slate-300"
            }`}
          >
            Flood Risk
          </button>
          <button
            type="button"
            onClick={() => setFilterType("rainfall")}
            className={`rounded-xl px-3 py-1.5 text-xs font-semibold transition ${
              filterType === "rainfall"
                ? "bg-brand-600 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-dark-hover dark:text-slate-300"
            }`}
          >
            Heavy Rainfall
          </button>
          <button
            type="button"
            onClick={() => setFilterType("weather")}
            className={`rounded-xl px-3 py-1.5 text-xs font-semibold transition ${
              filterType === "weather"
                ? "bg-brand-600 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-dark-hover dark:text-slate-300"
            }`}
          >
            Weather
          </button>
          <button
            type="button"
            onClick={() => setFilterType("environmental")}
            className={`rounded-xl px-3 py-1.5 text-xs font-semibold transition ${
              filterType === "environmental"
                ? "bg-brand-600 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-dark-hover dark:text-slate-300"
            }`}
          >
            Environmental
          </button>
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-4">
        {filteredAlerts.map((alert) => (
          <div
            key={alert.id}
            className={`rounded-2xl border p-5 shadow-sm transition hover:shadow-md ${
              alert.severity === "critical"
                ? "border-red-200 bg-white dark:border-red-900/60 dark:bg-dark-card"
                : alert.severity === "warning"
                ? "border-amber-200 bg-white dark:border-amber-900/60 dark:bg-dark-card"
                : "border-slate-200 bg-white dark:border-dark-border dark:bg-dark-card"
            }`}
          >
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              {/* Left Details */}
              <div className="space-y-2.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                      alert.severity === "critical"
                        ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
                        : alert.severity === "warning"
                        ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                        : "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                    }`}
                  >
                    {alert.severityLabel}
                  </span>

                  <span className="rounded-md bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700 dark:bg-dark-hover dark:text-slate-300">
                    {alert.type}
                  </span>

                  <span className="rounded-md border border-slate-200 px-2 py-0.5 text-[11px] font-semibold text-slate-600 dark:border-dark-border dark:text-slate-400">
                    Sensor: {alert.source}
                  </span>

                  <span
                    className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                      alert.status === "Active"
                        ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-400"
                        : alert.status === "Acknowledged"
                        ? "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300"
                        : "bg-slate-100 text-slate-600 dark:bg-dark-hover dark:text-slate-400"
                    }`}
                  >
                    Status: {alert.status}
                  </span>
                </div>

                <h3 className="text-base font-bold text-slate-900 dark:text-white">
                  {alert.title}
                </h3>

                <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
                  {alert.description}
                </p>

                {/* Evidence Stats Strip */}
                <div className="flex flex-wrap gap-4 pt-1 text-xs text-slate-600 dark:text-slate-300">
                  <div className="flex items-center space-x-1.5">
                    <MapPin className="h-3.5 w-3.5 text-brand-600" />
                    <span>Location: <strong>{alert.location}</strong></span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <Clock className="h-3.5 w-3.5 text-slate-400" />
                    <span>Timestamp: {alert.timestamp}</span>
                  </div>
                  <div>
                    Inundated Extent: <strong className="text-brand-600 dark:text-brand-400">{alert.inundatedArea}</strong>
                  </div>
                  <div>
                    Population Impact: <strong>{alert.affectedPopulation}</strong>
                  </div>
                </div>

                {/* Recommendation Box */}
                <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-2.5 text-xs text-slate-700 dark:border-dark-border dark:bg-dark-bg/50 dark:text-slate-300">
                  <span className="font-semibold text-slate-900 dark:text-white">Advisory: </span>
                  {alert.recommendation}
                </div>
              </div>

              {/* Right Action Buttons */}
              <div className="flex flex-row lg:flex-col gap-2 flex-shrink-0">
                <button
                  type="button"
                  onClick={() => navigate("/workspace")}
                  className="flex flex-1 lg:flex-none items-center justify-center space-x-1.5 rounded-xl bg-brand-600 px-4 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-brand-700"
                >
                  <span>Analyze in Workspace</span>
                  <ExternalLink className="h-3.5 w-3.5" />
                </button>

                {alert.status === "Active" && (
                  <button
                    type="button"
                    onClick={() => handleAcknowledge(alert.id)}
                    className="flex flex-1 lg:flex-none items-center justify-center space-x-1.5 rounded-xl border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50 dark:border-dark-border dark:bg-dark-card dark:text-slate-200 dark:hover:bg-dark-hover"
                  >
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                    <span>Acknowledge</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
