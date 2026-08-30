import React from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowRight,
  TrendingUp,
  Image as ImageIcon,
  Layers,
  ClipboardList,
  Activity,
  Bell,
  AlertTriangle,
  CloudRain,
  Radio,
  CheckCircle2,
  Info,
  MapPin,
  ExternalLink,
  ShieldCheck
} from "lucide-react";
import {
  MOCK_STATS,
  MOCK_ALERTS,
  MOCK_RECENT_ANALYSES,
  MOCK_MONITORING_AREAS,
} from "../mock/mockData";

export default function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* 1. Welcome & Start Analysis Hero Banner */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-white flex items-center gap-2">
            Welcome to SatQuery AI <span className="text-2xl">🛰️</span>
          </h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            AI-powered satellite intelligence for better decisions and a safer tomorrow.
          </p>
        </div>

        <Link
          to="/workspace"
          className="group relative flex items-center justify-between gap-4 overflow-hidden rounded-2xl bg-gradient-to-r from-brand-600 via-brand-700 to-blue-700 p-4 sm:p-5 text-white shadow-lg shadow-brand-600/20 transition-all duration-300 hover:shadow-xl hover:shadow-brand-600/30 hover:scale-[1.01] lg:w-96"
        >
          <div className="z-10">
            <h2 className="text-base font-bold flex items-center gap-1.5">
              Start Analysis
            </h2>
            <p className="mt-0.5 text-xs text-brand-100 line-clamp-2">
              Analyze areas, compare time periods and get AI insights.
            </p>
          </div>
          <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-white/20 backdrop-blur-md transition-transform duration-300 group-hover:translate-x-1">
            <ArrowRight className="h-5 w-5" />
          </div>
        </Link>
      </div>

      {/* 2. Metric Overview Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Card 1: Analyses Today */}
        <div className="flex items-center space-x-4 rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm transition hover:shadow-md dark:border-dark-border dark:bg-dark-card">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-blue-50 text-brand-600 dark:bg-blue-950/60 dark:text-brand-400">
            <ImageIcon className="h-6 w-6" />
          </div>
          <div>
            <p className="text-xs font-medium text-slate-400 dark:text-slate-400">
              Analyses Today
            </p>
            <p className="text-2xl font-bold text-slate-900 dark:text-white">
              {MOCK_STATS.analysesToday}
            </p>
            <p className="mt-0.5 text-xs font-medium text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
              <TrendingUp className="h-3 w-3" />
              <span>{MOCK_STATS.analysesChange}</span>
            </p>
          </div>
        </div>

        {/* Card 2: Data Processed */}
        <div className="flex items-center space-x-4 rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm transition hover:shadow-md dark:border-dark-border dark:bg-dark-card">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600 dark:bg-emerald-950/60 dark:text-emerald-400">
            <Layers className="h-6 w-6" />
          </div>
          <div>
            <p className="text-xs font-medium text-slate-400 dark:text-slate-400">
              Data Processed
            </p>
            <p className="text-2xl font-bold text-slate-900 dark:text-white">
              {MOCK_STATS.dataProcessed}
            </p>
            <p className="mt-0.5 text-xs font-medium text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
              <TrendingUp className="h-3 w-3" />
              <span>{MOCK_STATS.dataProcessedChange}</span>
            </p>
          </div>
        </div>

        {/* Card 3: Active Areas */}
        <div className="flex items-center space-x-4 rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm transition hover:shadow-md dark:border-dark-border dark:bg-dark-card">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-purple-50 text-purple-600 dark:bg-purple-950/60 dark:text-purple-400">
            <ClipboardList className="h-6 w-6" />
          </div>
          <div className="flex-1">
            <p className="text-xs font-medium text-slate-400 dark:text-slate-400">
              Active Areas
            </p>
            <p className="text-2xl font-bold text-slate-900 dark:text-white">
              {MOCK_STATS.activeAreas}
            </p>
            <Link
              to="/workspace"
              className="mt-0.5 text-xs font-medium text-brand-600 hover:text-brand-700 dark:text-brand-400 flex items-center gap-0.5"
            >
              <span>View all areas</span>
              <ArrowRight className="h-3 w-3" />
            </Link>
          </div>
        </div>

        {/* Card 4: Avg. Confidence */}
        <div className="flex items-center space-x-4 rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm transition hover:shadow-md dark:border-dark-border dark:bg-dark-card">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-amber-50 text-amber-600 dark:bg-amber-950/60 dark:text-amber-400">
            <Activity className="h-6 w-6" />
          </div>
          <div>
            <p className="text-xs font-medium text-slate-400 dark:text-slate-400">
              Avg. Confidence
            </p>
            <p className="text-2xl font-bold text-slate-900 dark:text-white">
              {MOCK_STATS.avgConfidence}
            </p>
            <p className="mt-0.5 text-xs font-medium text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
              <TrendingUp className="h-3 w-3" />
              <span>{MOCK_STATS.avgConfidenceChange}</span>
            </p>
          </div>
        </div>
      </div>

      {/* 3. Recent Alerts Section */}
      <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm dark:border-dark-border dark:bg-dark-card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-red-50 text-red-600 dark:bg-red-950/60 dark:text-red-400">
              <Bell className="h-4 w-4" />
            </div>
            <h2 className="text-base font-bold text-slate-900 dark:text-white">
              Recent Alerts
            </h2>
            <span className="flex h-5 min-w-[20px] items-center justify-center rounded-full bg-red-500 px-1.5 text-[11px] font-bold text-white">
              3
            </span>
          </div>

          <Link
            to="/alerts"
            className="text-xs font-semibold text-brand-600 hover:text-brand-700 dark:text-brand-400 flex items-center gap-1"
          >
            <span>View all alerts</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {MOCK_ALERTS.slice(0, 3).map((alert) => (
            <div
              key={alert.id}
              onClick={() => navigate(`/alerts`)}
              className="cursor-pointer group flex flex-col justify-between rounded-xl border border-slate-100 bg-slate-50/60 p-4 transition-all hover:border-slate-300 hover:bg-slate-50 dark:border-dark-border dark:bg-dark-bg/40 dark:hover:bg-dark-hover"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span
                    className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${
                      alert.severity === "critical"
                        ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
                        : alert.severity === "warning"
                        ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                        : "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                    }`}
                  >
                    {alert.type}
                  </span>
                </div>

                <div className="mt-3 flex items-start space-x-3">
                  <div
                    className={`mt-0.5 rounded-lg p-2 ${
                      alert.severity === "critical"
                        ? "bg-red-100 text-red-600 dark:bg-red-950 dark:text-red-400"
                        : alert.severity === "warning"
                        ? "bg-amber-100 text-amber-600 dark:bg-amber-950 dark:text-amber-400"
                        : "bg-blue-100 text-blue-600 dark:bg-blue-950 dark:text-blue-400"
                    }`}
                  >
                    {alert.severity === "critical" ? (
                      <AlertTriangle className="h-5 w-5" />
                    ) : (
                      <CloudRain className="h-5 w-5" />
                    )}
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900 group-hover:text-brand-600 dark:text-slate-100 dark:group-hover:text-brand-400 transition">
                      {alert.title}
                    </h3>
                    <p className="mt-2 text-[11px] font-medium text-slate-400">
                      {alert.timestamp}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 4 & 5. Two Columns: Recent Analyses & Active Monitoring Areas */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Left Column: Recent Analyses (7 cols) */}
        <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm dark:border-dark-border dark:bg-dark-card lg:col-span-7">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <ClipboardList className="h-4 w-4 text-brand-600" />
              <h2 className="text-base font-bold text-slate-900 dark:text-white">
                Recent Analyses
              </h2>
            </div>
            <Link
              to="/reports"
              className="text-xs font-semibold text-brand-600 hover:text-brand-700 dark:text-brand-400 flex items-center gap-1"
            >
              <span>View all</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-100 text-slate-400 dark:border-dark-border">
                  <th className="pb-2.5 font-medium">Title / Area</th>
                  <th className="pb-2.5 font-medium">Type</th>
                  <th className="pb-2.5 font-medium">Date</th>
                  <th className="pb-2.5 font-medium">Confidence</th>
                  <th className="pb-2.5 font-medium text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-dark-border">
                {MOCK_RECENT_ANALYSES.map((item) => (
                  <tr
                    key={item.id}
                    onClick={() => navigate(`/workspace`)}
                    className="cursor-pointer group transition hover:bg-slate-50 dark:hover:bg-dark-hover/60"
                  >
                    <td className="py-3">
                      <div className="flex items-center space-x-3">
                        <div className="h-9 w-9 flex-shrink-0 overflow-hidden rounded-lg bg-slate-100 border border-slate-200 dark:border-dark-border">
                          <img
                            src={item.thumbnail}
                            alt={item.title}
                            className="h-full w-full object-cover"
                          />
                        </div>
                        <div>
                          <p className="font-semibold text-slate-900 group-hover:text-brand-600 dark:text-slate-100 dark:group-hover:text-brand-400">
                            {item.title}
                          </p>
                          <p className="text-[11px] text-slate-400">{item.area}</p>
                        </div>
                      </div>
                    </td>
                    <td className="py-3">
                      <span
                        className={`inline-block rounded px-2 py-0.5 text-[10px] font-semibold ${
                          item.typeColor === "red"
                            ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
                            : item.typeColor === "blue"
                            ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                            : item.typeColor === "purple"
                            ? "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300"
                            : "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                        }`}
                      >
                        {item.type}
                      </span>
                    </td>
                    <td className="py-3 text-slate-600 dark:text-slate-300">
                      <div>{item.date}</div>
                      <div className="text-[10px] text-slate-400">{item.time}</div>
                    </td>
                    <td className="py-3">
                      <span className="rounded-md bg-emerald-50 px-2 py-0.5 font-bold text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400">
                        {item.confidence}%
                      </span>
                    </td>
                    <td className="py-3 text-right">
                      <span className="inline-flex items-center rounded-full bg-emerald-100 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-800 dark:bg-emerald-950 dark:text-emerald-400">
                        {item.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right Column: Active Monitoring Areas (5 cols) */}
        <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm dark:border-dark-border dark:bg-dark-card lg:col-span-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Radio className="h-4 w-4 text-brand-600" />
                <h2 className="text-base font-bold text-slate-900 dark:text-white">
                  Active Monitoring Areas
                </h2>
              </div>
              <Link
                to="/workspace"
                className="text-xs font-semibold text-brand-600 hover:text-brand-700 dark:text-brand-400 flex items-center gap-1"
              >
                <span>View all</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>

            <div className="space-y-3">
              {MOCK_MONITORING_AREAS.slice(0, 3).map((area) => (
                <div
                  key={area.id}
                  onClick={() => navigate(`/workspace`)}
                  className="cursor-pointer group flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50/60 p-3 transition hover:border-slate-300 hover:bg-slate-50 dark:border-dark-border dark:bg-dark-bg/40 dark:hover:bg-dark-hover"
                >
                  <div className="flex items-center space-x-3">
                    <div className="h-12 w-12 flex-shrink-0 overflow-hidden rounded-lg bg-slate-200 border border-slate-200 dark:border-dark-border relative">
                      <img
                        src={area.thumbnail}
                        alt={area.name}
                        className="h-full w-full object-cover"
                      />
                      <span className="absolute inset-0 bg-brand-600/10" />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="h-2 w-2 rounded-full bg-red-500 ring-2 ring-white" />
                      </div>
                    </div>
                    <div>
                      <p className="text-xs font-bold text-slate-900 group-hover:text-brand-600 dark:text-slate-100 dark:group-hover:text-brand-400">
                        {area.name}
                      </p>
                      <p className="text-[11px] text-slate-400">{area.location}</p>
                      <div className="mt-1 flex items-center space-x-1.5 text-[10px] text-slate-500">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                        <span>{area.lastUpdated}</span>
                      </div>
                    </div>
                  </div>

                  <span
                    className={`rounded-full px-2.5 py-0.5 text-[10px] font-bold ${
                      area.riskLevel === "critical"
                        ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
                        : "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                    }`}
                  >
                    {area.risk}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 dark:border-dark-border">
            <Link
              to="/workspace"
              className="text-xs font-semibold text-brand-600 hover:text-brand-700 dark:text-brand-400 flex items-center gap-1.5"
            >
              <span>View all areas on map</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </div>

      {/* 6. Bottom System Operational Status Banner */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs text-slate-600 shadow-sm dark:border-dark-border dark:bg-dark-card dark:text-slate-300">
        <div className="flex items-center space-x-2.5">
          <Info className="h-4 w-4 text-brand-600 flex-shrink-0" />
          <span>
            All systems operational. Satellite data and AI services are running normally.
          </span>
        </div>
        <Link
          to="/settings"
          className="font-semibold text-brand-600 hover:text-brand-700 dark:text-brand-400 flex items-center gap-1 whitespace-nowrap"
        >
          <span>System Status Details</span>
          <ArrowRight className="h-3 w-3" />
        </Link>
      </div>
    </div>
  );
}
