import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Menu,
  ShieldCheck,
  Activity,
  Calendar,
  Sun,
  Moon,
  Bell,
  User,
  ChevronDown,
  Server,
  HardDrive,
  Cpu,
  CheckCircle2,
  X,
  ExternalLink,
  AlertTriangle
} from "lucide-react";
import { useTheme } from "../../context/ThemeContext";
import { MOCK_ALERTS } from "../../mock/mockData";

export default function TopNavbar({ onToggleSidebar, isSidebarCollapsed }) {
  const { theme, toggleTheme } = useTheme();
  const [showNotifications, setShowNotifications] = useState(false);
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [showSystemModal, setShowSystemModal] = useState(false);

  const activeAlertCount = MOCK_ALERTS.filter(
    (a) => a.severity === "critical" || a.severity === "warning"
  ).length;

  return (
    <>
      <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-slate-200 bg-white/95 px-4 backdrop-blur-md transition-colors duration-200 dark:border-dark-border dark:bg-dark-card/95">
        {/* Left Side: Collapse Toggle & Branding */}
        <div className="flex items-center space-x-3 sm:space-x-4">
          <button
            type="button"
            onClick={onToggleSidebar}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 text-slate-600 transition hover:bg-slate-100 hover:text-slate-900 dark:border-dark-border dark:text-slate-300 dark:hover:bg-dark-hover dark:hover:text-white"
            title={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-label="Toggle sidebar"
          >
            <Menu className="h-5 w-5" />
          </button>

          <Link to="/dashboard" className="flex items-center space-x-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-tr from-brand-600 to-cyan-500 shadow-sm shadow-brand-500/20 text-white">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-5 w-5"
              >
                <path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z" />
              </svg>
            </div>
            <div>
              <div className="flex items-center space-x-1.5">
                <span className="text-base font-bold tracking-tight text-slate-900 dark:text-white">
                  SatQuery AI
                </span>
                <span className="rounded bg-brand-100 px-1.5 py-0.2 text-[10px] font-semibold text-brand-700 dark:bg-brand-950/80 dark:text-brand-300">
                  SIH26167
                </span>
              </div>
              <p className="text-[11px] font-medium text-slate-400 dark:text-slate-400">
                Satellite Intelligence
              </p>
            </div>
          </Link>
        </div>

        {/* Center / Status Badges */}
        <div className="hidden items-center space-x-3 md:flex lg:space-x-5">
          {/* Air-Gapped Compliance Badge */}
          <div className="flex items-center space-x-2 rounded-full border border-emerald-200 bg-emerald-50/80 px-3 py-1 text-xs font-medium text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-400">
            <ShieldCheck className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
            <span className="font-semibold">Air-Gapped & Compliant</span>
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500"></span>
            </span>
          </div>

          {/* System Status Healthy */}
          <button
            onClick={() => setShowSystemModal(true)}
            className="flex items-center space-x-2 rounded-lg px-2.5 py-1 text-xs text-slate-600 transition hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
          >
            <span className="text-slate-400 dark:text-slate-500">System Status</span>
            <span className="flex items-center space-x-1.5 font-medium text-emerald-600 dark:text-emerald-400">
              <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
              <span>Healthy</span>
            </span>
          </button>

          {/* Date / Time */}
          <div className="flex items-center space-x-1.5 rounded-lg border border-slate-200/80 bg-slate-50/80 px-3 py-1 text-xs font-medium text-slate-600 dark:border-dark-border dark:bg-dark-bg/60 dark:text-slate-300">
            <Calendar className="h-3.5 w-3.5 text-slate-400" />
            <span>30 May 2025, 09:30 AM</span>
          </div>
        </div>

        {/* Right Side: Notifications, Theme Switch, Profile */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          {/* Notifications Trigger */}
          <div className="relative">
            <button
              type="button"
              onClick={() => {
                setShowNotifications(!showNotifications);
                setShowProfileMenu(false);
              }}
              className="relative flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 text-slate-600 transition hover:bg-slate-100 dark:border-dark-border dark:text-slate-300 dark:hover:bg-dark-hover"
              title="Recent Notifications"
            >
              <Bell className="h-4 w-4" />
              {activeAlertCount > 0 && (
                <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-[10px] font-bold text-white shadow-sm">
                  {activeAlertCount}
                </span>
              )}
            </button>

            {/* Notifications Dropdown */}
            {showNotifications && (
              <div className="absolute right-0 mt-2 w-80 sm:w-96 rounded-xl border border-slate-200 bg-white p-3 shadow-xl ring-1 ring-black/5 transition-all dark:border-dark-border dark:bg-dark-card">
                <div className="flex items-center justify-between border-b border-slate-100 pb-2.5 dark:border-dark-border">
                  <div className="flex items-center space-x-2">
                    <Bell className="h-4 w-4 text-brand-600" />
                    <span className="text-sm font-semibold text-slate-900 dark:text-white">
                      Active Sat-Alerts
                    </span>
                  </div>
                  <Link
                    to="/alerts"
                    onClick={() => setShowNotifications(false)}
                    className="text-xs font-medium text-brand-600 hover:text-brand-700 dark:text-brand-400"
                  >
                    View all
                  </Link>
                </div>
                <div className="mt-2 divide-y divide-slate-100 max-h-80 overflow-y-auto dark:divide-dark-border">
                  {MOCK_ALERTS.slice(0, 3).map((alert) => (
                    <div key={alert.id} className="py-2.5 first:pt-1 last:pb-1">
                      <div className="flex items-start space-x-2.5">
                        <div
                          className={`mt-0.5 rounded p-1 ${
                            alert.severity === "critical"
                              ? "bg-red-100 text-red-600 dark:bg-red-950 dark:text-red-400"
                              : alert.severity === "warning"
                              ? "bg-amber-100 text-amber-600 dark:bg-amber-950 dark:text-amber-400"
                              : "bg-blue-100 text-blue-600 dark:bg-blue-950 dark:text-blue-400"
                          }`}
                        >
                          <AlertTriangle className="h-3.5 w-3.5" />
                        </div>
                        <div className="flex-1">
                          <p className="text-xs font-medium text-slate-900 dark:text-slate-100">
                            {alert.title}
                          </p>
                          <div className="mt-1 flex items-center justify-between text-[11px] text-slate-400">
                            <span>{alert.location}</span>
                            <span>{alert.timestamp.split(",")[1]}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Dark / Light Mode Toggle */}
          <button
            type="button"
            onClick={toggleTheme}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 text-slate-600 transition hover:bg-slate-100 dark:border-dark-border dark:text-slate-300 dark:hover:bg-dark-hover"
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4 text-amber-400" />
            ) : (
              <Moon className="h-4 w-4 text-slate-600" />
            )}
          </button>

          {/* Operator Profile Menu */}
          <div className="relative">
            <button
              type="button"
              onClick={() => {
                setShowProfileMenu(!showProfileMenu);
                setShowNotifications(false);
              }}
              className="flex items-center space-x-2 rounded-lg border border-slate-200 bg-slate-50/80 px-2.5 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-100 dark:border-dark-border dark:bg-dark-bg/60 dark:text-slate-200 dark:hover:bg-dark-hover"
            >
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-brand-600 text-white font-semibold text-xs">
                OP
              </div>
              <span className="hidden sm:inline">Operator</span>
              <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
            </button>

            {showProfileMenu && (
              <div className="absolute right-0 mt-2 w-56 rounded-xl border border-slate-200 bg-white p-2 shadow-xl ring-1 ring-black/5 dark:border-dark-border dark:bg-dark-card">
                <div className="border-b border-slate-100 px-3 py-2 dark:border-dark-border">
                  <p className="text-xs font-semibold text-slate-900 dark:text-white">
                    Lead EO Analyst
                  </p>
                  <p className="text-[11px] text-slate-400">analyst@isro.gov.in (Air-Gapped)</p>
                </div>
                <div className="mt-1 space-y-1">
                  <button
                    onClick={() => {
                      setShowProfileMenu(false);
                      setShowSystemModal(true);
                    }}
                    className="flex w-full items-center space-x-2 rounded-lg px-3 py-2 text-xs text-slate-700 transition hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
                  >
                    <Server className="h-3.5 w-3.5 text-brand-600" />
                    <span>System Status & Nodes</span>
                  </button>
                  <Link
                    to="/settings"
                    onClick={() => setShowProfileMenu(false)}
                    className="flex w-full items-center space-x-2 rounded-lg px-3 py-2 text-xs text-slate-700 transition hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
                  >
                    <User className="h-3.5 w-3.5 text-slate-500" />
                    <span>Analyst Preferences</span>
                  </Link>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* System Status Modal */}
      {showSystemModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-4 backdrop-blur-sm">
          <div className="w-full max-w-lg rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl dark:border-dark-border dark:bg-dark-card">
            <div className="flex items-center justify-between border-b border-slate-100 pb-4 dark:border-dark-border">
              <div className="flex items-center space-x-2.5">
                <div className="rounded-lg bg-emerald-100 p-2 text-emerald-600 dark:bg-emerald-950 dark:text-emerald-400">
                  <Activity className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white">
                    Air-Gapped System Diagnostics
                  </h3>
                  <p className="text-xs text-slate-400">SIH26167 On-Premise GPU Node Status</p>
                </div>
              </div>
              <button
                onClick={() => setShowSystemModal(false)}
                className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-dark-hover"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mt-4 space-y-3">
              <div className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 p-3 dark:border-dark-border dark:bg-dark-bg/60">
                <div className="flex items-center space-x-3">
                  <Cpu className="h-5 w-5 text-brand-600" />
                  <div>
                    <p className="text-xs font-semibold text-slate-900 dark:text-white">
                      Inference Engine
                    </p>
                    <p className="text-[11px] text-slate-400">NVIDIA RTX 4090 / CUDA 12.2 (Local)</p>
                  </div>
                </div>
                <span className="flex items-center space-x-1 rounded-full bg-emerald-100 px-2 py-0.5 text-[11px] font-semibold text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400">
                  <CheckCircle2 className="h-3 w-3" />
                  <span>Ready</span>
                </span>
              </div>

              <div className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 p-3 dark:border-dark-border dark:bg-dark-bg/60">
                <div className="flex items-center space-x-3">
                  <HardDrive className="h-5 w-5 text-brand-600" />
                  <div>
                    <p className="text-xs font-semibold text-slate-900 dark:text-white">
                      Local GeoTIFF Raster Storage
                    </p>
                    <p className="text-[11px] text-slate-400">NVMe High Throughput (1.8 TB Free)</p>
                  </div>
                </div>
                <span className="flex items-center space-x-1 rounded-full bg-emerald-100 px-2 py-0.5 text-[11px] font-semibold text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400">
                  <CheckCircle2 className="h-3 w-3" />
                  <span>Mounted</span>
                </span>
              </div>

              <div className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 p-3 dark:border-dark-border dark:bg-dark-bg/60">
                <div className="flex items-center space-x-3">
                  <ShieldCheck className="h-5 w-5 text-emerald-600" />
                  <div>
                    <p className="text-xs font-semibold text-slate-900 dark:text-white">
                      Air-Gap Network Isolation
                    </p>
                    <p className="text-[11px] text-slate-400">Zero External Egress / NIC Hardware Lock</p>
                  </div>
                </div>
                <span className="flex items-center space-x-1 rounded-full bg-emerald-100 px-2 py-0.5 text-[11px] font-semibold text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400">
                  <CheckCircle2 className="h-3 w-3" />
                  <span>Enforced</span>
                </span>
              </div>
            </div>

            <div className="mt-6 flex justify-end">
              <button
                type="button"
                onClick={() => setShowSystemModal(false)}
                className="rounded-lg bg-brand-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-brand-700"
              >
                Close Status
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
