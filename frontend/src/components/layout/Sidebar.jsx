import React from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Map,
  Bell,
  FileText,
  Settings,
  HelpCircle,
} from "lucide-react";
import { MOCK_ALERTS } from "../../mock/mockData";

export default function Sidebar({ isCollapsed }) {
  const activeAlertCount = MOCK_ALERTS.filter(
    (a) => a.severity === "critical" || a.severity === "warning"
  ).length;

  const mainNavItems = [
    {
      name: "Dashboard",
      path: "/dashboard",
      icon: LayoutDashboard,
    },
    {
      name: "Geospatial AI Analysis",
      path: "/workspace",
      icon: Map,
    },
    {
      name: "Alerts",
      path: "/alerts",
      icon: Bell,
      badge: activeAlertCount,
    },
    {
      name: "Reports & History",
      path: "/reports",
      icon: FileText,
    },
    {
      name: "Settings",
      path: "/settings",
      icon: Settings,
    },
  ];

  const bottomNavItems = [
    {
      name: "Help & Support",
      path: "/help",
      icon: HelpCircle,
    },
  ];

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 flex flex-col border-r border-slate-200 bg-white transition-all duration-300 dark:border-dark-border dark:bg-dark-sidebar ${
        isCollapsed ? "w-20" : "w-64"
      }`}
    >
      {/* Top Branding Area inside sidebar */}
      <div className="flex h-16 items-center border-b border-slate-100 px-5 dark:border-dark-border">
        <NavLink to="/dashboard" className="flex items-center space-x-3 overflow-hidden">
          <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-brand-600 to-cyan-500 shadow-md shadow-brand-500/20 text-white">
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
          {!isCollapsed && (
            <div className="flex flex-col">
              <span className="text-base font-bold tracking-tight text-slate-900 dark:text-white">
                SatQuery AI
              </span>
              <span className="text-[11px] font-medium text-slate-400">
                Satellite Intelligence
              </span>
            </div>
          )}
        </NavLink>
      </div>

      {/* Main Nav Items */}
      <div className="flex flex-1 flex-col justify-between p-3.5">
        <nav className="space-y-1.5">
          {mainNavItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `group flex items-center rounded-xl px-3.5 py-3 text-sm font-medium transition-all duration-150 ${
                    isActive
                      ? "bg-brand-600 text-white shadow-sm shadow-brand-600/30 dark:bg-brand-600 dark:text-white"
                      : "text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-dark-hover dark:hover:text-white"
                  } ${isCollapsed ? "justify-center px-2" : "justify-between"}`
                }
                title={isCollapsed ? item.name : undefined}
              >
                <div className="flex items-center space-x-3">
                  <Icon className="h-5 w-5 flex-shrink-0" />
                  {!isCollapsed && <span>{item.name}</span>}
                </div>

                {!isCollapsed && item.badge !== undefined && item.badge > 0 && (
                  <span className="flex h-5 min-w-[20px] items-center justify-center rounded-full bg-red-500 px-1.5 text-[11px] font-bold text-white shadow-sm">
                    {item.badge}
                  </span>
                )}
                {isCollapsed && item.badge !== undefined && item.badge > 0 && (
                  <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-red-500 ring-2 ring-white dark:ring-dark-sidebar" />
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Separated Bottom Nav Item (Help & Support) */}
        <div className="border-t border-slate-100 pt-3 dark:border-dark-border">
          {bottomNavItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `group flex items-center rounded-xl px-3.5 py-3 text-sm font-medium transition-all duration-150 ${
                    isActive
                      ? "bg-brand-600 text-white shadow-sm shadow-brand-600/30 dark:bg-brand-600 dark:text-white"
                      : "text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-dark-hover dark:hover:text-white"
                  } ${isCollapsed ? "justify-center px-2" : "space-x-3"}`
                }
                title={isCollapsed ? item.name : undefined}
              >
                <Icon className="h-5 w-5 flex-shrink-0" />
                {!isCollapsed && <span>{item.name}</span>}
              </NavLink>
            );
          })}
        </div>
      </div>
    </aside>
  );
}
