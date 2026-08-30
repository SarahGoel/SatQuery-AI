import React from "react";
import { Search, Square, MessageSquare, Info, ArrowRight } from "lucide-react";

export default function EmptyStateWorkspace() {
  return (
    <div className="relative flex h-full w-full flex-col items-center justify-center overflow-hidden rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition-colors dark:border-dark-border dark:bg-dark-card">
      {/* Subtle Background Grid Glow */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(#3b82f6_1px,transparent_1px)] [background-size:24px_24px] opacity-[0.03] dark:opacity-[0.07]" />

      {/* Satellite Illustration */}
      <div className="relative mb-2 flex items-center justify-center">
        {/* Glow backdrop */}
        <div className="absolute h-36 w-36 rounded-full bg-brand-500/10 blur-2xl dark:bg-brand-600/20" />

        <svg
          viewBox="0 0 280 180"
          className="relative h-36 w-56 text-brand-500"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Stars / sparkle dots */}
          <circle cx="35" cy="40" r="1.5" className="fill-brand-400/60" />
          <circle cx="245" cy="55" r="1.5" className="fill-brand-400/60" />
          <circle cx="210" cy="20" r="1" className="fill-brand-300/40" />
          <circle cx="65" cy="120" r="1" className="fill-brand-300/40" />

          {/* Earth curvature */}
          <ellipse
            cx="140"
            cy="175"
            rx="105"
            ry="75"
            className="fill-slate-100 stroke-slate-300 dark:fill-[#0F172A] dark:stroke-slate-700/60"
            strokeWidth="1.5"
          />
          {/* Earth latitude / longitude grid lines */}
          <path
            d="M 45 155 Q 140 120 235 155"
            className="stroke-slate-200 dark:stroke-slate-800"
            strokeWidth="1"
            strokeDasharray="3 3"
          />
          <path
            d="M 60 170 Q 140 140 220 170"
            className="stroke-slate-200 dark:stroke-slate-800"
            strokeWidth="1"
            strokeDasharray="3 3"
          />
          <path
            d="M 140 100 Q 140 140 140 180"
            className="stroke-slate-200 dark:stroke-slate-800"
            strokeWidth="1"
            strokeDasharray="3 3"
          />

          {/* Continents outline representation */}
          <path
            d="M 100 135 Q 115 130 130 138 Q 145 130 155 142 Q 130 152 110 145 Z"
            className="fill-brand-100/50 dark:fill-brand-950/60 stroke-brand-300/40 dark:stroke-brand-700/40"
            strokeWidth="1"
          />
          <path
            d="M 165 145 Q 185 140 195 150 Q 180 162 165 155 Z"
            className="fill-brand-100/40 dark:fill-brand-950/50 stroke-brand-300/30 dark:stroke-brand-700/30"
            strokeWidth="0.8"
          />

          {/* Atmosphere glow arc */}
          <path
            d="M 40 160 A 110 80 0 0 1 240 160"
            className="stroke-brand-400/40 dark:stroke-brand-500/30"
            strokeWidth="2"
            strokeLinecap="round"
          />

          {/* Satellite Orbit Path */}
          <path
            d="M 45 95 C 80 50, 200 45, 235 90"
            className="stroke-brand-400/50 dark:stroke-brand-500/40"
            strokeWidth="1.2"
            strokeDasharray="4 4"
          />

          {/* Satellite Body (Centered around cx=140, cy=65) */}
          <g transform="translate(140, 65) rotate(-28)">
            {/* Left Solar Panel */}
            <rect
              x="-42"
              y="-10"
              width="24"
              height="20"
              rx="2"
              className="fill-brand-600 stroke-brand-400 dark:fill-brand-600 dark:stroke-brand-300"
              strokeWidth="1.2"
            />
            {/* Solar panel grid lines */}
            <line x1="-34" y1="-10" x2="-34" y2="10" className="stroke-brand-300/70" strokeWidth="0.8" />
            <line x1="-26" y1="-10" x2="-26" y2="10" className="stroke-brand-300/70" strokeWidth="0.8" />
            <line x1="-42" y1="0" x2="-18" y2="0" className="stroke-brand-300/70" strokeWidth="0.8" />

            {/* Panel connector left */}
            <line x1="-18" y1="0" x2="-10" y2="0" className="stroke-slate-400 dark:stroke-slate-300" strokeWidth="1.5" />

            {/* Central Satellite Chassis */}
            <rect
              x="-10"
              y="-12"
              width="20"
              height="24"
              rx="3"
              className="fill-slate-100 stroke-slate-400 dark:fill-slate-200 dark:stroke-slate-400"
              strokeWidth="1.2"
            />
            {/* Chassis sensor detail */}
            <circle cx="0" cy="-3" r="3.5" className="fill-brand-600" />
            <rect x="-6" y="4" width="12" height="4" rx="1" className="fill-slate-400 dark:fill-slate-500" />

            {/* Panel connector right */}
            <line x1="10" y1="0" x2="18" y2="0" className="stroke-slate-400 dark:stroke-slate-300" strokeWidth="1.5" />

            {/* Right Solar Panel */}
            <rect
              x="18"
              y="-10"
              width="24"
              height="20"
              rx="2"
              className="fill-brand-600 stroke-brand-400 dark:fill-brand-600 dark:stroke-brand-300"
              strokeWidth="1.2"
            />
            {/* Solar panel grid lines */}
            <line x1="26" y1="-10" x2="26" y2="10" className="stroke-brand-300/70" strokeWidth="0.8" />
            <line x1="34" y1="-10" x2="34" y2="10" className="stroke-brand-300/70" strokeWidth="0.8" />
            <line x1="18" y1="0" x2="42" y2="0" className="stroke-brand-300/70" strokeWidth="0.8" />

            {/* Downward sensor dish beam */}
            <path
              d="M -5 12 L 0 17 L 5 12"
              className="stroke-brand-500 fill-none"
              strokeWidth="1.2"
            />
          </g>

          {/* Cloud elements */}
          <path
            d="M 50 82 Q 55 76 62 76 Q 70 76 72 82 Q 78 82 80 87 Q 80 92 73 92 L 48 92 Q 42 92 42 87 Q 42 82 50 82 Z"
            className="fill-slate-200/70 dark:fill-slate-700/50"
          />
          <path
            d="M 205 70 Q 210 65 216 65 Q 223 65 225 70 Q 230 70 232 74 Q 232 79 226 79 L 204 79 Q 198 79 198 74 Q 198 70 205 70 Z"
            className="fill-slate-200/70 dark:fill-slate-700/50"
          />
        </svg>
      </div>

      {/* Main Title & Subtitle */}
      <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white sm:text-2xl">
        Ready to Analyze
      </h2>
      <p className="mt-1.5 max-w-md text-center text-xs leading-relaxed text-slate-500 dark:text-slate-400">
        Search for a location or select an area, then ask SatQuery AI a question to get satellite insights.
      </p>

      {/* 3-Step Process Flow */}
      <div className="mt-8 flex w-full max-w-xl items-center justify-between px-2 sm:px-6">
        {/* Step 1 */}
        <div className="flex flex-col items-center text-center">
          <div className="flex h-11 w-11 items-center justify-center rounded-full border border-brand-200 bg-brand-50 text-brand-600 shadow-sm transition-transform hover:scale-105 dark:border-brand-800/80 dark:bg-brand-950/40 dark:text-brand-400">
            <Search className="h-5 w-5" />
          </div>
          <span className="mt-2.5 text-xs font-bold text-slate-900 dark:text-slate-100">
            1. Search Location
          </span>
          <span className="mt-0.5 text-[11px] text-slate-400">
            Find a place or area
          </span>
        </div>

        {/* Connector Arrow */}
        <div className="mb-6 flex-shrink-0 text-slate-300 dark:text-slate-600">
          <ArrowRight className="h-4 w-4" />
        </div>

        {/* Step 2 */}
        <div className="flex flex-col items-center text-center">
          <div className="flex h-11 w-11 items-center justify-center rounded-full border border-brand-200 bg-brand-50 text-brand-600 shadow-sm transition-transform hover:scale-105 dark:border-brand-800/80 dark:bg-brand-950/40 dark:text-brand-400">
            <Square className="h-5 w-5" />
          </div>
          <span className="mt-2.5 text-xs font-bold text-slate-900 dark:text-slate-100">
            2. Select Area
          </span>
          <span className="mt-0.5 text-[11px] text-slate-400">
            Select the required area
          </span>
        </div>

        {/* Connector Arrow */}
        <div className="mb-6 flex-shrink-0 text-slate-300 dark:text-slate-600">
          <ArrowRight className="h-4 w-4" />
        </div>

        {/* Step 3 */}
        <div className="flex flex-col items-center text-center">
          <div className="flex h-11 w-11 items-center justify-center rounded-full border border-brand-200 bg-brand-50 text-brand-600 shadow-sm transition-transform hover:scale-105 dark:border-brand-800/80 dark:bg-brand-950/40 dark:text-brand-400">
            <MessageSquare className="h-5 w-5" />
          </div>
          <span className="mt-2.5 text-xs font-bold text-slate-900 dark:text-slate-100">
            3. Ask a Question
          </span>
          <span className="mt-0.5 text-[11px] text-slate-400">
            Get AI-powered insights
          </span>
        </div>
      </div>

      {/* Subtle Information Card */}
      <div className="mt-8 flex w-full max-w-md items-center space-x-3 rounded-xl border border-slate-200/80 bg-slate-50/70 p-3.5 shadow-2xs transition-colors dark:border-dark-border dark:bg-dark-bg/60">
        <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-brand-600 text-white shadow-xs">
          <Info className="h-4 w-4" />
        </div>
        <div>
          <p className="text-xs font-bold text-slate-800 dark:text-slate-200">
            No area selected
          </p>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            Search for a location to begin analysis.
          </p>
        </div>
      </div>
    </div>
  );
}
