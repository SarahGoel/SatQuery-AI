import React, { useState } from "react";
import LayerControl from "../components/workspace/LayerControl";
import EmptyStateWorkspace from "../components/workspace/EmptyStateWorkspace";
import ChatPanel from "../components/workspace/ChatPanel";
import MapViewer from "../components/MapViewer";
import {
  Maximize2,
  Minimize2,
  ChevronDown,
  Search,
  Filter,
  Calendar,
  Sparkles,
  Crosshair,
  Info,
  MapPin,
  X
} from "lucide-react";

export default function GeospatialAnalysis() {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [searchLocationOpen, setSearchLocationOpen] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterOpen, setFilterOpen] = useState(false);

  // Date selection states
  const [t1Date, setT1Date] = useState("");
  const [t2Date, setT2Date] = useState("");
  const [showT1Picker, setShowT1Picker] = useState(false);
  const [showT2Picker, setShowT2Picker] = useState(false);

  // Layer states
  const [baseImagery, setBaseImagery] = useState("optical");
  const [indexLayers, setIndexLayers] = useState({
    ndvi: false,
    ndwi: false,
    ndbi: false,
    ndmi: false,
  });
  const [vectorLayers, setVectorLayers] = useState({
    floodRisk: false,
    adminBoundary: false,
    roads: false,
    waterBodies: false,
  });
  const [otherLayers, setOtherLayers] = useState({
    cloudMask: false,
  });

  // Assistant Query Input
  const [assistantInput, setAssistantInput] = useState("");

  const presetLocations = [
    { id: "assam", name: "Brahmaputra Basin, Assam", coords: "26.2006° N, 92.9376° E", lon: 92.9376, lat: 26.2006 },
    { id: "kerala", name: "Western Ghats, Kerala", coords: "10.8505° N, 76.2711° E", lon: 76.2711, lat: 10.8505 },
    { id: "sundarbans", name: "Sundarbans Delta Zone", coords: "21.9497° N, 88.8056° E", lon: 88.8056, lat: 21.9497 },
    { id: "godavari", name: "Godavari River Basin", coords: "16.9891° N, 81.8040° E", lon: 81.8040, lat: 16.9891 },
    { id: "karnataka", name: "Coastal Karnataka", coords: "14.5479° N, 74.5566° E", lon: 74.5566, lat: 14.5479 },
    { id: "ladakh", name: "Pangong Tso & Ladakh Glaciers", coords: "33.7595° N, 78.6674° E", lon: 78.6674, lat: 33.7595 },
  ];

  const pad = 0.18;
  const selectedBbox = selectedLocation
    ? [
        selectedLocation.lon - pad,
        selectedLocation.lat - pad,
        selectedLocation.lon + pad,
        selectedLocation.lat + pad,
      ]
    : null;
  const selectedOverlay = selectedLocation
    ? {
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            properties: { name: selectedLocation.name, mask: "change-vector" },
            geometry: {
              type: "Polygon",
              coordinates: [[
                [selectedLocation.lon - 0.08, selectedLocation.lat - 0.05],
                [selectedLocation.lon + 0.1, selectedLocation.lat - 0.04],
                [selectedLocation.lon + 0.12, selectedLocation.lat + 0.08],
                [selectedLocation.lon - 0.06, selectedLocation.lat + 0.07],
                [selectedLocation.lon - 0.08, selectedLocation.lat - 0.05],
              ]],
            },
          },
        ],
      }
    : null;

  const handleToggleIndexLayer = (key) => {
    setIndexLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleToggleVectorLayer = (key) => {
    setVectorLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleToggleOtherLayer = (key) => {
    setOtherLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().catch(() => {});
      }
      setIsFullscreen(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 font-sans antialiased">
      {/* 1. Page Header */}
      <div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-center">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white sm:text-2xl">
            Geospatial AI Analysis
          </h1>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
            Ask questions, analyze satellite imagery, and extract actionable insights.
          </p>
        </div>

        {/* Open in Fullscreen Button */}
        <button
          type="button"
          onClick={toggleFullscreen}
          className="flex items-center space-x-2 rounded-xl border border-slate-200 bg-white px-3.5 py-1.5 text-xs font-semibold text-slate-700 shadow-2xs transition hover:border-slate-300 hover:bg-slate-50 dark:border-dark-border dark:bg-dark-card dark:text-slate-200 dark:hover:bg-dark-hover"
        >
          {isFullscreen ? (
            <>
              <Minimize2 className="h-3.5 w-3.5 text-slate-500 dark:text-slate-400" />
              <span>Exit Fullscreen</span>
            </>
          ) : (
            <>
              <Maximize2 className="h-3.5 w-3.5 text-slate-500 dark:text-slate-400" />
              <span>Open in Fullscreen</span>
            </>
          )}
        </button>
      </div>

      {/* 2. Analysis Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Left Toolbar Controls: Location Dropdown + Search Input + Filter */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Location Dropdown */}
          <div className="relative">
            <button
              type="button"
              onClick={() => setSearchLocationOpen(!searchLocationOpen)}
              className="flex items-center space-x-2 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-2xs transition hover:bg-slate-50 dark:border-dark-border dark:bg-dark-card dark:text-slate-200 dark:hover:bg-dark-hover"
            >
              <span>{selectedLocation ? selectedLocation.name.split(",")[0] : "Search Location"}</span>
              <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
            </button>

            {searchLocationOpen && (
              <div className="absolute left-0 top-full z-30 mt-1.5 w-64 rounded-xl border border-slate-200 bg-white p-1.5 shadow-xl ring-1 ring-black/5 dark:border-dark-border dark:bg-dark-card">
                <div className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  Select Preset Location
                </div>
                <div className="max-h-56 overflow-y-auto space-y-1">
                  {presetLocations.map((loc) => (
                    <button
                      key={loc.id}
                      type="button"
                      onClick={() => {
                        setSelectedLocation(loc);
                        setSearchQuery(loc.name);
                        setSearchLocationOpen(false);
                      }}
                      className="flex w-full items-start space-x-2 rounded-lg px-2.5 py-2 text-left text-xs transition hover:bg-slate-100 dark:hover:bg-dark-hover"
                    >
                      <MapPin className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-brand-600 dark:text-brand-400" />
                      <div>
                        <p className="font-medium text-slate-800 dark:text-slate-200">{loc.name}</p>
                        <p className="text-[10px] text-slate-400">{loc.coords}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Search Input Field */}
          <div className="relative flex w-64 items-center sm:w-80">
            <input
              type="text"
              placeholder="Search for a place or area..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full rounded-xl border border-slate-200 bg-white py-2 pl-3.5 pr-8 text-xs text-slate-800 placeholder-slate-400 shadow-2xs transition focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/20 dark:border-dark-border dark:bg-dark-card dark:text-slate-100"
            />
            {searchQuery ? (
              <button
                onClick={() => {
                  setSearchQuery("");
                  setSelectedLocation(null);
                }}
                className="absolute right-2.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            ) : (
              <Search className="pointer-events-none absolute right-2.5 h-3.5 w-3.5 text-slate-400" />
            )}
          </div>

          {/* Filter Button */}
          <button
            type="button"
            onClick={() => setFilterOpen(!filterOpen)}
            className="flex h-8.5 w-8.5 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 shadow-2xs transition hover:bg-slate-50 dark:border-dark-border dark:bg-dark-card dark:text-slate-300 dark:hover:bg-dark-hover"
            title="Filter Settings"
          >
            <Filter className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Right Toolbar Controls: T1 Date, vs, T2 Date, Compare Button */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {/* T1 Date Selector */}
          <div className="relative flex items-center rounded-xl border border-slate-200 bg-white px-2.5 py-1.5 shadow-2xs dark:border-dark-border dark:bg-dark-card">
            <span className="mr-2 rounded bg-brand-600 px-1.5 py-0.5 text-[10px] font-bold text-white shadow-xs">
              T1
            </span>
            <input
              type="text"
              readOnly
              value={t1Date || "Select date"}
              onClick={() => setShowT1Picker(!showT1Picker)}
              className="w-20 cursor-pointer bg-transparent text-xs font-medium text-slate-700 focus:outline-none dark:text-slate-200"
            />
            <button
              type="button"
              onClick={() => setShowT1Picker(!showT1Picker)}
              className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
            >
              <Calendar className="h-3.5 w-3.5" />
            </button>

            {showT1Picker && (
              <div className="absolute right-0 top-full z-30 mt-1.5 w-48 rounded-xl border border-slate-200 bg-white p-2 shadow-xl dark:border-dark-border dark:bg-dark-card">
                <p className="mb-1 text-[10px] font-semibold text-slate-400">Select T1 Date</p>
                <input
                  type="date"
                  onChange={(e) => {
                    setT1Date(e.target.value);
                    setShowT1Picker(false);
                  }}
                  className="w-full rounded border border-slate-200 p-1 text-xs dark:border-dark-border dark:bg-dark-bg dark:text-white"
                />
              </div>
            )}
          </div>

          <span className="text-xs font-bold text-slate-400 dark:text-slate-500">
            vs
          </span>

          {/* T2 Date Selector */}
          <div className="relative flex items-center rounded-xl border border-slate-200 bg-white px-2.5 py-1.5 shadow-2xs dark:border-dark-border dark:bg-dark-card">
            <span className="mr-2 rounded bg-brand-600 px-1.5 py-0.5 text-[10px] font-bold text-white shadow-xs">
              T2
            </span>
            <input
              type="text"
              readOnly
              value={t2Date || "Select date"}
              onClick={() => setShowT2Picker(!showT2Picker)}
              className="w-20 cursor-pointer bg-transparent text-xs font-medium text-slate-700 focus:outline-none dark:text-slate-200"
            />
            <button
              type="button"
              onClick={() => setShowT2Picker(!showT2Picker)}
              className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
            >
              <Calendar className="h-3.5 w-3.5" />
            </button>

            {showT2Picker && (
              <div className="absolute right-0 top-full z-30 mt-1.5 w-48 rounded-xl border border-slate-200 bg-white p-2 shadow-xl dark:border-dark-border dark:bg-dark-card">
                <p className="mb-1 text-[10px] font-semibold text-slate-400">Select T2 Date</p>
                <input
                  type="date"
                  onChange={(e) => {
                    setT2Date(e.target.value);
                    setShowT2Picker(false);
                  }}
                  className="w-full rounded border border-slate-200 p-1 text-xs dark:border-dark-border dark:bg-dark-bg dark:text-white"
                />
              </div>
            )}
          </div>

          {/* Compare Button */}
          <button
            type="button"
            className="flex items-center space-x-1.5 rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-2xs transition hover:bg-slate-50 dark:border-dark-border dark:bg-dark-card dark:text-slate-200 dark:hover:bg-dark-hover"
          >
            <Sparkles className="h-3.5 w-3.5 text-brand-600 dark:text-brand-400" />
            <span>Compare</span>
          </button>
        </div>
      </div>

      {/* 3. Main Workspace: Three-Column Layout */}
      <div className="flex h-[calc(100vh-16rem)] min-h-[520px] flex-col gap-5 lg:flex-row">
        {/* Left Column: Data & Layers (~320px) */}
        <div className="w-full flex-shrink-0 lg:w-80">
          <LayerControl
            baseImagery={baseImagery}
            onBaseImageryChange={setBaseImagery}
            indexLayers={indexLayers}
            onToggleIndexLayer={handleToggleIndexLayer}
            vectorLayers={vectorLayers}
            onToggleVectorLayer={handleToggleVectorLayer}
            otherLayers={otherLayers}
            onToggleOtherLayer={handleToggleOtherLayer}
          />
        </div>

        {/* Center Column: OpenLayers GIS client */}
        <div className="relative flex-1 min-w-0">
          <MapViewer
            geojsonOverlay={selectedOverlay}
            bboxCoordinates={selectedBbox || [68.0, 6.5, 97.5, 35.5]}
            baseImagery={baseImagery}
            onBaseImageryChange={setBaseImagery}
          />
          {!selectedLocation && (
            <div className="pointer-events-none absolute inset-x-0 bottom-10 z-20 flex justify-center px-4">
              <div className="pointer-events-auto max-h-[70%] max-w-xl overflow-y-auto rounded-2xl shadow-xl">
                <EmptyStateWorkspace />
              </div>
            </div>
          )}
        </div>

        {/* Right Column: AI Analysis Assistant (~430px) */}
        <div className="w-full flex-shrink-0 lg:w-[430px]">
          <ChatPanel
            externalInput={assistantInput}
            onInputChange={setAssistantInput}
          />
        </div>
      </div>

      {/* 4. Bottom Status Bar */}
      <div className="flex flex-col items-start justify-between gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs shadow-2xs transition-colors dark:border-dark-border dark:bg-dark-card sm:flex-row sm:items-center">
        {/* Left Status Items */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-slate-600 dark:text-slate-400">
          <div className="flex items-center space-x-1.5">
            <Crosshair className="h-3.5 w-3.5 text-slate-400 dark:text-slate-500" />
            <span className="font-semibold text-slate-800 dark:text-slate-200">
              {selectedLocation ? selectedLocation.name : "No area selected"}
            </span>
          </div>

          <span className="hidden sm:inline text-slate-300 dark:text-slate-700">|</span>

          <div>
            <span>Coordinates: </span>
            <span className="font-mono text-slate-800 dark:text-slate-200">
              {selectedLocation ? selectedLocation.coords : "--"}
            </span>
          </div>

          <span className="hidden sm:inline text-slate-300 dark:text-slate-700">|</span>

          <div>
            <span>Imagery: </span>
            <span className="font-medium text-slate-800 dark:text-slate-200">
              {baseImagery === "optical" ? "Sentinel-2 (Optical)" : "Sentinel-1 (SAR)"}
            </span>
          </div>

          <span className="hidden sm:inline text-slate-300 dark:text-slate-700">|</span>

          <div>
            <span>Resolution: </span>
            <span className="font-medium text-slate-800 dark:text-slate-200">--</span>
          </div>

          <span className="hidden sm:inline text-slate-300 dark:text-slate-700">|</span>

          <div>
            <span>Cloud Cover: </span>
            <span className="font-medium text-slate-800 dark:text-slate-200">--</span>
          </div>
        </div>

        {/* Right Status Tip */}
        <div className="flex items-center space-x-1.5 text-slate-500 dark:text-slate-400">
          <span>Tip:</span>
          <span className="cursor-pointer font-medium text-brand-600 hover:underline dark:text-brand-400">
            Select an area and ask a question to begin analysis.
          </span>
          <Info className="h-3.5 w-3.5 text-brand-600 dark:text-brand-400" />
        </div>
      </div>
    </div>
  );
}
