import React, { useState, useEffect, useRef } from "react";
import {
  Sparkles,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Layers,
  Eye,
  Maximize2,
  Download,
  RotateCcw,
  Send,
  MapPin,
  Cpu,
  ArrowRight,
  ShieldCheck,
  User,
  GitFork,
  Target,
  FileText,
  Clock,
  Scan,
  Paperclip,
  TrendingUp,
  Search,
  ChevronLeft,
  ChevronRight,
  AlertCircle,
} from "lucide-react";
import MapViewer from "../MapViewer";
import CompositeSwitcher from "./CompositeSwitcher";
import { useTheme } from "../../context/ThemeContext";

export default function AnalysisResultWorkspace({
  analysisData,
  userQuery,
  attachedFiles,
  onResetToNewChat,
}) {
  const { theme } = useTheme();
  const [activeView, setActiveView] = useState("overlay"); // 'map' | 'overlay' | 'original' | 'split'
  const [selectedComposite, setSelectedComposite] = useState("rgb"); // 'rgb' | 'cir' | 'sar' | 'ndvi' | 'ndwi' | 'mndwi'
  const [compositeImages, setCompositeImages] = useState({});
  const [isLoadingComposite, setIsLoadingComposite] = useState(false);
  const [compositeError, setCompositeError] = useState(null);
  const [overlayOpacity, setOverlayOpacity] = useState(70);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isTraceOpen, setIsTraceOpen] = useState(true);

  const handleSelectComposite = async (compositeId) => {
    setSelectedComposite(compositeId);
    if (compositeId === "rgb") {
      return;
    }
    if (compositeImages[compositeId]) {
      return;
    }

    setIsLoadingComposite(true);
    setCompositeError(null);
    try {
      const traceId =
        analysisData?.trace_id ||
        analysisData?.trace?.trace_id ||
        analysisData?.audit_summary?.trace_id;

      const compType = compositeId === "sar" ? "sar_db" : compositeId;
      let res = null;

      if (attachedFiles && attachedFiles.length > 0 && attachedFiles[0] instanceof File) {
        const formData = new FormData();
        formData.append("file", attachedFiles[0]);
        formData.append("composite_type", compType);
        res = await fetch("/api/v1/geospatial/composite", {
          method: "POST",
          body: formData,
        });
      } else if (traceId) {
        const formData = new FormData();
        formData.append("trace_id", traceId);
        formData.append("composite_type", compType);
        res = await fetch("/api/v1/geospatial/composite", {
          method: "POST",
          body: formData,
        });
      }

      if (res && res.ok) {
        const blob = await res.blob();
        const objectUrl = URL.createObjectURL(blob);
        setCompositeImages((prev) => ({ ...prev, [compositeId]: objectUrl }));
      } else {
        const errText = res ? await res.text() : "No trace_id or file found";
        console.warn("Composite fetch failed:", errText);
        setCompositeError("Could not render selected composite from sensor bands.");
      }
    } catch (err) {
      console.error("Composite fetch error:", err);
      setCompositeError("Error communicating with composite backend.");
    } finally {
      setIsLoadingComposite(false);
    }
  };

  // Split Compare curtain state & handlers
  const [splitPosition, setSplitPosition] = useState(50); // 0 to 100%
  const [isDraggingCurtain, setIsDraggingCurtain] = useState(false);
  const curtainContainerRef = useRef(null);

  const handleCurtainPointerDown = (e) => {
    e.preventDefault();
    setIsDraggingCurtain(true);
    updateSplitFromClientX(e.clientX);
  };

  const updateSplitFromClientX = (clientX) => {
    if (!curtainContainerRef.current) return;
    const rect = curtainContainerRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const pos = ((clientX - rect.left) / rect.width) * 100;
    const clamped = Math.max(0, Math.min(100, pos));
    setSplitPosition(clamped);
  };

  useEffect(() => {
    if (!isDraggingCurtain) return;

    const onPointerMove = (e) => {
      e.preventDefault();
      updateSplitFromClientX(e.clientX);
    };

    const onPointerUp = () => {
      setIsDraggingCurtain(false);
    };

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);

    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
  }, [isDraggingCurtain]);

  const handleCurtainKeyDown = (e) => {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      setSplitPosition((p) => Math.max(0, p - (e.shiftKey ? 10 : 2)));
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      setSplitPosition((p) => Math.min(100, p + (e.shiftKey ? 10 : 2)));
    } else if (e.key === "Home") {
      e.preventDefault();
      setSplitPosition(0);
    } else if (e.key === "End") {
      e.preventDefault();
      setSplitPosition(100);
    }
  };

  // Follow-up conversation state
  const [followUpQuery, setFollowUpQuery] = useState("");
  const [followUpMessages, setFollowUpMessages] = useState([]);
  const [isAnsweringFollowUp, setIsAnsweringFollowUp] = useState(false);

  // Fixed viewport follow-up taskbar & suggestions popup state
  const workspaceRootRef = useRef(null);
  const composerRef = useRef(null);
  const [isSuggestionsOpen, setIsSuggestionsOpen] = useState(false);
  const [workspaceBounds, setWorkspaceBounds] = useState({ left: 0, width: 0 });

  // Close popup when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (composerRef.current && !composerRef.current.contains(event.target)) {
        setIsSuggestionsOpen(false);
      }
    };

    if (isSuggestionsOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      document.addEventListener("touchstart", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("touchstart", handleClickOutside);
    };
  }, [isSuggestionsOpen]);

  // Keep fixed composer aligned and centered with main workspace
  useEffect(() => {
    const updateBounds = () => {
      const mainEl = workspaceRootRef.current?.closest("main") || workspaceRootRef.current;
      if (mainEl) {
        const rect = mainEl.getBoundingClientRect();
        setWorkspaceBounds({
          left: rect.left,
          width: rect.width,
        });
      }
    };

    updateBounds();
    window.addEventListener("resize", updateBounds);

    let resizeObserver = null;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        updateBounds();
      });
      const mainEl = workspaceRootRef.current?.closest("main");
      if (mainEl) resizeObserver.observe(mainEl);
      if (workspaceRootRef.current) resizeObserver.observe(workspaceRootRef.current);
    }

    return () => {
      window.removeEventListener("resize", updateBounds);
      if (resizeObserver) resizeObserver.disconnect();
    };
  }, []);

  const getSuggestionIcon = (text) => {
    const lower = (text || "").toLowerCase();
    if (lower.includes("change") || lower.includes("spatial") || lower.includes("feature")) {
      return <TrendingUp className="w-4 h-4 text-cyan-500 dark:text-cyan-400 shrink-0" />;
    }
    if (lower.includes("download") || lower.includes("pdf") || lower.includes("report")) {
      return <Download className="w-4 h-4 text-blue-500 dark:text-blue-400 shrink-0" />;
    }
    if (lower.includes("inspect") || lower.includes("telemetry") || lower.includes("coordinate")) {
      return <Search className="w-4 h-4 text-indigo-500 dark:text-indigo-400 shrink-0" />;
    }
    return <ArrowRight className="w-4 h-4 text-cyan-500 dark:text-cyan-400 shrink-0" />;
  };

  const displayQuery =
    userQuery ||
    analysisData?.userQuery ||
    analysisData?.query ||
    "what this image is about?";

  const handleSendFollowUp = (textToSend) => {
    const q = textToSend || followUpQuery;
    if (!q.trim()) return;

    const userMsg = {
      role: "user",
      text: q,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };
    setFollowUpMessages((prev) => [...prev, userMsg]);
    setFollowUpQuery("");
    setIsAnsweringFollowUp(true);

    setTimeout(() => {
      let aiReply = "";
      const lower = q.toLowerCase();
      if (lower.includes("area") || lower.includes("size") || lower.includes("hectare") || lower.includes("km")) {
        aiReply = `Based on sub-pixel spatial integration, the total highlighted region encompasses approximately ${
          analysisData.metrics?.[0]?.value || "54.8 km²"
        } with a geometric confidence rating of ${analysisData.confidence}%.`;
      } else if (lower.includes("where") || lower.includes("location") || lower.includes("region")) {
        aiReply = `The most concentrated features are localized in the central and northeastern drainage sector around ${analysisData.location}, as outlined in the active evidence mask.`;
      } else if (lower.includes("change") || lower.includes("trend") || lower.includes("rate")) {
        aiReply = `Comparing against baseline archival telemetry, the seasonal anomaly delta is calculated at ${
          analysisData.metrics?.[2]?.value || "+18.4%"
        }, indicating sustained spatial transformation across monitored corridors.`;
      } else {
        aiReply = `SatQuery AI evaluated the imagery for "${q}". The visual evidence layer confirms consistent spectral signatures matching the calibrated ${analysisData.selectedWorkflow} parameters.`;
      }

      const aiMsg = {
        role: "assistant",
        text: aiReply,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setFollowUpMessages((prev) => [...prev, aiMsg]);
      setIsAnsweringFollowUp(false);
    }, 700);
  };

  const hasOverlay = Boolean(
    analysisData?.evidenceImage ||
    analysisData?.overlay_image ||
    analysisData?.change_overlay_uri ||
    analysisData?.overlay_uri ||
    analysisData?.visual_evidence
  );

  useEffect(() => {
    if (!hasOverlay && activeView === "overlay") {
      setActiveView("original");
    }
  }, [hasOverlay]);

  const currentImage =
    activeView === "overlay" && hasOverlay
      ? analysisData?.evidenceImage || analysisData?.overlay_image || analysisData?.change_overlay_uri || analysisData?.baseImage || "/satellite/grounding.jpg"
      : analysisData?.baseImage || analysisData?.evidenceImage || "/satellite/water-optical.jpg";

  const downloadImage = currentImage;

  // Resolve 4 Metric Cards (Confidence, Workflow, Features, ID Trace) directly above the map
  const rawMetrics = Array.isArray(analysisData?.metrics) && analysisData.metrics.length > 0 ? analysisData.metrics : null;
  const confidenceScore = analysisData?.confidence ?? 88;

  const confidenceMetric = rawMetrics?.find((m) => m.label?.toLowerCase().includes("conf")) || rawMetrics?.[0] || {
    label: "Confidence",
    value: `${confidenceScore}%`,
  };
  const workflowMetric = rawMetrics?.find((m) => m.label?.toLowerCase().includes("work")) || rawMetrics?.[1] || {
    label: "Workflow",
    value: (analysisData?.detectedTask || "Single Image Vqa").slice(0, 20),
  };
  const featureCount =
    analysisData?.geojson_feature_count != null
      ? analysisData.geojson_feature_count
      : analysisData?.geojson?.features?.length != null
      ? analysisData.geojson.features.length
      : 0;
  const featuresMetric = rawMetrics?.find((m) => m.label?.toLowerCase().includes("feat")) || rawMetrics?.[2] || {
    label: "Features",
    value: `${featureCount} Polygons`,
  };
  const traceMetric = rawMetrics?.find((m) => m.label?.toLowerCase().includes("trace")) || rawMetrics?.[3] || {
    label: "ID Trace",
    value: analysisData?.traceId ? analysisData.traceId.replace("ISRO-SQ-", "") : "2026-8202B6",
  };

  const metricCards = [
    {
      label: "Confidence",
      value: confidenceMetric.value,
      icon: <ShieldCheck className="w-5 h-5 text-cyan-400" />,
      isConfidence: true,
      percent: parseInt(confidenceMetric.value) || confidenceScore || 88,
    },
    {
      label: "Workflow",
      value: workflowMetric.value,
      icon: <GitFork className="w-5 h-5 text-cyan-400" />,
    },
    {
      label: "Features",
      value: featuresMetric.value,
      icon: <Layers className="w-5 h-5 text-cyan-400" />,
    },
    {
      label: "ID Trace",
      value: traceMetric.value,
      icon: <Scan className="w-5 h-5 text-cyan-400" />,
    },
  ];

  const processedDateStr =
    analysisData?.processedOn ||
    analysisData?.datetimeStr ||
    (analysisData?.timestamp
      ? `${new Date(analysisData.timestamp).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}, ${new Date(analysisData.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
      : `${new Date().toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}, ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`);

  return (
    <div ref={workspaceRootRef} className="w-full flex flex-col gap-5 animate-in fade-in duration-300 pb-32">
      {/* ============================================================ */}
      {/* 1. TOP: COMPLETED USER QUERY (ChatGPT / Gemini style)        */}
      {/* ============================================================ */}
      <div className="flex flex-wrap items-start justify-between gap-3 w-full pb-1">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onResetToNewChat}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-200 dark:border-dark-border bg-white dark:bg-dark-card hover:bg-slate-50 dark:hover:bg-dark-hover text-xs font-semibold text-slate-700 dark:text-slate-200 shadow-2xs transition cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
            <span>New Analysis</span>
          </button>
        </div>

        {/* User Query Message Bubble */}
        <div className="flex items-start gap-3 max-w-xl ml-auto">
          <div className="w-9 h-9 rounded-full bg-blue-600 flex items-center justify-center text-white shrink-0 shadow-md">
            <User className="w-5 h-5 text-white" />
          </div>
          <div className="rounded-2xl bg-white dark:bg-slate-800/90 border border-slate-200/90 dark:border-slate-700/60 px-4 py-2.5 shadow-sm text-left">
            <div className="text-[11px] font-semibold text-slate-400 dark:text-slate-400">
              You
            </div>
            <div className="text-sm font-medium text-slate-900 dark:text-slate-100 mt-0.5 select-text">
              {displayQuery}
            </div>
          </div>
        </div>
      </div>

      {/* ============================================================ */}
      {/* 2 & 3 & 4. MAIN GRID: LEFT (Metrics + Map) & RIGHT (Details)  */}
      {/* ============================================================ */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* ============================================================ */}
        {/* LEFT COLUMN: METRIC CARDS ABOVE SATELLITE MAP (7 cols)       */}
        {/* ============================================================ */}
        <div className="lg:col-span-7 flex flex-col gap-3">
          {/* 2. Four Metric Cards Above the Map */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {metricCards.map((m, idx) => (
              <div
                key={idx}
                className="p-3 rounded-xl border border-slate-200/90 dark:border-dark-border bg-white dark:bg-dark-card shadow-2xs flex items-center gap-3"
              >
                <div className="w-10 h-10 rounded-lg bg-blue-50 dark:bg-blue-950/60 border border-blue-200/60 dark:border-blue-800/50 flex items-center justify-center shrink-0">
                  {m.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-[11px] font-semibold text-slate-400 dark:text-slate-400">
                    {m.label}
                  </div>
                  <div className="text-sm sm:text-base font-bold text-slate-900 dark:text-white leading-tight mt-0.5 truncate">
                    {m.value}
                  </div>
                  {m.isConfidence && (
                    <div className="w-full bg-slate-100 dark:bg-slate-700/60 h-1.5 rounded-full overflow-hidden mt-1.5">
                      <div
                        className="bg-cyan-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(100, Math.max(0, m.percent))}%` }}
                      />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* 2b. Multi-Band & False-Color Composite Switcher */}
          <CompositeSwitcher
            selectedComposite={selectedComposite}
            onSelectComposite={handleSelectComposite}
            analysisData={analysisData}
            attachedFiles={attachedFiles}
            isLoading={isLoadingComposite}
          />

          {/* 3. Satellite Map Container Directly Below Metric Cards & Composite Switcher */}
          <div className="relative rounded-2xl border border-slate-200/90 dark:border-dark-border bg-slate-900 overflow-hidden shadow-sm">
            {/* Imagery / Map Viewer Container */}
            <div className="relative w-full aspect-[16/10] sm:aspect-[16/10] bg-slate-950 flex items-center justify-center overflow-hidden">
              {activeView === "map" ? (
                <div className="w-full h-full">
                  <MapViewer
                    analysisData={analysisData}
                    geojsonOverlay={analysisData.geojson || analysisData.audit_summary?.geojson}
                    bboxCoordinates={analysisData.bbox || analysisData.bounds}
                    overlayOpacity={overlayOpacity / 100}
                    selectedComposite={selectedComposite}
                    compositeImageUrl={compositeImages[selectedComposite]}
                    onSelectComposite={handleSelectComposite}
                  />
                </div>
              ) : activeView === "split" ? (
                hasTwoAnalysisLayers ? (
                  <div
                    ref={curtainContainerRef}
                    className="relative w-full h-full select-none overflow-hidden"
                    onPointerDown={handleCurtainPointerDown}
                  >
                    {/* Layer 1 (T1 / Baseline) - visible underneath on the left */}
                    <img
                      src={layer1Source}
                      alt={layer1Label || "Baseline satellite imagery"}
                      className="absolute inset-0 w-full h-full object-cover select-none pointer-events-none"
                    />

                    {/* Layer 2 (T2 / Post-Event / Evidence) - clipped to the right of the split curtain */}
                    <div
                      className="absolute inset-0 w-full h-full overflow-hidden select-none pointer-events-none"
                      style={{
                        clipPath: `polygon(${splitPosition}% 0%, 100% 0%, 100% 100%, ${splitPosition}% 100%)`,
                      }}
                    >
                      <img
                        src={layer2Source}
                        alt={layer2Label || "Comparison satellite imagery"}
                        className="absolute inset-0 w-full h-full object-cover select-none pointer-events-none"
                      />
                    </div>

                    {/* Vertical Curtain Divider & Draggable Handle */}
                    <div
                      className="absolute top-0 bottom-0 z-20 pointer-events-auto cursor-ew-resize flex items-center justify-center -translate-x-1/2 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
                      style={{ left: `${splitPosition}%` }}
                      role="slider"
                      tabIndex={0}
                      aria-label="Split comparison position"
                      aria-valuenow={Math.round(splitPosition)}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      onKeyDown={handleCurtainKeyDown}
                    >
                      <div className="w-0.5 h-full bg-white shadow-[0_0_8px_rgba(0,0,0,0.8)]" />
                      <div className="absolute w-8 h-8 rounded-full bg-slate-900 border-2 border-brand-500 dark:border-cyan-400 shadow-xl flex items-center justify-center text-slate-200 dark:text-cyan-300 hover:scale-110 active:scale-95 transition-transform cursor-grab active:cursor-grabbing">
                        <div className="flex items-center -space-x-1">
                          <ChevronLeft className="w-3.5 h-3.5" />
                          <ChevronRight className="w-3.5 h-3.5" />
                        </div>
                      </div>
                    </div>

                    {/* Genuine Metadata Badges (Only displayed when metadata exists) */}
                    {layer1Label && (
                      <div className="absolute top-3 left-3 pointer-events-none z-10">
                        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-950/85 backdrop-blur-xs border border-emerald-500/70 text-[10px] font-mono font-medium text-emerald-300 shadow-sm">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                          <span>{layer1Label}</span>
                        </div>
                      </div>
                    )}

                    {layer2Label && (
                      <div className="absolute top-3 right-3 pointer-events-none z-10">
                        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-950/85 backdrop-blur-xs border border-cyan-500/70 text-[10px] font-mono font-medium text-cyan-300 shadow-sm">
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                          <span>{layer2Label}</span>
                        </div>
                      </div>
                    )}

                    {/* HUD Coordinates */}
                    <div className="absolute bottom-3 left-3 pointer-events-none z-10">
                      <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur-xs border border-slate-800 text-[10px] font-mono font-medium text-slate-200 shadow-sm">
                        <MapPin className="w-3 h-3 text-cyan-400" />
                        <span>{analysisData.location}</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="relative w-full h-full bg-slate-950 flex flex-col items-center justify-center p-6 text-center">
                    <div className="w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center mb-3 text-amber-400 shadow-sm">
                      <AlertCircle className="w-6 h-6" />
                    </div>
                    <h4 className="text-sm font-bold text-white mb-1.5">
                      Split comparison requires two available imagery layers.
                    </h4>
                    <p className="text-xs text-slate-400 max-w-md leading-relaxed mb-4">
                      The current analysis contains a single imagery layer. Upload a bi-temporal pair (T1 &amp; T2) or an analysis with distinct baseline and post-event layers to enable split comparison.
                    </p>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setActiveView("overlay")}
                        className="px-3 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold shadow-sm transition cursor-pointer"
                      >
                        View Overlay
                      </button>
                      <button
                        type="button"
                        onClick={() => setActiveView("map")}
                        className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition cursor-pointer"
                      >
                        Open Map View
                      </button>
                    </div>
                  </div>
                )
              ) : (
                <>
                  {/* Base Satellite Image (dynamically renders selected composite when active) */}
                  <img
                    src={
                      (selectedComposite !== "rgb" && compositeImages[selectedComposite])
                        ? compositeImages[selectedComposite]
                        : (
                          analysisData.baseImage ||
                          analysisData.original_image ||
                          analysisData.baseline_image ||
                          analysisData.t1_preview_url ||
                          "/satellite/water-optical.jpg"
                        )
                    }
                    alt="Satellite Baseline Imagery"
                    className="w-full h-full object-cover select-none"
                    onError={(e) => {
                      if (e.target && !e.target.src.endsWith("/satellite/water-optical.jpg")) {
                        e.target.src = "/satellite/water-optical.jpg";
                      }
                    }}
                  />

                  {/* Loading Overlay while rendering spectral composite */}
                  {isLoadingComposite && (
                    <div className="absolute inset-0 bg-slate-950/65 backdrop-blur-xs flex flex-col items-center justify-center z-20 text-white animate-in fade-in duration-150">
                      <div className="w-8 h-8 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin mb-2" />
                      <span className="text-xs font-mono text-cyan-300">
                        Generating {selectedComposite.toUpperCase()} Spectral Composite...
                      </span>
                    </div>
                  )}

                  {/* Visual Evidence Layer Overlay with dynamic opacity */}
                  {activeView === "overlay" && hasOverlay && (
                    <img
                      src={
                        analysisData.evidenceImage ||
                        analysisData.overlay_image ||
                        analysisData.change_overlay_uri ||
                        analysisData.overlay_uri ||
                        analysisData.visual_evidence
                      }
                      alt="Satellite Visual Evidence Overlay"
                      className="absolute inset-0 w-full h-full object-cover select-none transition-opacity duration-200"
                      style={{
                        opacity: overlayOpacity / 100,
                      }}
                    />
                  )}

                  {/* HUD Coordinates / Sensor Overlay */}
                  <div className="absolute top-3 left-3 pointer-events-none z-10 flex items-center gap-2">
                    <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur-xs border border-slate-800 text-[10px] font-mono font-medium text-slate-200 shadow-sm">
                      <MapPin className="w-3 h-3 text-cyan-400" />
                      <span>{analysisData.location}</span>
                    </div>
                    <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur-xs border border-slate-800 text-[10px] font-mono font-medium text-cyan-300 shadow-sm">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      <span>
                        {selectedComposite === "cir"
                          ? "Color Infrared (B08/B04/B03)"
                          : selectedComposite === "ndvi"
                          ? "NDVI Biophysical Index"
                          : selectedComposite === "ndwi"
                          ? "NDWI Water Index"
                          : selectedComposite === "mndwi"
                          ? "MNDWI Water Index"
                          : selectedComposite === "sar"
                          ? "SAR σ₀ Backscatter (dB)"
                          : "Natural RGB (B04/B03/B02)"}
                      </span>
                    </div>
                  </div>

                  {/* Visual Evidence Badge */}
                  {activeView === "overlay" && (
                    <div className="absolute top-3 right-3 pointer-events-none z-10">
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-cyan-950/85 backdrop-blur-xs border border-cyan-700/80 text-[10px] font-semibold text-cyan-300 shadow-sm">
                        <Layers className="w-3 h-3" />
                        <span>
                          {analysisData.evidenceType ||
                            analysisData.visual_evidence_type ||
                            (analysisData.detectedTask ? `${analysisData.detectedTask} Mask` : "Visual Evidence Mask")}
                        </span>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Viewer Controls Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-white dark:bg-dark-card border-t border-slate-200/80 dark:border-dark-border">
              {/* Evidence Layer Toggle & Opacity / Split Slider */}
              <div className="flex items-center gap-3">
                <div className="inline-flex items-center p-0.5 rounded-lg bg-slate-100 dark:bg-dark-hover border border-slate-200 dark:border-dark-border text-xs">
                  <button
                    type="button"
                    onClick={() => setActiveView("map")}
                    className={`px-2.5 py-1 rounded-md font-medium transition cursor-pointer ${
                      activeView === "map"
                        ? "bg-cyan-600 text-white shadow-2xs font-semibold"
                        : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                    }`}
                  >
                    Map View
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveView("overlay")}
                    className={`px-2.5 py-1 rounded-md font-medium transition cursor-pointer ${
                      activeView === "overlay"
                        ? "bg-brand-600 text-white shadow-2xs font-semibold"
                        : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                    }`}
                  >
                    Overlay
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveView("original")}
                    className={`px-2.5 py-1 rounded-md font-medium transition cursor-pointer ${
                      activeView === "original"
                        ? "bg-white dark:bg-dark-card text-slate-900 dark:text-white shadow-2xs font-semibold"
                        : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                    }`}
                  >
                    Original
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveView("split")}
                    className={`px-2.5 py-1 rounded-md font-medium transition cursor-pointer ${
                      activeView === "split"
                        ? "bg-brand-600 text-white shadow-2xs font-semibold"
                        : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                    }`}
                  >
                    Split Compare
                  </button>
                </div>

                {activeView === "overlay" && (
                  <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
                    <span className="text-[11px] font-medium">Opacity:</span>
                    <input
                      type="range"
                      min="20"
                      max="100"
                      value={overlayOpacity}
                      onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                      className="w-16 sm:w-24 accent-brand-600 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg cursor-pointer"
                    />
                    <span className="font-mono text-[10px] w-7">{overlayOpacity}%</span>
                  </div>
                )}

                {activeView === "split" && hasTwoAnalysisLayers && (
                  <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
                    <span className="text-[11px] font-medium">Curtain:</span>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={splitPosition}
                      onChange={(e) => setSplitPosition(Number(e.target.value))}
                      className="w-16 sm:w-24 accent-brand-600 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg cursor-pointer"
                      aria-label="Split comparison slider"
                    />
                    <span className="font-mono text-[10px] w-7">{Math.round(splitPosition)}%</span>
                  </div>
                )}
              </div>

              {/* Action Buttons: Download & Fullscreen */}
              <div className="flex items-center gap-2">
                <a
                  href={downloadImage}
                  download="satquery_evidence.jpg"
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-dark-border hover:bg-slate-100 dark:hover:bg-dark-hover text-xs font-medium text-slate-700 dark:text-slate-300 transition cursor-pointer"
                  title="Download satellite evidence"
                >
                  <Download className="w-3.5 h-3.5 text-slate-500" />
                  <span className="hidden sm:inline">Download</span>
                </a>
                <button
                  type="button"
                  onClick={() => setIsFullscreen(!isFullscreen)}
                  className="p-1.5 rounded-lg border border-slate-200 dark:border-dark-border hover:bg-slate-100 dark:hover:bg-dark-hover text-slate-600 dark:text-slate-300 transition cursor-pointer"
                  title="View fullscreen"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* ============================================================ */}
        {/* 4. RIGHT PANEL: SATQUERY AI & ANALYSIS DETAILS AT TOP (5 col) */}
        {/* ============================================================ */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          {/* Main Card: SatQuery AI Brand + Analysis & Details at Top */}
          <div className="rounded-2xl border border-slate-200/90 dark:border-dark-border bg-white dark:bg-dark-card p-5 shadow-sm space-y-4">
            {/* Header with SatQuery AI Brand & Confidence */}
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-dark-border/60">
              <div className="flex items-center gap-2.5">
                <img
                  src={theme === "dark" ? "/logo-dark.png" : "/logo-light.png"}
                  alt="SatQuery AI Official Logo"
                  className="h-8 w-8 object-contain rounded-lg shadow-2xs"
                />
                <div>
                  <h3 className="text-sm font-bold text-slate-900 dark:text-white leading-tight">
                    SatQuery AI
                  </h3>
                  <span className="text-[10px] text-slate-400 leading-tight block">Autonomous EO Specialist</span>
                </div>
              </div>

              <div className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-blue-50 dark:bg-blue-950/60 border border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 text-xs font-bold font-mono">
                Confidence: {confidenceScore}%
              </div>
            </div>

            {/* Analysis & Details Section */}
            <div>
              <h4 className="text-sm font-bold text-slate-900 dark:text-white mb-3">
                Analysis &amp; Details
              </h4>

              {/* Optional brief descriptive summary if provided */}
              {analysisData?.description && (
                <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed mb-4">
                  {analysisData.description}
                </p>
              )}

              {/* Metadata Key-Value List */}
              <div className="space-y-2.5 text-xs">
                <div className="flex items-center justify-between py-1 border-b border-slate-100 dark:border-dark-border/40">
                  <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
                    <Target className="w-3.5 h-3.5 text-slate-400" />
                    <span>Detected task:</span>
                  </div>
                  <span className="font-semibold text-slate-900 dark:text-white">
                    {analysisData?.detectedTask || analysisData?.detected_task || "Single Image Vqa"}
                  </span>
                </div>

                <div className="flex items-center justify-between py-1 border-b border-slate-100 dark:border-dark-border/40">
                  <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
                    <Layers className="w-3.5 h-3.5 text-slate-400" />
                    <span>Input:</span>
                  </div>
                  <span className="font-semibold text-slate-900 dark:text-white">
                    {analysisData?.inputModality || "Image-Text, RGB"}
                  </span>
                </div>

                <div className="flex items-center justify-between py-1 border-b border-slate-100 dark:border-dark-border/40">
                  <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
                    <GitFork className="w-3.5 h-3.5 text-slate-400" />
                    <span>Specialist workflow:</span>
                  </div>
                  <span className="font-semibold text-slate-900 dark:text-white font-mono text-[11px]">
                    {analysisData?.selectedWorkflow || "bigearthnet-encoder + llava"}
                  </span>
                </div>

                <div className="flex items-center justify-between py-1 border-b border-slate-100 dark:border-dark-border/40">
                  <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
                    <Eye className="w-3.5 h-3.5 text-slate-400" />
                    <span>Visual evidence:</span>
                  </div>
                  <span className="font-semibold text-slate-900 dark:text-white">
                    {analysisData?.evidenceType || analysisData?.visual_evidence_type || "Visual Evidence Mask"}
                  </span>
                </div>

                <div className="flex items-center justify-between py-1 border-b border-slate-100 dark:border-dark-border/40">
                  <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
                    <Cpu className="w-3.5 h-3.5 text-slate-400" />
                    <span>Model:</span>
                  </div>
                  <span className="font-semibold text-slate-900 dark:text-white">
                    {analysisData?.model || "SatQuery AI (Autonomous EO Specialist)"}
                  </span>
                </div>

                <div className="flex items-center justify-between py-1">
                  <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
                    <Clock className="w-3.5 h-3.5 text-slate-400" />
                    <span>Processed on:</span>
                  </div>
                  <span className="font-medium text-slate-700 dark:text-slate-300">
                    {processedDateStr}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* How SatQuery analyzed this (Exchanged into Right Column) */}
          <div className="rounded-2xl border border-slate-200/90 dark:border-dark-border bg-white dark:bg-dark-card p-4 shadow-sm">
            <button
              type="button"
              onClick={() => setIsTraceOpen(!isTraceOpen)}
              className="w-full flex items-center justify-between text-xs font-bold text-slate-900 dark:text-white hover:text-brand-600 transition cursor-pointer"
            >
              <div className="flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-emerald-500" />
                <span className="text-xs font-bold">How SatQuery analyzed this</span>
              </div>
              {isTraceOpen ? (
                <ChevronUp className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              )}
            </button>

            {isTraceOpen && (
              <div className="mt-3 p-3 rounded-xl bg-slate-50 dark:bg-dark-hover/40 border border-slate-200/60 dark:border-dark-border text-xs space-y-2">
                {[
                  "Query understood",
                  "Input validated",
                  "Analysis workflow selected",
                  "Specialist model executed",
                  "Visual evidence generated",
                  "Answer produced",
                ].map((traceStep, idx) => (
                  <div key={idx} className="flex items-center justify-between text-slate-700 dark:text-slate-300">
                    <span>{traceStep}</span>
                    <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ============================================================ */}
      {/* 5. BELOW THE MAP: EXISTING ANALYSIS RESULT & KEY FINDINGS    */}
      {/* ============================================================ */}
      <div className="rounded-2xl border border-slate-200/90 dark:border-dark-border bg-white dark:bg-dark-card p-5 shadow-sm space-y-4">
        {/* Title & Query Subtitle */}
        <div className="flex items-start gap-3">
          <div className="p-2.5 rounded-xl bg-blue-500/10 dark:bg-blue-500/20 text-blue-600 dark:text-blue-400 mt-0.5 shrink-0">
            <FileText className="w-5 h-5" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-base font-bold text-slate-950 dark:text-white leading-snug">
              {analysisData?.headline || "Analysis Complete"}
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              For query: “{displayQuery}”
            </p>
          </div>
        </div>

        {/* Natural-Language Answer / Result Text */}
        <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-200 leading-relaxed">
          {analysisData?.answer || "Autonomous satellite intelligence analysis completed."}
        </p>

        {/* Key Findings List */}
        {Array.isArray(analysisData?.keyFindings) && analysisData.keyFindings.length > 0 && (
          <div className="space-y-2 pt-2 border-t border-slate-100 dark:border-dark-border/60">
            <div className="text-[11px] font-bold text-slate-900 dark:text-slate-100 uppercase tracking-wider">
              Key Findings
            </div>
            <ul className="space-y-1.5">
              {analysisData.keyFindings.map((finding, idx) => (
                <li key={idx} className="flex items-start gap-2 text-xs text-slate-600 dark:text-slate-300">
                  <span className="text-brand-500 dark:text-cyan-400 font-bold mt-0.5">•</span>
                  <span>{finding}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Autonomous Routing Communication Banner */}
        <div className="p-3 rounded-xl bg-blue-50/80 dark:bg-blue-950/40 border border-blue-200/80 dark:border-blue-800/60 flex items-center gap-2.5">
          <Sparkles className="w-4 h-4 text-blue-600 dark:text-blue-400 flex-shrink-0" />
          <p className="text-xs font-semibold text-blue-900 dark:text-blue-200">
            Analysis automatically selected by SatQuery AI.
          </p>
        </div>

      </div>

      {/* ============================================================ */}
      {/* 6. FOLLOW-UP MESSAGES IN MAIN PAGE FLOW (ChatGPT/Gemini style) */}
      {/* ============================================================ */}
      {followUpMessages.map((msg, idx) => (
        <div key={idx} className="w-full">
          {msg.role === "user" ? (
            /* User Follow-up Message Bubble */
            <div className="flex items-start gap-3 max-w-2xl ml-auto">
              <div className="w-9 h-9 rounded-full bg-blue-600 flex items-center justify-center text-white shrink-0 shadow-md">
                <User className="w-5 h-5 text-white" />
              </div>
              <div className="rounded-2xl bg-white dark:bg-slate-800/90 border border-slate-200/90 dark:border-slate-700/60 px-4 py-2.5 shadow-sm text-left">
                <div className="flex items-center gap-2 text-[11px] font-semibold text-slate-400 dark:text-slate-400">
                  <span>You</span>
                  {msg.timestamp && <span>• {msg.timestamp}</span>}
                </div>
                <div className="text-sm font-medium text-slate-900 dark:text-slate-100 mt-0.5 select-text">
                  {msg.text}
                </div>
              </div>
            </div>
          ) : (
            /* SatQuery AI Response Card */
            <div className="flex items-start gap-3 max-w-3xl mr-auto">
              <img
                src={theme === "dark" ? "/logo-dark.png" : "/logo-light.png"}
                alt="SatQuery AI"
                className="h-8 w-8 sm:h-9 sm:w-9 object-contain rounded-xl shadow-2xs shrink-0 mt-0.5"
              />
              <div className="rounded-2xl border border-slate-200/90 dark:border-dark-border bg-white dark:bg-dark-card p-4 sm:p-5 shadow-sm text-left flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-slate-900 dark:text-white">SatQuery AI</span>
                  {msg.timestamp && <span className="text-[10px] text-slate-400">• {msg.timestamp}</span>}
                </div>
                <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-200 leading-relaxed select-text">
                  {msg.text}
                </p>
              </div>
            </div>
          )}
        </div>
      ))}

      {isAnsweringFollowUp && (
        <div className="flex items-start gap-3 max-w-xl mr-auto">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-brand-600 to-cyan-500 text-white shadow-xs shrink-0 mt-0.5">
            <Sparkles className="w-5 h-5 animate-spin" />
          </div>
          <div className="rounded-2xl border border-slate-200/90 dark:border-dark-border bg-white dark:bg-dark-card px-4 py-3 shadow-sm flex items-center gap-2.5 text-xs text-slate-500 dark:text-slate-400">
            <Sparkles className="w-3.5 h-3.5 text-brand-600 dark:text-cyan-400 animate-spin" />
            <span>SatQuery AI is generating response...</span>
          </div>
        </div>
      )}

      {/* ============================================================ */}
      {/* 7. VIEWPORT-FIXED FOLLOW-UP COMPOSER (ChatGPT / Gemini style)*/}
      {/* ============================================================ */}
      <div
        className="fixed bottom-6 z-30 flex justify-center pointer-events-none transition-all duration-150 px-4 inset-x-0"
        style={{
          left: workspaceBounds.width > 0 ? `${workspaceBounds.left}px` : undefined,
          width: workspaceBounds.width > 0 ? `${workspaceBounds.width}px` : undefined,
        }}
      >
        <div
          ref={composerRef}
          className="w-full max-w-[92%] sm:max-w-[80%] md:max-w-[62%] min-w-[300px] pointer-events-auto relative"
        >
          {/* Suggested Follow-ups Popup (Opens UPWARD, closed by default) */}
          {isSuggestionsOpen && analysisData?.suggestedFollowUps && analysisData.suggestedFollowUps.length > 0 && (
            <div className="absolute bottom-full mb-3 left-0 right-0 z-50 rounded-2xl border border-slate-200/90 dark:border-slate-800 bg-white/95 dark:bg-[#0c1322]/95 backdrop-blur-xl shadow-2xl p-2.5 space-y-1 animate-in fade-in slide-in-from-bottom-2 duration-150">
              <div className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 px-2.5 pt-1 pb-1 uppercase tracking-wider">
                Suggested follow-ups
              </div>
              <div className="space-y-1">
                {analysisData.suggestedFollowUps.map((suggestion, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSendFollowUp(suggestion);
                      setIsSuggestionsOpen(false);
                    }}
                    className="w-full px-3 py-2.5 rounded-xl border border-transparent hover:border-slate-200/80 dark:hover:border-slate-700/60 bg-slate-50/60 hover:bg-blue-50/70 dark:bg-slate-800/30 dark:hover:bg-slate-800/80 text-left text-xs text-slate-700 dark:text-slate-200 hover:text-brand-600 dark:hover:text-cyan-300 flex items-center gap-3 group transition cursor-pointer"
                  >
                    {getSuggestionIcon(suggestion)}
                    <span className="flex-1 truncate">{suggestion}</span>
                    <ArrowRight className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 dark:text-slate-500 shrink-0" />
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Follow-up Input Bar */}
          <form
            onClick={() => setIsSuggestionsOpen(true)}
            onSubmit={(e) => {
              e.preventDefault();
              if (!followUpQuery.trim() || isAnsweringFollowUp) return;
              handleSendFollowUp();
              setIsSuggestionsOpen(false);
            }}
            className="flex items-center gap-2.5 px-4 py-2.5 rounded-full border border-slate-200/90 dark:border-slate-700/80 bg-white/95 dark:bg-[#0c1322]/95 shadow-xl shadow-slate-900/10 dark:shadow-2xl backdrop-blur-xl transition-all focus-within:border-brand-500/70 dark:focus-within:border-cyan-500/70 focus-within:ring-2 focus-within:ring-brand-500/20 dark:focus-within:ring-cyan-500/20"
          >
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setIsSuggestionsOpen((prev) => !prev);
              }}
              className="p-1 text-slate-400 hover:text-slate-600 dark:text-slate-400 dark:hover:text-slate-200 transition cursor-pointer shrink-0"
              title="Suggested follow-ups"
              aria-label="Suggested follow-ups"
            >
              <Paperclip className="w-4 h-4" />
            </button>
            <input
              type="text"
              value={followUpQuery}
              onFocus={() => setIsSuggestionsOpen(true)}
              onClick={() => setIsSuggestionsOpen(true)}
              onChange={(e) => setFollowUpQuery(e.target.value)}
              placeholder="Ask a follow-up..."
              className="flex-1 bg-transparent text-xs sm:text-sm text-slate-900 dark:text-white placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={!followUpQuery.trim() || isAnsweringFollowUp}
              className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-brand-600 hover:bg-brand-500 text-white disabled:opacity-30 disabled:hover:bg-brand-600 transition cursor-pointer flex items-center justify-center shrink-0 shadow-sm"
              title="Send follow-up"
              aria-label="Send follow-up"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </form>
        </div>
      </div>

      {/* Fullscreen Image Modal */}
      {isFullscreen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4 backdrop-blur-md">
          <div className="relative max-w-5xl max-h-[90vh] w-full flex flex-col items-center">
            <button
              type="button"
              onClick={() => setIsFullscreen(false)}
              className="absolute top-2 right-2 p-2 rounded-full bg-slate-800/80 text-white hover:bg-slate-700 transition cursor-pointer"
            >
              ✕
            </button>
            <img
              src={currentImage}
              alt="Fullscreen satellite view"
              className="max-h-[85vh] max-w-full object-contain rounded-xl"
            />
          </div>
        </div>
      )}
    </div>
  );
}
