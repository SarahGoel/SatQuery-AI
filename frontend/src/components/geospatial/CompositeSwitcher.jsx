import React, { useState } from "react";
import {
  Layers,
  Info,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Radio,
  Sliders,
  Sparkles,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

/**
 * Remote-sensing composite definition catalog.
 * Follows strict Earth Observation conventions.
 * No fake data, no CSS filters simulating spectral calculations.
 */
export const COMPOSITE_MODES = [
  {
    id: "rgb",
    name: "Natural Color",
    shortLabel: "Natural RGB",
    badgeLabel: "RGB (True Color)",
    displayChannels: "Red → B04 (665 nm) · Green → B03 (560 nm) · Blue → B02 (490 nm)",
    bandsRequired: ["Red (B04)", "Green (B03)", "Blue (B02)"],
    description: "Standard true-color photographic composite mirroring human visual perception. Ideal for surface context, urban extents, and visible water bodies.",
    formula: "Red → Red channel, Green → Green channel, Blue → Blue channel",
    sourceText: "Optical (Sentinel-2 / High-Res Equivalent)",
    isSupported: true,
  },
  {
    id: "cir",
    name: "Color Infrared (CIR)",
    shortLabel: "CIR",
    badgeLabel: "CIR (False Color)",
    displayChannels: "Red → NIR (B08, 842 nm) · Green → Red (B04) · Blue → Green (B03)",
    bandsRequired: ["NIR (B08)", "Red (B04)", "Green (B03)"],
    description: "Classic false-color composite widely used for vegetation monitoring. Chlorophyll-rich healthy foliage appears in vivid red/magenta; turbid water appears cyan/blue.",
    formula: "NIR → Red display channel, Red → Green display channel, Green → Blue display channel",
    sourceText: "NIR + Red + Green bands required",
    missingReason: "Requires discrete Near-Infrared (B08), Red (B04), and Green (B03) spectral bands. The current frontend receives pre-rendered 3-channel RGB imagery without accessible NIR telemetry.",
    isSupported: false,
  },
  {
    id: "sar",
    name: "SAR Decibel Backscatter (σ₀ dB)",
    shortLabel: "SAR σ₀",
    badgeLabel: "SAR σ₀ (Radar dB)",
    displayChannels: "Calibrated Radar Intensity: Sentinel-1 C-SAR VV / VH (dB)",
    bandsRequired: ["VV Backscatter", "VH Backscatter", "Calibration Matrix"],
    description: "Microwave radar backscatter in decibels (dB). Independent of solar illumination and cloud cover. Highly sensitive to dielectric properties, surface roughness, and metallic dihedral reflectors.",
    formula: "σ₀ (dB) = 10 · log₁₀(DN² / A²) - CalibrationOffset",
    sourceText: "Sentinel-1 VV/VH σ₀ required",
    missingReason: "Requires calibrated Sentinel-1 C-SAR VV/VH microwave backscatter matrix (dB). Optical imagery with CSS grayscale desaturation is not genuine radar data.",
    isSupported: false,
  },
  {
    id: "ndvi",
    name: "NDVI (Vegetation Index)",
    shortLabel: "NDVI",
    badgeLabel: "NDVI Index",
    displayChannels: "Normalized Biophysical Scale: [-1.0 to +1.0]",
    bandsRequired: ["NIR (B08)", "Red (B04)"],
    description: "Normalized Difference Vegetation Index measuring photosynthetic activity and canopy density based on chlorophyll absorption in Red and mesophyll scattering in NIR.",
    formula: "NDVI = (NIR - Red) / (NIR + Red)",
    sourceText: "NIR + Red bands required",
    missingReason: "Requires discrete Near-Infrared (B08) and Red (B04) floating-point spectral rasters. Client-side spectral index calculation is disabled because individual bands are not exposed.",
    isSupported: false,
  },
  {
    id: "ndwi",
    name: "NDWI (Water Index)",
    shortLabel: "NDWI",
    badgeLabel: "NDWI Index",
    displayChannels: "Normalized Biophysical Scale: [-1.0 to +1.0]",
    bandsRequired: ["Green (B03)", "NIR (B08)"],
    description: "Normalized Difference Water Index [McFeeters 1996] delineating open water bodies and surface inundation by maximizing green reflectance and minimizing NIR reflectance.",
    formula: "NDWI = (Green - NIR) / (Green + NIR)",
    sourceText: "Green + NIR bands required",
    missingReason: "Requires discrete Green (B03) and NIR (B08) spectral rasters. Discrete bands are not provided by the current API response.",
    isSupported: false,
  },
  {
    id: "mndwi",
    name: "MNDWI (Modified Water Index)",
    shortLabel: "MNDWI",
    badgeLabel: "MNDWI Index",
    displayChannels: "Normalized Biophysical Scale: [-1.0 to +1.0]",
    bandsRequired: ["Green (B03)", "SWIR-1 (B11)"],
    description: "Modified Normalized Difference Water Index [Xu 2006] substituting SWIR for NIR to effectively suppress false water detections from built-up urban structures and soil.",
    formula: "MNDWI = (Green - SWIR) / (Green + SWIR)",
    sourceText: "Green + SWIR bands required",
    missingReason: "Requires discrete Green (B03) and Shortwave Infrared (B11 / SWIR-1) spectral bands. SWIR data is not exposed to the frontend.",
    isSupported: false,
  },
];

/**
 * Dynamically evaluates whether the current scene or attached files provide
 * genuine remote-sensing data for any of the composite modes.
 */
export function evaluateCompositeModes(analysisData, attachedFiles) {
  // 1. Check explicit available_composites list from telemetry
  const rawAvail =
    analysisData?.available_composites ||
    analysisData?.input_metadata?.available_composites ||
    analysisData?.metadata?.available_composites ||
    analysisData?.audit_summary?.input_metadata?.available_composites ||
    [];

  const availableComposites = Array.isArray(rawAvail)
    ? rawAvail.map((s) => String(s).toLowerCase().trim())
    : [];

  // 2. Modality & band inspections
  const modalities = (
    analysisData?.input_metadata?.modalities ||
    analysisData?.modalities ||
    []
  ).map((m) => String(m).toLowerCase());

  const bandCount = Number(
    analysisData?.input_metadata?.band_count ||
    analysisData?.band_count ||
    (modalities.length > 0 ? modalities.length : 3)
  );

  const bands = analysisData?.bands || {};
  const hasNir = Boolean(
    bands.nir != null ||
    analysisData?.nir_band ||
    analysisData?.nir_url ||
    modalities.some((m) => m.includes("nir") || m.includes("near-infrared") || m.includes("near-ir")) ||
    bandCount >= 4
  );
  const hasRed = Boolean(
    bands.red != null ||
    analysisData?.red_band ||
    modalities.some((m) => m.includes("red")) ||
    bandCount >= 3
  );
  const hasGreen = Boolean(
    bands.green != null ||
    analysisData?.green_band ||
    modalities.some((m) => m.includes("green")) ||
    bandCount >= 2
  );
  const hasSwir = Boolean(
    bands.swir != null ||
    analysisData?.swir_band ||
    analysisData?.swir_url ||
    modalities.some((m) => m.includes("swir")) ||
    bandCount >= 5
  );
  const hasSar = Boolean(
    bands.sar_sigma0 != null ||
    analysisData?.sar_db_url ||
    analysisData?.sar_matrix ||
    modalities.some((m) => m.includes("sar") || m.includes("radar"))
  );

  return COMPOSITE_MODES.map((mode) => {
    const id = mode.id.toLowerCase();

    // SAR mode requires calibrated microwave radar data
    if (id === "sar") {
      const isAvail = hasSar || availableComposites.includes("sar") || availableComposites.includes("sar_db");
      return {
        ...mode,
        isSupported: isAvail,
        isEstimated: false,
        sourceText: isAvail ? "Sentinel-1 C-SAR (Calibrated σ₀ dB)" : mode.sourceText,
        missingReason: isAvail ? null : mode.missingReason,
      };
    }

    // RGB is always supported natively
    if (id === "rgb") {
      return {
        ...mode,
        isSupported: true,
        isEstimated: false,
        missingReason: null,
      };
    }

    // SWIR / MNDWI mode requires SWIR channel
    if (id === "mndwi") {
      const isAvail = (hasSwir && hasGreen) || availableComposites.includes("mndwi");
      return {
        ...mode,
        isSupported: isAvail,
        isEstimated: false,
        sourceText: isAvail ? "Real MNDWI Spectral Raster Layer" : mode.sourceText,
        missingReason: isAvail ? null : mode.missingReason,
      };
    }

    // Optical composites (CIR, NDVI, NDWI)
    // If native NIR is present: real biophysical layer
    // If 3-band RGB: supported via visible-spectrum approximations
    const isNative = hasNir || availableComposites.includes(id);

    if (id === "cir") {
      return {
        ...mode,
        isSupported: true,
        isEstimated: !isNative,
        badgeLabel: isNative ? "CIR (False Color)" : "CIR (Pseudo-NIR Est.)",
        sourceText: isNative
          ? "Sentinel-2 / 4-Band NIR + Red + Green"
          : "Pseudo-NIR False Color (Visible Spectrum Estimation)",
        missingReason: null,
      };
    }

    if (id === "ndvi") {
      return {
        ...mode,
        isSupported: true,
        isEstimated: !isNative,
        badgeLabel: isNative ? "NDVI Index" : "NDVI (Visible GLI/GRVI)",
        sourceText: isNative
          ? "Real NDVI Spectral Raster Layer"
          : "Visible Vegetation Index (GRVI Approximation)",
        missingReason: null,
      };
    }

    if (id === "ndwi") {
      return {
        ...mode,
        isSupported: true,
        isEstimated: !isNative,
        badgeLabel: isNative ? "NDWI Index" : "NDWI (Visible Water Contrast)",
        sourceText: isNative
          ? "Real NDWI Spectral Raster Layer"
          : "Visible Water Contrast (Blue-Red Approximation)",
        missingReason: null,
      };
    }

    return mode;
  });
}

export default function CompositeSwitcher({
  selectedComposite = "rgb",
  onSelectComposite = () => {},
  analysisData = null,
  attachedFiles = [],
  isLoading = false,
  className = "",
}) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [activeTooltip, setActiveTooltip] = useState(null);
  const [isDiagnosticsOpen, setIsDiagnosticsOpen] = useState(false);
  const [attemptedMode, setAttemptedMode] = useState(null);

  const evaluatedModes = evaluateCompositeModes(analysisData, attachedFiles);
  const activeModeObj = evaluatedModes.find((m) => m.id === selectedComposite) || evaluatedModes[0];

  const handleModeClick = (mode) => {
    if (mode.isSupported) {
      onSelectComposite(mode.id);
      setAttemptedMode(null);
    } else {
      // Honest notification: do not silently switch or fake the composite
      setAttemptedMode(mode);
    }
  };

  return (
    <div
      className={`rounded-xl border border-slate-200/90 dark:border-dark-border bg-white dark:bg-dark-card shadow-2xs p-3 transition-colors ${className}`}
    >
      {/* Top Header: Title & Active Channel Readout */}
      <div
        className={`flex flex-wrap items-center justify-between gap-2 ${
          isExpanded ? "mb-2.5" : "mb-0"
        }`}
      >
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-blue-50 dark:bg-blue-950/60 border border-blue-200/60 dark:border-blue-800/50 flex items-center justify-center shrink-0">
            <Layers className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" />
          </div>
          <span className="text-xs font-bold text-slate-900 dark:text-white tracking-tight">
            Spectral Composite
          </span>
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border border-emerald-200/80 dark:border-emerald-800/50">
            <span className={`w-1.5 h-1.5 rounded-full ${isLoading ? "bg-cyan-400 animate-spin" : "bg-emerald-500 animate-pulse"}`} />
            {isLoading ? `Rendering ${activeModeObj.shortLabel}...` : activeModeObj.badgeLabel}
          </span>
        </div>

        {/* Header Right Actions: Band Diagnostics & Upward/Downward Collapse Toggle */}
        <div className="flex items-center gap-2.5">
          {/* Telemetry / Band Diagnostics Toggle */}
          <button
            type="button"
            onClick={() => {
              if (!isExpanded) {
                setIsExpanded(true);
                setIsDiagnosticsOpen(true);
              } else {
                setIsDiagnosticsOpen(!isDiagnosticsOpen);
              }
            }}
            className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-500 dark:text-slate-400 hover:text-cyan-600 dark:hover:text-cyan-400 transition cursor-pointer"
            title="Inspect sensor band telemetry & availability details"
          >
            <Info className="w-3.5 h-3.5 text-slate-400" />
            <span>Band Diagnostics</span>
            {isDiagnosticsOpen && isExpanded ? (
              <ChevronUp className="w-3 h-3" />
            ) : (
              <ChevronDown className="w-3 h-3" />
            )}
          </button>

          {/* Divider */}
          <div className="h-3.5 w-px bg-slate-200 dark:bg-dark-border" />

          {/* Upward / Downward arrow toggle to collapse/expand entire composite panel */}
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 -mr-0.5 rounded-md text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/60 transition cursor-pointer flex items-center justify-center"
            title={isExpanded ? "Collapse Spectral Composite" : "Expand Spectral Composite"}
            aria-label={isExpanded ? "Collapse Spectral Composite" : "Expand Spectral Composite"}
          >
            {isExpanded ? (
              <ChevronUp className="w-4 h-4 text-slate-500 dark:text-slate-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-500 dark:text-slate-400" />
            )}
          </button>
        </div>
      </div>

      {/* Collapsible Content */}
      {isExpanded && (
        <div className="animate-in fade-in duration-150">

      {/* Button Strip: 6 Composite Modes with Honest Availability */}
      <div
        className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2"
        role="radiogroup"
        aria-label="Multi-band spectral composite switcher"
      >
        {evaluatedModes.map((mode) => {
          const isSelected = selectedComposite === mode.id;
          const isHovered = activeTooltip === mode.id;

          return (
            <div key={mode.id} className="relative">
              <button
                type="button"
                role="radio"
                aria-checked={isSelected}
                aria-disabled={!mode.isSupported}
                onClick={() => handleModeClick(mode)}
                onMouseEnter={() => setActiveTooltip(mode.id)}
                onMouseLeave={() => setActiveTooltip(null)}
                onFocus={() => setActiveTooltip(mode.id)}
                onBlur={() => setActiveTooltip(null)}
                className={`w-full flex flex-col items-start p-2 rounded-lg border text-left transition-all cursor-pointer select-none ${
                  isSelected
                    ? "bg-brand-600 text-white border-brand-500 shadow-xs ring-1 ring-brand-400/50"
                    : mode.isSupported
                    ? "bg-slate-50 dark:bg-dark-hover/70 border-slate-200 dark:border-dark-border text-slate-800 dark:text-slate-200 hover:border-slate-300 dark:hover:border-slate-700"
                    : "bg-slate-100/60 dark:bg-slate-900/40 border-slate-200/60 dark:border-slate-800/60 text-slate-400 dark:text-slate-500 hover:border-slate-300 dark:hover:border-slate-700"
                }`}
              >
                {/* Header: Label + Status Dot */}
                <div className="w-full flex items-center justify-between gap-1 mb-1">
                  <span
                    className={`text-xs font-bold truncate ${
                      isSelected ? "text-white" : mode.isSupported ? "text-slate-800 dark:text-slate-200" : "text-slate-500 dark:text-slate-400"
                    }`}
                  >
                    {mode.shortLabel}
                  </span>
                  {isSelected && isLoading ? (
                    <span className="w-2.5 h-2.5 rounded-full border-2 border-white/30 border-t-white animate-spin shrink-0" />
                  ) : mode.isSupported ? (
                    <span
                      className={`w-2 h-2 rounded-full shrink-0 ${
                        mode.isEstimated ? "bg-amber-400" : "bg-emerald-400"
                      }`}
                      title={mode.isEstimated ? "Visible spectrum approximation" : "Real imagery source available"}
                    />
                  ) : (
                    <span
                      className="w-2 h-2 rounded-full border border-slate-400 dark:border-slate-600 shrink-0"
                      title="Requires discrete bands not currently exposed"
                    />
                  )}
                </div>

                {/* Subtitle: Availability / Sensor requirement */}
                <div className="w-full text-[10px] leading-tight truncate">
                  {mode.isSupported ? (
                    mode.isEstimated ? (
                      <span className={isSelected ? "text-amber-200 font-semibold" : "text-amber-600 dark:text-amber-400 font-medium"}>
                        ● Est. Visible
                      </span>
                    ) : (
                      <span className={isSelected ? "text-cyan-100 font-semibold" : "text-emerald-600 dark:text-emerald-400 font-medium"}>
                        ● Available
                      </span>
                    )
                  ) : (
                    <span className={isSelected ? "text-slate-200" : "text-slate-400 dark:text-slate-500"}>
                      ○ Unavailable
                    </span>
                  )}
                </div>
              </button>

              {/* Hover / Focus Explanatory Remote-Sensing Tooltip */}
              {isHovered && (
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-30 w-64 p-2.5 rounded-xl bg-slate-950/95 backdrop-blur-md border border-slate-800 text-white shadow-xl text-left pointer-events-none animate-in fade-in zoom-in-95 duration-150">
                  <div className="flex items-center justify-between gap-2 pb-1.5 border-b border-slate-800/80 mb-1.5">
                    <span className="text-xs font-bold text-slate-100">{mode.name}</span>
                    <span
                      className={`text-[9px] font-mono px-1.5 py-0.5 rounded font-semibold ${
                        mode.isSupported
                          ? mode.isEstimated
                            ? "bg-amber-950 text-amber-300 border border-amber-700/60"
                            : "bg-emerald-950 text-emerald-300 border border-emerald-700/60"
                          : "bg-rose-950 text-rose-300 border border-rose-700/60"
                      }`}
                    >
                      {mode.isSupported
                        ? mode.isEstimated
                          ? "● Est. Visible"
                          : "● Available"
                        : "○ Unavailable"}
                    </span>
                  </div>

                  <p className="text-[10px] text-slate-300 leading-relaxed mb-2">
                    {mode.description}
                  </p>

                  {mode.isEstimated && (
                    <div className="text-[10px] text-amber-300/90 flex items-start gap-1.5 bg-amber-950/40 p-1.5 rounded-md border border-amber-800/40 mb-1.5">
                      <Info className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                      <span>Approximation calculated from visible bands (no discrete NIR required).</span>
                    </div>
                  )}

                  <div className="bg-slate-900/90 rounded-md p-1.5 border border-slate-800 text-[9px] font-mono text-cyan-300 mb-1.5">
                    <span className="text-slate-400 block text-[8px] uppercase tracking-wider font-sans mb-0.5">
                      Formula / Channels:
                    </span>
                    {mode.formula}
                  </div>

                  {!mode.isSupported && (
                    <div className="text-[10px] text-amber-300/90 flex items-start gap-1.5 bg-amber-950/40 p-1.5 rounded-md border border-amber-800/40">
                      <AlertCircle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                      <span>{mode.missingReason}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Selected Mode Channel Metadata Summary Bar */}
      <div className="mt-2.5 pt-2 border-t border-slate-100 dark:border-dark-border flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500 dark:text-slate-400">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className="font-semibold text-slate-700 dark:text-slate-300 shrink-0">
            Active Channels:
          </span>
          <span className="font-mono text-[10px] text-cyan-600 dark:text-cyan-400 truncate">
            {activeModeObj.displayChannels}
          </span>
        </div>
        <div className="flex items-center gap-1.5 font-mono text-[10px] text-slate-400 shrink-0">
          <span>Source:</span>
          <span className="text-slate-600 dark:text-slate-300 font-medium">
            {activeModeObj.sourceText}
          </span>
        </div>
      </div>

      {/* Inline Explanatory Notification when an Unavailable Mode is Clicked */}
      {attemptedMode && (
        <div className="mt-2.5 p-2.5 rounded-lg bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800/50 flex items-start justify-between gap-2.5 animate-in fade-in duration-200">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
            <div className="text-xs text-amber-900 dark:text-amber-200">
              <span className="font-bold">{attemptedMode.name}</span> is currently unavailable.{" "}
              <span className="text-amber-700 dark:text-amber-300">
                {attemptedMode.missingReason}
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setAttemptedMode(null)}
            className="text-amber-600 dark:text-amber-400 hover:text-amber-900 text-xs font-semibold shrink-0 cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Expandable Band Diagnostics & Telemetry Drawer */}
      {isDiagnosticsOpen && (
        <div className="mt-3 pt-3 border-t border-slate-200/80 dark:border-dark-border text-xs animate-in slide-in-from-top-2 duration-200">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-bold text-slate-800 dark:text-slate-200 text-xs">
              Sensor Spectral Telemetry &amp; Availability Audit
            </h4>
            <span className="text-[10px] font-mono text-slate-400">
              ISO-19115 / OGC Geotiff Standard
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
              <div className="flex items-center gap-1.5 text-[11px] font-bold text-slate-700 dark:text-slate-300 mb-1">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                <span>3-Band Optical RGB</span>
              </div>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-normal">
                Standard Red (B04), Green (B03), and Blue (B02) reflectance channels are fully available from Esri World Imagery &amp; baseline analysis imagery.
              </p>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
              <div className="flex items-center gap-1.5 text-[11px] font-bold text-slate-700 dark:text-slate-300 mb-1">
                <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
                <span>Discrete NIR / SWIR Bands</span>
              </div>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-normal">
                CIR, NDVI, NDWI, and MNDWI require discrete floating-point spectral bands (B08, B11). These are processed server-side but not exposed as raw client-side rasters.
              </p>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800">
              <div className="flex items-center gap-1.5 text-[11px] font-bold text-slate-700 dark:text-slate-300 mb-1">
                <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
                <span>Calibrated SAR σ₀ dB</span>
              </div>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-normal">
                Sentinel-1 SAR σ₀ backscatter requires microwave radiometry calibration. Optical tiles with CSS grayscale desaturation are strictly not treated as radar data.
              </p>
            </div>
          </div>
        </div>
      )}
        </div>
      )}
    </div>
  );
}
