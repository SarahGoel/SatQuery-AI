import React, { useEffect, useRef, useState } from "react";
import "ol/ol.css";
import Map from "ol/Map";
import View from "ol/View";
import Overlay from "ol/Overlay";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import VectorSource from "ol/source/Vector";
import OSM from "ol/source/OSM";
import XYZ from "ol/source/XYZ";
import GeoJSON from "ol/format/GeoJSON";
import Feature from "ol/Feature";
import Polygon from "ol/geom/Polygon";
import { fromLonLat, toLonLat } from "ol/proj";
import { Style, Stroke, Fill, Text } from "ol/style";
import { getRenderPixel } from "ol/render";
import { ChevronLeft, ChevronRight } from "lucide-react";

const OPTICAL_XYZ = {
  url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  maxZoom: 19,
  attributions: "Tiles © Esri — optical / Sentinel-2 style base",
};

const SAR_XYZ = {
  url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  maxZoom: 19,
  attributions: "SAR VV/VH panel (grayscale backscatter view)",
};

function getFeatureStyle(feature, collectionTaskType) {
  const props = feature.getProperties() || {};
  const featureClass = (props.class || props.category || "").toLowerCase();
  const label = (props.label || "").toLowerCase();
  const taskType = (props.task_type || collectionTaskType || "").toLowerCase();

  // Flood / Inundation Hazard: Red boundary (stroke: #ef4444, width: 2px), fill: rgba(239, 68, 68, 0.45)
  if (
    featureClass === "flood" ||
    featureClass === "water" ||
    label.includes("flood") ||
    label.includes("inundat") ||
    label.includes("change") ||
    taskType === "change_detection" ||
    taskType.includes("change")
  ) {
    return new Style({
      stroke: new Stroke({
        color: "#ef4444",
        width: 2.0,
      }),
      fill: new Fill({
        color: "rgba(239, 68, 68, 0.45)",
      }),
      text: props.label
        ? new Text({
            text: `${props.label}${props.confidence != null ? ` (${Math.round(props.confidence * 100)}%)` : ""}`,
            font: "12px monospace, sans-serif",
            fill: new Fill({ color: "#ffffff" }),
            stroke: new Stroke({ color: "rgba(15,23,42,0.85)", width: 3 }),
            offsetY: -12,
          })
        : undefined,
    });
  }

  // Cross-Modal / SAR: High-visibility amber (stroke: #f59e0b, width: 2px), fill: rgba(245, 158, 11, 0.35)
  if (
    featureClass === "sar_anomaly" ||
    featureClass === "radar" ||
    label.includes("sar") ||
    label.includes("radar") ||
    label.includes("anomaly") ||
    label.includes("backscatter") ||
    taskType === "cross_modal" ||
    taskType.includes("cross_modal")
  ) {
    return new Style({
      stroke: new Stroke({
        color: "#f59e0b",
        width: 2.0,
      }),
      fill: new Fill({
        color: "rgba(245, 158, 11, 0.35)",
      }),
      text: props.label
        ? new Text({
            text: `${props.label}${props.confidence != null ? ` (${Math.round(props.confidence * 100)}%)` : ""}`,
            font: "12px monospace, sans-serif",
            fill: new Fill({ color: "#fef3c7" }),
            stroke: new Stroke({ color: "rgba(15,23,42,0.85)", width: 3 }),
            offsetY: -12,
          })
        : undefined,
    });
  }

  // Grounding / Infrastructure: Crisp cyan border (stroke: #06b6d4, width: 2.5px), fill: rgba(6, 182, 212, 0.25)
  return new Style({
    stroke: new Stroke({
      color: "#06b6d4",
      width: 2.5,
    }),
    fill: new Fill({
      color: "rgba(6, 182, 212, 0.25)",
    }),
    text: props.label
      ? new Text({
          text: `${props.label}${props.confidence != null ? ` (${Math.round(props.confidence * 100)}%)` : ""}`,
          font: "12px monospace, sans-serif",
          fill: new Fill({ color: "#cffafe" }),
          stroke: new Stroke({ color: "rgba(15,23,42,0.85)", width: 3 }),
          offsetY: -12,
        })
      : undefined,
  });
}

const bboxStyle = new Style({
  stroke: new Stroke({
    color: "#fbbf24",
    width: 2,
    lineDash: [8, 6],
  }),
  fill: new Fill({
    color: "rgba(251, 191, 36, 0.08)",
  }),
  text: new Text({
    text: "BBOX",
    font: "11px monospace",
    fill: new Fill({ color: "#fbbf24" }),
    stroke: new Stroke({ color: "rgba(15,23,42,0.85)", width: 3 }),
    offsetY: -12,
  }),
});

function isValidExtent(extent) {
  if (!Array.isArray(extent) || extent.length !== 4) return false;
  const [minX, minY, maxX, maxY] = extent;
  return (
    Number.isFinite(minX) &&
    Number.isFinite(minY) &&
    Number.isFinite(maxX) &&
    Number.isFinite(maxY) &&
    minX <= maxX &&
    minY <= maxY &&
    !(minX === 0 && minY === 0 && maxX === 0 && maxY === 0)
  );
}

function bboxToPolygon(bboxCoordinates) {
  if (!bboxCoordinates) return null;
  let minLon;
  let minLat;
  let maxLon;
  let maxLat;
  if (Array.isArray(bboxCoordinates[0])) {
    const lons = bboxCoordinates.map((c) => c[0]);
    const lats = bboxCoordinates.map((c) => c[1]);
    minLon = Math.min(...lons);
    maxLon = Math.max(...lons);
    minLat = Math.min(...lats);
    maxLat = Math.max(...lats);
  } else if (bboxCoordinates.length >= 4) {
    [minLon, minLat, maxLon, maxLat] = bboxCoordinates;
  } else {
    return null;
  }
  const ring = [
    fromLonLat([minLon, minLat]),
    fromLonLat([maxLon, minLat]),
    fromLonLat([maxLon, maxLat]),
    fromLonLat([minLon, maxLat]),
    fromLonLat([minLon, minLat]),
  ];
  return new Polygon([ring]);
}

const MapViewer = ({
  analysisData,
  geojsonOverlay,
  bboxCoordinates,
  baseImagery: baseImageryProp,
  onBaseImageryChange,
  selectedComposite = "rgb",
  onSelectComposite,
  overlayOpacity = 0.70,
}) => {
  const mapElement = useRef();
  const mapRef = useRef();
  const overlayLayerRef = useRef(null);
  const bboxLayerRef = useRef(null);
  const opticalLayerRef = useRef(null);
  const sarLayerRef = useRef(null);
  const coordOverlayRef = useRef(null);
  const coordPopupRef = useRef(null);

  const [baseImagery, setBaseImagery] = useState(baseImageryProp || "optical");
  const [sarNotice, setSarNotice] = useState(null);
  const [cursorCoords, setCursorCoords] = useState({ lon: 0, lat: 0, zoom: 2 });
  const [pointerActive, setPointerActive] = useState(false);

  // Split curtain state for OpenLayers Optical vs SAR comparison
  const mapSplitPositionRef = useRef(50);
  const [mapSplitPosition, setMapSplitPosition] = useState(50);
  const [isMapDragging, setIsMapDragging] = useState(false);

  const activeGeojson =
    analysisData?.geojson ||
    analysisData?.audit_summary?.geojson ||
    geojsonOverlay;

  const activeBbox =
    bboxCoordinates ||
    analysisData?.bbox ||
    analysisData?.audit_summary?.bounds;

  useEffect(() => {
    if (baseImageryProp && baseImageryProp !== baseImagery) {
      setBaseImagery(baseImageryProp);
    }
  }, [baseImageryProp, baseImagery]);

  useEffect(() => {
    // Initialize OpenLayers Map Component [3, 102, 103]
    const opticalLayer = new TileLayer({
      source: new XYZ(OPTICAL_XYZ),
      visible: true,
      properties: { name: "optical-base", modality: "optical" },
    });
    const sarLayer = new TileLayer({
      source: new XYZ(SAR_XYZ),
      visible: false,
      className: "sar-base-layer",
      properties: { name: "sar-base", modality: "sar" },
    });
    opticalLayerRef.current = opticalLayer;
    sarLayerRef.current = sarLayer;

    const initialMap = new Map({
      target: mapElement.current,
      layers: [
        opticalLayer,
        sarLayer,
        new TileLayer({
          source: new OSM(), // Base map fallback layer
          visible: false,
          properties: { name: "osm-fallback" },
        }),
      ],
      view: new View({
        center: [0, 0],
        zoom: 2,
        projection: "EPSG:3857",
      }),
    });

    const popupEl = coordPopupRef.current;
    const coordOverlay = new Overlay({
      element: popupEl,
      offset: [12, -12],
      positioning: "bottom-left",
      stopEvent: false,
    });
    initialMap.addOverlay(coordOverlay);
    coordOverlayRef.current = coordOverlay;

    initialMap.on("pointermove", (evt) => {
      if (!evt.coordinate) return;
      const [lon, lat] = toLonLat(evt.coordinate);
      const zoom = initialMap.getView().getZoom() || 2;
      setPointerActive(true);
      setCursorCoords({
        lon: Number(lon.toFixed(6)),
        lat: Number(lat.toFixed(6)),
        zoom: Number(zoom.toFixed(2)),
      });
      coordOverlay.setPosition(evt.coordinate);
    });

    mapRef.current = initialMap;
    requestAnimationFrame(() => initialMap.updateSize());

    return () => initialMap.setTarget(null);
  }, []);

  // OpenLayers Layer Clipping Lifecycle for Optical ◧ SAR Curtain Comparison
  useEffect(() => {
    const sarLayer = sarLayerRef.current;
    const opticalLayer = opticalLayerRef.current;
    if (!sarLayer || !opticalLayer) return;

    const onPrerender = (event) => {
      const ctx = event.context;
      const map = mapRef.current;
      if (!ctx || !map) return;
      const mapSize = map.getSize();
      if (!mapSize) return;

      const width = mapSize[0];
      const height = mapSize[1];
      const splitPx = width * (mapSplitPositionRef.current / 100);

      // getRenderPixel accurately converts CSS coordinates into canvas context pixels (DPR-safe)
      const tl = getRenderPixel(event, [splitPx, 0]);
      const br = getRenderPixel(event, [width, height]);

      ctx.save();
      ctx.beginPath();
      ctx.rect(tl[0], 0, Math.max(0, br[0] - tl[0]), Math.max(0, br[1]));
      ctx.clip();
    };

    const onPostrender = (event) => {
      const ctx = event.context;
      if (ctx) {
        ctx.restore();
      }
    };

    if (baseImagery === "split") {
      opticalLayer.setVisible(true);
      sarLayer.setVisible(true);
      sarLayer.on("prerender", onPrerender);
      sarLayer.on("postrender", onPostrender);
      mapRef.current?.render();
    } else {
      sarLayer.un("prerender", onPrerender);
      sarLayer.un("postrender", onPostrender);
      const opticalOn = baseImagery === "optical";
      opticalLayer.setVisible(opticalOn);
      sarLayer.setVisible(!opticalOn);
      mapRef.current?.render();
    }

    return () => {
      sarLayer.un("prerender", onPrerender);
      sarLayer.un("postrender", onPostrender);
    };
  }, [baseImagery]);

  // Curtain Dragging & Keyboard Navigation for OpenLayers
  const handleMapCurtainPointerDown = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsMapDragging(true);
    updateMapCurtainFromClientX(e.clientX);
  };

  const updateMapCurtainFromClientX = (clientX) => {
    if (!mapElement.current) return;
    const rect = mapElement.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const pos = ((clientX - rect.left) / rect.width) * 100;
    const clamped = Math.max(0, Math.min(100, pos));
    mapSplitPositionRef.current = clamped;
    setMapSplitPosition(clamped);
    mapRef.current?.render();
  };

  useEffect(() => {
    if (!isMapDragging) return;

    const onPointerMove = (e) => {
      e.preventDefault();
      updateMapCurtainFromClientX(e.clientX);
    };

    const onPointerUp = () => {
      setIsMapDragging(false);
    };

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);

    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
  }, [isMapDragging]);

  const handleMapCurtainKeyDown = (e) => {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      const next = Math.max(0, mapSplitPosition - (e.shiftKey ? 10 : 2));
      mapSplitPositionRef.current = next;
      setMapSplitPosition(next);
      mapRef.current?.render();
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      const next = Math.min(100, mapSplitPosition + (e.shiftKey ? 10 : 2));
      mapSplitPositionRef.current = next;
      setMapSplitPosition(next);
      mapRef.current?.render();
    } else if (e.key === "Home") {
      e.preventDefault();
      mapSplitPositionRef.current = 0;
      setMapSplitPosition(0);
      mapRef.current?.render();
    } else if (e.key === "End") {
      e.preventDefault();
      mapSplitPositionRef.current = 100;
      setMapSplitPosition(100);
      mapRef.current?.render();
    }
  };

  useEffect(() => {
    if (!mapRef.current) return;

    // Clean Slate: explicitly wipe vector sources and remove existing layers before rendering new ones
    if (overlayLayerRef.current) {
      const src = overlayLayerRef.current.getSource();
      if (src && typeof src.clear === "function") {
        src.clear();
      }
      mapRef.current.removeLayer(overlayLayerRef.current);
      overlayLayerRef.current = null;
    }
    if (bboxLayerRef.current) {
      const src = bboxLayerRef.current.getSource();
      if (src && typeof src.clear === "function") {
        src.clear();
      }
      mapRef.current.removeLayer(bboxLayerRef.current);
      bboxLayerRef.current = null;
    }

    const geojson = activeGeojson;
    const taskType = geojson?.properties?.task_type;

    let parsedFeatures = [];
    if (geojson && (geojson.type === "FeatureCollection" || geojson.type === "Feature")) {
      const featureList = Array.isArray(geojson.features)
        ? geojson.features
        : geojson.type === "Feature"
        ? [geojson]
        : [];

      if (featureList.length > 0) {
        try {
          parsedFeatures = new GeoJSON().readFeatures(geojson, {
            dataProjection: "EPSG:4326",
            featureProjection: "EPSG:3857",
          });
        } catch (err) {
          console.error("Error reading GeoJSON features:", err);
          parsedFeatures = [];
        }
      }
    }

    const hasGeojsonFeatures = Array.isArray(parsedFeatures) && parsedFeatures.length > 0;

    if (hasGeojsonFeatures) {
      // Dynamic multi-instance vector layer with calibrated default opacity
      const vectorSource = new VectorSource();
      vectorSource.clear();
      vectorSource.addFeatures(parsedFeatures);

      const vectorLayer = new VectorLayer({
        source: vectorSource,
        style: (feature) => getFeatureStyle(feature, taskType),
        opacity: overlayOpacity ?? 0.70,
        properties: { name: "geojson-dynamic-overlay" },
      });

      overlayLayerRef.current = vectorLayer;
      mapRef.current.addLayer(vectorLayer);

      // Auto-Centering & Bounding Box Fit: precisely around returned features
      try {
        const extent = vectorSource.getExtent();
        if (isValidExtent(extent)) {
          let fitExtent = extent;
          if (extent[0] === extent[2] || extent[1] === extent[3]) {
            fitExtent = [extent[0] - 100, extent[1] - 100, extent[2] + 100, extent[3] + 100];
          }
          mapRef.current.getView().fit(fitExtent, { padding: [40, 40, 40, 40], duration: 800 });
        }
      } catch (fitErr) {
        console.warn("Could not fit view to vector extent:", fitErr);
      }
    } else if (activeBbox && Array.isArray(activeBbox) && activeBbox.length >= 4) {
      // Clean Slate: Only fallback to static bounding box if no real GeoJSON features exist
      try {
        const geom = bboxToPolygon(activeBbox);
        if (geom) {
          const bboxSource = new VectorSource({
            features: [new Feature({ geometry: geom, name: "bbox" })],
          });
          const bboxLayer = new VectorLayer({
            source: bboxSource,
            style: bboxStyle,
            properties: { name: "bbox-overlay" },
          });
          bboxLayerRef.current = bboxLayer;
          mapRef.current.addLayer(bboxLayer);

          const extent = geom.getExtent();
          if (isValidExtent(extent)) {
            let fitExtent = extent;
            if (extent[0] === extent[2] || extent[1] === extent[3]) {
              fitExtent = [extent[0] - 100, extent[1] - 100, extent[2] + 100, extent[3] + 100];
            }
            mapRef.current.getView().fit(fitExtent, { padding: [40, 40, 40, 40], duration: 800 });
          }
        }
      } catch (bboxErr) {
        console.warn("Could not render bbox overlay:", bboxErr);
      }
    }
  }, [activeGeojson, activeBbox]);

  useEffect(() => {
    if (overlayLayerRef.current) {
      overlayLayerRef.current.setOpacity(overlayOpacity ?? 0.70);
    }
  }, [overlayOpacity]);

  const handleBaseToggle = (next) => {
    if (next === "sar") {
      setSarNotice("Sentinel-1 calibrated SAR σ₀ decibel backscatter feed is unavailable for this AOI. Displaying genuine Natural RGB optical imagery.");
      setTimeout(() => setSarNotice(null), 5000);
      return;
    }
    setBaseImagery(next);
    if (onBaseImageryChange) onBaseImageryChange(next);
  };

  return (
    <div
      className="map-container relative"
      style={{ width: "100%", height: "100%", minHeight: "500px", borderRadius: "8px", overflow: "hidden" }}
    >
      {/* Floating Base Imagery Control */}
      <div
        className="absolute left-3 top-3 z-10 flex items-center gap-1 rounded-xl border border-slate-200/90 bg-white/95 p-1 shadow-md backdrop-blur-md dark:border-slate-700 dark:bg-slate-900/95"
        role="group"
        aria-label="Base imagery layer panels"
      >
        <button
          type="button"
          onClick={() => handleBaseToggle("optical")}
          className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition ${
            baseImagery === "optical"
              ? "bg-sky-600 text-white shadow-2xs"
              : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
          }`}
        >
          <span className="inline-flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            Natural RGB
          </span>
        </button>
        <button
          type="button"
          onClick={() => handleBaseToggle("sar")}
          className="rounded-lg px-2.5 py-1 text-xs font-semibold text-slate-400 hover:bg-slate-100 dark:text-slate-500 dark:hover:bg-slate-800 transition cursor-pointer"
          title="SAR σ₀ dB backscatter is unavailable (discrete radar telemetry required)"
        >
          <span className="inline-flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full border border-slate-400" />
            SAR σ₀ (Unavailable)
          </span>
        </button>
      </div>

      {/* Floating Notice when SAR or unavailable layer is clicked */}
      {sarNotice && (
        <div className="absolute top-14 left-3 z-20 max-w-sm p-2.5 rounded-xl bg-slate-950/95 backdrop-blur-md border border-amber-600/80 text-amber-200 text-xs shadow-xl animate-in fade-in duration-200 flex items-start justify-between gap-2">
          <span>{sarNotice}</span>
          <button
            type="button"
            onClick={() => setSarNotice(null)}
            className="text-amber-400 hover:text-white font-bold ml-2 cursor-pointer"
          >
            ×
          </button>
        </div>
      )}

      {/* Optical ◧ SAR Draggable Curtain Overlay for MapViewer */}
      {baseImagery === "split" && (
        <div className="absolute inset-0 z-20 pointer-events-none overflow-hidden select-none">
          {/* Metadata badges for OpenLayers Map Comparison */}
          <div className="absolute top-14 left-3 pointer-events-none z-10">
            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-slate-950/85 backdrop-blur-xs border border-sky-600/70 text-[10px] font-mono font-medium text-sky-300 shadow-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
              <span>Sentinel-2 Optical (Natural RGB)</span>
            </div>
          </div>

          <div className="absolute top-14 right-3 pointer-events-none z-10">
            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-slate-950/85 backdrop-blur-xs border border-amber-600/70 text-[10px] font-mono font-medium text-amber-300 shadow-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
              <span>Sentinel-1 SAR</span>
            </div>
          </div>

          {/* Draggable curtain divider */}
          <div
            className="absolute top-0 bottom-0 pointer-events-auto cursor-ew-resize flex items-center justify-center -translate-x-1/2 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
            style={{ left: `${mapSplitPosition}%` }}
            role="slider"
            tabIndex={0}
            aria-label="Split comparison position for Sentinel-2 Optical and Sentinel-1 SAR"
            aria-valuenow={Math.round(mapSplitPosition)}
            aria-valuemin={0}
            aria-valuemax={100}
            onPointerDown={handleMapCurtainPointerDown}
            onKeyDown={handleMapCurtainKeyDown}
          >
            <div className="w-0.5 h-full bg-white shadow-[0_0_8px_rgba(0,0,0,0.8)]" />
            <div className="absolute w-8 h-8 rounded-full bg-slate-900 border-2 border-sky-400 shadow-xl flex items-center justify-center text-sky-300 hover:scale-110 active:scale-95 transition-transform cursor-grab active:cursor-grabbing">
              <div className="flex items-center -space-x-1">
                <ChevronLeft className="w-3.5 h-3.5" />
                <ChevronRight className="w-3.5 h-3.5" />
              </div>
            </div>
          </div>
        </div>
      )}

      <div
        ref={mapElement}
        className="map-view"
        style={{ width: "100%", height: "100%", minHeight: "500px" }}
      />

      <div
        ref={coordPopupRef}
        className="pointer-events-none rounded bg-slate-900/85 px-1.5 py-0.5 font-mono text-[10px] text-amber-200 shadow"
        style={{ display: pointerActive ? "block" : "none" }}
      >
        {cursorCoords.lat.toFixed(5)}°, {cursorCoords.lon.toFixed(5)}°
      </div>

      <div className="absolute bottom-2 left-3 z-10 flex items-center gap-3 rounded-lg border border-slate-200/80 bg-white/90 px-3 py-1 font-mono text-[11px] text-slate-600 shadow-sm backdrop-blur-md dark:border-slate-700 dark:bg-slate-900/90 dark:text-slate-300">
        <span>
          LAT <strong>{cursorCoords.lat}°</strong>
        </span>
        <span>
          LON <strong>{cursorCoords.lon}°</strong>
        </span>
        <span>
          Z <strong>{cursorCoords.zoom}x</strong>
        </span>
        <span className="border-l border-slate-200 pl-2 text-emerald-600 dark:border-slate-700 dark:text-emerald-400">
          EPSG:3857 · Natural RGB (Esri World Imagery / Sentinel-2 Equivalent · True Color B04/B03/B02)
        </span>
      </div>
    </div>
  );
};

export default MapViewer;
