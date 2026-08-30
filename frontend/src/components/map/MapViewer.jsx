import React, { useEffect, useRef, useState } from "react";
import Map from "ol/Map";
import View from "ol/View";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import OSM from "ol/source/OSM";
import XYZ from "ol/source/XYZ";
import VectorSource from "ol/source/Vector";
import { fromLonLat, toLonLat } from "ol/proj";
import Draw, { createBox } from "ol/interaction/Draw";
import { Circle as CircleStyle, Fill, Stroke, Style } from "ol/style";
import Feature from "ol/Feature";
import Polygon from "ol/geom/Polygon";
import Point from "ol/geom/Point";
import {
  ZoomIn,
  ZoomOut,
  Maximize,
  Minimize,
  Compass,
  Layers,
  Square,
  Pentagon,
  MapPin,
  Trash2,
  Search,
  Sliders,
  Eye,
  EyeOff
} from "lucide-react";

export default function MapViewer({
  selectedROI,
  activeLayers,
  compareMode,
  swipePosition,
  onSwipeChange,
  activeFeatures,
  onROICreated,
}) {
  const mapElement = useRef(null);
  const mapRef = useRef(null);
  const vectorSourceRef = useRef(new VectorSource());
  const evidenceSourceRef = useRef(new VectorSource());
  const drawInteractionRef = useRef(null);

  const [activeDrawTool, setActiveDrawTool] = useState(null); // 'box', 'polygon', 'point', null
  const [basemap, setBasemap] = useState("satellite"); // 'satellite', 'osm', 'carto'
  const [coords, setCoords] = useState({ lon: 92.9376, lat: 26.2006, zoom: 8 });
  const [searchQuery, setSearchQuery] = useState("");
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Initialize Map
  useEffect(() => {
    if (!mapElement.current) return;

    // Basemap sources
    const osmSource = new OSM();
    const satelliteSource = new XYZ({
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      maxZoom: 19,
    });
    const cartoSource = new XYZ({
      url: "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
      maxZoom: 19,
    });

    const baseTileLayer = new TileLayer({
      source: satelliteSource,
      properties: { name: "basemap" },
    });

    const vectorLayer = new VectorLayer({
      source: vectorSourceRef.current,
      style: new Style({
        fill: new Fill({ color: "rgba(59, 130, 246, 0.2)" }),
        stroke: new Stroke({ color: "#2563eb", width: 2.5, lineDash: [6, 6] }),
        image: new CircleStyle({
          radius: 7,
          fill: new Fill({ color: "#2563eb" }),
          stroke: new Stroke({ color: "#ffffff", width: 2 }),
        }),
      }),
    });

    const evidenceLayer = new VectorLayer({
      source: evidenceSourceRef.current,
      style: (feature) => {
        const type = feature.get("type");
        if (type === "water") {
          return new Style({
            fill: new Fill({ color: "rgba(59, 130, 246, 0.45)" }),
            stroke: new Stroke({ color: "#1d4ed8", width: 2 }),
          });
        }
        if (type === "risk") {
          return new Style({
            fill: new Fill({ color: "rgba(239, 68, 68, 0.35)" }),
            stroke: new Stroke({ color: "#dc2626", width: 2, lineDash: [4, 4] }),
          });
        }
        return new Style({
          fill: new Fill({ color: "rgba(245, 158, 11, 0.35)" }),
          stroke: new Stroke({ color: "#d97706", width: 2 }),
        });
      },
    });

    const map = new Map({
      target: mapElement.current,
      layers: [baseTileLayer, evidenceLayer, vectorLayer],
      view: new View({
        center: fromLonLat([92.9376, 26.2006]),
        zoom: 8,
      }),
      controls: [], // custom HUD
    });

    map.on("pointermove", (evt) => {
      const lonLat = toLonLat(evt.coordinate);
      setCoords({
        lon: Number(lonLat[0].toFixed(4)),
        lat: Number(lonLat[1].toFixed(4)),
        zoom: Math.round(map.getView().getZoom() || 8),
      });
    });

    mapRef.current = map;

    return () => {
      map.setTarget(null);
    };
  }, []);

  // Update Basemap Layer
  useEffect(() => {
    if (!mapRef.current) return;
    const layers = mapRef.current.getLayers().getArray();
    const baseLayer = layers.find((l) => l.get("name") === "basemap");
    if (!baseLayer) return;

    if (basemap === "satellite") {
      baseLayer.setSource(
        new XYZ({
          url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
          maxZoom: 19,
        })
      );
    } else if (basemap === "carto") {
      baseLayer.setSource(
        new XYZ({
          url: "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
          maxZoom: 19,
        })
      );
    } else {
      baseLayer.setSource(new OSM());
    }
  }, [basemap]);

  // Handle ROI and Region Fly-to
  useEffect(() => {
    if (!mapRef.current || !selectedROI) return;
    const view = mapRef.current.getView();
    view.animate({
      center: fromLonLat([selectedROI.lon, selectedROI.lat]),
      zoom: selectedROI.zoom || 9,
      duration: 1000,
    });

    // Populate mock evidence features in ROI
    evidenceSourceRef.current.clear();

    const centerCoord = fromLonLat([selectedROI.lon, selectedROI.lat]);
    const offset1 = fromLonLat([selectedROI.lon + 0.12, selectedROI.lat + 0.08]);
    const offset2 = fromLonLat([selectedROI.lon - 0.15, selectedROI.lat - 0.05]);

    // Water polygon
    const waterPoly = new Polygon([
      [
        fromLonLat([selectedROI.lon - 0.18, selectedROI.lat - 0.08]),
        fromLonLat([selectedROI.lon + 0.1, selectedROI.lat - 0.02]),
        fromLonLat([selectedROI.lon + 0.22, selectedROI.lat + 0.12]),
        fromLonLat([selectedROI.lon - 0.05, selectedROI.lat + 0.15]),
        fromLonLat([selectedROI.lon - 0.18, selectedROI.lat - 0.08]),
      ],
    ]);
    const f1 = new Feature({ geometry: waterPoly, type: "water" });

    // Risk perimeter polygon
    const riskPoly = new Polygon([
      [
        fromLonLat([selectedROI.lon + 0.05, selectedROI.lat + 0.02]),
        fromLonLat([selectedROI.lon + 0.25, selectedROI.lat + 0.04]),
        fromLonLat([selectedROI.lon + 0.28, selectedROI.lat + 0.18]),
        fromLonLat([selectedROI.lon + 0.08, selectedROI.lat + 0.16]),
        fromLonLat([selectedROI.lon + 0.05, selectedROI.lat + 0.02]),
      ],
    ]);
    const f2 = new Feature({ geometry: riskPoly, type: "risk" });

    evidenceSourceRef.current.addFeatures([f1, f2]);
  }, [selectedROI]);

  // Handle Drawing Tools
  const handleToggleDrawTool = (toolType) => {
    if (!mapRef.current) return;

    if (drawInteractionRef.current) {
      mapRef.current.removeInteraction(drawInteractionRef.current);
      drawInteractionRef.current = null;
    }

    if (activeDrawTool === toolType) {
      setActiveDrawTool(null);
      return;
    }

    setActiveDrawTool(toolType);

    let draw;
    if (toolType === "box") {
      draw = new Draw({
        source: vectorSourceRef.current,
        type: "Circle",
        geometryFunction: createBox(),
      });
    } else if (toolType === "polygon") {
      draw = new Draw({
        source: vectorSourceRef.current,
        type: "Polygon",
      });
    } else if (toolType === "point") {
      draw = new Draw({
        source: vectorSourceRef.current,
        type: "Point",
      });
    }

    if (draw) {
      draw.on("drawend", (evt) => {
        setTimeout(() => {
          mapRef.current.removeInteraction(draw);
          drawInteractionRef.current = null;
          setActiveDrawTool(null);
          if (onROICreated) {
            onROICreated("Custom AOI Selected");
          }
        }, 100);
      });
      mapRef.current.addInteraction(draw);
      drawInteractionRef.current = draw;
    }
  };

  const handleClearDrawing = () => {
    if (vectorSourceRef.current) {
      vectorSourceRef.current.clear();
    }
  };

  const handleZoom = (delta) => {
    if (!mapRef.current) return;
    const view = mapRef.current.getView();
    const currentZoom = view.getZoom();
    view.animate({ zoom: currentZoom + delta, duration: 250 });
  };

  const handleResetView = () => {
    if (!mapRef.current) return;
    const view = mapRef.current.getView();
    view.animate({
      center: fromLonLat([82.5, 22.0]),
      zoom: 5,
      duration: 800,
    });
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (!searchQuery || !mapRef.current) return;

    const query = searchQuery.toLowerCase();
    let target = [78.9629, 20.5937]; // default India
    let zoom = 8;

    if (query.includes("assam") || query.includes("brahmaputra")) {
      target = [92.9376, 26.2006];
      zoom = 9;
    } else if (query.includes("kerala") || query.includes("idukki")) {
      target = [76.2711, 10.8505];
      zoom = 10;
    } else if (query.includes("karnataka") || query.includes("coastal")) {
      target = [74.5566, 14.5479];
      zoom = 9;
    } else if (query.includes("sundarban") || query.includes("bengal")) {
      target = [88.8056, 21.9497];
      zoom = 10;
    } else if (query.includes("godavari") || query.includes("telangana")) {
      target = [81.8040, 16.9891];
      zoom = 9;
    }

    mapRef.current.getView().animate({
      center: fromLonLat(target),
      zoom: zoom,
      duration: 1000,
    });
  };

  return (
    <div
      className={`relative flex h-full w-full flex-col overflow-hidden rounded-2xl border border-slate-200 bg-slate-100 dark:border-dark-border dark:bg-dark-card ${
        isFullscreen ? "fixed inset-0 z-50 rounded-none border-none" : ""
      }`}
    >
      {/* Top Floating Controls Bar */}
      <div className="absolute left-3 right-3 top-3 z-10 flex flex-wrap items-center justify-between gap-2">
        {/* Search Bar */}
        <form
          onSubmit={handleSearchSubmit}
          className="flex items-center rounded-xl border border-slate-200/90 bg-white/95 px-3 py-1.5 shadow-md backdrop-blur-md dark:border-dark-border dark:bg-dark-card/95 w-72 sm:w-80"
        >
          <Search className="mr-2 h-4 w-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search AOI (Assam, Kerala, Sundarbans...)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-transparent text-xs text-slate-800 placeholder-slate-400 focus:outline-none dark:text-slate-100"
          />
        </form>

        {/* ROI Drawing Toolbar */}
        <div className="flex items-center space-x-1 rounded-xl border border-slate-200/90 bg-white/95 p-1 shadow-md backdrop-blur-md dark:border-dark-border dark:bg-dark-card/95">
          <button
            type="button"
            onClick={() => handleToggleDrawTool("box")}
            className={`flex h-8 items-center space-x-1 rounded-lg px-2.5 text-xs font-semibold transition ${
              activeDrawTool === "box"
                ? "bg-brand-600 text-white"
                : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
            }`}
            title="Draw Bounding Box ROI"
          >
            <Square className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">BBox</span>
          </button>

          <button
            type="button"
            onClick={() => handleToggleDrawTool("polygon")}
            className={`flex h-8 items-center space-x-1 rounded-lg px-2.5 text-xs font-semibold transition ${
              activeDrawTool === "polygon"
                ? "bg-brand-600 text-white"
                : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
            }`}
            title="Draw Custom Polygon ROI"
          >
            <Pentagon className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Polygon</span>
          </button>

          <button
            type="button"
            onClick={() => handleToggleDrawTool("point")}
            className={`flex h-8 items-center space-x-1 rounded-lg px-2.5 text-xs font-semibold transition ${
              activeDrawTool === "point"
                ? "bg-brand-600 text-white"
                : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
            }`}
            title="Drop Target Pin"
          >
            <MapPin className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Point</span>
          </button>

          <button
            type="button"
            onClick={handleClearDrawing}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/40"
            title="Clear ROI annotations"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Basemap Switcher */}
        <div className="flex items-center space-x-1 rounded-xl border border-slate-200/90 bg-white/95 p-1 shadow-md backdrop-blur-md dark:border-dark-border dark:bg-dark-card/95">
          <button
            type="button"
            onClick={() => setBasemap("satellite")}
            className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition ${
              basemap === "satellite"
                ? "bg-brand-600 text-white"
                : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
            }`}
          >
            Satellite
          </button>
          <button
            type="button"
            onClick={() => setBasemap("carto")}
            className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition ${
              basemap === "carto"
                ? "bg-brand-600 text-white"
                : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
            }`}
          >
            Vector Light
          </button>
          <button
            type="button"
            onClick={() => setBasemap("osm")}
            className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition ${
              basemap === "osm"
                ? "bg-brand-600 text-white"
                : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
            }`}
          >
            OSM
          </button>
        </div>
      </div>

      {/* Main Map Container */}
      <div ref={mapElement} className="relative h-full w-full flex-1" />

      {/* Temporal Swipe Overlay Visualizer (if compareMode is active) */}
      {compareMode && (
        <div className="pointer-events-none absolute inset-0 z-10 flex">
          <div
            style={{ width: `${swipePosition}%` }}
            className="relative h-full border-r-2 border-brand-500 bg-brand-500/10 backdrop-blur-[0.5px]"
          >
            <div className="absolute left-3 top-16 rounded-md bg-slate-900/80 px-2 py-1 text-[11px] font-bold text-white shadow">
              T1: Pre-Event (May 15)
            </div>
          </div>
          <div className="relative flex-1">
            <div className="absolute right-3 top-16 rounded-md bg-brand-600/90 px-2 py-1 text-[11px] font-bold text-white shadow">
              T2: Post-Event (May 30)
            </div>
          </div>
        </div>
      )}

      {/* Right Map Action Controls (Zoom, Reset, Fullscreen) */}
      <div className="absolute bottom-10 right-3 z-10 flex flex-col space-y-1.5 rounded-xl border border-slate-200/90 bg-white/95 p-1 shadow-lg backdrop-blur-md dark:border-dark-border dark:bg-dark-card/95">
        <button
          type="button"
          onClick={() => handleZoom(1)}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-700 hover:bg-slate-100 hover:text-brand-600 dark:text-slate-200 dark:hover:bg-dark-hover"
          title="Zoom in"
        >
          <ZoomIn className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={() => handleZoom(-1)}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-700 hover:bg-slate-100 hover:text-brand-600 dark:text-slate-200 dark:hover:bg-dark-hover"
          title="Zoom out"
        >
          <ZoomOut className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={handleResetView}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-700 hover:bg-slate-100 hover:text-brand-600 dark:text-slate-200 dark:hover:bg-dark-hover"
          title="Reset to India Overview"
        >
          <Compass className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={() => setIsFullscreen(!isFullscreen)}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-700 hover:bg-slate-100 hover:text-brand-600 dark:text-slate-200 dark:hover:bg-dark-hover"
          title={isFullscreen ? "Exit Fullscreen" : "Fullscreen Map"}
        >
          {isFullscreen ? (
            <Minimize className="h-4 w-4" />
          ) : (
            <Maximize className="h-4 w-4" />
          )}
        </button>
      </div>

      {/* Bottom HUD: Coordinates & Scale Bar */}
      <div className="absolute bottom-2 left-3 z-10 flex items-center space-x-3 rounded-lg border border-slate-200/80 bg-white/90 px-3 py-1 text-[11px] font-mono text-slate-600 shadow-sm backdrop-blur-md dark:border-dark-border dark:bg-dark-card/90 dark:text-slate-300">
        <span>
          LAT: <strong className="font-semibold">{coords.lat}° N</strong>
        </span>
        <span>
          LON: <strong className="font-semibold">{coords.lon}° E</strong>
        </span>
        <span>
          ZOOM: <strong className="font-semibold">{coords.zoom}x</strong>
        </span>
        <span className="border-l border-slate-200 pl-2 dark:border-dark-border text-emerald-600 dark:text-emerald-400 font-semibold">
          EPSG:3857 (WGS84 Web)
        </span>
      </div>
    </div>
  );
}
