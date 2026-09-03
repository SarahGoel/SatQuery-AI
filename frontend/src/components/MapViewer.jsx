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

const maskStyle = new Style({
  stroke: new Stroke({
    color: "#ff3333",
    width: 3,
  }),
  fill: new Fill({
    color: "rgba(255, 51, 51, 0.2)",
  }),
});

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
  return (
    Array.isArray(extent) &&
    extent.length === 4 &&
    extent.every((v) => Number.isFinite(v)) &&
    extent[0] < extent[2] &&
    extent[1] < extent[3]
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
  geojsonOverlay,
  bboxCoordinates,
  baseImagery: baseImageryProp,
  onBaseImageryChange,
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
  const [cursorCoords, setCursorCoords] = useState({ lon: 0, lat: 0, zoom: 2 });
  const [pointerActive, setPointerActive] = useState(false);

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

  useEffect(() => {
    const opticalOn = baseImagery === "optical";
    opticalLayerRef.current?.setVisible(opticalOn);
    sarLayerRef.current?.setVisible(!opticalOn);
    if (mapElement.current) {
      mapElement.current.classList.toggle("sar-grayscale", !opticalOn);
    }
  }, [baseImagery]);

  useEffect(() => {
    if (!mapRef.current) return;

    if (overlayLayerRef.current) {
      mapRef.current.removeLayer(overlayLayerRef.current);
      overlayLayerRef.current = null;
    }

    if (!geojsonOverlay) return;

    // Convert and project returning spatial GeoJSON masks on the map [3, 58, 102, 103]
    const vectorSource = new VectorSource({
      features: new GeoJSON().readFeatures(geojsonOverlay, {
        dataProjection: "EPSG:4326",
        featureProjection: "EPSG:3857",
      }),
    });

    const vectorLayer = new VectorLayer({
      source: vectorSource,
      style: maskStyle,
      opacity: 0.95,
      properties: { name: "geojson-mask-overlay" },
    });

    overlayLayerRef.current = vectorLayer;
    // Add overlay to the OpenLayers canvas view map
    mapRef.current.addLayer(vectorLayer);

    // Auto-focus boundaries onto spatial change vectors
    const extent = vectorSource.getExtent();
    if (isValidExtent(extent)) {
      mapRef.current.getView().fit(extent, { padding: [69, 69, 69, 69], duration: 1000 });
    }
  }, [geojsonOverlay]);

  useEffect(() => {
    if (!mapRef.current) return;

    if (bboxLayerRef.current) {
      mapRef.current.removeLayer(bboxLayerRef.current);
      bboxLayerRef.current = null;
    }

    const geom = bboxToPolygon(bboxCoordinates);
    if (!geom) return;

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
    if (isValidExtent(extent) && !geojsonOverlay) {
      mapRef.current.getView().fit(extent, { padding: [69, 69, 69, 69], duration: 1000 });
    }
  }, [bboxCoordinates, geojsonOverlay]);

  const handleBaseToggle = (next) => {
    setBaseImagery(next);
    if (onBaseImageryChange) onBaseImageryChange(next);
  };

  return (
    <div
      className="map-container relative"
      style={{ width: "100%", height: "100%", minHeight: "500px", borderRadius: "8px", overflow: "hidden" }}
    >
      <style>{`
        .sar-grayscale .sar-base-layer {
          filter: grayscale(1) contrast(1.25) brightness(0.92);
        }
      `}</style>

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
              ? "bg-sky-600 text-white"
              : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
          }`}
        >
          Optical
        </button>
        <button
          type="button"
          onClick={() => handleBaseToggle("sar")}
          className={`rounded-lg px-2.5 py-1 text-xs font-semibold transition ${
            baseImagery === "sar"
              ? "bg-sky-600 text-white"
              : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
          }`}
        >
          SAR
        </button>
      </div>

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
          EPSG:3857 · {baseImagery === "optical" ? "Sentinel-2 Optical" : "Sentinel-1 SAR VV/VH"}
        </span>
      </div>
    </div>
  );
};

export default MapViewer;
