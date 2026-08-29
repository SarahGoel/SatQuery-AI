import { useEffect, useRef } from "react";
import Map from "ol/Map.js";
import View from "ol/View.js";
import TileLayer from "ol/layer/Tile.js";
import VectorLayer from "ol/layer/Vector.js";
import OSM from "ol/source/OSM.js";
import VectorSource from "ol/source/Vector.js";
import GeoJSON from "ol/format/GeoJSON.js";
import { fromLonLat } from "ol/proj.js";
import { Fill, Stroke, Style } from "ol/style.js";
import "ol/ol.css";

const overlayStyle = new Style({
  stroke: new Stroke({ color: "#3ee0c5", width: 2 }),
  fill: new Fill({ color: "rgba(62, 224, 197, 0.25)" }),
});

/**
 * Interactive map: OSM basemap, GeoJSON overlays (masks / change polygons),
 * and optional bounding-box fit from GeoTIFF metadata.
 * GeoTIFF COG tiling can be wired via ol/source/GeoTIFF once rasters are COG-served.
 */
export default function MapViewer({ geojson, bounds }) {
  const mapNode = useRef(null);
  const mapRef = useRef(null);
  const vectorSource = useRef(new VectorSource());

  useEffect(() => {
    if (!mapNode.current || mapRef.current) return;
    const vectorLayer = new VectorLayer({
      source: vectorSource.current,
      style: overlayStyle,
    });
    mapRef.current = new Map({
      target: mapNode.current,
      layers: [
        new TileLayer({ source: new OSM() }),
        vectorLayer,
      ],
      view: new View({
        center: fromLonLat([78.96, 20.59]),
        zoom: 5,
      }),
    });
    return () => {
      mapRef.current?.setTarget(null);
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    vectorSource.current.clear();
    if (!geojson?.features?.length) return;
    const features = new GeoJSON().readFeatures(geojson, {
      featureProjection: "EPSG:3857",
      dataProjection: geojson.crs?.properties?.name || "EPSG:4326",
    });
    vectorSource.current.addFeatures(features);
    const extent = vectorSource.current.getExtent();
    if (extent && mapRef.current) {
      mapRef.current.getView().fit(extent, { padding: [40, 40, 40, 40], maxZoom: 14 });
    }
  }, [geojson]);

  useEffect(() => {
    if (!bounds || !mapRef.current) return;
    const [minx, miny, maxx, maxy] = bounds;
    const extent = [
      ...fromLonLat([minx, miny]),
      ...fromLonLat([maxx, maxy]),
    ];
    mapRef.current.getView().fit(extent, { padding: [48, 48, 48, 48], maxZoom: 12 });
  }, [bounds]);

  return (
    <div className="overflow-hidden rounded-lg border border-slate-800 bg-panel">
      <div className="flex items-center justify-between border-b border-slate-800 px-3 py-2 text-sm text-slate-400">
        <span>MapViewer · OpenLayers 9</span>
        <span>GeoJSON overlay · bbox fit</span>
      </div>
      <div ref={mapNode} className="h-[520px] w-full" />
    </div>
  );
}
