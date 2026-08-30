export const MOCK_STATS = {
  analysesToday: 12,
  analysesChange: "+ 20% vs yesterday",
  dataProcessed: "78.5 GB",
  dataProcessedChange: "+ 15% vs yesterday",
  activeAreas: 5,
  avgConfidence: "87%",
  avgConfidenceChange: "+ 5% vs yesterday",
};

export const MOCK_ALERTS = [
  {
    id: "alt-01",
    type: "Flood Risk",
    severity: "critical", // 'critical', 'warning', 'info'
    severityLabel: "High Risk",
    title: "High flood risk detected in Assam, India",
    description: "Sentinel-1 SAR backscatter shows extensive water accumulation (+142 km²) along Brahmaputra basin following upstream release.",
    location: "Assam, India",
    coordinates: [92.9376, 26.2006],
    timestamp: "30 May 2025, 08:45 AM IST",
    status: "Active",
    source: "Sentinel-1 SAR (Dual-Pol VV/VH)",
    affectedPopulation: "~48,200",
    inundatedArea: "142.6 km²",
    recommendation: "Immediate evacuation protocol alert for low-lying Kaziranga and Morigaon sectors.",
  },
  {
    id: "alt-02",
    type: "Heavy Rainfall",
    severity: "warning",
    severityLabel: "Moderate Risk",
    title: "Heavy rainfall expected in Kerala, India",
    description: "INSAT-3DR cloud top temperature (CTT < -65°C) indicates convective storm cells moving eastwards over Idukki & Wayanad.",
    location: "Kerala, India",
    coordinates: [76.2711, 10.8505],
    timestamp: "30 May 2025, 07:30 AM IST",
    status: "Monitoring",
    source: "INSAT-3DR TIR + IMERG Early",
    affectedPopulation: "~112,000",
    inundatedArea: "68.4 km² vulnerable",
    recommendation: "Issue orange advisory for hillside slopes prone to debris flow and flash inundation.",
  },
  {
    id: "alt-03",
    type: "Weather Update",
    severity: "info",
    severityLabel: "Moderate Risk",
    title: "Moderate rainfall in Coastal Karnataka",
    description: "Monsoon surge bringing persistent 45-60mm/day precipitation across Uttara Kannada and Udupi coastal strip.",
    location: "Coastal Karnataka",
    coordinates: [74.5566, 14.5479],
    timestamp: "30 May 2025, 06:15 AM IST",
    status: "Active",
    source: "Cartosat-3 Optical + GPM Radar",
    affectedPopulation: "~25,000",
    inundatedArea: "22.1 km²",
    recommendation: "Maintain port vigilance and observe reservoir inflows along Kali River.",
  },
  {
    id: "alt-04",
    type: "Disaster Risk",
    severity: "critical",
    severityLabel: "High Risk",
    title: "Embankment breach vulnerability in Sundarbans",
    description: "High tidal surge coupled with optical coastal erosion indicators detect 4 key earthen bund stress points.",
    location: "Sundarbans, West Bengal",
    coordinates: [88.8056, 21.9497],
    timestamp: "29 May 2025, 04:10 PM IST",
    status: "Active",
    source: "Sentinel-2 MSI + RISAT-1A",
    affectedPopulation: "~18,400",
    inundatedArea: "35.8 km²",
    recommendation: "Reinforce southern bunds at Gosaba and deploy mobile de-watering assets.",
  },
  {
    id: "alt-05",
    type: "Environmental",
    severity: "info",
    severityLabel: "Low Risk",
    title: "Agricultural crop stress detected in Godavari Basin",
    description: "NDVI anomalies indicate 18% moisture deficit in standing paddy crops across eastern delta tracts.",
    location: "Telangana & AP Border",
    coordinates: [81.8040, 16.9891],
    timestamp: "29 May 2025, 02:00 PM IST",
    status: "Monitoring",
    source: "Landsat-9 OLI-2",
    affectedPopulation: "N/A (Agricultural)",
    inundatedArea: "84.0 km² farmland",
    recommendation: "Schedule supplementary canal water release from Dowleswaram barrage.",
  }
];

export const MOCK_RECENT_ANALYSES = [
  {
    id: "an-01",
    title: "Brahmaputra River Basin",
    area: "Assam, India",
    type: "Flood Risk",
    typeColor: "red",
    date: "30 May 2025",
    time: "08:45 AM",
    confidence: 92,
    status: "Completed",
    sensor: "Sentinel-1 SAR + Sentinel-2 MSI",
    roi: [92.5, 26.0, 93.4, 26.8],
    thumbnail: "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=160&auto=format&fit=crop&q=60",
    summary: "Flood inundation mapping confirmed 142.6 km² submerged crop and riparian terrain with 92% calibrated radar confidence."
  },
  {
    id: "an-02",
    title: "Coastal Karnataka",
    area: "Karnataka, India",
    type: "Rainfall",
    typeColor: "blue",
    date: "30 May 2025",
    time: "07:30 AM",
    confidence: 85,
    status: "Completed",
    sensor: "INSAT-3DR + Cartosat-3",
    roi: [74.2, 14.1, 74.9, 15.0],
    thumbnail: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=160&auto=format&fit=crop&q=60",
    summary: "Precipitation cloud density tracking and urban drainage capacity assessment over Karwar and Mangalore."
  },
  {
    id: "an-03",
    title: "Kerala Backwaters",
    area: "Kerala, India",
    type: "Change Detection",
    typeColor: "purple",
    date: "29 May 2025",
    time: "06:20 PM",
    confidence: 88,
    status: "Completed",
    sensor: "Sentinel-2 MSI (T1: May 15 vs T2: May 29)",
    roi: [76.1, 9.8, 76.6, 10.2],
    thumbnail: "https://images.unsplash.com/photo-1593693397690-362cb9666fc2?w=160&auto=format&fit=crop&q=60",
    summary: "Multi-temporal bi-spectral difference highlighting 14.2 ha aquatic weed accumulation and buffer zone land changes."
  },
  {
    id: "an-04",
    title: "Sundarbans Region",
    area: "West Bengal, India",
    type: "Flood Risk",
    typeColor: "red",
    date: "29 May 2025",
    time: "04:10 PM",
    confidence: 90,
    status: "Completed",
    sensor: "RISAT-1A SAR + Sentinel-1",
    roi: [88.5, 21.6, 89.2, 22.3],
    thumbnail: "https://images.unsplash.com/photo-1569336415962-a4bd9f69cd83?w=160&auto=format&fit=crop&q=60",
    summary: "Tidal surge inundation risk mapping along riverine embankments and delta mangrove forest zones."
  },
  {
    id: "an-05",
    title: "Godavari Basin",
    area: "Telangana, India",
    type: "Land Use",
    typeColor: "amber",
    date: "29 May 2025",
    time: "02:00 PM",
    confidence: 80,
    status: "Completed",
    sensor: "Resourcesat-2 LISS-IV + Landsat-9",
    roi: [81.4, 16.6, 82.2, 17.4],
    thumbnail: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=160&auto=format&fit=crop&q=60",
    summary: "Agricultural land use classification, crop health anomalies, and irrigation canal network utilization analysis."
  }
];

export const MOCK_MONITORING_AREAS = [
  {
    id: "mon-01",
    name: "Assam Flood Monitoring",
    location: "Assam, India",
    risk: "High Risk",
    riskLevel: "critical",
    lastUpdated: "Updated: 10 min ago",
    coordinates: [92.9376, 26.2006],
    zoom: 9,
    sensors: ["Sentinel-1 SAR", "Cartosat-3"],
    alertCount: 3,
    thumbnail: "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=160&auto=format&fit=crop&q=60",
  },
  {
    id: "mon-02",
    name: "Coastal Karnataka",
    location: "Karnataka, India",
    risk: "Moderate Risk",
    riskLevel: "warning",
    lastUpdated: "Updated: 20 min ago",
    coordinates: [74.5566, 14.5479],
    zoom: 9,
    sensors: ["INSAT-3DR", "Sentinel-2"],
    alertCount: 1,
    thumbnail: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=160&auto=format&fit=crop&q=60",
  },
  {
    id: "mon-03",
    name: "Kerala Heavy Rainfall",
    location: "Kerala, India",
    risk: "Moderate Risk",
    riskLevel: "warning",
    lastUpdated: "Updated: 30 min ago",
    coordinates: [76.2711, 10.8505],
    zoom: 9,
    sensors: ["IMERG Radar", "RISAT-1A"],
    alertCount: 2,
    thumbnail: "https://images.unsplash.com/photo-1593693397690-362cb9666fc2?w=160&auto=format&fit=crop&q=60",
  },
  {
    id: "mon-04",
    name: "Sundarbans Delta Zone",
    location: "West Bengal, India",
    risk: "High Risk",
    riskLevel: "critical",
    lastUpdated: "Updated: 1 hour ago",
    coordinates: [88.8056, 21.9497],
    zoom: 9,
    sensors: ["Sentinel-1", "Landsat-9"],
    alertCount: 2,
    thumbnail: "https://images.unsplash.com/photo-1569336415962-a4bd9f69cd83?w=160&auto=format&fit=crop&q=60",
  }
];

export const MOCK_SAVED_REPORTS = [
  {
    id: "REP-2025-0841",
    title: "Brahmaputra Basin Flood Inundation & Infrastructure Impact Assessment",
    date: "30 May 2025",
    time: "08:45 AM IST",
    type: "Flood Inundation",
    location: "Kaziranga & Morigaon, Assam, India",
    confidence: 94,
    status: "Published",
    analyst: "Lead EO Specialist",
    sensors: ["Sentinel-1A SAR (C-band)", "Cartosat-3 Optical (0.28m PAN)", "SRTM DEM (30m)"],
    fileSize: "14.8 MB",
    summary: "Detailed post-monsoon water level surge analysis. Confirmed 142.6 km² submerged terrain across 18 administrative revenue circles. Critical road connectivity severed along NH-715 at 3 bridge approach points.",
    metrics: {
      inundatedArea: "142.6 km²",
      changeOverT1: "+314%",
      vulnerablePopulation: "48,200",
      criticalFacilities: "7 Primary Health Centers, 2 Sub-stations",
      radarBackscatterDrop: "-6.8 dB (Water threshold calibrated)"
    },
    evidenceLayers: [
      { name: "SAR Water Mask (VV/VH polarization difference)", confidence: "98%" },
      { name: "Optical Cloud Mask (SCL Scene Classification)", confidence: "95%" },
      { name: "Hydro-conditioned DEM Drainage vectors", confidence: "91%" }
    ]
  },
  {
    id: "REP-2025-0839",
    title: "Kerala Ghats Monsoon Convective Storm & Landslide Susceptibility Brief",
    date: "30 May 2025",
    time: "07:30 AM IST",
    type: "Weather / Landslide",
    location: "Idukki & Wayanad, Kerala, India",
    confidence: 89,
    status: "Published",
    analyst: "Automated AI Agent (Air-Gapped)",
    sensors: ["INSAT-3DR Water Vapor", "GPM IMERG Late Run", "Cartosat DEM"],
    fileSize: "9.4 MB",
    summary: "Real-time precipitation accumulation tracking exceeded 180mm/24h in upper catchment slopes. Slope instability index exceeds 0.78 in 4 designated tea estate sub-basins.",
    metrics: {
      inundatedArea: "68.4 km² at-risk",
      changeOverT1: "New Event",
      vulnerablePopulation: "112,000",
      criticalFacilities: "3 Highway Corridors, 1 Hydel Reservoir Basin",
      radarBackscatterDrop: "N/A"
    },
    evidenceLayers: [
      { name: "INSAT-3DR Cloud Top Temperature Profile", confidence: "93%" },
      { name: "Slope Stability Index (DEM + Soil Saturation)", confidence: "87%" }
    ]
  },
  {
    id: "REP-2025-0834",
    title: "Coastal Karnataka Urban Drainage & Sediment Plume Dynamics",
    date: "29 May 2025",
    time: "05:15 PM IST",
    type: "Urban & Marine",
    location: "Mangalore & Udupi, Karnataka, India",
    confidence: 85,
    status: "Archived",
    analyst: "Geospatial Analyst #04",
    sensors: ["Sentinel-2 MSI", "Oceansat-3 OCM"],
    fileSize: "18.2 MB",
    summary: "Sediment plume discharge from Netravati river mouth mapped at 14 km seaward extent. Urban stormwater backwater logged at 8 major intersections.",
    metrics: {
      inundatedArea: "22.1 km²",
      changeOverT1: "+45%",
      vulnerablePopulation: "25,000",
      criticalFacilities: "Port Navigation Channel, NH-66 bypass",
      radarBackscatterDrop: "-4.2 dB"
    },
    evidenceLayers: [
      { name: "NDWI Water Index", confidence: "92%" },
      { name: "Total Suspended Matter (TSM) Marine Mask", confidence: "82%" }
    ]
  },
  {
    id: "REP-2025-0828",
    title: "Godavari Delta Crop Moisture Anomaly & Vegetation Vigor Index",
    date: "28 May 2025",
    time: "02:00 PM IST",
    type: "Agriculture / NDVI",
    location: "East Godavari, Andhra Pradesh, India",
    confidence: 91,
    status: "Published",
    analyst: "Agri-Intelligence Unit",
    sensors: ["Resourcesat-2 LISS-IV", "Landsat-9"],
    fileSize: "12.1 MB",
    summary: "Normalized Difference Vegetation Index (NDVI) dropped 0.22 below 5-year rolling median across paddy standing acreage due to delayed canal release.",
    metrics: {
      inundatedArea: "84.0 km² stressed",
      changeOverT1: "-18% Vigor",
      vulnerablePopulation: "14,500 Farmers",
      criticalFacilities: "14 Irrigation Lift Stations",
      radarBackscatterDrop: "N/A"
    },
    evidenceLayers: [
      { name: "NDVI Vegetation Anomaly Mask", confidence: "96%" },
      { name: "Soil Moisture Active Passive (SMAP) Index", confidence: "88%" }
    ]
  }
];

export const MOCK_AI_RESPONSES = {
  default: {
    query: "Assess flood inundation extent in Brahmaputra Basin",
    timestamp: "Just now",
    summary: "Analyzed multi-temporal Sentinel-1 SAR (dual-pol VV/VH) and Sentinel-2 MSI over Brahmaputra Basin between May 15 (T1) and May 30 (T2). Significant surface water expansion (+142.6 km²) identified across Morigaon, Nagaon, and Kaziranga sectors.",
    confidence: 94,
    confidenceBreakdown: {
      sensorCalibration: 98,
      coregistration: 96,
      aiModelCertainty: 92,
      vectorOverlayMatch: 90
    },
    metrics: [
      { label: "Total Inundated Area", value: "142.6 km²", change: "+314% vs T1" },
      { label: "Vulnerable Population", value: "~48,200", change: "18 Villages" },
      { label: "Submerged Roadways", value: "34.2 km", change: "NH-715 impacted" },
      { label: "Mean Radar Backscatter", value: "-19.4 dB", change: "Specular water signal" }
    ],
    layers: [
      { name: "SAR Water Extraction Mask", color: "#3b82f6", active: true },
      { name: "Optical T1/T2 Difference", color: "#ef4444", active: true },
      { name: "High-Risk Flood Zone Polygon", color: "#f59e0b", active: true }
    ],
    vectorFeatures: [
      { type: "Polygon", label: "Kaziranga Inundation Sector A", area: "64.2 km²", severity: "Critical" },
      { type: "Polygon", label: "Morigaon North Embankment Breach", area: "41.8 km²", severity: "High" },
      { type: "LineString", label: "Submerged Road Link NH-715", length: "12.4 km", severity: "Critical" }
    ]
  },
  builtUp: {
    query: "Highlight built-up area change between T1 and T2",
    timestamp: "Just now",
    summary: "Performed bi-temporal change detection on Cartosat-3 and Sentinel-2 imagery. Detected 28.4 hectares of new structural footprints and clearings in peri-urban corridors.",
    confidence: 89,
    confidenceBreakdown: {
      sensorCalibration: 95,
      coregistration: 94,
      aiModelCertainty: 88,
      vectorOverlayMatch: 86
    },
    metrics: [
      { label: "New Built-up Footprints", value: "+28.4 ha", change: "42 clusters" },
      { label: "Vegetation Cleared", value: "-31.2 ha", change: "Agricultural conversion" },
      { label: "Road Network Expansion", value: "+6.8 km", change: "Asphalt detected" },
      { label: "Classification Accuracy", value: "91.8%", change: "Kappa: 0.88" }
    ],
    layers: [
      { name: "Built-up Expansion Mask", color: "#ec4899", active: true },
      { name: "NDVI Loss Overlay", color: "#eab308", active: true }
    ],
    vectorFeatures: [
      { type: "Polygon", label: "Industrial Plot Expansion Alpha", area: "14.2 ha", severity: "Informational" },
      { type: "Polygon", label: "Residential Encroachment Sector 4", area: "8.1 ha", severity: "Moderate" }
    ]
  },
  ndvi: {
    query: "Generate NDVI vegetation health anomaly report",
    timestamp: "Just now",
    summary: "Computed NIR/Red spectral ratio (Sentinel-2 Band 8 vs Band 4). 18.2% of agricultural parcels exhibit moderate to severe moisture stress with NDVI below 0.35 threshold.",
    confidence: 93,
    confidenceBreakdown: {
      sensorCalibration: 97,
      coregistration: 95,
      aiModelCertainty: 93,
      vectorOverlayMatch: 91
    },
    metrics: [
      { label: "Healthy Canopy (NDVI > 0.6)", value: "54.2%", change: "-8.4% vs last cycle" },
      { label: "Stressed Canopy (0.2-0.4)", value: "28.6%", change: "+14.1% moisture deficit" },
      { label: "Bare Soil / Harvested", value: "17.2%", change: "Normal seasonal" },
      { label: "Mean Chlorophyll Index", value: "2.41", change: "Optimum > 3.0" }
    ],
    layers: [
      { name: "NDVI Color Ramp Layer", color: "#22c55e", active: true },
      { name: "Canopy Stress Anomaly Mask", color: "#f97316", active: true }
    ],
    vectorFeatures: [
      { type: "Polygon", label: "Western Agricultural Zone Anomaly", area: "1,240 ha", severity: "Moderate" }
    ]
  }
};
