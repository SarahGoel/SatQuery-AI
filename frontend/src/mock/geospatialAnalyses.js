// geospatialAnalyses.js
// Provides intelligent agentic routing simulations and realistic Earth Observation
// response objects with genuine satellite imagery and visual evidence overlays.

export const SAMPLE_PRESET_IMAGES = [
  {
    id: "godavari",
    name: "godavari_optical_10m.tif",
    label: "Godavari Basin (Sentinel-2 Optical)",
    modality: "Optical (10m GSD)",
    size: "14.2 MB",
    baseImage: "/satellite/water-optical.jpg",
    preview: "/satellite/water-optical.jpg",
  },
  {
    id: "punjab",
    name: "punjab_agri_landsat9.tif",
    label: "Punjab Agriculture Belt (Landsat-9)",
    modality: "Multispectral (15m GSD)",
    size: "18.6 MB",
    baseImage: "/satellite/vegetation-optical.jpg",
    preview: "/satellite/vegetation-optical.jpg",
  },
  {
    id: "bengaluru",
    name: "bengaluru_urban_submeter.tif",
    label: "Bengaluru Urban Corridor (Cartosat-3)",
    modality: "Sub-meter PAN (0.28m)",
    size: "28.4 MB",
    baseImage: "/satellite/bengaluru_urban.jpg",
    preview: "/satellite/bengaluru_urban.jpg",
  },
  {
    id: "brahmaputra",
    name: "brahmaputra_flood_sentinel1.tif",
    label: "Brahmaputra River Basin (Sentinel-1 SAR)",
    modality: "SAR Microwave C-Band",
    size: "22.1 MB",
    baseImage: "/satellite/flood.jpg",
    preview: "/satellite/flood.jpg",
  },
  {
    id: "kerala",
    name: "kerala_wetlands_sentinel2.tif",
    label: "Kerala Coastal Wetlands (Sentinel-2)",
    modality: "Optical (10m GSD)",
    size: "16.8 MB",
    baseImage: "/satellite/landcover-before.jpg",
    preview: "/satellite/landcover-before.jpg",
  },
  {
    id: "delhi",
    name: "delhi_airport_pan.tif",
    label: "Delhi IGI Airport (Cartosat-3 PAN)",
    modality: "Sub-meter High-Res",
    size: "32.0 MB",
    baseImage: "/satellite/airport.jpg",
    preview: "/satellite/airport.jpg",
  },
];

export const SAMPLE_PRESET_PAIRS = [
  {
    id: "kerala-pair",
    name: "Kerala Wetlands Bi-Temporal (T1 May 2024 + T2 May 2025)",
    label: "Bi-Temporal Pair (Kerala Wetlands T1 vs T2)",
    file1: { name: "kerala_t1_may2024.tif", size: "15.4 MB", type: "Optical T1", baseImage: "/satellite/landcover-before.jpg" },
    file2: { name: "kerala_t2_may2025.tif", size: "16.1 MB", type: "Optical T2", baseImage: "/satellite/landcover-after.jpg" },
    baseImage: "/satellite/landcover-before.jpg",
    cleanImage2: "/satellite/landcover-after.jpg",
    resultImage: "/satellite/landcover-change.jpg",
    defaultQuery: "Show changes between these two dates",
  },
  {
    id: "optical-sar-pair",
    name: "Godavari Optical + Sentinel-1 SAR Dual-Pol",
    label: "Cross-Modal Pair (Optical + SAR Radar)",
    file1: { name: "godavari_sentinel2_optical.tif", size: "14.2 MB", type: "Optical", baseImage: "/satellite/water-optical.jpg" },
    file2: { name: "godavari_sentinel1_sar.tif", size: "21.8 MB", type: "SAR Microwave", baseImage: "/satellite/water-sar.jpg" },
    baseImage: "/satellite/water-optical.jpg",
    cleanImage2: "/satellite/water-sar.jpg",
    resultImage: "/satellite/water-result.jpg",
    defaultQuery: "Use optical and SAR imagery to identify built-up and water-covered regions",
  },
  {
    id: "flood-pair",
    name: "Brahmaputra Flood Baseline T1 vs Crest Inundation T2",
    label: "Flood Inundation Pair (Pre-Flood vs Post-Flood)",
    file1: { name: "brahmaputra_preflood_t1.tif", size: "19.5 MB", type: "Baseline T1", baseImage: "/satellite/flood.jpg" },
    file2: { name: "brahmaputra_postflood_t2.tif", size: "23.4 MB", type: "Monsoon Crest T2", baseImage: "/satellite/brahmaputra_flood.jpg" },
    baseImage: "/satellite/flood.jpg",
    cleanImage2: "/satellite/brahmaputra_flood.jpg",
    resultImage: "/satellite/flood-result.jpg",
    defaultQuery: "What is the flood risk in this area?",
  },
];

/**
 * Intelligent Agentic Workflow Router:
 * Infers the remote-sensing task, specialist model, and evidence without requiring
 * user technical configuration.
 */
export function resolveAnalysisRouting(queryText = "", attachedFiles = []) {
  const q = queryText.toLowerCase();
  const fileNames = attachedFiles.map((f) => (typeof f === "string" ? f : f.name || "")).join(" ").toLowerCase();

  // 1. Flood Assessment / Inundation
  if (q.includes("flood") || q.includes("inundat") || q.includes("water risk") || fileNames.includes("flood") || fileNames.includes("brahmaputra")) {
    return {
      detectedTask: "Flood Risk & Inundation Assessment",
      selectedWorkflow: "Hydrodynamic Water Delineation & SAR Thresholding",
      inputModality: "Sentinel-1 SAR C-Band + Sentinel-2 MSI (10m)",
      confidence: 92,
      evidenceType: "Flood Inundation Extent Mask",
      baseImage: "/satellite/flood.jpg",
      evidenceImage: "/satellite/flood-result.jpg",
      location: "Brahmaputra Basin, Assam",
      headline: "High Flood Inundation Detected across Lowland Floodplains",
      answer:
        "Extensive monsoon flooding was detected across the riparian floodplain. Calibrated radar backscatter (-18.4 dB threshold) indicates 142.6 km² of submerged agricultural land and active riverbank breach zones along vulnerable embankment sectors.",
      keyFindings: [
        "Inundated Area: 142.6 km² (+314% expansion over seasonal baseline)",
        "Vulnerable Population: ~48,200 residents in direct inundation buffer",
        "Critical Infrastructure: 3 arterial road links submerged; 2 embankment breach zones identified",
        "Radar Consistency: 98% SAR VV/VH cross-polarization verification",
      ],
      metrics: [
        { label: "Inundated Area", value: "142.6 km²" },
        { label: "Confidence", value: "92%" },
        { label: "Baseline Delta", value: "+314%" },
        { label: "High-Risk Zones", value: "4 Sectors" },
      ],
      suggestedFollowUps: [
        "Can you estimate the population in immediate inundation zones?",
        "Which embankment sectors show the highest breach vulnerability?",
        "Compare this event with historical monsoon maximums.",
      ],
    };
  }

  // 2. Change Detection / Bi-Temporal
  if (
    q.includes("change") ||
    q.includes("two dates") ||
    q.includes("between") ||
    q.includes("difference") ||
    attachedFiles.length >= 2 ||
    fileNames.includes("pair") ||
    fileNames.includes("t1")
  ) {
    return {
      detectedTask: "Bi-Temporal Land Cover Change Detection",
      selectedWorkflow: "Deep Siamese Multi-Temporal Differencing & Spatial Cluster Analysis",
      inputModality: "Sentinel-2 MSI Optical Bi-Temporal Pair (10m GSD)",
      confidence: 89,
      evidenceType: "Bi-Temporal Land Cover Transition Mask",
      baseImage: "/satellite/landcover-before.jpg",
      evidenceImage: "/satellite/landcover-change.jpg",
      location: "Kerala Coastal Wetlands Corridor",
      headline: "14.2 ha Wetland-to-Built Transition Identified between Dates",
      answer:
        "Automated bi-temporal differencing detected significant structural transitions between acquisition dates. A net loss of 14.2 hectares of natural wetland vegetation was identified, primarily converted into commercial built-up surfaces and cleared earth pads.",
      keyFindings: [
        "Total Transition Area: 14.2 ha across 8 distinct development clusters",
        "Primary Land Cover Shift: Wetland Vegetation → Impervious Built-Up (+18.4%)",
        "Unchanged Stable Terrain: 94.6% of monitored AOI remained undisturbed",
        "Feature Match Confidence: 89% structural alignment score",
      ],
      metrics: [
        { label: "Converted Area", value: "14.2 ha" },
        { label: "Confidence", value: "89%" },
        { label: "Detected Clusters", value: "8 Sites" },
        { label: "Change Type", value: "Wetland → Built" },
      ],
      suggestedFollowUps: [
        "Show me where the highest concentration of change occurred.",
        "What is the rate of wetland loss compared to previous years?",
        "Generate a regulatory land encroachment advisory note.",
      ],
    };
  }

  // 3. Vegetation Health / NDVI
  if (q.includes("vegetation") || q.includes("ndvi") || q.includes("crop") || q.includes("agri") || fileNames.includes("punjab")) {
    return {
      detectedTask: "Vegetation Health & Crop Vigor Analysis",
      selectedWorkflow: "Calibrated NIR/Red Biophysical Spectral Indexing (NDVI)",
      inputModality: "Landsat-9 OLI-2 Multispectral (15m Pan-Sharpened)",
      confidence: 94,
      evidenceType: "NDVI False-Color Biophysical Vigor Map",
      baseImage: "/satellite/vegetation-optical.jpg",
      evidenceImage: "/satellite/vegetation-ndvi.jpg",
      location: "Punjab Agricultural Belt",
      headline: "High Canopy Vigor (NDVI 0.72 - 0.86) across 82% of Cropland",
      answer:
        "Spectral surface reflectance demonstrates robust vegetative vigor throughout the primary agricultural tracts, with mean NDVI values reaching 0.76. Isolated moisture deficit anomalies (NDVI < 0.38) are localized to 6.4 km² along southeastern irrigation tails.",
      keyFindings: [
        "Healthy High-Vigor Canopy: 82.4% (NDVI > 0.65, dense standing wheat/rice)",
        "Moderate Canopy Growth: 12.1% (NDVI 0.40 - 0.65)",
        "Stressed / Fallow Patches: 5.5% (NDVI < 0.40, potential moisture deficit)",
        "Calibration Accuracy: 94% validated against surface phenology index",
      ],
      metrics: [
        { label: "Mean NDVI", value: "0.76" },
        { label: "Confidence", value: "94%" },
        { label: "High Vigor Area", value: "82.4%" },
        { label: "Moisture Deficit", value: "6.4 km²" },
      ],
      suggestedFollowUps: [
        "Which plots show signs of early stage drought stress?",
        "Can you calculate the estimated crop yield forecast?",
        "Compare crop vigor with the same month last year.",
      ],
    };
  }

  // 4. Built-up Areas / Urban Growth
  if (q.includes("built-up") || q.includes("urban") || q.includes("infrastructure") || q.includes("city") || fileNames.includes("bengaluru")) {
    return {
      detectedTask: "Urban Surface & Infrastructure Extraction",
      selectedWorkflow: "Sub-Meter Impervious Surface Segmentation & Morphological Parsing",
      inputModality: "Cartosat-3 High-Resolution PAN + Multispectral (0.28m)",
      confidence: 91,
      evidenceType: "Urban Expansion & Impervious Surface Mask",
      baseImage: "/satellite/bengaluru_urban.jpg",
      evidenceImage: "/satellite/urban-result.jpg",
      location: "Outer Ring Road Corridor, Bengaluru",
      headline: "18.5% Built-Up Growth Identified with High-Density Expansion",
      answer:
        "High-resolution sub-meter spatial feature segmentation identified 428 new commercial and residential structures across the monitoring sector. Impervious surface fraction increased by 18.5% with significant transport corridor densification.",
      keyFindings: [
        "New Structural Units: 428 building footprints extracted with 91% precision",
        "Impervious Surface Fraction: 68.2% total footprint (+18.5% annual growth)",
        "Green Space Fragmentation: 12.8% reduction in peripheral tree cover",
        "Spatial Resolution: 0.28m sub-meter resolving power",
      ],
      metrics: [
        { label: "New Structures", value: "428" },
        { label: "Confidence", value: "91%" },
        { label: "Growth Rate", value: "+18.5%" },
        { label: "Impervious Ratio", value: "68.2%" },
      ],
      suggestedFollowUps: [
        "Highlight newly built transport arteries and road expansions.",
        "What is the total loss in permeable soil area?",
        "Classify structures into commercial vs residential.",
      ],
    };
  }

  // 5. Optical + SAR Cross-Modal Fusion
  if (q.includes("sar") || q.includes("optical + sar") || q.includes("cross-modal") || q.includes("together") || fileNames.includes("sar")) {
    return {
      detectedTask: "Cross-Modal Optical + SAR Synergistic Delineation",
      selectedWorkflow: "Multi-Sensor Co-Registration & SAR Microwave/Optical Fusion",
      inputModality: "Sentinel-2 MSI Optical (10m) + Sentinel-1 C-SAR Dual-Pol",
      confidence: 93,
      evidenceType: "Fused Cross-Modal Water & Built Delineation Layer",
      baseImage: "/satellite/water-optical.jpg",
      evidenceImage: "/satellite/water-result.jpg",
      location: "Godavari Basin & River Confluence",
      headline: "Synergistic Radar/Optical Fusion Resolved Water & Structural Bounds",
      answer:
        "By fusing all-weather SAR microwave backscatter with cloud-filtered optical multispectral bands, SatQuery AI achieved sharp delineation of standing water bodies (54.8 km²) regardless of atmospheric haze, while resolving adjacent built-up settlements.",
      keyFindings: [
        "Water Mask Precision: 98% radar specularity match; zero cloud false-positives",
        "Standing Water Surface: 54.8 km² across 6 interconnected reservoir channels",
        "Adjacent Built Settlements: 24.3 km² confirmed via high double-bounce radar response",
        "Cross-Modal Coregistration: RMSE < 0.35 pixels across all reference tie points",
      ],
      metrics: [
        { label: "Water Area", value: "54.8 km²" },
        { label: "Confidence", value: "93%" },
        { label: "Sensors Fused", value: "Opt + SAR" },
        { label: "Co-Registration", value: "< 0.35 px" },
      ],
      suggestedFollowUps: [
        "Show separate optical vs SAR backscatter views.",
        "How much water surface was obscured by optical clouds?",
        "Export the fused hydro-boundary vector polygon.",
      ],
    };
  }

  // 6. Object Grounding / VQA / Scene Description
  if (q.includes("describe") || q.includes("objects") || q.includes("grounding") || q.includes("find") || q.includes("airport") || fileNames.includes("airport") || fileNames.includes("delhi")) {
    return {
      detectedTask: "Visual Grounding & Remote-Sensing Scene Intelligence",
      selectedWorkflow: "Open-Vocabulary Spatial Grounding Transformer (VQA Grounding)",
      inputModality: "Cartosat-3 PAN Sub-Meter High-Resolution (0.28m GSD)",
      confidence: 90,
      evidenceType: "Spatial Grounding Bounding Boxes & Runway Vectors",
      baseImage: "/satellite/airport.jpg",
      evidenceImage: "/satellite/airport-result.jpg",
      location: "IGI Airport Runway 29R & Apron Complex, New Delhi",
      headline: "42 Commercial Aircraft Grounded & Runway 29R Verified Clear",
      answer:
        "Visual grounding analysis resolved 42 commercial aircraft stationed across Terminal 3 aprons, categorized by wingspan. Active runways 29R and 29L were confirmed operational with unobstructed touchdown zones and clear taxiway corridors.",
      keyFindings: [
        "Total Aircraft Counted: 42 airframes grounded with spatial bounding vectors",
        "Runway Status: Active Runway 29R clear with zero foreign object debris",
        "Apron Occupancy: 78% bay utilization across Western concourses",
        "Detection Model: Zero-shot open-vocabulary spatial grounding transformer",
      ],
      metrics: [
        { label: "Aircraft Count", value: "42" },
        { label: "Confidence", value: "90%" },
        { label: "Runway Status", value: "Operational" },
        { label: "GSD Resolution", value: "0.28m" },
      ],
      suggestedFollowUps: [
        "Filter grounding boxes by narrow-body vs wide-body airliners.",
        "Are any maintenance vehicles positioned in active taxi corridors?",
        "Generate high-resolution crop of Terminal 3 gate 14.",
      ],
    };
  }

  // Default: Water Detection & Water Bodies
  return {
    detectedTask: "Water Body Detection & Boundary Delineation",
    selectedWorkflow: "MNDWI Water Spectral Extraction & Edge Segmentation",
    inputModality: "Sentinel-2 MSI Optical 10m Multispectral",
    confidence: 88,
    evidenceType: "Water Detection Segmentation Mask",
    baseImage: "/satellite/water-optical.jpg",
    evidenceImage: "/satellite/water-result.jpg",
    location: "Godavari Basin, Maharashtra",
    headline: "Water-Covered Regions Detected across the Study Area (54.8 km²)",
    answer:
      "Water-covered regions were detected across the study area and reservoir boundaries. Multispectral Modified Normalized Difference Water Index (MNDWI) segmentation successfully identified 54.8 km² of open water surface across 6 primary channels with high boundary fidelity.",
    keyFindings: [
      "Total Extracted Water Area: 54.8 km² across 6 interconnected basins",
      "Perimeter Boundary Delineation: 114.2 km shoreline resolved with sub-pixel edge alignment",
      "Water Quality Indicator: Low turbidity spectral signature in central reservoir deep water",
      "False Positive Rejection: Shadows and cloud artifacts successfully eliminated",
    ],
    metrics: [
      { label: "Water Surface", value: "54.8 km²" },
      { label: "Confidence", value: "88%" },
      { label: "Primary Basins", value: "6" },
      { label: "Shoreline", value: "114.2 km" },
    ],
    suggestedFollowUps: [
      "Can you calculate the surface water volume changes over time?",
      "Are there signs of seasonal shrinkage compared to pre-monsoon?",
      "Export GeoJSON vector coordinates for all detected water polygons.",
    ],
  };
}
