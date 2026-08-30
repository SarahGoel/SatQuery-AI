import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  HelpCircle,
  BookOpen,
  Terminal,
  Compass,
  Activity,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  MessageSquare,
  Sparkles,
  Layers
} from "lucide-react";

export default function HelpSupport() {
  const [openFaq, setOpenFaq] = useState(0);

  const faqs = [
    {
      q: "What makes SatQuery AI fully air-gapped?",
      a: "SatQuery AI runs entirely on on-premise hardware using local weights (UNet-SAR-EO) and local GIS tiles. No telemetry, imagery, or queries leave your classified network perimeter.",
    },
    {
      q: "How are confidence scores calculated?",
      a: "Confidence is computed as a weighted harmonic composite of: 1) Radiometric sensor calibration (35%), 2) Sub-pixel coregistration accuracy between T1 and T2 passes (30%), 3) Model inference softmax certainty (25%), and 4) Vector ground overlay match (10%).",
    },
    {
      q: "Which satellite sensors are supported out-of-the-box?",
      a: "We support ISRO Cartosat-1/2/3, Resourcesat-2 LISS-IV, RISAT-1A SAR, INSAT-3DR meteorological infrared, Sentinel-1 SAR (C-band dual-pol), and Sentinel-2 MSI multi-spectral bands.",
    },
    {
      q: "How do I perform temporal swipe comparison?",
      a: "Navigate to Geospatial AI Analysis, toggle 'Enable Swipe' under the Temporal Pair section on the left panel, and adjust the slider across the center map to compare T1 baseline with T2 post-event imagery.",
    },
  ];

  const exampleQueries = [
    {
      title: "Flood Extent & Inundation Mapping",
      prompt: "Assess flood inundation extent in Brahmaputra Basin over Kaziranga sector between T1 and T2.",
      category: "Disaster / SAR",
    },
    {
      title: "Bi-Temporal Built-up Change",
      prompt: "Highlight built-up area and road infrastructure expansion between May 15 and May 30.",
      category: "Urban / Optical",
    },
    {
      title: "Agricultural Crop Moisture Deficit",
      prompt: "Generate NDVI vegetation health anomaly report and identify standing crop stress parcels.",
      category: "Agri / Spectral",
    },
    {
      title: "Embankment & Riverbank Vulnerability",
      prompt: "Identify critical river embankment breach risks and compute vulnerable village population.",
      category: "Hydrology",
    },
  ];

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white flex items-center gap-2.5">
          <HelpCircle className="h-6 w-6 text-brand-600" />
          <span>Help & Support Center</span>
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          User guides, query syntax reference, map controls, and confidence score explanations for SatQuery AI.
        </p>
      </div>

      {/* 4 Feature Guide Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm dark:border-dark-border dark:bg-dark-card space-y-2.5">
          <div className="flex items-center space-x-2 text-brand-600">
            <BookOpen className="h-5 w-5" />
            <h2 className="text-sm font-bold text-slate-900 dark:text-white">
              1. Getting Started Workflow
            </h2>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
            Select an AOI from the preset list or use the BBox/Polygon drawing tools on the interactive map. Choose sensor visualization (Optical, SAR, NDVI, NDWI) and ask your question in the natural language assistant panel.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm dark:border-dark-border dark:bg-dark-card space-y-2.5">
          <div className="flex items-center space-x-2 text-emerald-600">
            <Compass className="h-5 w-5" />
            <h2 className="text-sm font-bold text-slate-900 dark:text-white">
              2. Map Controls & Annotations
            </h2>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
            Use the top toolbar to draw bounding boxes or custom polygons. The temporal swipe slider lets you compare pre- and post-disaster passes seamlessly. Use the search bar to rapidly zoom to any Indian river basin or district.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm dark:border-dark-border dark:bg-dark-card space-y-2.5">
          <div className="flex items-center space-x-2 text-purple-600">
            <Activity className="h-5 w-5" />
            <h2 className="text-sm font-bold text-slate-900 dark:text-white">
              3. Understanding Confidence Scores
            </h2>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
            Every analysis result includes an explainable confidence gauge with breakdown bars for sensor calibration, radiometric alignment, and model certainty so analysts have complete auditability.
          </p>
        </div>

        <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-sm dark:border-dark-border dark:bg-dark-card space-y-2.5">
          <div className="flex items-center space-x-2 text-amber-600">
            <ShieldCheck className="h-5 w-5" />
            <h2 className="text-sm font-bold text-slate-900 dark:text-white">
              4. Air-Gapped Operation
            </h2>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
            The platform is engineered specifically for secure, air-gapped disaster management centers and national command rooms without dependencies on external cloud APIs.
          </p>
        </div>
      </div>

      {/* Example Queries Section */}
      <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm dark:border-dark-border dark:bg-dark-card space-y-4">
        <div className="flex items-center space-x-2">
          <Terminal className="h-5 w-5 text-brand-600" />
          <h2 className="text-base font-bold text-slate-900 dark:text-white">
            Recommended Example Queries
          </h2>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {exampleQueries.map((item, idx) => (
            <div
              key={idx}
              className="flex flex-col justify-between rounded-xl border border-slate-100 bg-slate-50 p-3.5 dark:border-dark-border dark:bg-dark-bg/60 space-y-2"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-900 dark:text-white">
                    {item.title}
                  </span>
                  <span className="rounded bg-brand-100 px-1.5 py-0.2 text-[10px] font-semibold text-brand-700 dark:bg-brand-950 dark:text-brand-300">
                    {item.category}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-600 dark:text-slate-300 font-mono bg-white p-2 rounded-lg border border-slate-200/70 dark:border-dark-border dark:bg-dark-card">
                  "{item.prompt}"
                </p>
              </div>

              <Link
                to="/workspace"
                className="text-[11px] font-semibold text-brand-600 hover:text-brand-700 dark:text-brand-400 flex items-center gap-1"
              >
                <span>Try this query in Workspace</span>
                <ExternalLink className="h-3 w-3" />
              </Link>
            </div>
          ))}
        </div>
      </div>

      {/* FAQ Accordions */}
      <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm dark:border-dark-border dark:bg-dark-card space-y-4">
        <div className="flex items-center space-x-2">
          <MessageSquare className="h-5 w-5 text-brand-600" />
          <h2 className="text-base font-bold text-slate-900 dark:text-white">
            Frequently Asked Questions
          </h2>
        </div>

        <div className="divide-y divide-slate-100 dark:divide-dark-border">
          {faqs.map((faq, idx) => {
            const isOpen = openFaq === idx;
            return (
              <div key={idx} className="py-3.5 first:pt-0 last:pb-0">
                <button
                  type="button"
                  onClick={() => setOpenFaq(isOpen ? -1 : idx)}
                  className="flex w-full items-center justify-between text-left text-xs sm:text-sm font-semibold text-slate-900 dark:text-white"
                >
                  <span>{faq.q}</span>
                  {isOpen ? (
                    <ChevronUp className="h-4 w-4 text-slate-400 flex-shrink-0 ml-2" />
                  ) : (
                    <ChevronDown className="h-4 w-4 text-slate-400 flex-shrink-0 ml-2" />
                  )}
                </button>
                {isOpen && (
                  <p className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-300">
                    {faq.a}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
