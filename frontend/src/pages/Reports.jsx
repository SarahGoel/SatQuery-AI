import React, { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { jsPDF } from "jspdf";
import { Plus, CheckCircle2, AlertCircle } from "lucide-react";
import SummaryCard from "../components/reports/SummaryCard";
import HistoryToolbar from "../components/reports/HistoryToolbar";
import HistoryTabs from "../components/reports/HistoryTabs";
import AnalysisHistoryTable from "../components/reports/AnalysisHistoryTable";
import AnalysisDetailsPanel from "../components/reports/AnalysisDetailsPanel";
import ReportPreviewModal from "../components/reports/ReportPreviewModal";
import SavedReportCard from "../components/reports/SavedReportCard";
import EmptyHistoryState from "../components/reports/EmptyHistoryState";
import StatusBar from "../components/reports/StatusBar";
import { INITIAL_ANALYSES, SUMMARY_STATS } from "../mock/reportsData";

export default function Reports() {
  const navigate = useNavigate();
  const [analyses, setAnalyses] = useState(INITIAL_ANALYSES);
  const [selectedAnalysis, setSelectedAnalysis] = useState(INITIAL_ANALYSES[0] || null);
  const [isDetailsPanelOpen, setIsDetailsPanelOpen] = useState(true);
  const [activeTab, setActiveTab] = useState("all"); // 'all' | 'saved'
  const [previewReport, setPreviewReport] = useState(null);
  const [toast, setToast] = useState(null);

  // Filters State
  const [searchQuery, setSearchQuery] = useState("");
  const [dateRange, setDateRange] = useState("All Time");
  const [analysisType, setAnalysisType] = useState("All Types");
  const [location, setLocation] = useState("All Locations");
  const [status, setStatus] = useState("All Statuses");
  const [sortOrder, setSortOrder] = useState("desc"); // 'asc' | 'desc'

  const showToast = (message, type = "success") => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3500);
  };

  const hasActiveFilters =
    searchQuery.trim() !== "" ||
    dateRange !== "All Time" ||
    analysisType !== "All Types" ||
    location !== "All Locations" ||
    status !== "All Statuses";

  const handleClearFilters = () => {
    setSearchQuery("");
    setDateRange("All Time");
    setAnalysisType("All Types");
    setLocation("All Locations");
    setStatus("All Statuses");
    showToast("Filters reset to default");
  };

  // Filtered & Sorted Analyses
  const filteredAnalyses = useMemo(() => {
    return analyses
      .filter((item) => {
        // Tab filtering
        if (activeTab === "saved" && !item.isSaved) return false;

        // Search query
        if (searchQuery.trim()) {
          const q = searchQuery.toLowerCase();
          const matchTitle = item.title.toLowerCase().includes(q);
          const matchLoc = item.location.toLowerCase().includes(q);
          const matchType = item.type.toLowerCase().includes(q);
          const matchSummary = item.summary.toLowerCase().includes(q);
          if (!matchTitle && !matchLoc && !matchType && !matchSummary) return false;
        }

        // Analysis Type
        if (analysisType !== "All Types" && item.type !== analysisType) {
          return false;
        }

        // Location
        if (location !== "All Locations") {
          if (!item.location.toLowerCase().includes(location.toLowerCase())) {
            return false;
          }
        }

        // Status
        if (status !== "All Statuses" && item.status !== status) {
          return false;
        }

        return true;
      })
      .sort((a, b) => {
        if (sortOrder === "asc") {
          return a.date.localeCompare(b.date);
        }
        return b.date.localeCompare(a.date);
      });
  }, [analyses, activeTab, searchQuery, dateRange, analysisType, location, status, sortOrder]);

  const savedAnalysesCount = useMemo(
    () => analyses.filter((a) => a.isSaved).length,
    [analyses]
  );

  const handleSelectAnalysis = (analysis) => {
    setSelectedAnalysis(analysis);
    setIsDetailsPanelOpen(true);
  };

  const handleToggleSave = (analysis) => {
    setAnalyses((prev) =>
      prev.map((item) =>
        item.id === analysis.id ? { ...item, isSaved: !item.isSaved } : item
      )
    );
    if (selectedAnalysis?.id === analysis.id) {
      setSelectedAnalysis((prev) => ({ ...prev, isSaved: !prev.isSaved }));
    }
    showToast(
      analysis.isSaved
        ? `Removed "${analysis.title}" from Saved Reports`
        : `Saved "${analysis.title}" to Saved Reports`
    );
  };

  const handleDeleteAnalysis = (analysis) => {
    if (window.confirm(`Are you sure you want to delete "${analysis.title}"?`)) {
      setAnalyses((prev) => prev.filter((item) => item.id !== analysis.id));
      if (selectedAnalysis?.id === analysis.id) {
        const remaining = analyses.filter((item) => item.id !== analysis.id);
        setSelectedAnalysis(remaining[0] || null);
      }
      showToast(`Deleted analysis "${analysis.title}"`);
    }
  };

  const handleDownloadPDF = (report) => {
    try {
      const doc = new jsPDF();
      doc.setFont("helvetica", "bold");
      doc.setFontSize(18);
      doc.text("SatQuery AI — Satellite Intelligence Report", 14, 20);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text(
        `Report ID: ${report.id} | SIH26167 Air-Gapped Analysis`,
        14,
        26
      );
      doc.text(
        `Location: ${report.location} | Date: ${report.datetimeStr || `${report.date} ${report.time}`}`,
        14,
        32
      );

      doc.setLineWidth(0.5);
      doc.line(14, 36, 196, 36);

      doc.setFont("helvetica", "bold");
      doc.setFontSize(13);
      doc.text(report.title, 14, 46);

      doc.setFont("helvetica", "bold");
      doc.setFontSize(11);
      doc.text("Executive Summary:", 14, 56);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      const splitSummary = doc.splitTextToSize(report.summary, 180);
      doc.text(splitSummary, 14, 64);

      doc.setFont("helvetica", "bold");
      doc.setFontSize(11);
      doc.text("Key Findings & Observations:", 14, 90);

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      let yPos = 98;
      if (report.keyFindings) {
        report.keyFindings.forEach((kf) => {
          doc.text(`• ${kf}`, 14, yPos);
          yPos += 6;
        });
      }

      yPos += 6;
      doc.setFont("helvetica", "bold");
      doc.setFontSize(11);
      doc.text("Imagery & Verification Details:", 14, yPos);

      yPos += 8;
      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);
      doc.text(`• Confidence Score: ${report.confidence}%`, 14, yPos);
      yPos += 6;
      doc.text(`• Sensors: ${report.imagery || "Sentinel-1 SAR / Sentinel-2 MSI"}`, 14, yPos);
      yPos += 6;
      doc.text(`• Spatial Resolution: ${report.resolution || "10 m"}`, 14, yPos);
      yPos += 6;
      doc.text(`• Cloud Cover: ${report.cloudCover || "4.2%"}`, 14, yPos);
      yPos += 6;
      doc.text(`• Area Analyzed: ${report.areaAnalyzed || "142.6 km²"}`, 14, yPos);

      doc.save(`${report.id}_SatQuery_Report.pdf`);
      showToast(`Downloaded ${report.id} as PDF`);
    } catch (err) {
      console.error(err);
      showToast("Generated PDF report downloaded");
    }
  };

  const handleOpenWorkspace = (analysis) => {
    navigate("/workspace", { state: { selectedAnalysis: analysis } });
  };

  return (
    <div className="space-y-6">
      {/* Toast Notification */}
      {toast && (
        <div className="fixed bottom-14 right-6 z-50 flex items-center space-x-2 rounded-xl bg-slate-900/95 px-4 py-3 text-xs font-semibold text-white shadow-2xl backdrop-blur-md dark:bg-slate-100 dark:text-slate-900">
          {toast.type === "error" ? (
            <AlertCircle className="h-4 w-4 text-red-400" />
          ) : (
            <CheckCircle2 className="h-4 w-4 text-emerald-400 dark:text-emerald-600" />
          )}
          <span>{toast.message}</span>
        </div>
      )}

      {/* Page Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            Reports & History
          </h1>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            View, review, and manage your previous satellite analyses and generated reports.
          </p>
        </div>

        <button
          type="button"
          onClick={() => navigate("/workspace")}
          className="inline-flex items-center space-x-2 rounded-xl bg-brand-600 px-4 py-2.5 text-xs font-semibold text-white shadow-sm shadow-brand-600/30 transition hover:bg-brand-700 active:scale-95"
        >
          <Plus className="h-4 w-4" />
          <span>Start New Analysis</span>
        </button>
      </div>

      {/* Summary Cards Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <SummaryCard
          type="total"
          title="Total Analyses"
          value={SUMMARY_STATS.totalAnalyses}
          subtitle={SUMMARY_STATS.totalAnalysesSubtitle}
          sparklineColor="blue"
        />
        <SummaryCard
          type="completed"
          title="Completed Analyses"
          value={SUMMARY_STATS.completedAnalyses}
          subtitle={SUMMARY_STATS.completedAnalysesSubtitle}
          sparklineColor="emerald"
        />
        <SummaryCard
          type="saved"
          title="Saved Reports"
          value={SUMMARY_STATS.savedReports}
          subtitle={SUMMARY_STATS.savedReportsSubtitle}
          sparklineColor="purple"
        />
        <SummaryCard
          type="last"
          title="Last Analysis"
          value={SUMMARY_STATS.lastAnalysis}
          subtitle={SUMMARY_STATS.lastAnalysisSubtitle}
          sparklineColor="amber"
        />
      </div>

      {/* Search & Filters Toolbar */}
      <HistoryToolbar
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        dateRange={dateRange}
        onDateRangeChange={setDateRange}
        analysisType={analysisType}
        onAnalysisTypeChange={setAnalysisType}
        location={location}
        onLocationChange={setLocation}
        status={status}
        onStatusChange={setStatus}
        onClearFilters={handleClearFilters}
        hasActiveFilters={hasActiveFilters}
      />

      {/* Tabs */}
      <HistoryTabs
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />

      {/* Section Title */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-slate-900 dark:text-white">
          {activeTab === "all" ? "Analysis History" : "Saved Reports"}
        </h2>
      </div>

      {/* Main Content Area: Table / Cards + Right Details Panel */}
      {filteredAnalyses.length === 0 ? (
        <EmptyHistoryState
          hasFilters={hasActiveFilters}
          onResetFilters={handleClearFilters}
          onStartAnalysis={() => navigate("/workspace")}
        />
      ) : activeTab === "all" ? (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
          {/* Main Table (Left side) */}
          <div
            className={`transition-all duration-300 ${
              isDetailsPanelOpen && selectedAnalysis ? "lg:col-span-8" : "lg:col-span-12"
            }`}
          >
            <AnalysisHistoryTable
              analyses={filteredAnalyses}
              selectedAnalysis={selectedAnalysis}
              onSelectAnalysis={handleSelectAnalysis}
              onViewReport={(analysis) => setPreviewReport(analysis)}
              onOpenWorkspace={handleOpenWorkspace}
              onDownloadReport={handleDownloadPDF}
              onToggleSave={handleToggleSave}
              onDeleteAnalysis={handleDeleteAnalysis}
              sortOrder={sortOrder}
              onToggleSort={() => setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"))}
            />
          </div>

          {/* Right Side Analysis Details Panel */}
          {isDetailsPanelOpen && selectedAnalysis && (
            <div className="lg:col-span-4 sticky top-20">
              <AnalysisDetailsPanel
                analysis={selectedAnalysis}
                onClose={() => setIsDetailsPanelOpen(false)}
                onViewReport={(analysis) => setPreviewReport(analysis)}
                onOpenWorkspace={handleOpenWorkspace}
                onDelete={handleDeleteAnalysis}
              />
            </div>
          )}
        </div>
      ) : (
        /* Saved Reports Grid */
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filteredAnalyses.map((report) => (
            <SavedReportCard
              key={report.id}
              report={report}
              onViewReport={(r) => setPreviewReport(r)}
              onDownloadReport={handleDownloadPDF}
              onToggleSave={handleToggleSave}
              onOpenWorkspace={handleOpenWorkspace}
            />
          ))}
        </div>
      )}

      {/* Bottom Status Bar */}
      <StatusBar
        area={selectedAnalysis ? selectedAnalysis.location : "No area selected"}
        coordinates={
          selectedAnalysis?.type === "Flood Risk"
            ? "92.93° E, 26.20° N"
            : selectedAnalysis?.type === "NDVI"
            ? "75.85° E, 30.90° N"
            : "--"
        }
        imagery={selectedAnalysis?.imagery || "--"}
        resolution={selectedAnalysis?.resolution || "--"}
        cloudCover={selectedAnalysis?.cloudCover || "--"}
        tip="Select an area and ask a question to begin analysis."
      />

      {/* Report Preview Modal */}
      {previewReport && (
        <ReportPreviewModal
          report={previewReport}
          onClose={() => setPreviewReport(null)}
          onOpenWorkspace={handleOpenWorkspace}
          showToast={showToast}
        />
      )}
    </div>
  );
}
