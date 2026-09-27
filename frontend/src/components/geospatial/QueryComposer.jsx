import React, { useState, useRef, useEffect } from "react";
import {
  Paperclip,
  PlusCircle,
  Mic,
  MicOff,
  Send,
  X,
  ChevronDown,
  Upload,
} from "lucide-react";
import { SAMPLE_PRESET_IMAGES, SAMPLE_PRESET_PAIRS } from "../../mock/geospatialAnalyses";

export default function QueryComposer({
  query,
  onQueryChange,
  attachedFiles,
  onAddFiles,
  onRemoveFile,
  onSubmit,
  isAnalyzing,
}) {
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const pairInputRef = useRef(null);
  const [showAttachMenu, setShowAttachMenu] = useState(false);
  const [showPairMenu, setShowPairMenu] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const attachMenuRef = useRef(null);
  const pairMenuRef = useRef(null);

  // Close dropdown menus on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (attachMenuRef.current && !attachMenuRef.current.contains(event.target)) {
        setShowAttachMenu(false);
      }
      if (pairMenuRef.current && !pairMenuRef.current.contains(event.target)) {
        setShowPairMenu(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Speech Recognition (Web Speech API or simulated fallback)
  const handleToggleVoice = () => {
    if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
      if (!isListening) {
        setIsListening(true);
        const sampleVoices = [
          "Detect water bodies in this area",
          "What is the flood risk in this area?",
          "Analyze vegetation health and crop stress",
        ];
        const randomVoice = sampleVoices[Math.floor(Math.random() * sampleVoices.length)];
        setTimeout(() => {
          onQueryChange(randomVoice);
          setIsListening(false);
        }, 1200);
      }
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (isListening) {
      setIsListening(false);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = "en-US";

      recognition.onstart = () => setIsListening(true);
      recognition.onresult = (event) => {
        const transcript = event?.results?.[0]?.[0]?.transcript || "";
        if (transcript) {
          onQueryChange(query ? `${query} ${transcript}` : transcript);
        }
        setIsListening(false);
      };
      recognition.onerror = () => setIsListening(false);
      recognition.onend = () => setIsListening(false);

      recognition.start();
    } catch {
      setIsListening(false);
    }
  };

  // File Upload Handlers
  const handleNativeFileUpload = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      const formatted = files.map((f) => ({
        id: `user-${Date.now()}-${f.name}`,
        name: f.name,
        size: `${(f.size / (1024 * 1024)).toFixed(1)} MB`,
        modality: f.name.endsWith(".tif") || f.name.endsWith(".tiff") ? "GeoTIFF Satellite" : "Optical Imagery",
        baseImage: f.type.startsWith("image/") ? URL.createObjectURL(f) : "/satellite/water-optical.jpg",
        file: f,
        rawFile: f,
      }));
      onAddFiles(formatted);
    }
    setShowAttachMenu(false);
    e.target.value = "";
  };

  const handleNativePairUpload = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      const formatted = files.slice(0, 2).map((f, i) => ({
        id: `pair-${Date.now()}-${i}`,
        name: f.name,
        size: `${(f.size / (1024 * 1024)).toFixed(1)} MB`,
        modality: i === 0 ? "Temporal T1 / Optical" : "Temporal T2 / SAR",
        baseImage: f.type.startsWith("image/")
          ? URL.createObjectURL(f)
          : i === 0
          ? "/satellite/landcover-before.jpg"
          : "/satellite/landcover-after.jpg",
        file: f,
        rawFile: f,
      }));
      onAddFiles(formatted);
    }
    setShowPairMenu(false);
    e.target.value = "";
  };

  const handleSelectPreset = (preset) => {
    onAddFiles([preset]);
    setShowAttachMenu(false);
  };

  const handleSelectPairPreset = (pair) => {
    const cleanImage2 =
      pair.file2?.baseImage ||
      pair.cleanImage2 ||
      (pair.id === "kerala-pair"
        ? "/satellite/landcover-after.jpg"
        : pair.id === "optical-sar-pair"
        ? "/satellite/water-sar.jpg"
        : pair.id === "flood-pair"
        ? "/satellite/brahmaputra_flood.jpg"
        : pair.baseImage);

    onAddFiles([
      {
        id: `${pair.id}-1`,
        name: pair.file1.name,
        size: pair.file1.size,
        modality: pair.file1.type,
        baseImage: pair.file1?.baseImage || pair.baseImage,
      },
      {
        id: `${pair.id}-2`,
        name: pair.file2.name,
        size: pair.file2.size,
        modality: pair.file2.type,
        baseImage: cleanImage2,
      },
    ]);
    if (!query.trim() && pair.defaultQuery) {
      onQueryChange(pair.defaultQuery);
    }
    setShowPairMenu(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if ((query.trim() || attachedFiles.length > 0) && !isAnalyzing) {
        onSubmit();
      }
    }
  };

  const canSubmit = (query.trim().length > 0 || attachedFiles.length > 0) && !isAnalyzing;

  return (
    <div className="w-full max-w-4xl mx-auto transition-all">
      {/* Hidden Native File Inputs */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".tif,.tiff,.gtiff,.png,.jpg,.jpeg"
        className="hidden"
        onChange={handleNativeFileUpload}
      />
      <input
        ref={pairInputRef}
        type="file"
        multiple
        accept=".tif,.tiff,.gtiff,.png,.jpg,.jpeg"
        className="hidden"
        onChange={handleNativePairUpload}
      />

      {/* Main Large Rounded Composer Container */}
      <div className="relative rounded-2xl sm:rounded-3xl border border-slate-200/90 dark:border-blue-500/30 bg-white dark:bg-[#0A1224]/95 shadow-sm dark:shadow-[0_0_30px_rgba(37,99,235,0.09)] hover:shadow-md transition-shadow p-3.5 sm:p-4.5 focus-within:ring-2 focus-within:ring-blue-500/20 focus-within:border-blue-500 dark:focus-within:border-blue-400">
        
        {/* Attached Files Strip (Requirement 8) */}
        {attachedFiles.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 mb-2 pb-2.5 border-b border-slate-100 dark:border-slate-800/80">
            <span className="text-[11px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider pl-1">
              Attached Imagery ({attachedFiles.length})
            </span>
            {attachedFiles.map((file) => (
              <div
                key={file.id || file.name}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-blue-50/80 dark:bg-blue-950/60 border border-blue-200/60 dark:border-blue-800/80 text-xs font-medium text-blue-800 dark:text-blue-300 shadow-2xs group"
              >
                <span className="text-sm">🛰️</span>
                <span className="max-w-[140px] sm:max-w-[200px] truncate text-slate-800 dark:text-slate-200">
                  {file.name}
                </span>
                {file.size && (
                  <span className="text-[10px] text-slate-400 dark:text-slate-500 hidden sm:inline">
                    ({file.size})
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => onRemoveFile(file.id || file.name)}
                  className="p-0.5 ml-0.5 rounded-full hover:bg-blue-200/70 dark:hover:bg-blue-900 text-blue-500 hover:text-blue-900 dark:hover:text-blue-200 transition cursor-pointer"
                  title="Remove image"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Query Textarea */}
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            attachedFiles.length > 0
              ? "What would you like to know about these images?"
              : "Ask anything about your satellite imagery..."
          }
          rows={2}
          className="w-full resize-none border-0 bg-transparent text-sm sm:text-base text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none focus:ring-0 leading-relaxed font-normal"
        />

        {/* Composer Controls Bottom Bar */}
        <div className="flex items-center justify-between pt-2.5 border-t border-slate-100/80 dark:border-slate-800/80 mt-1">
          {/* Left Buttons: Attach Image(s) and Add Image Pair */}
          <div className="flex items-center gap-2 relative">
            {/* 1. Attach Image(s) Dropdown */}
            <div className="relative" ref={attachMenuRef}>
              <button
                type="button"
                onClick={() => {
                  setShowAttachMenu(!showAttachMenu);
                  setShowPairMenu(false);
                }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-200 dark:border-slate-700/60 bg-slate-50/90 dark:bg-[#101B34] hover:bg-slate-100 dark:hover:bg-[#162548] text-xs font-medium text-slate-700 dark:text-slate-200 shadow-2xs transition cursor-pointer"
              >
                <Paperclip className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />
                <span>Attach Image(s)</span>
                <ChevronDown className="w-3 h-3 text-slate-400" />
              </button>

              {/* Attach Dropdown Menu */}
              {showAttachMenu && (
                <div className="absolute left-0 bottom-full mb-2 w-72 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-[#0B1528] shadow-xl z-50 p-2 text-xs animate-in fade-in zoom-in-95 duration-150">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg hover:bg-blue-50 dark:hover:bg-[#132242] text-slate-800 dark:text-slate-200 text-left font-medium transition cursor-pointer"
                  >
                    <Upload className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                    <div>
                      <div className="font-semibold text-slate-900 dark:text-white">Upload from Device</div>
                      <div className="text-[10px] text-slate-400">GeoTIFF, TIFF, PNG, JPEG</div>
                    </div>
                  </button>

                  <div className="my-1.5 border-t border-slate-100 dark:border-slate-800" />

                  <div className="px-2.5 py-1 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                    Quick Sample Datasets
                  </div>

                  <div className="max-h-48 overflow-y-auto space-y-0.5">
                    {SAMPLE_PRESET_IMAGES.map((img) => (
                      <button
                        key={img.id}
                        type="button"
                        onClick={() => handleSelectPreset(img)}
                        className="w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-[#132242] text-left transition text-slate-700 dark:text-slate-300 cursor-pointer"
                      >
                        <div className="truncate">
                          <div className="font-medium text-slate-800 dark:text-slate-200 truncate">
                            {img.label}
                          </div>
                          <div className="text-[10px] text-slate-400">{img.modality}</div>
                        </div>
                        <span className="text-[10px] font-mono text-slate-400 ml-2">{img.size}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* 2. Add Image Pair Dropdown */}
            <div className="relative" ref={pairMenuRef}>
              <button
                type="button"
                onClick={() => {
                  setShowPairMenu(!showPairMenu);
                  setShowAttachMenu(false);
                }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-200 dark:border-slate-700/60 bg-slate-50/90 dark:bg-[#101B34] hover:bg-slate-100 dark:hover:bg-[#162548] text-xs font-medium text-slate-700 dark:text-slate-200 shadow-2xs transition cursor-pointer"
              >
                <PlusCircle className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />
                <span>Add Image Pair</span>
              </button>

              {/* Pair Dropdown Menu */}
              {showPairMenu && (
                <div className="absolute left-0 bottom-full mb-2 w-80 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-[#0B1528] shadow-xl z-50 p-2 text-xs animate-in fade-in zoom-in-95 duration-150">
                  <button
                    type="button"
                    onClick={() => pairInputRef.current?.click()}
                    className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg hover:bg-blue-50 dark:hover:bg-[#132242] text-slate-800 dark:text-slate-200 text-left font-medium transition cursor-pointer"
                  >
                    <Upload className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                    <div>
                      <div className="font-semibold text-slate-900 dark:text-white">Upload Custom Pair</div>
                      <div className="text-[10px] text-slate-400">T1/T2 Bi-temporal or Optical + SAR</div>
                    </div>
                  </button>

                  <div className="my-1.5 border-t border-slate-100 dark:border-slate-800" />

                  <div className="px-2.5 py-1 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                    Preset Multi-Image Pairs
                  </div>

                  <div className="space-y-1">
                    {SAMPLE_PRESET_PAIRS.map((pair) => (
                      <button
                        key={pair.id}
                        type="button"
                        onClick={() => handleSelectPairPreset(pair)}
                        className="w-full p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-[#132242] text-left transition cursor-pointer"
                      >
                        <div className="font-semibold text-slate-800 dark:text-slate-200 text-xs">
                          {pair.label}
                        </div>
                        <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">
                          {pair.file1.type} + {pair.file2.type}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Controls: Microphone & Blue Send Button */}
          <div className="flex items-center gap-2">
            {/* Microphone Button */}
            <button
              type="button"
              onClick={handleToggleVoice}
              className={`p-2 rounded-xl transition cursor-pointer ${
                isListening
                  ? "bg-red-100 text-red-600 dark:bg-red-950/50 dark:text-red-400 animate-pulse"
                  : "text-slate-500 dark:text-slate-300 hover:text-slate-800 dark:hover:text-white hover:bg-slate-100 dark:bg-[#101B34] dark:hover:bg-[#162548]"
              }`}
              title={isListening ? "Listening... click to stop" : "Voice Query Input"}
            >
              {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            </button>

            {/* Send / Analyze Button */}
            <button
              type="button"
              disabled={!canSubmit}
              onClick={onSubmit}
              className={`inline-flex items-center justify-center p-2.5 sm:px-4 sm:py-2 rounded-xl text-xs font-semibold shadow-sm transition-all ${
                canSubmit
                  ? "bg-blue-600 hover:bg-blue-500 text-white cursor-pointer active:scale-95 shadow-blue-500/25"
                  : "bg-slate-200 dark:bg-slate-800/80 text-slate-400 dark:text-slate-600 cursor-not-allowed"
              }`}
              title="Analyze satellite imagery"
            >
              <Send className="w-4 h-4 sm:mr-1.5" />
              <span className="hidden sm:inline">Analyze</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
