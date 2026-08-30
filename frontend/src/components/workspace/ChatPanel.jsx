import React, { useState } from "react";
import { Sparkles, ChevronUp, ChevronDown, MessageSquare, ArrowRight, Send } from "lucide-react";

export default function ChatPanel({
  externalInput = "",
  onInputChange = () => {},
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [inputValue, setInputValue] = useState(externalInput || "");

  const examplePrompts = [
    "What is the flood risk in this area?",
    "Show changes between T1 and T2",
    "Detect water bodies in this area",
    "Analyze vegetation health (NDVI)",
  ];

  const handleSelectExample = (prompt) => {
    setInputValue(prompt);
    onInputChange(prompt);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    // Frontend only: Keep text or keep UI responsive
  };

  return (
    <div className="flex h-full w-full flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm transition-colors dark:border-dark-border dark:bg-dark-card">
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-slate-100 p-4 dark:border-dark-border">
        <div className="flex items-center space-x-2">
          <Sparkles className="h-4 w-4 text-brand-600 dark:text-brand-400" />
          <h2 className="text-sm font-bold text-slate-900 dark:text-white">
            AI Analysis Assistant
          </h2>
        </div>
        <button
          type="button"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="rounded-lg p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-dark-hover dark:hover:text-slate-200"
          title="Toggle Assistant Panel"
        >
          {isCollapsed ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronUp className="h-4 w-4" />
          )}
        </button>
      </div>

      {!isCollapsed && (
        <div className="flex flex-1 flex-col justify-between overflow-y-auto p-4">
          {/* Top / Center Empty State Content */}
          <div className="flex flex-col items-center pt-4">
            {/* Glowing Avatar Icon */}
            <div className="relative mb-3 flex items-center justify-center">
              {/* Sparkle decorative points */}
              <div className="absolute -left-3 top-2 h-1 w-1 rounded-full bg-brand-400 opacity-60" />
              <div className="absolute -right-3 top-4 h-1.5 w-1.5 rounded-full bg-brand-400 opacity-70" />
              <div className="absolute bottom-1 -left-2 h-1 w-1 rounded-full bg-brand-300 opacity-50" />
              <div className="absolute -top-2 right-1 h-1 w-1 rounded-full bg-brand-400 opacity-80" />

              {/* Halo glow */}
              <div className="absolute h-16 w-16 rounded-full bg-brand-500/15 blur-lg dark:bg-brand-500/25" />

              {/* Center circle */}
              <div className="relative flex h-14 w-14 items-center justify-center rounded-full bg-brand-50 shadow-inner dark:bg-brand-950/60 border border-brand-200/80 dark:border-brand-800/80 text-brand-600 dark:text-brand-400">
                <MessageSquare className="h-6 w-6 fill-brand-600/20 text-brand-600 dark:text-brand-400" />
              </div>
            </div>

            {/* Title & Description */}
            <h3 className="text-sm font-bold text-slate-900 dark:text-white">
              Start a conversation
            </h3>
            <p className="mt-1.5 max-w-xs text-center text-xs leading-relaxed text-slate-500 dark:text-slate-400">
              Ask a question about this area. I can help you analyze satellite imagery, detect changes, assess risks, and more.
            </p>

            {/* Try These Examples Section */}
            <div className="mt-6 w-full space-y-2">
              <p className="text-center text-xs font-semibold text-slate-700 dark:text-slate-300">
                Try these examples
              </p>
              <div className="space-y-2 pt-1">
                {examplePrompts.map((prompt, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSelectExample(prompt)}
                    className="group flex w-full items-center justify-between rounded-xl border border-slate-200/80 bg-white p-2.5 text-left text-xs font-medium text-slate-700 shadow-2xs transition hover:border-brand-500/60 hover:bg-brand-50/40 hover:text-brand-900 dark:border-dark-border dark:bg-dark-bg/40 dark:text-slate-300 dark:hover:border-brand-600/60 dark:hover:bg-dark-hover dark:hover:text-brand-200"
                  >
                    <span>{prompt}</span>
                    <ArrowRight className="h-3.5 w-3.5 text-slate-400 transition-transform group-hover:translate-x-0.5 group-hover:text-brand-600 dark:group-hover:text-brand-400" />
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Bottom Chat Input Form */}
          <form
            onSubmit={handleSubmit}
            className="mt-4 pt-3"
          >
            <div className="flex items-center space-x-2 rounded-xl border border-slate-200 bg-slate-50/80 px-3 py-2 shadow-inner focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-500/20 dark:border-dark-border dark:bg-dark-bg/60">
              <input
                type="text"
                placeholder="Ask anything about this area..."
                value={inputValue}
                onChange={(e) => {
                  setInputValue(e.target.value);
                  onInputChange(e.target.value);
                }}
                className="flex-1 bg-transparent text-xs text-slate-900 placeholder-slate-400 focus:outline-none dark:text-white"
              />
              <button
                type="submit"
                className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg bg-brand-600 text-white shadow-xs transition hover:bg-brand-700 disabled:opacity-50"
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
