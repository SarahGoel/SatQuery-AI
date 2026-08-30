import React, { useState, useRef, useEffect } from "react";
import { ChevronDown, Check } from "lucide-react";

export default function FilterDropdown({
  label,
  value,
  options,
  onChange,
  icon: Icon,
  rightIcon: RightIcon,
}) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const isFiltered =
    value &&
    value !== "All" &&
    value !== "All Types" &&
    value !== "All Locations" &&
    value !== "All Statuses" &&
    value !== "All Time";

  const displayLabel = isFiltered ? value : label;

  return (
    <div className="relative inline-block text-left" ref={dropdownRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center space-x-2 rounded-xl border px-3.5 py-2 text-xs font-medium transition-all ${
          isFiltered
            ? "border-brand-500 bg-brand-50/50 text-brand-700 dark:border-brand-500/80 dark:bg-brand-950/30 dark:text-brand-300"
            : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50 dark:border-dark-border dark:bg-dark-card dark:text-slate-200 dark:hover:bg-dark-hover"
        }`}
      >
        {Icon && <Icon className="h-3.5 w-3.5 text-slate-400 dark:text-slate-500" />}
        <span>{displayLabel}</span>
        {RightIcon ? (
          <RightIcon className="h-3.5 w-3.5 text-slate-400 dark:text-slate-500" />
        ) : (
          <ChevronDown
            className={`h-3.5 w-3.5 text-slate-400 transition-transform duration-200 ${
              isOpen ? "rotate-180" : ""
            }`}
          />
        )}
      </button>

      {isOpen && (
        <div className="absolute left-0 z-50 mt-1.5 w-48 rounded-xl border border-slate-200 bg-white p-1.5 shadow-xl ring-1 ring-black/5 dark:border-dark-border dark:bg-dark-card">
          <div className="max-h-60 overflow-y-auto space-y-0.5">
            {options.map((option) => {
              const isSelected = value === option;
              return (
                <button
                  key={option}
                  type="button"
                  onClick={() => {
                    onChange(option);
                    setIsOpen(false);
                  }}
                  className={`flex w-full items-center justify-between rounded-lg px-3 py-1.5 text-xs transition ${
                    isSelected
                      ? "bg-brand-50 font-semibold text-brand-700 dark:bg-brand-950/60 dark:text-brand-300"
                      : "text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-dark-hover"
                  }`}
                >
                  <span>{option}</span>
                  {isSelected && (
                    <Check className="h-3.5 w-3.5 text-brand-600 dark:text-brand-400" />
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
