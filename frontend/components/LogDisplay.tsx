"use client";

import React, { useState, useEffect, useRef } from "react";

interface LogDisplayProps {
  loading: boolean;
  logs: string[];
}

const LogDisplay: React.FC<LogDisplayProps> = ({ loading, logs = [] }) => {
  const [isOpen, setIsOpen] = useState(true);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the bottom when new logs are added
  useEffect(() => {
    if (isOpen && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, isOpen]);

  // Determine the status message
  const getStatus = () => {
    if (loading) {
      return (
        <div className="flex items-center">
          <svg
            className="animate-spin -ml-1 mr-3 h-5 w-5 text-gray-700"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            ></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
          </svg>
          <span>Processing...</span>
        </div>
      );
    }
    if (logs.length > 0) {
      const lastLog = logs[logs.length - 1] || "";
      if (lastLog.includes("[ERROR]")) {
        return (
          <span className="text-red-600 font-bold">
            Finished with an error.
          </span>
        );
      }
      return <span className="text-green-700 font-bold">Finished.</span>;
    }
    return "Awaiting request...";
  };

  return (
    <div className="rounded-lg border border-gray-300 bg-white/50 shadow-sm transition-all duration-300 w-full">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-3 text-left font-semibold text-gray-800 flex justify-between items-center"
      >
        <div>Status: {getStatus()}</div>
        <span
          className={`transform transition-transform duration-200 ${
            isOpen ? "rotate-180" : "rotate-0"
          }`}
        >
          ▼ {/* Down arrow character */}
        </span>
      </button>

      {isOpen && (
        <div
          ref={logContainerRef}
          className="p-3 border-t border-gray-300 bg-gray-50 max-h-64 overflow-y-auto text-sm font-mono"
          style={{ color: "#243931" }}
        >
          {logs.length > 0 ? (
            logs.map((log, index) => (
              <p
                key={index}
                className="whitespace-pre-wrap break-words text-xs leading-relaxed hover:bg-gray-200"
              >
                {log}
              </p>
            ))
          ) : (
            <p className="text-gray-500 italic">No logs to display yet.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default LogDisplay;
