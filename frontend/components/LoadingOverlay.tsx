"use client";

import React, { useEffect, useState } from "react";
import Image from "next/image";
import LogDisplay from "./LogDisplay";

interface LoadingOverlayProps {
  isLoading: boolean;
  logs: string[];
}

export default function LoadingOverlay({
  isLoading,
  logs,
}: LoadingOverlayProps) {
  const [visible, setVisible] = useState(false);
  const [isExiting, setIsExiting] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (isLoading) {
      setVisible(true);
      setIsExiting(false);
      setProgress(0);

      // Simulate a linear progress animation to 100%
      let start: number | null = null;
      const duration = 105000; // 1 min 45 sec
      const step = (timestamp: number) => {
        if (!start) start = timestamp;
        const elapsed = timestamp - start;
        const newProgress = Math.min((elapsed / duration) * 100, 100);
        setProgress(newProgress);
        if (newProgress < 100) {
          requestAnimationFrame(step);
        }
      };
      requestAnimationFrame(step);
    } else if (visible) {
      setProgress(100);
      setTimeout(() => {
        setIsExiting(true);
        setTimeout(() => setVisible(false), 500);
      }, 800);
    }
  }, [isLoading, visible]);

  if (!visible) return null;

  return (
    <>
      <style>{`
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes fadeOut { from { opacity: 1; } to { opacity: 0; } }
        @keyframes spinBlob { 0% { transform: rotate(0deg) scale(1.2); } 100% { transform: rotate(360deg) scale(1.2); } }
        @keyframes rotateLogo { 0% { transform: rotate(0deg); } 100% { transform: rotate(-360deg); } }

        .animate-fade-in { animation: fadeIn 0.5s ease-out forwards; }
        .animate-fade-out { animation: fadeOut 0.5s ease-in forwards; }
        .rotate-logo { animation: rotateLogo 8s linear infinite; }
      `}</style>

      <div
        className={`absolute top-0 left-0 w-full h-full flex items-center justify-center z-[999] overflow-hidden rounded-2xl ${
          isExiting ? "animate-fade-out" : "animate-fade-in"
        }`}
        style={{
          pointerEvents: isLoading ? "auto" : "none",
          backgroundColor: "rgba(20, 30, 25, 0.85)",
        }}
      >
        {/* Glowing Background */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            height: "100%",
            width: "100%",
            zIndex: 0,
            overflow: "hidden",
            borderRadius: "inherit",
            filter: "blur(100px)",
          }}
        >
          <div
            style={{
              borderRadius: "9999px",
              position: "absolute",
              inset: 0,
              margin: "auto",
              width: "150%",
              height: "150%",
              backgroundColor: "#fff",
              transform: "scale(1.2)",
              opacity: 0.6,
            }}
          >
            <div
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                margin: "auto",
                background:
                  "conic-gradient(from 0deg, #08f, #0f6, #bbffa1, #4c00ff, #ab2666, #09f)",
                animation: "spinBlob 8s linear infinite",
                opacity: 0.4,
              }}
            />
          </div>
        </div>

        {/* Foreground Content */}
        <div className="flex flex-col items-center justify-center z-10 space-y-8 w-4/5">
          <div className="rotate-logo w-20 h-20 relative">
            <Image
              src="/logo.png"
              alt="Logo"
              fill
              style={{ objectFit: "contain" }}
            />
          </div>

          <div
            style={{
              color: "#F0EDD1",
              fontSize: "1.75rem",
              fontWeight: "500",
              textAlign: "center",
            }}
          >
            Finding you the best EcoRoute
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-gray-700/50 rounded-full h-2.5">
            <div
              className="bg-[#ACC08D] h-2.5 rounded-full"
              style={{
                width: `${progress}%`,
                transition: "width 1s ease-out",
              }}
            />
          </div>

          {/* Log Display */}
          <div className="w-full mt-4">
            <LogDisplay loading={isLoading} logs={logs} />
          </div>
        </div>
      </div>
    </>
  );
}
