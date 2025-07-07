"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";

interface LoadingOverlayProps {
  isLoading: boolean;
}

// User-friendly steps translated from the backend process
const userFriendlySteps = [
  "Connecting to our routing servers...",
  "Building the local road network...",
  "Pinpointing your start and end points...",
  "Checking real-time traffic conditions...",
  "Analyzing turns and intersections...",
  "Calculating the eco-cost for each road...",
  "Finding the most fuel-efficient path...",
  "Comparing with standard routes...",
  "Generating your travel statistics...",
  "Finalizing your eco-route...",
];

// Total animation duration set to 1 minute 45 seconds (105,000 ms)
const TOTAL_DURATION = 105000;
const STEP_INTERVAL = TOTAL_DURATION / (userFriendlySteps.length + 1); // Interval per step

export default function LoadingOverlay({ isLoading }: LoadingOverlayProps) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(false);
  const [isExiting, setIsExiting] = useState(false);

  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;

    if (isLoading) {
      setVisible(true);
      setCurrentStepIndex(0);
      setProgress(0);
      setIsExiting(false);

      timer = setInterval(() => {
        setCurrentStepIndex((prevIndex) => {
          if (prevIndex < userFriendlySteps.length - 1) {
            setIsExiting(true);
            setTimeout(() => {
              setCurrentStepIndex(prevIndex + 1);
              setProgress(((prevIndex + 2) / userFriendlySteps.length) * 99);
              setIsExiting(false);
            }, 500);
            return prevIndex;
          }
          if (timer) clearInterval(timer);
          return prevIndex;
        });
      }, STEP_INTERVAL);
    } else if (visible) {
      if (timer) clearInterval(timer);
      setIsExiting(true);
      setTimeout(() => {
        setCurrentStepIndex(userFriendlySteps.length);
        setProgress(100);
        setIsExiting(false);
        setTimeout(() => {
          setVisible(false);
        }, 1200);
      }, 500);
    }

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isLoading]);
  

  if (!visible) return null;

  const displayText =
    currentStepIndex < userFriendlySteps.length
      ? userFriendlySteps[currentStepIndex]
      : "Route found!";

  return (
    <>
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes fadeOut {
          from { opacity: 1; }
          to { opacity: 0; }
        }
        
        @keyframes slideUpOut {
            from {
                transform: translateY(0);
                opacity: 1;
            }
            to {
                transform: translateY(-100%);
                opacity: 0;
            }
        }

        @keyframes slideDownIn {
            from {
                transform: translateY(100%);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        @keyframes spinBlob {
          0% { transform: rotate(0deg) scale(1.2); }
          100% { transform: rotate(360deg) scale(1.2); }
        }

        @keyframes rotateLogo {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(-360deg); }
        }

        .animate-fade-in { animation: fadeIn 0.5s ease-out forwards; }
        .animate-fade-out { animation: fadeOut 0.5s ease-in forwards; }
        
        .text-container {
            height: 2.5rem; /* Set a fixed height to prevent layout shifts */
            overflow: hidden;
        }

        .text-anim {
            display: inline-block;
            width: 100%;
        }

        .anim-slide-up {
            animation: slideUpOut 0.5s ease-in forwards;
        }

        .anim-slide-down {
            animation: slideDownIn 0.5s ease-out forwards;
        }

        .rotate-logo { animation: rotateLogo 8s linear infinite; }
      `}</style>

      <div
        className="absolute top-0 left-0 w-full h-full flex items-center justify-center z-[999] overflow-hidden rounded-2xl animate-fade-in"
        style={{
          pointerEvents: isLoading ? "auto" : "none",
          backgroundColor: "rgba(20, 30, 25, 0.85)",
        }}
      >
        {/* Glowing background blob */}
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

        {/* Foreground content */}
        <div className="flex flex-col items-center justify-center z-10 space-y-8 w-4/5">
          <div className="rotate-logo w-20 h-20 relative">
            <Image
              src="/logo.png"
              alt="Logo"
              fill
              style={{ objectFit: "contain" }}
            />
          </div>

          <div className="text-container">
            <div
              className={`text-anim ${
                isExiting ? "anim-slide-up" : "anim-slide-down"
              }`}
              style={{
                color: "#233830",
                fontSize: "1.75rem",
                fontWeight: "500",
                textAlign: "center",
              }}
            >
              {displayText}
            </div>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-gray-700/50 rounded-full h-2.5">
            <div
              className="bg-[#ACC08D] h-2.5 rounded-full"
              style={{
                width: `${progress}%`,
                transition: "width 0.5s ease-in-out",
              }}
            ></div>
          </div>
        </div>
      </div>
    </>
  );
}
