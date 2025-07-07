"use client";

import React, { useState, useEffect, useRef } from "react";
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
const STEP_INTERVAL = TOTAL_DURATION / (userFriendlySteps.length + 1);

export default function LoadingOverlay({ isLoading }: LoadingOverlayProps) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(false);
  const [isExiting, setIsExiting] = useState(false);

  const animationFrameId = useRef<number | null>(null);
  const stepTimerId = useRef<NodeJS.Timeout | null>(null);

  // Effect to control step transitions
  useEffect(() => {
    if (isLoading) {
      // Reset and start loading animation
      setVisible(true);
      setIsExiting(false);
      setCurrentStepIndex(0);
      setProgress(0);

      // Interval to advance to the next step
      stepTimerId.current = setInterval(() => {
        setCurrentStepIndex((prevIndex) => {
          if (prevIndex < userFriendlySteps.length - 1) {
            return prevIndex + 1;
          }
          // If it reaches the last step, clear the interval
          if (stepTimerId.current) clearInterval(stepTimerId.current);
          return prevIndex;
        });
      }, STEP_INTERVAL);
    } else if (visible) {
      // Handle loading completion
      if (stepTimerId.current) clearInterval(stepTimerId.current);
      if (animationFrameId.current)
        cancelAnimationFrame(animationFrameId.current);

      // Animate to 100% and fade out
      setProgress(100);
      setCurrentStepIndex(userFriendlySteps.length); // Show "Route found!"

      setTimeout(() => {
        setIsExiting(true); // Start fade-out animation for the whole overlay
        setTimeout(() => setVisible(false), 500);
      }, 800);
    }

    return () => {
      if (stepTimerId.current) clearInterval(stepTimerId.current);
      if (animationFrameId.current)
        cancelAnimationFrame(animationFrameId.current);
    };
  }, [isLoading, visible]);

  // Effect for continuous progress bar animation
  useEffect(() => {
    if (!isLoading) return;

    const startProgress = progress;
    // The target is the beginning of the *next* step's progress range
    const targetProgress =
      ((currentStepIndex + 1) / (userFriendlySteps.length + 1)) * 100;

    let startTime: number | null = null;

    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp;
      const elapsedTime = timestamp - startTime;

      // Calculate the fraction of the interval that has passed
      const animationProgress = Math.min(elapsedTime / STEP_INTERVAL, 1);

      // Interpolate the progress value
      const newProgress =
        startProgress + (targetProgress - startProgress) * animationProgress;

      setProgress(newProgress);

      if (elapsedTime < STEP_INTERVAL) {
        animationFrameId.current = requestAnimationFrame(animate);
      }
    };

    animationFrameId.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [currentStepIndex, isLoading]);

  if (!visible) return null;

  const displayText =
    currentStepIndex < userFriendlySteps.length
      ? userFriendlySteps[currentStepIndex]
      : "Route found!";

  return (
    <>
      <style>{`
        /* ... existing keyframes and styles ... */
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
            height: 2.5rem; /* Fixed height for text area */
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .text-anim {
            display: inline-block;
            width: 100%;
            animation: slideDownIn 0.4s ease-out forwards;
        }
        
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
              key={currentStepIndex} // Re-trigger animation on step change
              className="text-anim"
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
                // Use a faster transition for the final jump to 100%
                transition: progress === 100 ? "width 0.4s ease-out" : "none",
              }}
            ></div>
          </div>
        </div>
      </div>
    </>
  );
}