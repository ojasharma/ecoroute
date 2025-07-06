"use client";

import React from "react";
import Image from "next/image";

interface LoadingOverlayProps {
  isLoading: boolean;
}

export default function LoadingOverlay({ isLoading }: LoadingOverlayProps) {
  if (!isLoading) return null;

  return (
    <>
      <style>{`
        @keyframes spinBlob {
          0% {
            transform: rotate(0deg) scale(1.2);
          }
          100% {
            transform: rotate(360deg) scale(1.2);
          }
        }

        @keyframes rotateLogo {
          0% {
            transform: rotate(0deg);
          }
          100% {
            transform: rotate(360deg);
          }
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        .fade-in {
          animation: fadeIn 0.6s ease-in-out forwards;
        }

        .rotate-logo {
          animation: rotateLogo 6s linear infinite;
        }
      `}</style>

      <div
        className="absolute top-0 left-0 w-full h-full flex items-center justify-center z-[999] overflow-hidden rounded-2xl fade-in"
        style={{
          pointerEvents: "none",
          backgroundColor: "rgba(20, 30, 25, 0.75)", // Darkened background
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
              opacity: 0.6, // Reduced opacity
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
                opacity: 0.4, // Reduced blob opacity
              }}
            />
          </div>
        </div>

        {/* Foreground content */}
        <div className="flex flex-col items-center justify-center z-10 space-y-4">
          {/* Rotating logo */}
          <div className="rotate-logo w-16 h-16 relative">
            <Image
              src="/logo.png"
              alt="Logo"
              fill
              style={{ objectFit: "contain" }}
            />
          </div>

          {/* Loading text with softer shadow */}
          <div
            style={{
              color: "#F0EDD1",
              fontSize: "2rem",
              fontWeight: "bold",
              textShadow: "0 0 5px rgba(0, 0, 0, 0.5)", // Softer shadow
            }}
          >
            Finding you the best route...
          </div>
        </div>
      </div>
    </>
  );
}
