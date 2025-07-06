"use client";

import React, { useState } from "react";

const LOGO_URL = "/logo.png";

interface AnimatedButtonProps {
  disabled?: boolean;
  onClick?: () => void;
}

const AnimatedButton: React.FC<AnimatedButtonProps> = ({
  disabled = false,
  onClick,
}) => {
  const [isHovering, setIsHovering] = useState(false);

  const handleMouseEnter = () => {
    if (!disabled) setIsHovering(true);
  };

  const handleMouseLeave = () => {
    if (!disabled) setIsHovering(false);
  };

  return (
    <button
      disabled={disabled}
      onClick={disabled ? undefined : onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={`group relative flex items-center justify-center rounded-2xl px-8 py-4 text-lg font-semibold overflow-hidden transition-all duration-300 ease-in-out ${
        disabled
          ? "cursor-not-allowed bg-gray-400 text-gray-200"
          : "cursor-pointer bg-[#ACC08D] text-[#233830] hover:opacity-90"
      }`}
      style={{
        boxShadow: disabled ? "none" : "0 6px 20px rgba(0,0,0,0.25)",
      }}
    >
      <img
        src={LOGO_URL}
        alt="logo"
        className={`w-7 h-7 mr-3 transition-transform duration-700 ease-in-out ${
          isHovering ? "rotate-[360deg] scale-110" : ""
        } ${disabled ? "opacity-40 grayscale" : ""}`}
        onError={(e) => {
          e.currentTarget.src =
            "https://placehold.co/28x28/233830/ACC08D?text=B";
          e.currentTarget.onerror = null;
        }}
      />

      <div className="relative h-6 w-28 overflow-hidden">
        <span
          className={`absolute inset-0 flex items-center justify-center transition-transform duration-500 ease-in-out ${
            isHovering ? "-translate-y-full" : "translate-y-0"
          }`}
        >
          Find Route
        </span>

        <span
          className={`absolute inset-0 flex items-center justify-center transition-transform duration-500 ease-in-out ${
            isHovering ? "translate-y-0" : "translate-y-full"
          }`}
        >
          Let's Go!
        </span>
      </div>
    </button>
  );
};

export default AnimatedButton;
