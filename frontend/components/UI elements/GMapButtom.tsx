"use client";

import React, { useState } from "react";

const GOOGLE_MAPS_LOGO_URL = "/googlemap.png";

interface GoogleMapButtonProps {
  disabled?: boolean;
  onClick?: () => void;
}

const GoogleMapButton: React.FC<GoogleMapButtonProps> = ({
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
      className={`group relative flex items-center justify-center rounded-xl px-4 py-2 text-[10px] font-medium overflow-hidden transition-all duration-300 ease-in-out ${
        disabled
          ? "cursor-not-allowed bg-gray-400 text-gray-200"
          : "cursor-pointer bg-[#4285F4] text-white hover:bg-[#3367D6]"
      }`}
    >
      <img
        src={GOOGLE_MAPS_LOGO_URL}
        alt="Google Maps"
        className={`w-5 h-5 mr-2 transition-transform duration-700 ease-in-out ${
          isHovering ? "rotate-[360deg] scale-110" : ""
        } ${disabled ? "opacity-40 grayscale" : ""}`}
        onError={(e) => {
          e.currentTarget.src =
            "https://placehold.co/20x20/4285F4/FFFFFF?text=G";
          e.currentTarget.onerror = null;
        }}
      />

      <div className="relative h-5 w-28 overflow-hidden">
        <span
          className={`absolute inset-0 flex items-center justify-center transition-transform duration-500 ease-in-out ${
            isHovering ? "-translate-y-full" : "translate-y-0"
          }`}
        >
          Add to Google Map
        </span>

        <span
          className={`absolute inset-0 flex items-center justify-center transition-transform duration-500 ease-in-out ${
            isHovering ? "translate-y-0" : "translate-y-full"
          }`}
        >
          Open Maps!
        </span>
      </div>
    </button>
  );
};

export default GoogleMapButton;
