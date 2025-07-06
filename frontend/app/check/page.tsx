"use client";

import React, { useEffect, useState } from "react";
import {
  GeoapifyContext,
  GeoapifyGeocoderAutocomplete,
} from "@geoapify/react-geocoder-autocomplete";
import "@geoapify/geocoder-autocomplete/styles/minimal.css";

export default function TestAutocomplete() {
  const [error, setError] = useState<string | null>(null);
  const apiKey = process.env.NEXT_PUBLIC_GEOAPIFY_API_KEY;

  // Log API key availability for debugging
  useEffect(() => {
    if (!apiKey) {
      console.error(
        "Geoapify API key is missing. Please set NEXT_PUBLIC_GEOAPIFY_API_KEY in your .env.local file."
      );
      setError("Geoapify API key is missing.");
    }
  }, [apiKey]);

  // Debug component mount and React version
  useEffect(() => {
    console.log(
      "TestAutocomplete component mounted, React version:",
      React.version
    );
  }, []);

  if (error) {
    return <div className="text-red-500 text-center">{error}</div>;
  }

  return (
    <GeoapifyContext apiKey={apiKey}>
      <div className="min-h-screen flex flex-col items-center justify-center gap-4 bg-gray-100 p-10">
        <div className="w-[300px] px-4 py-2 border rounded-xl bg-white">
          <GeoapifyGeocoderAutocomplete
            placeholder="Search Source"
            placeSelect={(place) => console.log("Source selected:", place)}
          />
        </div>
        <div className="w-[300px] px-4 py-2 border rounded-xl bg-white">
          <GeoapifyGeocoderAutocomplete
            placeholder="Search Destination"
            placeSelect={(place) => console.log("Destination selected:", place)}
          />
        </div>
      </div>
    </GeoapifyContext>
  );
}
