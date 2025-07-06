"use client";

import React, { useState } from "react";
import { GeoapifyContext } from "@geoapify/react-geocoder-autocomplete";
import axios from "axios";
import MapControls from "@/components/MapControls"; // Adjust path if needed
import MapDisplay from "@/components/MapDisplay"; // Adjust path if needed

export default function Page() {
  const [source, setSource] = useState<any>(null);
  const [destination, setDestination] = useState<any>(null);
  const [vehicle, setVehicle] = useState("motorcycle");
  const [loading, setLoading] = useState(false);

  const apiKey = process.env.NEXT_PUBLIC_GEOAPIFY_API_KEY;

  const handleRouteFind = () => {
    if (!source || !destination) {
      alert("Please select both source and destination.");
      return;
    }

    setLoading(true);

    const origin = [source.properties.lat, source.properties.lon];
    const destinationCoords = [
      destination.properties.lat,
      destination.properties.lon,
    ];

    axios
      .post("http://127.0.0.1:8000/route", {
        origin,
        destination: destinationCoords,
        vehicle,
      })
      .then((response) => {
        console.log("Route response:", response.data);
        // You'll likely want to do something with the route data here,
        // such as drawing it on the map.
      })
      .catch((error) => {
        console.error("Error fetching route:", error);
        alert("Failed to fetch route.");
      })
      .finally(() => {
        setLoading(false);
      });
  };

  if (!apiKey) {
    return (
      <div className="text-red-500 text-center">
        Geoapify API key is missing.
      </div>
    );
  }

  return (
    <GeoapifyContext apiKey={apiKey}>
      <div
        className="min-h-screen flex items-center justify-center"
        style={{ backgroundColor: "#365241" }}
      >
        <div
          className="rounded-2xl w-[90vw] h-[95vh] p-8"
          style={{ backgroundColor: "#F0EDD1", color: "#243931" }}
        >
          <h1 className="text-5xl font-bold mb-4">Dashboard</h1>

          <div className="flex justify-between items-start gap-4">
            <MapControls
              source={source}
              setSource={setSource}
              destination={destination}
              setDestination={setDestination}
              vehicle={vehicle}
              setVehicle={setVehicle}
              handleRouteFind={handleRouteFind}
              loading={loading}
            />

            <MapDisplay
              source={source}
              destination={destination}
              loading={loading}
            />
          </div>
        </div>
      </div>
    </GeoapifyContext>
  );
}
