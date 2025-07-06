"use client";

import React, { useRef, useEffect, useState } from "react";
import L, { Marker } from "leaflet";
import {
  GeoapifyContext,
  GeoapifyGeocoderAutocomplete,
} from "@geoapify/react-geocoder-autocomplete";
import "@geoapify/geocoder-autocomplete/styles/minimal.css";
import "leaflet/dist/leaflet.css";
import axios from "axios";

// Custom icon creation function
const createCustomIcon = (
  iconUrl: string,
  size: [number, number] = [32, 32],
  anchor: [number, number] = [16, 32]
) => {
  return L.icon({
    iconUrl: iconUrl,
    iconSize: size,
    iconAnchor: anchor,
    popupAnchor: [0, -32] as [number, number],
    shadowUrl:
      "https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png",
    shadowSize: [41, 41] as [number, number],
    shadowAnchor: [12, 41] as [number, number],
  });
};

const sourceIcon = createCustomIcon("/marker1.png");
const destinationIcon = createCustomIcon("/marker2.png");

export default function Page() {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const sourceMarkerRef = useRef<Marker | null>(null);
  const destinationMarkerRef = useRef<Marker | null>(null);

  const [source, setSource] = useState<any>(null);
  const [destination, setDestination] = useState<any>(null);
  const [vehicle, setVehicle] = useState("motorcycle");

  const apiKey = process.env.NEXT_PUBLIC_GEOAPIFY_API_KEY;

  useEffect(() => {
    if (!mapRef.current) return;

    const map = L.map(mapRef.current).setView([28.6139, 77.209], 5);
    mapInstanceRef.current = map;

    L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_labels_under/{z}/{x}/{y}.png",
      {
        attribution:
          '© <a href="https://carto.com/">CARTO</a> | © OpenStreetMap contributors',
      }
    ).addTo(map);

    return () => {
      if (sourceMarkerRef.current) {
        sourceMarkerRef.current.remove();
        sourceMarkerRef.current = null;
      }
      if (destinationMarkerRef.current) {
        destinationMarkerRef.current.remove();
        destinationMarkerRef.current = null;
      }
      map.remove();
    };
  }, []);

  useEffect(() => {
    if (!mapInstanceRef.current) return;
    const map = mapInstanceRef.current;

    if (sourceMarkerRef.current) {
      sourceMarkerRef.current.remove();
      sourceMarkerRef.current = null;
    }

    if (source?.properties?.lat && source?.properties?.lon) {
      console.log(
        "Source coordinates:",
        source.properties.lat,
        source.properties.lon
      );

      const marker = L.marker([source.properties.lat, source.properties.lon], {
        title: source.properties.formatted || "Source",
        icon: sourceIcon,
      }).addTo(map);
      sourceMarkerRef.current = marker;

      if (destination?.properties?.lat && destination?.properties?.lon) {
        const bounds = L.latLngBounds(
          [source.properties.lat, source.properties.lon],
          [destination.properties.lat, destination.properties.lon]
        );
        map.fitBounds(bounds, { padding: [50, 50] });
      } else {
        map.setView([source.properties.lat, source.properties.lon], 10);
      }
    }
  }, [source, destination]);

  useEffect(() => {
    if (!mapInstanceRef.current) return;
    const map = mapInstanceRef.current;

    if (destinationMarkerRef.current) {
      destinationMarkerRef.current.remove();
      destinationMarkerRef.current = null;
    }

    if (destination?.properties?.lat && destination?.properties?.lon) {
      console.log(
        "Destination coordinates:",
        destination.properties.lat,
        destination.properties.lon
      );

      const marker = L.marker(
        [destination.properties.lat, destination.properties.lon],
        {
          title: destination.properties.formatted || "Destination",
          icon: destinationIcon,
        }
      ).addTo(map);
      destinationMarkerRef.current = marker;

      if (source?.properties?.lat && source?.properties?.lon) {
        const bounds = L.latLngBounds(
          [source.properties.lat, source.properties.lon],
          [destination.properties.lat, destination.properties.lon]
        );
        map.fitBounds(bounds, { padding: [50, 50] });
      } else {
        map.setView(
          [destination.properties.lat, destination.properties.lon],
          10
        );
      }
    }
  }, [destination, source]);

const handleRouteFind = () => {
  if (!source || !destination) {
    alert("Please select both source and destination.");
    return;
  }

  const payload = {
    origin: source.geometry.coordinates.reverse(),       // [lat, lon]
    destination: destination.geometry.coordinates.reverse(), // [lat, lon]
    vehicle,
  };

  console.log("Payload for route request:", payload);

  axios
    .post("http://127.0.0.1:8000/route", payload)
    .then((response) => {
      console.log("Route response:", response.data);
    })
    .catch((error) => {
      console.error("Error fetching route:", error);
      alert("Failed to fetch route.");
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
            {/* Left Panel */}
            <div
              className="rounded-2xl p-6 flex flex-col gap-4"
              style={{
                backgroundColor: "#233830",
                width: "23vw",
                height: "75vh",
                color: "#F0EDD1",
              }}
            >
              <div>
                <label
                  className="block mb-1 text-sm font-medium"
                  htmlFor="source"
                >
                  Source
                </label>
                <div
                  className="w-full rounded-2xl px-4 py-2"
                  style={{
                    backgroundColor: "rgba(255, 255, 255, 0.1)",
                    backdropFilter: "blur(10px)",
                    border: "1px solid rgba(255, 255, 255, 0.2)",
                    position: "relative",
                    zIndex: 99,
                  }}
                >
                  <GeoapifyGeocoderAutocomplete
                    placeholder="Enter source"
                    placeSelect={(place) => setSource(place)}
                  />
                </div>
              </div>

              <div>
                <label
                  className="block mb-1 text-sm font-medium"
                  htmlFor="destination"
                >
                  Destination
                </label>
                <div
                  className="w-full rounded-2xl px-4 py-2"
                  style={{
                    backgroundColor: "rgba(255, 255, 255, 0.1)",
                    backdropFilter: "blur(10px)",
                    border: "1px solid rgba(255, 255, 255, 0.2)",
                    position: "relative",
                    zIndex: 9,
                  }}
                >
                  <GeoapifyGeocoderAutocomplete
                    placeholder="Enter destination"
                    placeSelect={(place) => setDestination(place)}
                  />
                </div>
              </div>

              <div>
                <label
                  className="block mb-1 text-sm font-medium"
                  htmlFor="vehicle"
                >
                  Vehicle Type
                </label>
                <select
                  id="vehicle"
                  value={vehicle}
                  onChange={(e) => setVehicle(e.target.value)}
                  className="w-full rounded-2xl px-4 py-2"
                  style={{
                    backgroundColor: "#ACC08D",
                    color: "#233830",
                    border: "1px solid rgba(255, 255, 255, 0.2)",
                  }}
                >
                  <option value="motorcycle">Motorcycle</option>
                  <option value="pass. car">Passenger Car</option>
                  <option value="LCV">Light Commercial Vehicle</option>
                  <option value="coach">Coach</option>
                  <option value="HGV">Heavy Goods Vehicle</option>
                  <option value="urban bus">Urban Bus</option>
                </select>
              </div>

              <button
                onClick={handleRouteFind}
                className="mt-auto rounded-2xl px-6 py-3 text-lg font-semibold"
                style={{ backgroundColor: "#ACC08D", color: "#233830" }}
              >
                Find Route
              </button>
            </div>

            {/* Map Panel */}
            <div
              className="rounded-2xl overflow-hidden"
              style={{
                backgroundColor: "#233830",
                width: "60vw",
                height: "75vh",
              }}
            >
              <div ref={mapRef} style={{ width: "100%", height: "100%" }} />
            </div>
          </div>
        </div>
      </div>
    </GeoapifyContext>
  );
}
