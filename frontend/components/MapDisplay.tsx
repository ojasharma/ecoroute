"use client";

import React, { useRef, useEffect } from "react";
import L, { Marker } from "leaflet";
import "leaflet/dist/leaflet.css";
import LoadingOverlay from "./LoadingOverlay"; // Import the new component

interface MapDisplayProps {
  source: any;
  destination: any;
  loading: boolean;
}

const createCustomIcon = (
  iconUrl: string,
  size: [number, number] = [32, 32],
  anchor: [number, number] = [16, 32]
) => {
  return L.icon({
    iconUrl,
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

export default function MapDisplay({
  source,
  destination,
  loading,
}: MapDisplayProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const sourceMarkerRef = useRef<Marker | null>(null);
  const destinationMarkerRef = useRef<Marker | null>(null);

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

  return (
    <div
      className="relative rounded-2xl overflow-hidden"
      style={{
        backgroundColor: "#233830",
        width: "60vw",
        height: "75vh",
      }}
    >
      <div ref={mapRef} style={{ width: "100%", height: "100%" }} />

      {/* Render the LoadingOverlay component */}
      <LoadingOverlay isLoading={loading} />
    </div>
  );
  
}
