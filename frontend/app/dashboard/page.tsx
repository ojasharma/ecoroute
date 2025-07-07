"use client";

import React, { useState, useEffect } from "react";
import { GeoapifyContext } from "@geoapify/react-geocoder-autocomplete";
import { LatLngTuple } from "leaflet";
import MapControls from "@/components/MapControls";
import MapDisplay from "@/components/MapDisplay";
import LogDisplay from "@/components/LogDisplay";
import GoogleMapButton from "@/components/UI elements/GMapButtom"; // Add this

export default function Page() {
  const [source, setSource] = useState<any>(null);
  const [destination, setDestination] = useState<any>(null);
  const [vehicle, setVehicle] = useState("motorcycle");
  const [loading, setLoading] = useState(false);

  const [ecoRoute, setEcoRoute] = useState<LatLngTuple[]>([]);
  const [googleRoute, setGoogleRoute] = useState<LatLngTuple[]>([]);
  const [logs, setLogs] = useState<string[]>([]);

  const apiKey = process.env.NEXT_PUBLIC_GEOAPIFY_API_KEY;
  const eventSourceRef = React.useRef<EventSource | null>(null);

  const handleRouteFind = () => {
    if (!source || !destination) {
      alert("Please select both source and destination.");
      return;
    }

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setLoading(true);
    setLogs([]);
    setEcoRoute([]);
    setGoogleRoute([]);

    const origin = `${source.properties.lat},${source.properties.lon}`;
    const destinationCoords = `${destination.properties.lat},${destination.properties.lon}`;
    const url = `http://127.0.0.1:8000/route/stream?origin=${origin}&destination=${destinationCoords}&vehicle=${vehicle}`;

    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      const parsedData = JSON.parse(event.data);

      if (parsedData.type === 'log') {
        setLogs(prev => [...prev, parsedData.message]);
      } else if (parsedData.type === 'result') {
        const { eco_route, google_route } = parsedData.data;
        setEcoRoute(eco_route as LatLngTuple[]);
        setGoogleRoute(google_route as LatLngTuple[]);
        setLoading(false);
        eventSource.close();
      } else if (parsedData.type === 'error') {
        setLogs(prev => [...prev, `[ERROR] ${parsedData.message}`]);
        alert(`Error: ${parsedData.message}`);
        setLoading(false);
        eventSource.close();
      }
    };

    eventSource.onerror = (err) => {
      console.error("EventSource failed:", err);
      setLogs(prev => [...prev, "[ERROR] Connection to server lost."]);
      setLoading(false);
      eventSource.close();
    };
  };

  const handleGoogleMapOpen = () => {
    if (!ecoRoute || ecoRoute.length < 2) return;

    let path_coords = ecoRoute;
    const max_points = 25;

    if (path_coords.length > max_points) {
      const step = Math.floor(path_coords.length / (max_points - 1));
      path_coords = path_coords.filter((_, index) => index % step === 0);
      if (path_coords[path_coords.length - 1] !== ecoRoute[ecoRoute.length - 1]) {
        path_coords.push(ecoRoute[ecoRoute.length - 1]);
      }
    }

    const origin = `${path_coords[0][0]},${path_coords[0][1]}`;
    const destination = `${path_coords[path_coords.length - 1][0]},${path_coords[path_coords.length - 1][1]}`;
    const waypoints = path_coords
      .slice(1, -1)
      .map(([lat, lon]) => `${lat},${lon}`)
      .join("|");

    const url = `https://www.google.com/maps/dir/?api=1&origin=${origin}&destination=${destination}&waypoints=${waypoints}`;
    window.open(url, "_blank");
  };

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

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

          <div className="flex justify-between items-start gap-4 h-[calc(100%-60px)]">
            <div className="flex flex-col gap-4 w-1/3 h-full">
              <MapControls
                source={source}
                setSource={setSource}
                destination={destination}
                setDestination={setDestination}
                vehicle={vehicle}
                setVehicle={setVehicle}
                handleRouteFind={handleRouteFind}
                loading={loading}
                routeReady={ecoRoute.length > 0 || googleRoute.length > 0}
              />

              <GoogleMapButton
                disabled={ecoRoute.length === 0}
                onClick={handleGoogleMapOpen}
              />

              {(loading || logs.length > 0) && (
                <LogDisplay loading={loading} logs={logs} />
              )}
            </div>

            <div className="w-2/3 h-full">
              <MapDisplay
                source={source}
                destination={destination}
                loading={loading}
                ecoRoute={ecoRoute}
                googleRoute={googleRoute}
              />
            </div>
          </div>
        </div>
      </div>
    </GeoapifyContext>
  );
}
