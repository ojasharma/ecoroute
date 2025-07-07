"use client";

import React, { useState, useEffect } from "react"; // Add useEffect
import { GeoapifyContext } from "@geoapify/react-geocoder-autocomplete";
import { LatLngTuple } from "leaflet";
import MapControls from "@/components/MapControls";
import MapDisplay from "@/components/MapDisplay";
import LogDisplay from "@/components/LogDisplay";

export default function Page() {
  const [source, setSource] = useState<any>(null);
  const [destination, setDestination] = useState<any>(null);
  const [vehicle, setVehicle] = useState("motorcycle");
  const [loading, setLoading] = useState(false);

  const [ecoRoute, setEcoRoute] = useState<LatLngTuple[]>([]);
  const [googleRoute, setGoogleRoute] = useState<LatLngTuple[]>([]);
  const [logs, setLogs] = useState<string[]>([]);

  const apiKey = process.env.NEXT_PUBLIC_GEOAPIFY_API_KEY;

  // Store the EventSource connection in a ref to manage it across renders
  const eventSourceRef = React.useRef<EventSource | null>(null);

  const handleRouteFind = () => {
    if (!source || !destination) {
      alert("Please select both source and destination.");
      return;
    }

    // Close any existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setLoading(true);
    setLogs([]);
    setEcoRoute([]);
    setGoogleRoute([]);

    const origin = `${source.properties.lat},${source.properties.lon}`;
    const destinationCoords = `${destination.properties.lat},${destination.properties.lon}`;

    // Construct the URL with query parameters
    const url = `http://127.0.0.1:8000/route/stream?origin=${origin}&destination=${destinationCoords}&vehicle=${vehicle}`;
    
    // Create a new EventSource connection
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    // Handle incoming messages (events)
    eventSource.onmessage = (event) => {
      const parsedData = JSON.parse(event.data);

      if (parsedData.type === 'log') {
        setLogs(prevLogs => [...prevLogs, parsedData.message]);
      } else if (parsedData.type === 'result') {
        const { eco_route, google_route } = parsedData.data;
        setEcoRoute(eco_route as LatLngTuple[]);
        setGoogleRoute(google_route as LatLngTuple[]);
        setLoading(false); // Stop loading when we get the final result
        eventSource.close(); // Close the connection
      } else if (parsedData.type === 'error') {
        setLogs(prevLogs => [...prevLogs, `[ERROR] ${parsedData.message}`]);
        alert(`Error: ${parsedData.message}`);
        setLoading(false);
        eventSource.close();
      }
    };

    // Handle connection errors
    eventSource.onerror = (err) => {
      console.error("EventSource failed:", err);
      setLogs(prevLogs => [...prevLogs, "[ERROR] Connection to server lost."]);
      setLoading(false);
      eventSource.close();
    };
  };

  // Cleanup effect to close the connection when the component unmounts
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // ... (the rest of your component's return statement remains the same)
  // ...
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