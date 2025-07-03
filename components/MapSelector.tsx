// components/MapSelector.tsx
"use client";

import { useState } from "react";
import { Coordinate } from "@/types/route";
import dynamic from "next/dynamic";
import { useMapEvents} from "react-leaflet";
import "@/lib/fixLeafletIcons";
import "leaflet/dist/leaflet.css";

// Dynamically load Leaflet components to prevent SSR issues
const MapContainer = dynamic(() => import("react-leaflet").then(mod => mod.MapContainer), { ssr: false });
const TileLayer = dynamic(() => import("react-leaflet").then(mod => mod.TileLayer), { ssr: false });
const Marker = dynamic(() => import("react-leaflet").then(mod => mod.Marker), { ssr: false });

dynamic(() => import("react-leaflet").then(mod => mod.Polyline), { ssr: false });
// Wrap useMapEvents in a client-only subcomponent
const LocationPicker = dynamic(() =>
    Promise.resolve(function LocationPicker({ onPick }: { onPick: (coord: Coordinate) => void }) {
        useMapEvents({
            click(e) {
                onPick([e.latlng.lat, e.latlng.lng]);
            },
        });
        return null;
    }), { ssr: false }
);

interface Props {
    onSelect: (from: Coordinate, to: Coordinate) => void;
}

export default function MapSelector({ onSelect }: Props) {
    const [from, setFrom] = useState<Coordinate | null>(null);
    const [to, setTo] = useState<Coordinate | null>(null);

    const handlePick = (coord: Coordinate) => {
        if (!from) setFrom(coord);
        else if (!to) setTo(coord);
    };

    const handleReset = () => {
        setFrom(null);
        setTo(null);
    };

    const handleSubmit = () => {
        if (from && to) onSelect(from, to);
    };

    return (
        <div className="space-y-4">
            <div className="rounded-xl overflow-hidden border border-white/20 shadow-md">
                <MapContainer
                    center={[30.7333, 76.7794]}
                    zoom={13}
                    scrollWheelZoom
                    className="h-[400px] w-full"
                >
                    <TileLayer
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        attribution='&copy; OpenStreetMap contributors'
                    />
                    <LocationPicker onPick={handlePick} />
                    {from && <Marker position={from} />}
                    {to && <Marker position={to} />}
                </MapContainer>
            </div>

            <div className="flex justify-between">
                <button
                    onClick={handleSubmit}
                    disabled={!from || !to}
                    className="bg-cyan-500 hover:bg-cyan-600 text-white px-4 py-2 rounded-md font-medium disabled:opacity-40 disabled:cursor-not-allowed"
                >
                    Get CO₂ Friendly Route
                </button>
                <button
                    onClick={handleReset}
                    className="text-sm text-red-300 hover:text-red-500"
                >
                    Reset Selection
                </button>
            </div>
        </div>
    );
}
