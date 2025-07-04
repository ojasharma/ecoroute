"use client";

import { useState } from "react";
import MapSelector from "@/components/MapSelector";
import { Coordinate, RouteResponse } from "@/types/route";

export default function DashboardPage() {
    const [route, setRoute] = useState<Coordinate[] | null>(null);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    const handleRouteFetch = async (from: Coordinate, to: Coordinate) => {
        setError("");
        setLoading(true);
        try {
            const res = await fetch("/api/carbonroute", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ from, to }),
            });

            if (!res.ok) throw new Error("Failed to fetch route");

            const data: RouteResponse = await res.json();
            setRoute(data.path);
        } catch (err) {
            console.error(err);
            setError("Unable to fetch carbon-optimized route.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-900 to-blue-950 text-white px-4 py-8 flex justify-center items-center">
            <div className="bg-white/10 backdrop-blur-xl border border-white/20 p-8 rounded-2xl shadow-2xl w-full max-w-3xl">
                <h2 className="text-3xl font-bold text-center mb-6">
                    CO₂-Optimized Route Planner
                </h2>

                <p className="text-sm text-white/70 text-center mb-6">
                    Click on the map to select your <span className="font-semibold">starting</span> and <span className="font-semibold">ending</span> locations.
                </p>

                <MapSelector onSelect={handleRouteFetch} />

                <div className="mt-6 text-center">
                    {loading && <p className="text-cyan-300">Calculating optimal route...</p>}
                    {route && !loading && (
                        <p className="text-green-400 font-semibold">Route received successfully!</p>
                    )}
                    {error && <p className="text-red-400">{error}</p>}
                </div>
            </div>
        </div>
    );
}
