"use client";

import React from "react";

export default function ColorPalette() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-white p-6">
      <h1 className="text-3xl font-bold mb-10">Color Palette</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl">
        <div
          className="flex flex-col items-center justify-center rounded-xl shadow-lg p-6"
          style={{ backgroundColor: "#89C559" }}
        >
          <span className="text-xl font-semibold text-black">#89C559</span>
        </div>

        <div
          className="flex flex-col items-center justify-center rounded-xl shadow-lg p-6"
          style={{ backgroundColor: "#436029" }}
        >
          <span className="text-xl font-semibold text-white">#436029</span>
        </div>

        <div
          className="flex flex-col items-center justify-center rounded-xl shadow-lg p-6"
          style={{ backgroundColor: "#F0EDD1" }}
        >
          <span className="text-xl font-semibold text-black">#F0EDD1</span>
        </div>

        <div
          className="flex flex-col items-center justify-center rounded-xl shadow-lg p-6"
          style={{ backgroundColor: "#0C100E" }}
        >
          <span className="text-xl font-semibold text-white">#0C100E</span>
        </div>
      </div>
    </div>
  );
}
