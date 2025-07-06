"use client";

import React from "react";

export default function HomePage() {
  const styles = {
    body: {
      margin: 0,
      padding: 0,
      boxSizing: "border-box",
      outline: 0,
      fontFamily: "sans-serif",
    } as React.CSSProperties,

    homePage: {
      backgroundColor: "#000000",
      height: "100vh",
      display: "grid",
      placeItems: "center",
    } as React.CSSProperties,

    content: {
      display: "flex",
      flexDirection: "column",
      gap: "16px",
      position: "relative",
      zIndex: 1,
    } as React.CSSProperties,

    quote: {
      color: "white",
      fontWeight: "bold",
      fontStyle: "italic",
      fontSize: "64px",
      padding: "10px",
      backgroundColor: "#101010",
      borderRadius: "10px",
    } as React.CSSProperties,

    by: {
      color: "#00e1ff",
    } as React.CSSProperties,

    blobOuterContainer: {
      position: "fixed",
      height: "100%",
      width: "100%",
      zIndex: 0,
      inset: 0,
      margin: "auto",
      filter: "blur(100px)",
    } as React.CSSProperties,

    blobInnerContainer: {
      borderRadius: "99999px",
      position: "absolute",
      inset: 0,
      margin: "auto",
      width: "100vw",
      height: "100vh",
      minWidth: "1000px",
      overflow: "hidden",
      backgroundColor: "#fff",
      transform: "scale(0.8)",
    } as React.CSSProperties,

    blob: {
      position: "absolute",
      width: "100vw",
      height: "100vh",
      inset: 0,
      margin: "auto",
      background:
        "conic-gradient(from 0deg, #08f, #f60, #bbffa1, #4c00ff, #ab2666, #09f)",
      animation: "spinBlob 8s linear infinite",
    } as React.CSSProperties,
  };

  return (
    <>
      <style>{`
        @keyframes spinBlob {
          0% {
            transform: rotate(0deg) scale(2);
          }
          100% {
            transform: rotate(360deg) scale(2);
          }
        }
      `}</style>

      <div style={styles.blobOuterContainer}>
        <div style={styles.blobInnerContainer}>
          <div style={styles.blob}></div>
        </div>
      </div>

      <div style={styles.homePage}>
        <div style={styles.content}>
          {/* Uncomment if you want quote */}
          {/* <div style={styles.quote}>CSS ONLY</div>
          <div style={styles.by}>- Sun Tzu, The Art of War</div> */}
        </div>
      </div>
    </>
  );
}
