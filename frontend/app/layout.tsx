import type { Metadata } from "next";
import "./globals.css";
import "leaflet/dist/leaflet.css";
import Navbar from "@/components/NavBar";

export const metadata: Metadata = {
  title: "ECO ROUTE",
  description: "Path finder for least CO2 emition",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head />
      <body className="antialiased">
        <Navbar />
        <main className="ml-20">{children}</main>
      </body>
    </html>
  );
}
