"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Clock, Users, Settings, LogOut } from "lucide-react";
import React from "react";

// Navigation items array
const navItems = [
  { href: "/dashboard", icon: <Home size={22} />, label: "Dashboard" },
  { href: "/history", icon: <Clock size={22} />, label: "History" },
  { href: "/drivers", icon: <Users size={22} />, label: "Drivers" },
  { href: "/settings", icon: <Settings size={22} />, label: "Settings" },
];

// Props interface for AnimatedLabel
interface AnimatedLabelProps {
  text: string;
  delay: string;
}

// Reusable component for the animated text labels
const AnimatedLabel: React.FC<AnimatedLabelProps> = ({ text, delay }) => (
  <div className="overflow-hidden">
    <span
      className={`
        whitespace-nowrap text-sm font-medium
        opacity-0 -translate-y-2 transform
        transition-all duration-0 delay-0
        group-hover:opacity-100 group-hover:translate-y-0 group-hover:duration-300 ${delay}
      `}
    >
      {text}
    </span>
  </div>
);

export default function Navbar() {
  const pathname = usePathname();

  const handleLogout = () => {
    // Add your logout logic here
    console.log("Logout clicked");
  };

  return (
    <aside className="fixed top-0 left-0 h-screen w-20 bg-[#233830] text-[#F0EDD1] flex flex-col z-50 hover:w-56 transition-all duration-300 ease-in-out group">
      {/* Logo and Brand Name */}
      <div className="flex items-center h-20 w-full flex-shrink-0 pl-6">
        <Link href="/dashboard" className="flex items-center outline-none">
          <Image
            src="/logo.png"
            alt="Eco Route Logo"
            width={32}
            height={32}
          />
          <div className="overflow-hidden ml-3">
            <span
              className={`
                whitespace-nowrap text-lg font-bold
                opacity-0 -translate-y-2 transform
                transition-all duration-0 delay-0
                group-hover:opacity-100 group-hover:translate-y-0 group-hover:duration-300 group-hover:delay-150
              `}
            >
              ECO ROUTE
            </span>
          </div>
        </Link>
      </div>

      {/* Main Navigation Links */}
      <nav className="flex flex-col flex-grow items-center gap-2 p-2">
        {navItems.map((item) => (
          <Link key={item.href} href={item.href} passHref legacyBehavior>
            <a
              className={`flex items-center w-full h-12 rounded-xl transition-all duration-200 ease-in-out
                ${
                  pathname === item.href
                    ? "bg-[#F0EDD1] text-[#233830] shadow-lg"
                    : "hover:bg-white/10 text-[#F0EDD1]"
                }
              `}
              title={item.label}
            >
              <div className="flex items-center justify-center w-16 flex-shrink-0">
                {item.icon}
              </div>
              <AnimatedLabel text={item.label} delay="group-hover:delay-200" />
            </a>
          </Link>
        ))}
      </nav>

      {/* Logout Button */}
      <div className="p-2">
        <button
          onClick={handleLogout}
          className="flex items-center w-full h-12 rounded-md hover:bg-white/10 text-[#F0EDD1] transition-colors duration-200 ease-in-out"
          title="Logout"
        >
          <div className="flex items-center justify-center w-16 flex-shrink-0">
            <LogOut size={22} />
          </div>
          <AnimatedLabel text="Logout" delay="group-hover:delay-200" />
        </button>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 h-10 w-full flex items-center justify-center">
        <div className="overflow-hidden">
          <span
            className={`
              whitespace-nowrap text-[10px] text-white/30
              opacity-0 translate-y-2 transform
              transition-all duration-0 delay-0
              group-hover:opacity-100 group-hover:translate-y-0 group-hover:duration-300 group-hover:delay-200
            `}
          >
            © 2025 ECO
          </span>
        </div>
      </div>
    </aside>
  );
}
