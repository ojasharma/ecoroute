"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Clock, Users, Settings, LogOut } from "lucide-react";
import React from "react";

const navItems = [
  { href: "/dashboard", icon: <Home size={22} />, label: "Dashboard" },
  { href: "/history", icon: <Clock size={22} />, label: "History" },
  { href: "/drivers", icon: <Users size={22} />, label: "Drivers" },
  { href: "/settings", icon: <Settings size={22} />, label: "Settings" },
];

interface AnimatedLabelProps {
  text: string;
  active?: boolean;
}

const AnimatedLabel: React.FC<AnimatedLabelProps> = ({ text, active }) => (
  <div className="overflow-hidden">
    <span
      className={`whitespace-nowrap text-sm font-medium
        opacity-0 -translate-y-2 transform
        transition-all group-hover:opacity-100 group-hover:translate-y-0 group-hover:duration-300
        ${active ? "text-[#233830]" : "text-[#ACC08D]"}
        transition-colors duration-300
      `}
    >
      {text}
    </span>
  </div>
);

export default function Navbar() {
  const pathname = usePathname();

  const handleLogout = () => {
    console.log("Logout clicked");
  };

  return (
    <aside className="fixed top-0 left-0 h-screen w-20 bg-[#233830] text-[#ACC08D] flex flex-col z-500 hover:w-56 transition-all duration-300 ease-in-out group">
      {/* Logo and Brand */}
      <div className="flex items-center h-20 w-full flex-shrink-0 pl-6">
        <Link href="/dashboard" className="flex items-center outline-none">
          <Image src="/logo.png" alt="Eco Route Logo" width={32} height={32} />
          <div className="overflow-hidden ml-3">
            <span
              className={`whitespace-nowrap text-lg font-bold text-[#ACC08D]
                opacity-0 -translate-y-2 transform
                transition-all group-hover:opacity-100 group-hover:translate-y-0 group-hover:duration-300
              `}
            >
              ECO ROUTE
            </span>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col flex-grow items-center gap-2">
        {navItems.map((item) => (
          <Link key={item.href} href={item.href} passHref legacyBehavior>
            <a
              className={`flex items-center h-12 rounded-r-2xl transition-all duration-200 ease-in-out
                ${
                  pathname === item.href
                    ? "bg-[#ACC08D] text-[#233830] shadow-lg mr-2"
                    : "hover:bg-white/10 text-[#ACC08D]"
                }
                transition-colors w-[calc(100%-0.5rem)]
              `}
              title={item.label}
            >
              <div className="flex items-center justify-center w-16 flex-shrink-0">
                {React.cloneElement(item.icon, {
                  color: pathname === item.href ? "#233830" : "#ACC08D",
                })}
              </div>
              <AnimatedLabel
                text={item.label}
                active={pathname === item.href}
              />
            </a>
          </Link>
        ))}
      </nav>

      {/* Logout */}
      <div>
        <button
          onClick={handleLogout}
          className="flex items-center w-[calc(100%-0.5rem)] h-12 rounded-r-2xl hover:bg-white/10 text-[#ACC08D] transition-colors duration-200 ease-in-out mr-2"
          title="Logout"
        >
          <div className="flex items-center justify-center w-16 flex-shrink-0">
            <LogOut size={22} color="#ACC08D" />
          </div>
          <AnimatedLabel text="Logout" />
        </button>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 h-10 w-full flex items-center justify-center">
        <div className="overflow-hidden">
          <span
            className={`whitespace-nowrap text-[10px] text-[#ACC08D]/30
              opacity-0 translate-y-2 transform
              transition-all group-hover:opacity-100 group-hover:translate-y-0 group-hover:duration-300
            `}
          >
            © 2025 ECO
          </span>
        </div>
      </div>
    </aside>
  );
}
