"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Layers,
  Cpu,
  Clock,
  GitBranch,
  BarChart3,
  AlertTriangle,
  Fingerprint,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/workloads", label: "Workload Profiles", icon: Layers },
  { href: "/gpus", label: "GPU Health", icon: Cpu },
  { href: "/temporal", label: "Temporal", icon: Clock },
  { href: "/correlations", label: "Correlations", icon: GitBranch },
  { href: "/distributions", label: "Distributions", icon: BarChart3 },
  { href: "/anomalies", label: "Anomalies", icon: AlertTriangle },
  { href: "/profiling", label: "Profiling", icon: Fingerprint },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-56 shrink-0 border-r border-zinc-800 bg-zinc-950 flex flex-col">
      <div className="px-4 py-5 border-b border-zinc-800">
        <h1 className="text-sm font-semibold text-zinc-100 tracking-tight">
          GPU Telemetry EDA
        </h1>
        <p className="text-xs text-zinc-500 mt-0.5">H100 Behavioral Profiling</p>
      </div>
      <nav className="flex-1 py-3 px-2 space-y-0.5">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-md text-sm transition-colors ${
                active
                  ? "bg-zinc-800 text-zinc-100"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
              }`}
            >
              <Icon size={16} />
              {label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
