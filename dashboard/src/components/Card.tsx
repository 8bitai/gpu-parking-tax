import type { ReactNode } from "react";

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 ${className}`}
    >
      {children}
    </div>
  );
}

export function KpiCard({
  label,
  value,
  unit,
}: {
  label: string;
  value: string | number;
  unit?: string;
}) {
  return (
    <Card>
      <p className="text-xs text-zinc-500 uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-semibold text-zinc-100 mt-1">
        {value}
        {unit && <span className="text-sm text-zinc-400 ml-1">{unit}</span>}
      </p>
    </Card>
  );
}
