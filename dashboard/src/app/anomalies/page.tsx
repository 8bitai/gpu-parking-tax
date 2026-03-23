"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchApi } from "@/lib/api";
import { Card } from "@/components/Card";
import PlotlyChart from "@/components/PlotlyChart";

interface XidEvent {
  timestamp: string;
  gpu: string;
  Hostname: string;
  DCGM_FI_DEV_XID_ERRORS: number;
}

interface ViolationRow {
  gpu: string;
  Hostname: string;
  power?: number;
  thermal?: number;
}

interface EccTrend {
  dates: string[];
  metrics: Record<string, number[]>;
}

const GPU_COLORS = [
  "#8b5cf6", "#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#3b82f6",
  "#ec4899", "#84cc16", "#f97316", "#6366f1", "#14b8a6", "#eab308",
  "#a855f7", "#22d3ee", "#fb923c", "#4ade80",
];

export default function AnomaliesPage() {
  const { data: xidEvents } = useQuery({
    queryKey: ["anomalies-xid"],
    queryFn: () => fetchApi<XidEvent[]>("/api/anomalies/xid-events"),
  });

  const { data: violations } = useQuery({
    queryKey: ["anomalies-violations"],
    queryFn: () => fetchApi<ViolationRow[]>("/api/anomalies/violations-summary"),
  });

  const { data: eccTrend } = useQuery({
    queryKey: ["anomalies-ecc"],
    queryFn: () => fetchApi<EccTrend>("/api/anomalies/ecc-trend"),
  });

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-zinc-100">Anomalies</h1>

      {/* XID Timeline */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          XID Error Timeline
        </h2>
        {xidEvents && xidEvents.length > 0 ? (
          <PlotlyChart
            className="h-72"
            data={(() => {
              const gpus = [...new Set(xidEvents.map((e) => e.gpu))];
              return gpus.map((g, i) => {
                const events = xidEvents.filter((e) => e.gpu === g);
                return {
                  x: events.map((e) => e.timestamp),
                  y: events.map((e) => e.DCGM_FI_DEV_XID_ERRORS),
                  type: "scatter" as const,
                  mode: "markers" as const,
                  name: `GPU ${g}`,
                  marker: { color: GPU_COLORS[i % GPU_COLORS.length], size: 6 },
                };
              });
            })()}
            layout={{
              xaxis: { title: "Time" },
              yaxis: { title: "XID Error Code" },
            }}
          />
        ) : (
          <p className="text-sm text-emerald-500">No XID errors detected.</p>
        )}
      </Card>

      {/* Violations Summary */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          Violation Summary by GPU
        </h2>
        {violations && (
          <PlotlyChart
            className="h-72"
            data={[
              {
                type: "bar" as const,
                name: "Power",
                x: violations.map((v) => `GPU ${v.gpu}`),
                y: violations.map((v) => v.power ?? 0),
                marker: { color: "#f59e0b" },
              },
              {
                type: "bar" as const,
                name: "Thermal",
                x: violations.map((v) => `GPU ${v.gpu}`),
                y: violations.map((v) => v.thermal ?? 0),
                marker: { color: "#ef4444" },
              },
            ]}
            layout={{
              barmode: "stack" as const,
              xaxis: { title: "GPU" },
              yaxis: { title: "Violation Count" },
            }}
          />
        )}
      </Card>

      {/* ECC Trend */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          ECC Error Trend
        </h2>
        {eccTrend && Object.keys(eccTrend.metrics).length > 0 ? (
          <PlotlyChart
            className="h-72"
            data={Object.entries(eccTrend.metrics).map(([metric, vals], i) => ({
              x: eccTrend.dates,
              y: vals,
              type: "scatter" as const,
              mode: "lines+markers" as const,
              name: metric.replace(/DCGM_FI_(?:DEV_)?/g, ""),
              marker: { color: GPU_COLORS[i % GPU_COLORS.length], size: 4 },
            }))}
            layout={{
              xaxis: { title: "Date" },
              yaxis: { title: "Max ECC Errors" },
            }}
          />
        ) : (
          <p className="text-sm text-emerald-500">No ECC errors detected.</p>
        )}
      </Card>
    </div>
  );
}
