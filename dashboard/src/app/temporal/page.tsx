"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchApi } from "@/lib/api";
import { Card } from "@/components/Card";
import PlotlyChart from "@/components/PlotlyChart";
import { getWorkloadColor } from "@/lib/colors";

interface DiurnalData {
  workloads: string[];
  hours: number[];
  matrix: (number | null)[][];
  metric: string;
}

interface WeeklyData {
  days: string[];
  workloads: Record<string, Record<string, number>>;
  metric: string;
}

interface DriftData {
  dates: string[];
  mean: number[];
  std: number[];
  metric: string;
}

export default function TemporalPage() {
  const [metric, setMetric] = useState("DCGM_FI_DEV_GPU_UTIL");

  const { data: metrics } = useQuery({
    queryKey: ["metrics-list"],
    queryFn: () => fetchApi<string[]>("/api/distributions/metrics"),
  });

  const { data: diurnal } = useQuery({
    queryKey: ["temporal-diurnal", metric],
    queryFn: () => fetchApi<DiurnalData>(`/api/temporal/diurnal?metric=${metric}`),
  });

  const { data: weekly } = useQuery({
    queryKey: ["temporal-weekly", metric],
    queryFn: () => fetchApi<WeeklyData>(`/api/temporal/weekly?metric=${metric}`),
  });

  const { data: drift } = useQuery({
    queryKey: ["temporal-drift", metric],
    queryFn: () => fetchApi<DriftData>(`/api/temporal/drift?metric=${metric}`),
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold text-zinc-100">Temporal Patterns</h1>
        <select
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
          className="bg-zinc-800 border border-zinc-700 text-zinc-300 text-xs rounded px-2 py-1"
        >
          {metrics?.map((m) => (
            <option key={m} value={m}>
              {m.replace(/DCGM_FI_(?:DEV_|PROF_(?:PIPE_)?)?/g, "")}
            </option>
          ))}
        </select>
      </div>

      {/* Diurnal Heatmap */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          Diurnal Pattern (Hour × Workload)
        </h2>
        {diurnal && (
          <PlotlyChart
            className="h-72"
            data={[
              {
                z: diurnal.matrix,
                x: diurnal.hours.map((h) => `${h}:00`),
                y: diurnal.workloads,
                type: "heatmap" as const,
                colorscale: "Viridis",
                colorbar: { tickfont: { color: "#a1a1aa" } },
              },
            ]}
            layout={{
              xaxis: { title: "Hour of Day" },
              yaxis: { automargin: true },
            }}
          />
        )}
      </Card>

      {/* Weekly Bar */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">Weekly Pattern</h2>
        {weekly && (
          <PlotlyChart
            className="h-72"
            data={Object.entries(weekly.workloads).map(([wl, dayValues]) => ({
              type: "bar" as const,
              name: wl,
              x: weekly.days,
              y: weekly.days.map((d) => dayValues[d] ?? 0),
              marker: { color: getWorkloadColor(wl) },
            }))}
            layout={{
              barmode: "group" as const,
              xaxis: { title: "Day" },
              yaxis: { title: metric.replace(/DCGM_FI_(?:DEV_|PROF_(?:PIPE_)?)?/g, "") },
            }}
          />
        )}
      </Card>

      {/* Drift Detection */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          Rolling Mean/Std Drift
        </h2>
        {drift && (
          <PlotlyChart
            className="h-72"
            data={[
              {
                x: drift.dates,
                y: drift.mean,
                type: "scatter" as const,
                mode: "lines" as const,
                name: "Mean",
                line: { color: "#8b5cf6" },
              },
              {
                x: drift.dates,
                y: drift.mean.map((m, i) => m + (drift.std[i] ?? 0)),
                type: "scatter" as const,
                mode: "lines" as const,
                name: "+1σ",
                line: { color: "#8b5cf6", dash: "dot", width: 1 },
                showlegend: false,
              },
              {
                x: drift.dates,
                y: drift.mean.map((m, i) => m - (drift.std[i] ?? 0)),
                type: "scatter" as const,
                mode: "lines" as const,
                name: "-1σ",
                line: { color: "#8b5cf6", dash: "dot", width: 1 },
                fill: "tonexty" as const,
                fillcolor: "rgba(139, 92, 246, 0.1)",
                showlegend: false,
              },
            ]}
            layout={{
              xaxis: { title: "Date" },
              yaxis: { title: metric.replace(/DCGM_FI_(?:DEV_|PROF_(?:PIPE_)?)?/g, "") },
            }}
          />
        )}
      </Card>
    </div>
  );
}
