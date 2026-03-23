"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchApi } from "@/lib/api";
import { Card } from "@/components/Card";
import PlotlyChart from "@/components/PlotlyChart";

interface GpuHealth {
  gpu: string;
  Hostname: string;
  UUID: string;
  mean_util: number;
  mean_temp: number;
  mean_power: number;
  ecc_sbe_total?: number;
  ecc_dbe_total?: number;
  xid_errors?: number;
  power_violations?: number;
  thermal_violations?: number;
}

interface HeatmapData {
  gpus: string[];
  dates: string[];
  matrix: (number | null)[][];
  metric: string;
}

export default function GpusPage() {
  const [heatMetric, setHeatMetric] = useState("DCGM_FI_DEV_GPU_UTIL");

  const { data: health } = useQuery({
    queryKey: ["gpu-health"],
    queryFn: () => fetchApi<GpuHealth[]>("/api/gpus/health-summary"),
  });

  const { data: heatmap } = useQuery({
    queryKey: ["gpu-heatmap", heatMetric],
    queryFn: () =>
      fetchApi<HeatmapData>(`/api/gpus/heatmap?metric=${heatMetric}`),
  });

  const { data: metrics } = useQuery({
    queryKey: ["metrics-list"],
    queryFn: () => fetchApi<string[]>("/api/distributions/metrics"),
  });

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-zinc-100">GPU Health</h1>

      {/* GPU Cards Grid */}
      {health && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
          {health.map((gpu, i) => (
            <Card key={gpu.UUID || `${gpu.Hostname}-${gpu.gpu}-${i}`} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-zinc-200">
                  GPU {gpu.gpu}
                </span>
                <span className="text-xs text-zinc-500">{gpu.Hostname}</span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <p className="text-lg font-semibold text-zinc-100">
                    {gpu.mean_util?.toFixed(0)}%
                  </p>
                  <p className="text-xs text-zinc-500">Util</p>
                </div>
                <div>
                  <p className="text-lg font-semibold text-zinc-100">
                    {gpu.mean_temp?.toFixed(0)}°
                  </p>
                  <p className="text-xs text-zinc-500">Temp</p>
                </div>
                <div>
                  <p className="text-lg font-semibold text-zinc-100">
                    {gpu.mean_power?.toFixed(0)}W
                  </p>
                  <p className="text-xs text-zinc-500">Power</p>
                </div>
              </div>
              {(gpu.ecc_sbe_total || gpu.xid_errors || gpu.power_violations) ? (
                <div className="text-xs text-zinc-500 border-t border-zinc-800 pt-2 space-y-0.5">
                  {gpu.ecc_sbe_total ? <p>ECC SBE: {gpu.ecc_sbe_total}</p> : null}
                  {gpu.xid_errors ? <p>XID: {gpu.xid_errors}</p> : null}
                  {gpu.power_violations ? <p>Power Viol: {gpu.power_violations}</p> : null}
                  {gpu.thermal_violations ? <p>Thermal Viol: {gpu.thermal_violations}</p> : null}
                </div>
              ) : (
                <p className="text-xs text-emerald-600 border-t border-zinc-800 pt-2">
                  No errors
                </p>
              )}
            </Card>
          ))}
        </div>
      )}

      {/* Selectable Heatmap */}
      <Card>
        <div className="flex items-center gap-3 mb-2">
          <h2 className="text-sm font-medium text-zinc-300">GPU × Time Heatmap</h2>
          <select
            value={heatMetric}
            onChange={(e) => setHeatMetric(e.target.value)}
            className="bg-zinc-800 border border-zinc-700 text-zinc-300 text-xs rounded px-2 py-1"
          >
            {metrics?.map((m) => (
              <option key={m} value={m}>
                {m.replace(/DCGM_FI_(?:DEV_|PROF_(?:PIPE_)?)?/g, "")}
              </option>
            ))}
          </select>
        </div>
        {heatmap && (
          <PlotlyChart
            className="h-96"
            data={[
              {
                z: heatmap.matrix,
                x: heatmap.dates,
                y: heatmap.gpus,
                type: "heatmap" as const,
                colorscale: "Viridis",
                colorbar: { tickfont: { color: "#a1a1aa" } },
              },
            ]}
            layout={{
              xaxis: { title: "Date" },
              yaxis: { title: "GPU", automargin: true },
            }}
          />
        )}
      </Card>
    </div>
  );
}
