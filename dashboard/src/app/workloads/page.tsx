"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchApi } from "@/lib/api";
import { Card } from "@/components/Card";
import PlotlyChart from "@/components/PlotlyChart";
import { getWorkloadColor, WORKLOAD_COLORS } from "@/lib/colors";

const RADAR_FEATURES = [
  "tensor_dominance",
  "fp16_fp32_ratio",
  "compute_memory_ratio",
  "clock_ratio",
  "thermal_headroom",
  "power_efficiency",
  "memory_utilization_pct",
];

export default function WorkloadsPage() {
  const { data: profiles } = useQuery({
    queryKey: ["workload-profiles"],
    queryFn: () =>
      fetchApi<Record<string, Record<string, number>>>("/api/workloads/profiles"),
  });

  const { data: distributions } = useQuery({
    queryKey: ["workload-distributions"],
    queryFn: () =>
      fetchApi<Record<string, Record<string, number[]>>>("/api/workloads/distributions"),
  });

  const { data: pipeData } = useQuery({
    queryKey: ["workload-pipe"],
    queryFn: () =>
      fetchApi<Record<string, Record<string, number>>>("/api/workloads/pipe-breakdown"),
  });

  const { data: compareData } = useQuery({
    queryKey: ["workload-compare"],
    queryFn: () =>
      fetchApi<
        { workload_type: string; metric: string; mean: number; std: number; median: number }[]
      >("/api/workloads/compare"),
  });

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-zinc-100">Workload Profiles</h1>

      {/* Radar Chart */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          Workload Fingerprint Radar
        </h2>
        {profiles && (
          <PlotlyChart
            className="h-96"
            data={Object.entries(profiles).map(([wl, vals]) => {
              const r = RADAR_FEATURES.map((f) => vals[f] ?? 0);
              return {
                type: "scatterpolar" as const,
                r: [...r, r[0]],
                theta: [...RADAR_FEATURES.map((f) => f.replace(/_/g, " ")), RADAR_FEATURES[0].replace(/_/g, " ")],
                fill: "toself" as const,
                name: wl,
                line: { color: getWorkloadColor(wl) },
                fillcolor: getWorkloadColor(wl) + "30",
              };
            })}
            layout={{
              polar: {
                bgcolor: "transparent",
                radialaxis: { visible: true, color: "#52525b", gridcolor: "#27272a" },
                angularaxis: { color: "#a1a1aa", gridcolor: "#27272a" },
              },
            }}
          />
        )}
      </Card>

      {/* Violin Plots */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          Metric Distributions by Workload
        </h2>
        {distributions && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {Object.entries(distributions).map(([metric, wlData]) => (
              <PlotlyChart
                key={metric}
                className="h-72"
                data={Object.entries(wlData).map(([wl, vals]) => ({
                  type: "violin" as const,
                  y: vals,
                  name: wl,
                  box: { visible: true },
                  meanline: { visible: true },
                  line: { color: getWorkloadColor(wl) },
                  fillcolor: getWorkloadColor(wl) + "40",
                }))}
                layout={{
                  title: { text: metric.replace(/DCGM_FI_(?:DEV_|PROF_(?:PIPE_)?)?/g, ""), font: { size: 11, color: "#a1a1aa" } },
                  showlegend: false,
                  margin: { l: 50, r: 10, t: 30, b: 30 },
                }}
              />
            ))}
          </div>
        )}
      </Card>

      {/* Pipe Breakdown */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          Profiling Pipe Breakdown
        </h2>
        {pipeData && (
          <PlotlyChart
            className="h-72"
            data={[
              "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
              "DCGM_FI_PROF_PIPE_FP16_ACTIVE",
              "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
              "DCGM_FI_PROF_PIPE_FP64_ACTIVE",
            ].map((pipe) => ({
              type: "bar" as const,
              name: pipe.replace("DCGM_FI_PROF_PIPE_", "").replace("_ACTIVE", ""),
              x: Object.keys(pipeData),
              y: Object.values(pipeData).map((d) => d[pipe] ?? 0),
            }))}
            layout={{ barmode: "stack" as const, xaxis: { title: "Workload" }, yaxis: { title: "Activity" } }}
          />
        )}
      </Card>

      {/* Comparison Table */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          Workload Comparison
        </h2>
        {Array.isArray(compareData) && (
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-xs text-zinc-300">
              <thead className="sticky top-0 bg-zinc-900">
                <tr>
                  <th className="text-left p-2">Workload</th>
                  <th className="text-left p-2">Metric</th>
                  <th className="text-right p-2">Mean</th>
                  <th className="text-right p-2">Std</th>
                  <th className="text-right p-2">Median</th>
                </tr>
              </thead>
              <tbody>
                {compareData.slice(0, 100).map((row, i) => (
                  <tr key={i} className="border-t border-zinc-800">
                    <td className="p-2">{row.workload_type}</td>
                    <td className="p-2 font-mono text-zinc-400">
                      {row.metric?.replace(/DCGM_FI_(?:DEV_|PROF_(?:PIPE_)?)?/g, "")}
                    </td>
                    <td className="p-2 text-right">{row.mean?.toFixed(2)}</td>
                    <td className="p-2 text-right">{row.std?.toFixed(2)}</td>
                    <td className="p-2 text-right">{row.median?.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
