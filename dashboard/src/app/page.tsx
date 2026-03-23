"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchApi } from "@/lib/api";
import { KpiCard, Card } from "@/components/Card";
import PlotlyChart from "@/components/PlotlyChart";
import { getWorkloadColor } from "@/lib/colors";

interface Summary {
  total_rows: number;
  gpu_count: number;
  node_count: number;
  date_range: { start: string; end: string };
  workload_counts: Record<string, number>;
}

interface Kpis {
  mean_gpu_util: number;
  mean_power_w: number;
  mean_temp_c: number;
  mean_mem_util: number;
  mean_fb_util_pct: number;
}

interface TimelineData {
  [workload: string]: { hours: string[]; values: number[] };
}

interface HeatmapData {
  gpus: string[];
  dates: string[];
  matrix: (number | null)[][];
}

export default function OverviewPage() {
  const { data: summary } = useQuery({
    queryKey: ["overview-summary"],
    queryFn: () => fetchApi<Summary>("/api/overview/summary"),
  });

  const { data: kpis } = useQuery({
    queryKey: ["overview-kpis"],
    queryFn: () => fetchApi<Kpis>("/api/overview/kpis"),
  });

  const { data: timeline } = useQuery({
    queryKey: ["overview-timeline"],
    queryFn: () => fetchApi<TimelineData>("/api/overview/utilization-timeline"),
  });

  const { data: heatmap } = useQuery({
    queryKey: ["overview-heatmap"],
    queryFn: () => fetchApi<HeatmapData>("/api/overview/gpu-heatmap"),
  });

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-zinc-100">Overview</h1>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <KpiCard
          label="Date Range"
          value={
            summary?.date_range?.start
              ? `${summary.date_range.start.slice(0, 10)} — ${summary.date_range.end?.slice(0, 10)}`
              : "..."
          }
        />
        <KpiCard label="GPUs" value={summary?.gpu_count ?? "..."} />
        <KpiCard label="Nodes" value={summary?.node_count ?? "..."} />
        <KpiCard
          label="Total Rows"
          value={summary ? (summary.total_rows / 1e6).toFixed(1) + "M" : "..."}
        />
        <KpiCard label="Avg GPU Util" value={kpis?.mean_gpu_util ?? "..."} unit="%" />
        <KpiCard label="Avg Power" value={kpis?.mean_power_w ?? "..."} unit="W" />
      </div>

      {/* Workload Breakdown */}
      {summary?.workload_counts && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {Object.entries(summary.workload_counts).map(([wl, count]) => (
            <div
              key={wl}
              className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3 flex items-center gap-3"
            >
              <div
                className="w-3 h-3 rounded-full shrink-0"
                style={{ backgroundColor: getWorkloadColor(wl) }}
              />
              <div>
                <p className="text-xs text-zinc-400">{wl}</p>
                <p className="text-sm font-medium text-zinc-200">
                  {(count / 1000).toFixed(1)}K
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Utilization Timeline */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          GPU Utilization by Workload (Hourly)
        </h2>
        {timeline && (
          <PlotlyChart
            className="h-80"
            data={Object.entries(timeline).map(([wl, d]) => ({
              x: d.hours,
              y: d.values,
              type: "scatter" as const,
              mode: "lines" as const,
              name: wl,
              stackgroup: "one",
              line: { color: getWorkloadColor(wl), width: 0 },
              fillcolor: getWorkloadColor(wl) + "80",
            }))}
            layout={{ title: "", xaxis: { title: "Time" }, yaxis: { title: "GPU Util %" } }}
          />
        )}
      </Card>

      {/* GPU Heatmap */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          GPU × Day Utilization Heatmap
        </h2>
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
                colorbar: { title: { text: "Util %", font: { color: "#a1a1aa" } } as object, tickfont: { color: "#a1a1aa" } },
              },
            ]}
            layout={{
              title: "",
              xaxis: { title: "Date" },
              yaxis: { title: "GPU", automargin: true },
            }}
          />
        )}
      </Card>
    </div>
  );
}
