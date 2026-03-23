"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchApi } from "@/lib/api";
import { Card } from "@/components/Card";
import PlotlyChart from "@/components/PlotlyChart";
import { getWorkloadColor } from "@/lib/colors";

interface ViolinData {
  metric: string;
  data: Record<string, number[]>;
}

interface HistData {
  metric: string;
  bin_edges: number[];
  data: Record<string, number[]>;
}

interface QuantileRow {
  workload_type: string;
  metric: string;
  p1: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  p99: number;
}

export default function DistributionsPage() {
  const [metric, setMetric] = useState("DCGM_FI_DEV_GPU_UTIL");
  const [bins, setBins] = useState(50);

  const { data: metrics } = useQuery({
    queryKey: ["metrics-list"],
    queryFn: () => fetchApi<string[]>("/api/distributions/metrics"),
  });

  const { data: violin } = useQuery({
    queryKey: ["dist-violin", metric],
    queryFn: () => fetchApi<ViolinData>(`/api/distributions/violin?metric=${metric}`),
  });

  const { data: hist } = useQuery({
    queryKey: ["dist-hist", metric, bins],
    queryFn: () =>
      fetchApi<HistData>(`/api/distributions/histogram?metric=${metric}&bins=${bins}`),
  });

  const { data: quantiles } = useQuery({
    queryKey: ["dist-quantiles", metric],
    queryFn: () =>
      fetchApi<QuantileRow[]>(`/api/distributions/quantiles?metric=${metric}`),
  });

  const shortName = (m: string) =>
    m.replace(/DCGM_FI_(?:DEV_|PROF_(?:PIPE_)?)?/g, "");

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold text-zinc-100">Distributions</h1>
        <select
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
          className="bg-zinc-800 border border-zinc-700 text-zinc-300 text-xs rounded px-2 py-1"
        >
          {metrics?.map((m) => (
            <option key={m} value={m}>{shortName(m)}</option>
          ))}
        </select>
      </div>

      {/* Violin + Box */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          Violin + Box Plot: {shortName(metric)}
        </h2>
        {violin && (
          <PlotlyChart
            className="h-80"
            data={Object.entries(violin.data).map(([wl, vals]) => ({
              type: "violin" as const,
              y: vals,
              name: wl,
              box: { visible: true },
              meanline: { visible: true },
              line: { color: getWorkloadColor(wl) },
              fillcolor: getWorkloadColor(wl) + "40",
            }))}
            layout={{ yaxis: { title: shortName(metric) } }}
          />
        )}
      </Card>

      {/* Histogram */}
      <Card>
        <div className="flex items-center gap-3 mb-2">
          <h2 className="text-sm font-medium text-zinc-300">Histogram</h2>
          <label className="text-xs text-zinc-500">
            Bins:
            <input
              type="range"
              min={10}
              max={200}
              value={bins}
              onChange={(e) => setBins(Number(e.target.value))}
              className="ml-2 w-24"
            />
            <span className="ml-1">{bins}</span>
          </label>
        </div>
        {hist && (
          <PlotlyChart
            className="h-72"
            data={Object.entries(hist.data).map(([wl, counts]) => {
              const centers = hist.bin_edges
                .slice(0, -1)
                .map((e, i) => (e + hist.bin_edges[i + 1]) / 2);
              return {
                type: "bar" as const,
                x: centers,
                y: counts,
                name: wl,
                marker: { color: getWorkloadColor(wl), opacity: 0.7 },
              };
            })}
            layout={{
              barmode: "overlay" as const,
              xaxis: { title: shortName(metric) },
              yaxis: { title: "Count" },
            }}
          />
        )}
      </Card>

      {/* Quantile Table */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">Quantiles</h2>
        {quantiles && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-zinc-300">
              <thead className="bg-zinc-900">
                <tr>
                  <th className="text-left p-2">Workload</th>
                  <th className="text-right p-2">p1</th>
                  <th className="text-right p-2">p5</th>
                  <th className="text-right p-2">p25</th>
                  <th className="text-right p-2 font-semibold">p50</th>
                  <th className="text-right p-2">p75</th>
                  <th className="text-right p-2">p95</th>
                  <th className="text-right p-2">p99</th>
                </tr>
              </thead>
              <tbody>
                {quantiles.map((q, i) => (
                  <tr key={i} className="border-t border-zinc-800">
                    <td className="p-2">{q.workload_type}</td>
                    <td className="p-2 text-right">{q.p1?.toFixed(2)}</td>
                    <td className="p-2 text-right">{q.p5?.toFixed(2)}</td>
                    <td className="p-2 text-right">{q.p25?.toFixed(2)}</td>
                    <td className="p-2 text-right font-semibold">{q.p50?.toFixed(2)}</td>
                    <td className="p-2 text-right">{q.p75?.toFixed(2)}</td>
                    <td className="p-2 text-right">{q.p95?.toFixed(2)}</td>
                    <td className="p-2 text-right">{q.p99?.toFixed(2)}</td>
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
