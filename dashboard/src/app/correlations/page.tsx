"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchApi } from "@/lib/api";
import { Card } from "@/components/Card";
import PlotlyChart from "@/components/PlotlyChart";
import { getWorkloadColor } from "@/lib/colors";

interface MatrixData {
  metrics: string[];
  matrix: number[][];
}

interface ScatterData {
  x_metric: string;
  y_metric: string;
  data: Record<string, { x: number[]; y: number[] }>;
}

interface PairData {
  metric_1: string;
  metric_2: string;
  correlation: number;
}

export default function CorrelationsPage() {
  const [xMetric, setXMetric] = useState("DCGM_FI_DEV_GPU_UTIL");
  const [yMetric, setYMetric] = useState("DCGM_FI_DEV_POWER_USAGE");

  const { data: matrixData } = useQuery({
    queryKey: ["corr-matrix"],
    queryFn: () => fetchApi<MatrixData>("/api/correlations/matrix"),
  });

  const { data: scatterData } = useQuery({
    queryKey: ["corr-scatter", xMetric, yMetric],
    queryFn: () =>
      fetchApi<ScatterData>(
        `/api/correlations/scatter?x=${xMetric}&y=${yMetric}`
      ),
  });

  const { data: topPairs } = useQuery({
    queryKey: ["corr-top-pairs"],
    queryFn: () => fetchApi<PairData[]>("/api/correlations/top-pairs?n=20"),
  });

  const { data: metrics } = useQuery({
    queryKey: ["metrics-list"],
    queryFn: () => fetchApi<string[]>("/api/distributions/metrics"),
  });

  const shortName = (m: string) =>
    m.replace(/DCGM_FI_(?:DEV_|PROF_(?:PIPE_)?)?/g, "");

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-zinc-100">Correlations</h1>

      {/* Correlation Heatmap */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          Correlation Matrix
        </h2>
        {matrixData && (
          <PlotlyChart
            className="h-[600px]"
            data={[
              {
                z: matrixData.matrix,
                x: matrixData.metrics.map(shortName),
                y: matrixData.metrics.map(shortName),
                type: "heatmap" as const,
                colorscale: "RdBu",
                colorbar: { tickfont: { color: "#a1a1aa" } },
                zmin: -1,
                zmax: 1,
              },
            ]}
            layout={{
              xaxis: { tickangle: -45, automargin: true, tickfont: { size: 9 } },
              yaxis: { automargin: true, tickfont: { size: 9 } },
              margin: { l: 120, r: 20, t: 20, b: 120 },
            }}
          />
        )}
      </Card>

      {/* Interactive Scatter */}
      <Card>
        <div className="flex items-center gap-3 mb-2">
          <h2 className="text-sm font-medium text-zinc-300">Scatter Plot</h2>
          <select
            value={xMetric}
            onChange={(e) => setXMetric(e.target.value)}
            className="bg-zinc-800 border border-zinc-700 text-zinc-300 text-xs rounded px-2 py-1"
          >
            {metrics?.map((m) => (
              <option key={m} value={m}>{shortName(m)}</option>
            ))}
          </select>
          <span className="text-xs text-zinc-500">vs</span>
          <select
            value={yMetric}
            onChange={(e) => setYMetric(e.target.value)}
            className="bg-zinc-800 border border-zinc-700 text-zinc-300 text-xs rounded px-2 py-1"
          >
            {metrics?.map((m) => (
              <option key={m} value={m}>{shortName(m)}</option>
            ))}
          </select>
        </div>
        {scatterData && (
          <PlotlyChart
            className="h-96"
            data={Object.entries(scatterData.data).map(([wl, d]) => ({
              x: d.x,
              y: d.y,
              type: "scatter" as const,
              mode: "markers" as const,
              name: wl,
              marker: { color: getWorkloadColor(wl), size: 4, opacity: 0.6 },
            }))}
            layout={{
              xaxis: { title: shortName(xMetric) },
              yaxis: { title: shortName(yMetric) },
            }}
          />
        )}
      </Card>

      {/* Top Correlated Pairs Table */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          Top Correlated Pairs
        </h2>
        {topPairs && (
          <div className="overflow-x-auto max-h-80">
            <table className="w-full text-xs text-zinc-300">
              <thead className="sticky top-0 bg-zinc-900">
                <tr>
                  <th className="text-left p-2">Metric 1</th>
                  <th className="text-left p-2">Metric 2</th>
                  <th className="text-right p-2">Correlation</th>
                </tr>
              </thead>
              <tbody>
                {topPairs
                  .filter((p) => p.correlation != null)
                  .map((p, i) => (
                  <tr key={i} className="border-t border-zinc-800">
                    <td className="p-2 font-mono">{shortName(p.metric_1)}</td>
                    <td className="p-2 font-mono">{shortName(p.metric_2)}</td>
                    <td className="p-2 text-right">
                      <span
                        className={
                          p.correlation > 0 ? "text-emerald-400" : "text-red-400"
                        }
                      >
                        {p.correlation.toFixed(3)}
                      </span>
                    </td>
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
