"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchApi } from "@/lib/api";
import { Card } from "@/components/Card";
import PlotlyChart from "@/components/PlotlyChart";
import { getWorkloadColor } from "@/lib/colors";

interface PcaData {
  data: Record<string, { pc1: number[]; pc2: number[] }>;
  explained_variance: number[];
  loadings: Record<string, { pc1: number; pc2: number }>;
}

interface FingerprintData {
  [workload: string]: {
    [feature: string]: { mean: number; std: number };
  };
}

export default function ProfilingPage() {
  const { data: pca } = useQuery({
    queryKey: ["profiling-pca"],
    queryFn: () => fetchApi<PcaData>("/api/profiling/pca"),
  });

  const { data: fingerprints } = useQuery({
    queryKey: ["profiling-fingerprints"],
    queryFn: () => fetchApi<FingerprintData>("/api/profiling/fingerprints"),
  });

  const shortName = (m: string) =>
    m.replace(/DCGM_FI_(?:DEV_|PROF_(?:PIPE_)?)?/g, "");

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-zinc-100">
        Profiling Fingerprints
      </h1>

      {/* PCA Scatter */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          2D PCA — Workload Separability
          {pca && (
            <span className="text-xs text-zinc-500 ml-2">
              (Explained variance: {(pca.explained_variance[0] * 100).toFixed(1)}% +{" "}
              {(pca.explained_variance[1] * 100).toFixed(1)}%)
            </span>
          )}
        </h2>
        {pca && (
          <PlotlyChart
            className="h-[500px]"
            data={Object.entries(pca.data).map(([wl, coords]) => ({
              x: coords.pc1,
              y: coords.pc2,
              type: "scatter" as const,
              mode: "markers" as const,
              name: wl,
              marker: { color: getWorkloadColor(wl), size: 4, opacity: 0.6 },
            }))}
            layout={{
              xaxis: {
                title: `PC1 (${(pca.explained_variance[0] * 100).toFixed(1)}%)`,
              },
              yaxis: {
                title: `PC2 (${(pca.explained_variance[1] * 100).toFixed(1)}%)`,
              },
            }}
          />
        )}
      </Card>

      {/* Explained Variance */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <h2 className="text-sm font-medium text-zinc-300 mb-2">
            Explained Variance
          </h2>
          {pca && (
            <PlotlyChart
              className="h-48"
              data={[
                {
                  type: "bar" as const,
                  x: pca.explained_variance.map((_, i) => `PC${i + 1}`),
                  y: pca.explained_variance.map((v) => v * 100),
                  marker: { color: "#8b5cf6" },
                },
              ]}
              layout={{
                yaxis: { title: "Variance %" },
                margin: { l: 50, r: 10, t: 10, b: 30 },
              }}
            />
          )}
        </Card>

        {/* Feature Loadings */}
        <Card>
          <h2 className="text-sm font-medium text-zinc-300 mb-2">
            Feature Loadings (PC1 vs PC2)
          </h2>
          {pca && (
            <PlotlyChart
              className="h-48"
              data={[
                {
                  type: "scatter" as const,
                  mode: "text+markers" as const,
                  x: Object.values(pca.loadings).map((l) => l.pc1),
                  y: Object.values(pca.loadings).map((l) => l.pc2),
                  text: Object.keys(pca.loadings).map(shortName),
                  textposition: "top center" as const,
                  textfont: { size: 8, color: "#a1a1aa" },
                  marker: { color: "#06b6d4", size: 6 },
                },
              ]}
              layout={{
                xaxis: { title: "PC1 Loading" },
                yaxis: { title: "PC2 Loading" },
                margin: { l: 50, r: 10, t: 10, b: 40 },
              }}
            />
          )}
        </Card>
      </div>

      {/* Fingerprint Table */}
      <Card>
        <h2 className="text-sm font-medium text-zinc-300 mb-2">
          Workload Fingerprints
        </h2>
        {fingerprints && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-zinc-300">
              <thead className="bg-zinc-900">
                <tr>
                  <th className="text-left p-2">Workload</th>
                  {Object.keys(
                    Object.values(fingerprints)[0] ?? {}
                  ).map((feat) => (
                    <th key={feat} className="text-right p-2">
                      {shortName(feat)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(fingerprints).map(([wl, feats]) => (
                  <tr key={wl} className="border-t border-zinc-800">
                    <td className="p-2 flex items-center gap-2">
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: getWorkloadColor(wl) }}
                      />
                      {wl}
                    </td>
                    {Object.values(feats).map((v, i) => (
                      <td key={i} className="p-2 text-right">
                        {v.mean.toFixed(3)}
                        <span className="text-zinc-600 ml-1">
                          ±{v.std.toFixed(3)}
                        </span>
                      </td>
                    ))}
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
