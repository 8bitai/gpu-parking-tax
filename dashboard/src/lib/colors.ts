export const WORKLOAD_COLORS: Record<string, string> = {
  llm_inference: "#8b5cf6",
  embedding: "#06b6d4",
  voice_ai: "#f59e0b",
  computer_vision: "#10b981",
  other: "#6b7280",
  idle: "#374151",
};

export function getWorkloadColor(workload: string): string {
  return WORKLOAD_COLORS[workload] || "#6b7280";
}

export const PLOTLY_DARK_LAYOUT: Record<string, unknown> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#a1a1aa", family: "var(--font-geist-sans), sans-serif", size: 12 },
  xaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  yaxis: { gridcolor: "#27272a", zerolinecolor: "#3f3f46" },
  margin: { l: 60, r: 20, t: 40, b: 40 },
  legend: { bgcolor: "transparent", font: { color: "#a1a1aa" } },
};
