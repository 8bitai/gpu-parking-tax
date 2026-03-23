"use client";

import dynamic from "next/dynamic";
import { PLOTLY_DARK_LAYOUT } from "@/lib/colors";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyObj = Record<string, any>;

interface PlotlyChartProps {
  data: AnyObj[];
  layout?: AnyObj;
  className?: string;
}

export default function PlotlyChart({
  data,
  layout = {},
  className = "",
}: PlotlyChartProps) {
  const base = PLOTLY_DARK_LAYOUT as AnyObj;
  const mergedLayout = {
    ...base,
    ...layout,
    xaxis: { ...base.xaxis, ...layout.xaxis },
    yaxis: { ...base.yaxis, ...layout.yaxis },
  };

  return (
    <div className={className}>
      <Plot
        data={data}
        layout={mergedLayout}
        config={{ responsive: true, displayModeBar: false }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
