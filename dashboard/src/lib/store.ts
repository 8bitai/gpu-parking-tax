import { create } from "zustand";

interface FilterState {
  dateRange: [string, string] | null;
  workloadTypes: string[];
  gpuSelection: string[];
  nodeSelection: string[];
  resolution: "1h" | "1d";
  setDateRange: (range: [string, string] | null) => void;
  setWorkloadTypes: (types: string[]) => void;
  setGpuSelection: (gpus: string[]) => void;
  setNodeSelection: (nodes: string[]) => void;
  setResolution: (res: "1h" | "1d") => void;
}

export const useFilterStore = create<FilterState>((set) => ({
  dateRange: null,
  workloadTypes: [],
  gpuSelection: [],
  nodeSelection: [],
  resolution: "1h",
  setDateRange: (range) => set({ dateRange: range }),
  setWorkloadTypes: (types) => set({ workloadTypes: types }),
  setGpuSelection: (gpus) => set({ gpuSelection: gpus }),
  setNodeSelection: (nodes) => set({ nodeSelection: nodes }),
  setResolution: (res) => set({ resolution: res }),
}));
