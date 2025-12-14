export type Approach = "north" | "south" | "east" | "west";

export type VehicleType = "car" | "motorcycle" | "bus" | "truck";

export interface Point {
  x: number;
  y: number;
}

export interface ROI {
  points: Point[];
}

export interface CountingLine {
  start: Point;
  end: Point;
}

export interface CameraConfig {
  approach: Approach;
  roi: ROI;
  counting_line: CountingLine;
  source: string;
}

export interface UploadRequest {
  north: string;
  south: string;
  east: string;
  west: string;
}

export interface RunRequest {
  configs: {
    [key in Approach]?: CameraConfig;
  };
}

export interface VehicleCounts {
  car: number;
  motorcycle: number;
  bus: number;
  truck: number;
  total: number;
}

export interface LiveCount {
  approach: Approach;
  vehicles: VehicleCounts;
  total: number;
}

export interface Phase {
  approach: Approach;
  green: number;
  yellow: number;
  red: number;
}

export interface CyclePlan {
  cycle_seconds: number;
  phases: Phase[];
  version: number;
}

export interface OptimizationDelta {
  approach: Approach;
  prev_green: number;
  new_green: number;
  delta: number;
}

export interface SystemState {
  running: boolean;
  cycle_plan?: CyclePlan;
  live_counts: LiveCount[];
  phase_active?: Approach;
  remaining_seconds: number;
}

export interface WSMessage {
  type:
    | "live_counts"
    | "detections_meta"
    | "phase_update"
    | "cycle_plan"
    | "optimization_delta"
    | "system_state";
  data: any;
}
