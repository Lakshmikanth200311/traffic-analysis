"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import VideoFeed from "@/components/VideoFeed";

// Types
type Approach = "north" | "south" | "east" | "west";

interface VehicleCounts {
  car: number;
  motorcycle: number;
  bus: number;
  truck: number;
  total: number;
}

interface LiveCount {
  approach: Approach;
  vehicles: VehicleCounts;
  total: number;
}

interface Phase {
  approach: Approach;
  green: number;
  yellow: number;
  red: number;
}

interface CyclePlan {
  cycle_seconds: number;
  phases: Phase[];
  version: number;
}

interface SystemState {
  running: boolean;
  cycle_plan?: CyclePlan;
  live_counts: LiveCount[];
  phase_active?: Approach;
  remaining_seconds: number;
}

interface OptimizationDelta {
  approach: Approach;
  prev_green: number;
  new_green: number;
  delta: number;
}

// Constants
const approaches: Approach[] = ["north", "south", "east", "west"];
const defaultCounts: VehicleCounts = {
  car: 0,
  motorcycle: 0,
  bus: 0,
  truck: 0,
  total: 0,
};

// API Service
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const apiService = {
  async getState(): Promise<SystemState> {
    const response = await fetch(`${API_BASE_URL}/state`);
    if (!response.ok) throw new Error("Failed to fetch state");
    return response.json();
  },

  async uploadVideos(files: File[]): Promise<{ [key: string]: string }> {
    const formData = new FormData();
    const approaches = ["north", "south", "east", "west"];

    files.forEach((file, index) => {
      if (index < approaches.length) {
        formData.append(approaches[index], file);
      }
    });

    const response = await fetch(`${API_BASE_URL}/upload-files`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Upload failed");
    return response.json();
  },

  async runModel(data: { configs: any }) {
    console.log("🚀 Sending run request with configs:", data.configs);
    const response = await fetch(`${API_BASE_URL}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      const errorText = await response.text();
      console.error("❌ Run failed:", errorText);
      throw new Error(`Failed to start model: ${response.statusText}`);
    }
    return response.json();
  },

  async stopModel() {
    const response = await fetch(`${API_BASE_URL}/stop`, {
      method: "POST",
    });
    if (!response.ok) throw new Error("Failed to stop model");
    return response.json();
  },
};

// WebSocket Hook
const useWebSocket = (onMessage: (data: any) => void) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // Clear any existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    console.log("🔌 Attempting WebSocket connection...");
    setConnectionStatus("connecting");

    try {
      // ✅ FINAL WebSocket URL logic (local + cloud safe)
      const WS_URL =
      process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";
    
    console.log("🔌 WebSocket URL:", WS_URL);
    
    const ws = new WebSocket(WS_URL);


      ws.onopen = () => {
        console.log("✅ WebSocket connected successfully");
        setIsConnected(true);
        setConnectionStatus("connected");
        reconnectAttemptsRef.current = 0;

        // Send initial ping
        ws.send(JSON.stringify({ type: "ping" }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log("📨 WebSocket message:", data.type);
          onMessage(data);
        } catch (error) {
          console.error("❌ Error parsing WebSocket message:", error);
        }
      };

      ws.onclose = (event) => {
        console.log("🔌 WebSocket disconnected:", event.code, event.reason);
        setIsConnected(false);
        setConnectionStatus("disconnected");

        // Attempt reconnection with exponential backoff
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(
            1000 * Math.pow(2, reconnectAttemptsRef.current),
            30000
          );
          reconnectAttemptsRef.current++;

          console.log(
            `🔄 Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`
          );

          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
          }
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        } else {
          console.error("❌ Max reconnection attempts reached");
        }
      };

      ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
        setIsConnected(false);
        setConnectionStatus("error");
      };

      wsRef.current = ws;
    } catch (error) {
      console.error("❌ WebSocket connection failed:", error);
      setConnectionStatus("error");
    }
  }, [onMessage]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { isConnected, connectionStatus, reconnect: connect };
};

export default function TrafficOptimizer() {
  const [systemState, setSystemState] = useState<SystemState>({
    running: false,
    live_counts: [],
    remaining_seconds: 0,
  });
  const [optimizationDeltas, setOptimizationDeltas] = useState<
    OptimizationDelta[]
  >([]);
  const [uploadedFiles, setUploadedFiles] = useState<{ [key: string]: string }>(
    {}
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [darkMode, setDarkMode] = useState(true);

  const handleWebSocketMessage = useCallback((data: any) => {
    console.log("📨 WebSocket message type:", data.type);

    switch (data.type) {
      case "connection_established":
        console.log("✅ WebSocket connection confirmed");
        setError("");
        // Update state with initial data if provided
        if (data.data.system_state) {
          setSystemState(data.data.system_state);
        }
        break;
      case "system_state":
        console.log("🔄 Updating system state");
        setSystemState(data.data);
        setError("");
        break;
      case "live_counts":
        console.log("🚗 Updating live counts");
        setSystemState((prev) => ({
          ...prev,
          live_counts: data.data.counts,
        }));
        break;
      case "phase_update":
        console.log(
          "🚦 Phase update:",
          data.data.phase,
          data.data.remaining_seconds
        );
        setSystemState((prev) => ({
          ...prev,
          phase_active: data.data.phase,
          remaining_seconds: data.data.remaining_seconds,
        }));
        break;
      case "cycle_plan":
        console.log("📋 New cycle plan received");
        setSystemState((prev) => ({
          ...prev,
          cycle_plan: data.data.plan,
        }));
        break;
      case "optimization_delta":
        console.log("📈 Optimization deltas received");
        setOptimizationDeltas((prev) => [...prev, ...data.data]);
        break;
      case "pong":
        console.log("🏓 Received pong from server");
        break;
      default:
        console.log("❓ Unknown message type:", data.type);
    }
  }, []);

  const { isConnected, connectionStatus } = useWebSocket(
    handleWebSocketMessage
  );

  // Load initial state
  useEffect(() => {
    const loadInitialState = async () => {
      try {
        const state = await apiService.getState();
        setSystemState(state);
      } catch (err) {
        setError("Cannot connect to backend server");
      }
    };
    loadInitialState();
  }, []);

  // File upload handler
  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const files = event.target.files;
    if (!files || files.length !== 4) {
      alert("Please select exactly 4 video files");
      return;
    }

    try {
      setIsLoading(true);
      const filePaths = await apiService.uploadVideos(Array.from(files));
      setUploadedFiles(filePaths);
      alert("Videos uploaded successfully!");
    } catch (err) {
      setError("Failed to upload videos");
      alert("Upload failed. Please check backend connection.");
    } finally {
      setIsLoading(false);
    }
  };

  // Stop optimization
  const handleStop = async () => {
    try {
      setIsLoading(true);
      await apiService.stopModel();
    } catch (err) {
      setError("Failed to stop optimization");
    } finally {
      setIsLoading(false);
    }
  };

  // Update the getDefaultCameraConfigs function in your frontend:
  const getDefaultCameraConfigs = (filePaths: { [key: string]: string }) => {
    const defaultROI = {
      points: [
        { x: 0, y: 0 },
        { x: 1280, y: 0 },
        { x: 1280, y: 720 },
        { x: 0, y: 720 },
      ],
    };

    const defaultCountingLine = {
      start: { x: 600, y: 0 },
      end: { x: 600, y: 720 },
    };

    // FIX: Return the configs object directly, not nested under "configs"
    return {
      north: {
        approach: "north",
        roi: defaultROI,
        counting_line: defaultCountingLine,
        source: filePaths.north || "file:///uploads/north.mp4",
      },
      south: {
        approach: "south",
        roi: defaultROI,
        counting_line: defaultCountingLine,
        source: filePaths.south || "file:///uploads/south.mp4",
      },
      east: {
        approach: "east",
        roi: defaultROI,
        counting_line: defaultCountingLine,
        source: filePaths.east || "file:///uploads/east.mp4",
      },
      west: {
        approach: "west",
        roi: defaultROI,
        counting_line: defaultCountingLine,
        source: filePaths.west || "file:///uploads/west.mp4",
      },
    };
  };

  // Also update the handleStart function:
  const handleStart = async () => {
    if (Object.keys(uploadedFiles).length === 0) {
      alert("Please upload videos first");
      return;
    }

    try {
      setIsLoading(true);
      const configs = getDefaultCameraConfigs(uploadedFiles);

      // FIX: Send the configs directly, not wrapped in another object
      await apiService.runModel({ configs });
    } catch (err) {
      setError("Failed to start optimization");
      console.error("Start error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  // Get approach icon
  const getApproachIcon = (approach: Approach) => {
    switch (approach) {
      case "north":
        return "⬆️";
      case "south":
        return "⬇️";
      case "east":
        return "➡️";
      case "west":
        return "⬅️";
    }
  };

  // Get status color
  const getStatusColor = (isActive: boolean, remainingSeconds?: number) => {
    if (!isActive) return "bg-gray-500";
    if (remainingSeconds && remainingSeconds <= 3) return "bg-yellow-500";
    return "bg-green-500";
  };

  return (
    <div
      className={`min-h-screen transition-colors duration-300 ${
        darkMode ? "bg-gray-900 text-gray-100" : "bg-gray-50 text-gray-900"
      }`}
    >
      <div className="container mx-auto p-6 space-y-6">
        {/* Header */}
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">🚦 AI Traffic Optimizer</h1>
            <p className="text-muted-foreground mt-1">
              Real-time traffic light optimization using YOLO and adaptive
              timing
            </p>
          </div>
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`px-4 py-2 rounded-lg transition ${
              darkMode
                ? "bg-gray-700 hover:bg-gray-600"
                : "bg-gray-200 hover:bg-gray-300"
            }`}
          >
            {darkMode ? "🌙 Dark" : "☀️ Light"}
          </button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Sidebar - Control Panel */}
          <div className="lg:col-span-1 space-y-6">
            {/* Control Panel Card */}
            <div
              className={`rounded-lg border shadow-sm ${
                darkMode
                  ? "bg-gray-800 border-gray-700"
                  : "bg-white border-gray-200"
              }`}
            >
              <div className="p-6">
                <h2 className="text-xl font-semibold mb-4">Control Panel</h2>
                {/* Error Display */}
                {error && (
                  <div className="p-3 bg-red-100 border border-red-300 rounded-lg mb-4">
                    <div className="text-red-700 text-sm font-medium">
                      <div className="font-bold">Error:</div>
                      {error}
                    </div>
                  </div>
                )}
                {/* Connection Status */}

                <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg mb-4">
                  <span className="font-medium">WebSocket Connection</span>
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-3 h-3 rounded-full ${
                        isConnected
                          ? "bg-green-500 animate-pulse"
                          : connectionStatus === "connecting"
                          ? "bg-yellow-500 animate-pulse"
                          : "bg-red-500"
                      }`}
                    />
                    <span className="text-sm font-medium">
                      {connectionStatus === "connected"
                        ? "Connected"
                        : connectionStatus === "connecting"
                        ? "Connecting..."
                        : connectionStatus === "error"
                        ? "Connection Error"
                        : "Disconnected"}
                    </span>
                  </div>
                </div>

                {/* System Status */}
                <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg mb-4">
                  <span className="font-medium">Optimization System</span>
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        systemState.running
                          ? "bg-green-500 animate-pulse"
                          : "bg-gray-500"
                      }`}
                    />
                    <span className="text-sm">
                      {systemState.running ? "Running" : "Stopped"}
                    </span>
                  </div>
                </div>
                {/* File Upload */}
                <div className="space-y-2 mb-4">
                  <label className="text-sm font-medium">
                    Upload 4 Traffic Videos
                  </label>
                  <input
                    type="file"
                    multiple
                    accept="video/*"
                    onChange={handleFileUpload}
                    disabled={isLoading || !isConnected}
                    className="w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
                  />
                  {isLoading && (
                    <div className="text-sm text-blue-600 flex items-center gap-2">
                      <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600"></div>
                      Uploading videos...
                    </div>
                  )}
                  {Object.keys(uploadedFiles).length > 0 && (
                    <div className="text-sm text-green-600">
                      ✅ {Object.keys(uploadedFiles).length}/4 videos uploaded
                    </div>
                  )}
                </div>
                {/* Control Buttons */}
                <div className="flex gap-2 pt-2">
                  <button
                    onClick={handleStart}
                    disabled={
                      !isConnected ||
                      isLoading ||
                      Object.keys(uploadedFiles).length === 0
                    }
                    className={`flex-1 px-4 py-2 rounded-lg font-semibold transition ${
                      !isConnected ||
                      isLoading ||
                      Object.keys(uploadedFiles).length === 0
                        ? "bg-gray-400 cursor-not-allowed"
                        : "bg-blue-600 hover:bg-blue-700 text-white"
                    }`}
                  >
                    {isLoading ? (
                      <div className="flex items-center gap-2 justify-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        Starting...
                      </div>
                    ) : (
                      "🚀 Start Model"
                    )}
                  </button>
                  <button
                    onClick={handleStop}
                    disabled={!isConnected || !systemState.running || isLoading}
                    className={`flex-1 px-4 py-2 rounded-lg font-semibold transition border ${
                      !isConnected || !systemState.running || isLoading
                        ? "border-gray-400 text-gray-400 cursor-not-allowed"
                        : "border-gray-300 hover:bg-gray-100 text-gray-700"
                    }`}
                  >
                    {isLoading ? (
                      <div className="flex items-center gap-2 justify-center">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                        Stopping...
                      </div>
                    ) : (
                      "⏹️ Stop"
                    )}
                  </button>
                </div>
                {/* Quick Actions */}
                <div className="pt-4 border-t mt-4">
                  <h4 className="text-sm font-medium mb-2">Quick Actions</h4>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() =>
                        window.open(
                          (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1")
                            .replace("/api/v1", "") + "/docs",
                          "_blank"
                        )
                      }
                      className="px-3 py-2 text-sm border rounded-lg hover:bg-gray-100 transition"
                    >
                      📚 API Docs
                    </button>
                    <button
                      onClick={() => setUploadedFiles({})}
                      disabled={systemState.running}
                      className="px-3 py-2 text-sm border rounded-lg hover:bg-gray-100 transition disabled:opacity-50"
                    >
                      🔄 Reset
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Optimization Summary */}
            <div
              className={`rounded-lg border shadow-sm ${
                darkMode
                  ? "bg-gray-800 border-gray-700"
                  : "bg-white border-gray-200"
              }`}
            >
              <div className="p-6">
                <h2 className="text-xl font-semibold mb-4">
                  Optimization Summary
                </h2>
                {optimizationDeltas.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    No optimization data yet
                  </div>
                ) : (
                  <div className="space-y-4">
                    {optimizationDeltas
                      .slice(-8)
                      .reverse()
                      .map((delta, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
                        >
                          <div className="flex items-center gap-3">
                            <span className="text-xl">
                              {delta.approach === "north" && "⬆️"}
                              {delta.approach === "south" && "⬇️"}
                              {delta.approach === "east" && "➡️"}
                              {delta.approach === "west" && "⬅️"}
                            </span>
                            <div>
                              <div className="font-medium capitalize">
                                {delta.approach}
                              </div>
                              <div className="text-sm text-muted-foreground">
                                {delta.prev_green}s → {delta.new_green}s
                              </div>
                            </div>
                          </div>
                          <div
                            className={`text-right ${
                              delta.delta > 0
                                ? "text-green-600"
                                : delta.delta < 0
                                ? "text-red-600"
                                : "text-gray-600"
                            }`}
                          >
                            <div className="flex items-center gap-1 font-semibold">
                              <span>
                                {delta.delta > 0
                                  ? "↗️"
                                  : delta.delta < 0
                                  ? "↘️"
                                  : "➡️"}
                              </span>
                              <span>
                                {delta.delta > 0 ? "+" : ""}
                                {delta.delta}s
                              </span>
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {Math.abs(delta.delta)}s change
                            </div>
                          </div>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            {/* Video Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {approaches.map((approach) => {
                const liveCount = systemState.live_counts.find(
                  (lc) => lc.approach === approach
                );
                const counts = liveCount?.vehicles || defaultCounts;
                const isActive = systemState.phase_active === approach;

                return (
                  <div
                    key={approach}
                    className={`rounded-lg border shadow-sm ${
                      darkMode
                        ? "bg-gray-800 border-gray-700"
                        : "bg-white border-gray-200"
                    } ${isActive ? "ring-2 ring-blue-500" : ""}`}
                  >
                    <div className="p-6">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                          <span>{getApproachIcon(approach)}</span>
                          {approach.toUpperCase()}
                        </h3>
                        <div
                          className={`w-3 h-3 rounded-full ${getStatusColor(
                            isActive,
                            systemState.remaining_seconds
                          )} animate-pulse`}
                        />
                      </div>

                      {isActive &&
                        systemState.remaining_seconds !== undefined && (
                          <div className="text-sm text-muted-foreground mb-4">
                            {systemState.remaining_seconds}s remaining
                          </div>
                        )}

                      {/* LIVE VIDEO FEED - REPLACED PLACEHOLDER */}
                      <VideoFeed approach={approach} isActive={isActive} />

                      {/* Vehicle counts */}
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="flex justify-between">
                          <span>🚗 Cars:</span>
                          <span className="font-semibold">{counts.car}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>🏍️ Motorcycles:</span>
                          <span className="font-semibold">
                            {counts.motorcycle}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>🚌 Buses:</span>
                          <span className="font-semibold">{counts.bus}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>🚚 Trucks:</span>
                          <span className="font-semibold">{counts.truck}</span>
                        </div>
                        <div className="col-span-2 flex justify-between border-t pt-1 mt-1">
                          <span className="font-semibold">Total:</span>
                          <span className="font-bold text-lg">
                            {counts.total}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            {/* Phase Timeline */}
            <div
              className={`rounded-lg border shadow-sm ${
                darkMode
                  ? "bg-gray-800 border-gray-700"
                  : "bg-white border-gray-200"
              }`}
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold">Phase Timeline</h2>
                  {systemState.cycle_plan && (
                    <span className="text-sm text-muted-foreground">
                      Cycle v{systemState.cycle_plan.version} •{" "}
                      {systemState.cycle_plan.cycle_seconds}s total
                    </span>
                  )}
                </div>

                {!systemState.cycle_plan ? (
                  <div className="text-center text-muted-foreground py-8">
                    No active cycle plan
                  </div>
                ) : (
                  <div className="space-y-4">
                    {systemState.cycle_plan.phases.map((phase, index) => {
                      const isActive =
                        phase.approach === systemState.phase_active;

                      return (
                        <div key={index} className="flex items-center gap-4">
                          <div className="flex items-center gap-2 w-24">
                            <span>{getApproachIcon(phase.approach)}</span>
                            <span className="capitalize font-medium">
                              {phase.approach}
                            </span>
                          </div>

                          <div className="flex-1">
                            <div className="flex gap-1 mb-1">
                              {/* Green phase */}
                              <div
                                className={`h-8 rounded-l transition-all duration-300 ${
                                  isActive && systemState.remaining_seconds > 3
                                    ? "bg-green-500"
                                    : isActive
                                    ? "bg-yellow-500"
                                    : "bg-gray-300"
                                }`}
                                style={{
                                  width: `${
                                    (phase.green /
                                      systemState.cycle_plan!.cycle_seconds) *
                                    100
                                  }%`,
                                }}
                              />

                              {/* Yellow phase */}
                              <div
                                className="h-8 bg-yellow-500 transition-all duration-300"
                                style={{
                                  width: `${
                                    (phase.yellow /
                                      systemState.cycle_plan!.cycle_seconds) *
                                    100
                                  }%`,
                                }}
                              />

                              {/* Red phase */}
                              <div
                                className="h-8 bg-red-500 rounded-r transition-all duration-300"
                                style={{
                                  width: `${
                                    (phase.red /
                                      systemState.cycle_plan!.cycle_seconds) *
                                    100
                                  }%`,
                                }}
                              />
                            </div>

                            <div className="flex justify-between text-xs text-muted-foreground">
                              <span>G: {phase.green}s</span>
                              <span>Y: {phase.yellow}s</span>
                              <span>R: {phase.red}s</span>
                            </div>
                          </div>

                          {isActive && (
                            <div className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm font-semibold">
                              {systemState.remaining_seconds}s
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}

                {systemState.phase_active && (
                  <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                    <div className="text-sm text-blue-800">
                      <strong>Active:</strong>{" "}
                      {systemState.phase_active.toUpperCase()} approach •
                      <strong> Remaining:</strong>{" "}
                      {systemState.remaining_seconds} seconds
                    </div>
                  </div>
                )}
              </div>
            </div>
            {/* Status Bar */}
            <div
              className={`p-4 rounded-lg shadow ${
                darkMode ? "bg-gray-800" : "bg-white"
              }`}
            >
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        isConnected ? "bg-green-500" : "bg-red-500"
                      }`}
                    />
                    <span>
                      WebSocket: {isConnected ? "Connected" : "Disconnected"}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        systemState.running
                          ? "bg-green-500 animate-pulse"
                          : "bg-gray-500"
                      }`}
                    />
                    <span>
                      System: {systemState.running ? "Running" : "Stopped"}
                    </span>
                  </div>
                </div>
                <div className="text-muted-foreground">
                  Cycle: v{systemState.cycle_plan?.version || 0}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
