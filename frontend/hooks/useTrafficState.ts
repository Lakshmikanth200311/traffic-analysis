"use client";

import { useState, useEffect, useCallback } from "react";
import { useWebSocket } from "./useWebSocket";
import { trafficApi } from "../lib/api";
import {
  SystemState,
  LiveCount,
  CyclePlan,
  OptimizationDelta,
  WSMessage,
} from "../types/traffic";

export const useTrafficState = () => {
  const [systemState, setSystemState] = useState<SystemState>({
    running: false,
    live_counts: [],
    remaining_seconds: 0,
  });

  const [optimizationDeltas, setOptimizationDeltas] = useState<
    OptimizationDelta[]
  >([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch initial state
  useEffect(() => {
    const fetchInitialState = async () => {
      try {
        setIsLoading(true);
        setError(null);
        console.log("📡 Fetching initial system state...");
        const state = await trafficApi.getState();
        console.log("✅ Initial state received:", state);
        setSystemState(state);
      } catch (err: any) {
        console.error("❌ Failed to fetch initial state:", err);
        const errorMessage = err.message || "Failed to connect to server";
        setError(errorMessage);
      } finally {
        setIsLoading(false);
      }
    };

    fetchInitialState();
  }, []);

  const handleWebSocketMessage = useCallback((message: WSMessage) => {
    console.log("📨 Processing WebSocket message:", message.type);

    switch (message.type) {
      case "system_state":
        console.log("🔄 Updating system state:", message.data);
        setSystemState(message.data);
        setError(null);
        break;

      case "live_counts":
        console.log("📊 Updating live counts:", message.data.counts);
        setSystemState((prev) => ({
          ...prev,
          live_counts: message.data.counts,
        }));
        break;

      case "phase_update":
        console.log(
          "🚦 Phase update:",
          message.data.phase,
          message.data.remaining_seconds
        );
        setSystemState((prev) => ({
          ...prev,
          phase_active: message.data.phase,
          remaining_seconds: message.data.remaining_seconds,
        }));
        break;

      case "cycle_plan":
        console.log("📋 Cycle plan updated:", message.data.plan);
        setSystemState((prev) => ({
          ...prev,
          cycle_plan: message.data.plan,
        }));
        break;

      case "optimization_delta":
        console.log("⚡ Optimization deltas received:", message.data);
        setOptimizationDeltas((prev) => [...prev, ...message.data]);
        break;

      default:
        console.log("⚠️ Unknown message type:", message.type);
    }
  }, []);

  const { isConnected } = useWebSocket({
    onMessage: handleWebSocketMessage,
    onError: (error) => {
      console.error("❌ WebSocket error:", error);
      setError("WebSocket connection failed - check if backend is running");
    },
    onClose: (event) => {
      console.log("🔌 WebSocket closed:", event.code, event.reason);
      if (event.code !== 1000) {
        setError("WebSocket disconnected unexpectedly");
      }
    },
    onOpen: () => {
      console.log("✅ WebSocket connected successfully");
      setError(null);
    },
  });

  // Start the traffic optimization
  const startOptimization = useCallback(async (configs: any) => {
    try {
      setIsLoading(true);
      setError(null);
      console.log("▶️ Starting optimization with configs:", configs);
      await trafficApi.runModel({ configs });
      console.log("✅ Optimization started successfully");
      // State will be updated via WebSocket
    } catch (err: any) {
      console.error("❌ Failed to start optimization:", err);
      const errorMessage =
        err.response?.data?.detail ||
        err.message ||
        "Failed to start optimization";
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Stop the traffic optimization
  const stopOptimization = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      console.log("⏹️ Stopping optimization...");
      await trafficApi.stopModel();
      console.log("✅ Optimization stopped successfully");
      // State will be updated via WebSocket
    } catch (err: any) {
      console.error("❌ Failed to stop optimization:", err);
      const errorMessage =
        err.response?.data?.detail ||
        err.message ||
        "Failed to stop optimization";
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Upload videos
  const uploadVideos = useCallback(
    async (filePaths: { [key: string]: string }) => {
      try {
        setIsLoading(true);
        setError(null);
        console.log("📤 Uploading videos:", filePaths);
        await trafficApi.uploadVideos(filePaths);
        console.log("✅ Videos uploaded successfully");
      } catch (err: any) {
        console.error("❌ Failed to upload videos:", err);
        const errorMessage =
          err.response?.data?.detail ||
          err.message ||
          "Failed to upload videos";
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  return {
    systemState,
    optimizationDeltas,
    isConnected,
    isLoading,
    error,
    startOptimization,
    stopOptimization,
    uploadVideos,
    setError,
  };
};
