"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { WSMessage } from "../types/traffic";

interface UseWebSocketProps {
  onMessage: (message: WSMessage) => void;
  onError?: (error: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onOpen?: () => void;
}

export const useWebSocket = ({
  onMessage,
  onError,
  onClose,
  onOpen,
}: UseWebSocketProps) => {
  const [isConnected, setIsConnected] = useState(false);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const shouldConnect = useRef(true);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connection attempts
    if (
      ws.current?.readyState === WebSocket.CONNECTING ||
      ws.current?.readyState === WebSocket.OPEN
    ) {
      console.log("⚠️ Connection already in progress or established");
      return;
    }

    if (!shouldConnect.current || !mountedRef.current) {
      console.log("⚠️ Component unmounted or connection disabled");
      return;
    }

    // Use environment variable or default to localhost
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";

    console.log(`🔌 Attempting WebSocket connection to: ${wsUrl}`);

    try {
      // Clean up any existing connection first
      if (ws.current) {
        ws.current.onopen = null;
        ws.current.onmessage = null;
        ws.current.onclose = null;
        ws.current.onerror = null;
        if (
          ws.current.readyState === WebSocket.OPEN ||
          ws.current.readyState === WebSocket.CONNECTING
        ) {
          ws.current.close();
        }
        ws.current = null;
      }

      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        if (!mountedRef.current) {
          console.log("⚠️ Connection opened but component unmounted");
          ws.current?.close();
          return;
        }

        console.log("✅ WebSocket connected successfully");
        setIsConnected(true);
        setReconnectAttempts(0);
        onOpen?.();
      };

      ws.current.onmessage = (event) => {
        if (!mountedRef.current) return;

        try {
          const message: WSMessage = JSON.parse(event.data);
          console.log("📨 WebSocket message received:", message.type);
          onMessage(message);
        } catch (error) {
          console.error(
            "❌ Error parsing WebSocket message:",
            error,
            event.data
          );
        }
      };

      ws.current.onclose = (event) => {
        if (!mountedRef.current) {
          console.log("🔌 WebSocket closed after component unmount");
          return;
        }

        console.log("🔌 WebSocket disconnected:", event.code, event.reason);
        setIsConnected(false);
        onClose?.(event);

        // Only attempt reconnection if:
        // 1. Component is still mounted
        // 2. It wasn't a clean close (code 1000)
        // 3. We haven't exceeded max attempts
        // 4. shouldConnect is still true
        if (
          shouldConnect.current &&
          event.code !== 1000 &&
          reconnectAttempts < 10 &&
          mountedRef.current
        ) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
          console.log(
            `🔄 Reconnecting in ${delay}ms... (Attempt ${
              reconnectAttempts + 1
            })`
          );

          reconnectTimeout.current = setTimeout(() => {
            if (mountedRef.current && shouldConnect.current) {
              setReconnectAttempts((prev) => prev + 1);
              connect();
            }
          }, delay);
        }
      };

      ws.current.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
        onError?.(error);
      };
    } catch (error) {
      console.error("❌ WebSocket connection failed:", error);
      onError?.(error as Event);
    }
  }, [onMessage, onError, onClose, onOpen, reconnectAttempts]);

  const disconnect = useCallback(() => {
    console.log("🛑 Manually disconnecting WebSocket");
    shouldConnect.current = false;

    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }

    if (ws.current) {
      // Remove event listeners to prevent them from firing
      ws.current.onopen = null;
      ws.current.onmessage = null;
      ws.current.onclose = null;
      ws.current.onerror = null;

      if (
        ws.current.readyState === WebSocket.OPEN ||
        ws.current.readyState === WebSocket.CONNECTING
      ) {
        ws.current.close(1000, "Manual disconnect");
      }
      ws.current = null;
    }
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((message: WSMessage) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
      console.log("📤 Message sent:", message.type);
    } else {
      console.warn("⚠️ WebSocket not connected, cannot send message");
    }
  }, []);

  useEffect(() => {
    // Mark component as mounted
    mountedRef.current = true;
    shouldConnect.current = true;

    // Small delay to avoid React strict mode issues
    const connectionTimer = setTimeout(() => {
      if (mountedRef.current && shouldConnect.current) {
        connect();
      }
    }, 100);

    return () => {
      console.log("🧹 Cleaning up WebSocket connection");
      mountedRef.current = false;
      shouldConnect.current = false;

      clearTimeout(connectionTimer);

      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }

      if (ws.current) {
        ws.current.onopen = null;
        ws.current.onmessage = null;
        ws.current.onclose = null;
        ws.current.onerror = null;

        if (
          ws.current.readyState === WebSocket.OPEN ||
          ws.current.readyState === WebSocket.CONNECTING
        ) {
          ws.current.close(1000, "Component unmounting");
        }
        ws.current = null;
      }
    };
  }, []); // Empty dependency array - only run once

  return {
    isConnected,
    sendMessage,
    disconnect,
    reconnect: connect,
    reconnectAttempts,
  };
};
