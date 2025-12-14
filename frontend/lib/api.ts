import axios from "axios";
import {
  UploadRequest,
  RunRequest,
  SystemState,
  CameraConfig,
} from "../types/traffic";

// IMPORTANT: Use localhost, not 0.0.0.0
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL!;


console.log("🔧 API Base URL configured:", API_BASE_URL);

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(
      `🚀 Making ${config.method?.toUpperCase()} request to ${config.url}`
    );
    return config;
  },
  (error) => {
    console.error("❌ API request error:", error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(
      `✅ ${response.config.method?.toUpperCase()} ${
        response.config.url
      } - Success`
    );
    return response;
  },
  (error) => {
    console.error("❌ API response error:", error);

    if (error.code === "ECONNREFUSED" || error.code === "ERR_NETWORK") {
      const errorMsg =
        "🔌 Cannot connect to backend server. Please verify:\n" +
        "1. Backend is running and reachable\n" +
        "2. Check Render logs for backend errors\n" +
        "3. Verify NEXT_PUBLIC_API_URL is set correctly";
      console.error(errorMsg);
      throw new Error(
        "Cannot connect to backend server. Please make sure the server is running on port 8000."
      );
    }

    if (error.response) {
      console.error(
        "📊 Server responded with error:",
        error.response.status,
        error.response.data
      );
    } else if (error.request) {
      console.error("🌐 No response received from server");
      console.error("Request details:", {
        url: error.config?.url,
        method: error.config?.method,
        baseURL: error.config?.baseURL,
      });
    }

    throw error;
  }
);

export const trafficApi = {
  async uploadVideos(uploadData: UploadRequest) {
    try {
      const response = await api.post("/upload", uploadData);
      return response.data;
    } catch (error: any) {
      console.error("Upload videos failed:", error.message);
      throw error;
    }
  },

  async runModel(runData: RunRequest) {
    try {
      const response = await api.post("/run", runData);
      return response.data;
    } catch (error: any) {
      console.error("Run model failed:", error.message);
      throw error;
    }
  },

  async stopModel() {
    try {
      const response = await api.post("/stop");
      return response.data;
    } catch (error: any) {
      console.error("Stop model failed:", error.message);
      throw error;
    }
  },

  async getState(): Promise<SystemState> {
    try {
      const response = await api.get("/state");
      return response.data;
    } catch (error: any) {
      console.error("Get state failed:", error.message);
      throw error;
    }
  },

  async getPlan() {
    try {
      const response = await api.get("/plan");
      return response.data;
    } catch (error: any) {
      console.error("Get plan failed:", error.message);
      throw error;
    }
  },

  async healthCheck() {
    try {
      const response = await api.get("/healthz");
      return response.data;
    } catch (error: any) {
      console.error("Health check failed:", error.message);
      throw error;
    }
  },
};

// File upload utility
export const uploadFiles = async (
  files: File[]
): Promise<{ [key: string]: string }> => {
  const formData = new FormData();

  const approaches = ["north", "south", "east", "west"];

  files.forEach((file, index) => {
    if (index < approaches.length) {
      formData.append(approaches[index], file);
    }
  });

  try {
    console.log("📤 Uploading files to server...");
    // Remove /api/v1 prefix for file upload endpoint
    const uploadUrl = API_BASE_URL.replace("/api/v1", "") + "/upload-files";
    console.log("📤 Upload URL:", uploadUrl);

    const response = await fetch(uploadUrl, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Upload failed:", errorText);
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    const result = await response.json();
    console.log("✅ Files uploaded successfully:", result);
    return result;
  } catch (error: any) {
    console.error("❌ File upload error:", error.message);
    throw error;
  }
};

// Default camera configurations
export const getDefaultCameraConfigs = (filePaths: {
  [key: string]: string;
}): RunRequest => {
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

  return {
    configs: {
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
    },
  };
};

// Test connectivity on module load
if (typeof window !== "undefined") {
  // Only run in browser
  setTimeout(async () => {
    try {
      const testUrl = `${process.env.NEXT_PUBLIC_API_URL}/healthz`;
      console.log("🔍 Testing backend connectivity at:", testUrl);
      const response = await fetch(testUrl);
      if (response.ok) {
        const data = await response.json();
        console.log("✅ Backend is reachable:", data);
      } else {
        console.warn("⚠️ Backend responded with status:", response.status);
      }
    } catch (error: any) {
      console.error(
        "❌ Backend connectivity test failed:",
        error.message,
        "\n",
        "Please ensure backend is running at http://localhost:8000"
      );
    }
  }, 1000);
}
