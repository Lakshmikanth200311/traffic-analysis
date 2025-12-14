import React, { useState } from "react";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { uploadFiles, getDefaultCameraConfigs } from "@/lib/api";

interface ControlPanelProps {
  isRunning: boolean;
  isConnected: boolean;
  onStart?: (configs: any) => void;
  onStop?: () => void;
  onUpload?: (filePaths: { [key: string]: string }) => void;
  isLoading?: boolean;
  error?: string | null;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  isRunning,
  isConnected,
  onStart,
  onStop,
  onUpload,
  isLoading = false,
  error = null,
}) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<{ [key: string]: string }>(
    {}
  );

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const files = event.target.files;
    if (!files || files.length !== 4) {
      alert("Please select exactly 4 video files");
      return;
    }

    try {
      setIsUploading(true);

      // Convert FileList to array
      const filesArray = Array.from(files);

      // Upload files to server
      const filePaths = await uploadFiles(filesArray);
      setUploadedFiles(filePaths);

      // Notify parent component
      if (onUpload) {
        await onUpload(filePaths);
      }

      alert("Videos uploaded successfully!");
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Failed to upload videos. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleStart = () => {
    if (Object.keys(uploadedFiles).length === 0) {
      alert("Please upload videos first");
      return;
    }

    const configs = getDefaultCameraConfigs(uploadedFiles);
    if (onStart) {
      onStart(configs);
    }
  };

  const handleStop = () => {
    if (onStop) {
      onStop();
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Control Panel</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Error Display */}
        {error && (
          <div className="p-3 bg-red-100 border border-red-300 rounded-lg">
            <div className="text-red-700 text-sm font-medium">
              Error: {error}
            </div>
          </div>
        )}

        {/* Connection Status */}
        <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
          <span className="font-medium">WebSocket Connection</span>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? "bg-green-500" : "bg-red-500"
              }`}
            />
            <span className="text-sm">
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>

        {/* System Status */}
        <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
          <span className="font-medium">System Status</span>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isRunning ? "bg-green-500 animate-pulse" : "bg-gray-500"
              }`}
            />
            <span className="text-sm">{isRunning ? "Running" : "Stopped"}</span>
          </div>
        </div>

        {/* File Upload */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Upload 4 Traffic Videos</label>
          <input
            type="file"
            multiple
            accept="video/*"
            onChange={handleFileUpload}
            disabled={isUploading || isLoading}
            className="w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
          />
          {isUploading && (
            <div className="text-sm text-blue-600 flex items-center gap-2">
              <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600"></div>
              Uploading videos...
            </div>
          )}
          {Object.keys(uploadedFiles).length > 0 && (
            <div className="text-sm text-green-600">
              ✓ {Object.keys(uploadedFiles).length}/4 videos uploaded
            </div>
          )}
        </div>

        {/* Control Buttons */}
        <div className="flex gap-2 pt-2">
          <Button
            onClick={handleStart}
            disabled={
              isRunning ||
              !isConnected ||
              isLoading ||
              isUploading ||
              Object.keys(uploadedFiles).length === 0
            }
            className="flex-1"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                Starting...
              </div>
            ) : (
              "🚀 Start Model"
            )}
          </Button>
          <Button
            onClick={handleStop}
            disabled={!isRunning || isLoading}
            variant="outline"
            className="flex-1"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                Stopping...
              </div>
            ) : (
              "⏹️ Stop"
            )}
          </Button>
        </div>

        {/* Quick Actions */}
        <div className="pt-4 border-t">
          <h4 className="text-sm font-medium mb-2">Quick Actions</h4>
          <div className="grid grid-cols-2 gap-2">
            <Button variant="outline" size="sm" disabled>
              ⏸️ Pause
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setUploadedFiles({})}
              disabled={isRunning}
            >
              🔄 Reset
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
