"use client";
import React, { useEffect, useRef } from "react";
import { Approach } from "@/types/traffic";

// Add this component for video display
const VideoFeed: React.FC<{ approach: Approach; isActive: boolean }> = ({
  approach,
  isActive,
}) => {
  const videoRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (videoRef.current) {
      // Set up the video stream source
      videoRef.current.src = `http://localhost:8000/api/v1/video_feed/${approach}`;
    }

    return () => {
      // Cleanup if needed
      if (videoRef.current) {
        videoRef.current.src = "";
      }
    };
  }, [approach]);

  return (
    <div className="relative">
      <div className="aspect-video bg-black rounded-lg mb-4 overflow-hidden">
        <img
          ref={videoRef}
          alt={`Live feed for ${approach}`}
          className="w-full h-full object-cover"
          style={{
            border: isActive ? "3px solid #3B82F6" : "none",
            opacity: isActive ? 1 : 0.8,
          }}
        />
      </div>

      {/* Live indicator */}
      <div className="absolute top-3 left-3 flex items-center gap-2">
        <div className="flex items-center gap-1 bg-red-600 text-white px-2 py-1 rounded text-xs font-semibold">
          <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
          LIVE
        </div>
        <div
          className={`px-2 py-1 rounded text-xs font-semibold ${
            isActive ? "bg-green-600 text-white" : "bg-gray-600 text-gray-200"
          }`}
        >
          {isActive ? "ACTIVE" : "WAITING"}
        </div>
      </div>
    </div>
  );
};

export default VideoFeed;
