import React from "react";
import { Approach, VehicleCounts } from "../types/traffic";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface VideoPanelProps {
  approach: Approach;
  counts: VehicleCounts;
  isActive: boolean;
  remainingSeconds?: number;
}

export const VideoPanel: React.FC<VideoPanelProps> = ({
  approach,
  counts,
  isActive,
  remainingSeconds,
}) => {
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

  const getStatusColor = () => {
    if (!isActive) return "bg-gray-500";
    if (remainingSeconds && remainingSeconds <= 3) return "bg-yellow-500";
    return "bg-green-500";
  };

  return (
    <Card
      className={`relative overflow-hidden ${
        isActive ? "ring-2 ring-blue-500" : ""
      }`}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <span>{getApproachIcon(approach)}</span>
            {approach.toUpperCase()}
          </CardTitle>
          <div
            className={`w-3 h-3 rounded-full ${getStatusColor()} animate-pulse`}
          />
        </div>
        {isActive && remainingSeconds !== undefined && (
          <div className="text-sm text-muted-foreground">
            {remainingSeconds}s remaining
          </div>
        )}
      </CardHeader>

      <CardContent>
        {/* Video placeholder - would be replaced with actual video stream */}
        <div className="aspect-video bg-muted rounded-lg mb-4 flex items-center justify-center">
          <div className="text-center text-muted-foreground">
            <div className="text-2xl mb-2">🎥</div>
            <div>Live Feed</div>
            <div className="text-xs mt-1">{approach} approach</div>
          </div>
        </div>

        {/* Vehicle counts */}
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="flex justify-between">
            <span>🚗 Cars:</span>
            <span className="font-semibold">{counts.car}</span>
          </div>
          <div className="flex justify-between">
            <span>🏍️ Motorcycles:</span>
            <span className="font-semibold">{counts.motorcycle}</span>
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
            <span className="font-bold text-lg">{counts.total}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
