import React from "react";
import { OptimizationDelta } from "../types/traffic";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface OptimizationSummaryProps {
  deltas: OptimizationDelta[];
}

export const OptimizationSummary: React.FC<OptimizationSummaryProps> = ({
  deltas,
}) => {
  if (deltas.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Optimization Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            No optimization data yet
          </div>
        </CardContent>
      </Card>
    );
  }

  const getApproachIcon = (approach: string) => {
    switch (approach) {
      case "north":
        return "⬆️";
      case "south":
        return "⬇️";
      case "east":
        return "➡️";
      case "west":
        return "⬅️";
      default:
        return "🚦";
    }
  };

  const getDeltaColor = (delta: number) => {
    if (delta > 0) return "text-green-600";
    if (delta < 0) return "text-red-600";
    return "text-gray-600";
  };

  const getDeltaIcon = (delta: number) => {
    if (delta > 0) return "↗️";
    if (delta < 0) return "↘️";
    return "➡️";
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Optimization Summary</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {deltas
            .slice(-8)
            .reverse()
            .map((delta, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <span className="text-xl">
                    {getApproachIcon(delta.approach)}
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

                <div className={`text-right ${getDeltaColor(delta.delta)}`}>
                  <div className="flex items-center gap-1 font-semibold">
                    <span>{getDeltaIcon(delta.delta)}</span>
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

        {deltas.length > 8 && (
          <div className="mt-4 text-center text-sm text-muted-foreground">
            Showing latest 8 of {deltas.length} optimizations
          </div>
        )}
      </CardContent>
    </Card>
  );
};
