import React from "react";
import { CyclePlan, Approach } from "../types/traffic";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface PhaseTimelineProps {
  cyclePlan?: CyclePlan;
  activePhase?: Approach;
  remainingSeconds: number;
}

export const PhaseTimeline: React.FC<PhaseTimelineProps> = ({
  cyclePlan,
  activePhase,
  remainingSeconds,
}) => {
  if (!cyclePlan) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Phase Timeline</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            No active cycle plan
          </div>
        </CardContent>
      </Card>
    );
  }

  const getPhaseColor = (phaseApproach: Approach, isActive: boolean) => {
    if (!isActive) return "bg-gray-200";
    if (remainingSeconds <= 3) return "bg-yellow-500";
    return "bg-green-500";
  };

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

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Phase Timeline</span>
          <span className="text-sm font-normal text-muted-foreground">
            Cycle v{cyclePlan.version} • {cyclePlan.cycle_seconds}s total
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {cyclePlan.phases.map((phase, index) => {
            const isActive = phase.approach === activePhase;

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
                      className={`h-8 rounded-l ${getPhaseColor(
                        phase.approach,
                        isActive
                      )} transition-all duration-300`}
                      style={{
                        width: `${
                          (phase.green / cyclePlan.cycle_seconds) * 100
                        }%`,
                      }}
                    />

                    {/* Yellow phase */}
                    <div
                      className="h-8 bg-yellow-500 transition-all duration-300"
                      style={{
                        width: `${
                          (phase.yellow / cyclePlan.cycle_seconds) * 100
                        }%`,
                      }}
                    />

                    {/* Red phase */}
                    <div
                      className="h-8 bg-red-500 rounded-r transition-all duration-300"
                      style={{
                        width: `${
                          (phase.red / cyclePlan.cycle_seconds) * 100
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
                    {remainingSeconds}s
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {activePhase && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <div className="text-sm text-blue-800">
              <strong>Active:</strong> {activePhase.toUpperCase()} approach •
              <strong> Remaining:</strong> {remainingSeconds} seconds
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
