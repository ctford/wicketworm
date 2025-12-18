/**
 * Shared types for WicketWorm
 * Used across model-train and ui packages
 */

/**
 * A single ball event in a Test match
 */
export type BallEvent = {
  innings: 1 | 2 | 3 | 4;
  over: number;
  ball: number;
  runs: number;
  wickets: 0 | 1;
  extras?: number;
  kind?: string;
};

/**
 * Current state of a Test match at any point
 */
export type GameState = {
  matchId: string;
  innings: 1 | 2 | 3 | 4;
  runsFor: number;
  wicketsDown: number;
  ballsBowled: number;
  lead: number;
  target?: number;
  matchOversLimit: number;
  ballsRemaining: number;
  completedInnings: number;
  isChasing: boolean;
};

/**
 * Win/Draw/Loss probabilities at a specific point in the match
 */
export type ProbPoint = {
  xOver: number;
  innings: 1 | 2 | 3 | 4;
  over: number;
  pWin: number;
  pDraw: number;
  pLoss: number;
};

/**
 * Annotation for significant match events
 */
export type Annotation = {
  xOver: number;
  kind: "wicket" | "declare" | "inningsEnd" | "leadChange" | "milestone";
  label?: string;
};

/**
 * Trained model structure for browser inference
 */
export type ModelData = {
  coefficients: number[][];
  intercepts: number[];
  featureMeans: number[];
  featureStds: number[];
  featureNames: string[];
};

/**
 * X-axis offset for innings separation
 */
export const OVER_OFFSET = 500;

/**
 * Calculate x-axis position from innings and over
 */
export function calculateXOver(innings: 1 | 2 | 3 | 4, over: number): number {
  return (innings - 1) * OVER_OFFSET + over;
}
