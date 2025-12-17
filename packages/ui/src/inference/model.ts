import type { ModelData, GameState } from '@wicketworm/shared-types';

/**
 * Load trained model from JSON
 */
export async function loadModel(url: string = '/model.json'): Promise<ModelData> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load model: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Extract feature vector from game state
 */
export function extractFeatures(state: GameState): number[] {
  const runRate = state.ballsBowled > 0
    ? (state.runsFor / state.ballsBowled) * 6
    : 0;

  const runsPerWicket = state.runsFor / (state.wicketsDown + 1);

  const requiredRunRate = state.isChasing && state.target && state.ballsRemaining > 0
    ? ((state.target - state.runsFor) / state.ballsRemaining) * 6
    : 0;

  return [
    state.innings,
    state.wicketsDown,
    runRate,
    state.lead,
    state.ballsRemaining,
    runsPerWicket,
    state.isChasing ? 1 : 0,
    requiredRunRate
  ];
}

/**
 * Standardize features using model means and stds
 */
function standardize(features: number[], means: number[], stds: number[]): number[] {
  return features.map((f, i) => (f - means[i]) / stds[i]);
}

/**
 * Compute softmax over logits
 */
function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExps);
}

/**
 * Predict Win/Draw/Loss probabilities from game state
 */
export function predict(
  model: ModelData,
  state: GameState
): { pWin: number; pDraw: number; pLoss: number } {
  // Extract features
  const features = extractFeatures(state);

  // Standardize
  const standardized = standardize(features, model.featureMeans, model.featureStds);

  // Compute logits: W·x + b
  const logits = model.coefficients.map((weights, i) => {
    const dotProduct = weights.reduce((sum, w, j) => sum + w * standardized[j], 0);
    return dotProduct + model.intercepts[i];
  });

  // Apply softmax
  const probs = softmax(logits);

  return {
    pWin: probs[0],
    pDraw: probs[1],
    pLoss: probs[2]
  };
}
