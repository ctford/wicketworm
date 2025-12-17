import type { GameState, ProbPoint } from '@wicketworm/shared-types';
import { calculateXOver } from '@wicketworm/shared-types';
import { loadModel, predict } from './inference/model';
import { WormChart } from './chart/worm';
import adelaideTestData from './data/adelaide-test.json';

const statusEl = document.getElementById('status');

function updateStatus(message: string): void {
  if (statusEl) {
    statusEl.textContent = message;
  }
  console.log(message);
}

async function main() {
  try {
    updateStatus('Loading model...');

    // Load trained model
    const model = await loadModel();
    updateStatus('Model loaded successfully');

    // Load Adelaide Test match data
    const states = adelaideTestData.states as GameState[];
    updateStatus(`Loaded Adelaide Ashes Test - ${states.length} match states`);

    // Compute probabilities for each state
    updateStatus('Computing probabilities...');
    const probPoints: ProbPoint[] = states.map(state => {
      const probs = predict(model, state);
      return {
        xOver: calculateXOver(state.innings, state.ballsBowled / 6),
        innings: state.innings,
        over: state.ballsBowled / 6,
        pWin: probs.pWin,
        pDraw: probs.pDraw,
        pLoss: probs.pLoss
      };
    });

    updateStatus('Rendering worm chart...');

    // Create and render chart
    const chart = new WormChart('#chart', {
      height: 500
    });
    chart.render(probPoints);

    updateStatus(`${adelaideTestData.description} (${probPoints.length} points)`);

    // Log sample probabilities
    console.log('Sample probabilities:');
    console.log(`Start of match:`, probPoints[0]);
    console.log(`Mid-match:`, probPoints[Math.floor(probPoints.length / 2)]);
    console.log(`End of match:`, probPoints[probPoints.length - 1]);

  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    updateStatus(`Error: ${message}`);
    console.error(error);
  }
}

// Start the app
main();
