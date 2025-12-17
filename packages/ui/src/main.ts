import type { ProbPoint } from '@wicketworm/shared-types';
import { WormChart } from './chart/worm';
import ashesSeriesData from './data/ashes-series-2025.json';

const statusEl = document.getElementById('status');

function updateStatus(message: string): void {
  if (statusEl) {
    statusEl.textContent = message;
  }
  console.log(message);
}

async function main() {
  try {
    updateStatus('Loading Ashes series data...');

    const seriesData = ashesSeriesData as any;
    const container = document.getElementById('charts-container');

    if (!container) {
      throw new Error('Charts container not found');
    }

    // Clear loading message
    container.innerHTML = '';

    // Create a chart for each test
    for (const test of seriesData.tests) {
      // Create test container
      const testDiv = document.createElement('div');
      testDiv.className = 'test-chart';

      // Add header
      const headerDiv = document.createElement('div');
      headerDiv.className = 'test-header';

      const cityDiv = document.createElement('div');
      cityDiv.className = 'test-city';
      cityDiv.textContent = test.city;

      const datesDiv = document.createElement('div');
      datesDiv.className = 'test-dates';
      datesDiv.textContent = test.dates;

      const resultDiv = document.createElement('div');
      resultDiv.className = 'test-result';
      resultDiv.textContent = test.result;

      headerDiv.appendChild(cityDiv);
      headerDiv.appendChild(datesDiv);
      headerDiv.appendChild(resultDiv);

      // Add chart container
      const chartDiv = document.createElement('div');
      chartDiv.className = 'chart-svg';
      chartDiv.id = `chart-${test.matchId}`;

      testDiv.appendChild(headerDiv);
      testDiv.appendChild(chartDiv);
      container.appendChild(testDiv);

      // Render chart
      const chart = new WormChart(`#chart-${test.matchId}`, {
        height: 200,
        maxOvers: 450,  // Show 450 overs (5 days * 90 overs/day)
        inningsBoundaries: test.inningsBoundaries,
        wicketFalls: test.wicketFalls
      });

      const probPoints: ProbPoint[] = test.probabilities.map((p: any) => ({
        xOver: p.xOver,  // Use xOver with innings offset, not raw over number
        innings: p.innings,
        over: p.over,
        pWin: p.pWin,
        pDraw: p.pDraw,
        pLoss: p.pLoss
      }));

      chart.render(probPoints);

      console.log(`${test.city}: ${probPoints.length} points, ${test.days} day(s)`);
    }

    updateStatus(`${seriesData.series} - ${seriesData.tests.length} tests loaded`);

  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    updateStatus(`Error: ${message}`);
    console.error(error);
  }
}

// Start the app
main();
