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

    // Expose data globally for debugging
    (window as any).ashesData = seriesData;

    // Debug: Log Perth innings 4 data
    const perth = seriesData.tests.find((t: any) => t.city === 'Perth');
    if (perth) {
      const inn4_177 = perth.probabilities.find((p: any) => p.score === '177/1');
      if (inn4_177) {
        console.log('🏏 Perth 177/1 prediction:', (inn4_177.pWin * 100).toFixed(1) + '% AUS');
      }
    }

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
        wicketFalls: test.wicketFalls,
        matchEndOver: test.matchEndOver
      });

      // Keep real probabilities, add transition at match end, then extend with winner's color
      const realProbabilities = test.probabilities.filter((p: any) => p.score !== 'Match Complete');
      const matchCompletePoints = test.probabilities.filter((p: any) => p.score === 'Match Complete');

      let probsToRender = [...realProbabilities];

      // If match is complete, add final result at match end line, then extend to fill chart
      if (test.matchEndOver !== undefined && matchCompletePoints.length > 0) {
        const finalProbs = matchCompletePoints[0];

        // Add transition point at match end line with final probabilities
        probsToRender.push({
          ...finalProbs,
          xOver: test.matchEndOver
        });

        // Add all Match Complete extension points to fill the rest of the chart
        // with the winning team's color
        probsToRender.push(...matchCompletePoints);
      }

      const probPoints: ProbPoint[] = probsToRender.map((p: any) => ({
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
