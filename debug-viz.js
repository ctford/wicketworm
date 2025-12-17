const { chromium } = require('playwright');
const fs = require('fs');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  // Navigate to the app
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle' });

  // Wait for charts to render
  await page.waitForTimeout(2000);

  // Take screenshot
  await page.screenshot({
    path: 'visualization-debug.png',
    fullPage: true
  });

  // Extract data from the page
  const chartData = await page.evaluate(() => {
    const results = [];

    // Get all test charts
    const testCharts = document.querySelectorAll('.test-chart');

    testCharts.forEach((chart, index) => {
      const city = chart.querySelector('.test-city')?.textContent;
      const dates = chart.querySelector('.test-dates')?.textContent;
      const result = chart.querySelector('.test-result')?.textContent;

      results.push({
        index,
        city,
        dates,
        result
      });
    });

    return results;
  });

  console.log('Chart data from page:');
  console.log(JSON.stringify(chartData, null, 2));

  // Also log the raw JSON data being loaded
  const rawData = await page.evaluate(() => {
    return fetch('/src/data/ashes-series-2025.json')
      .then(r => r.json())
      .then(data => {
        // Return summary of each test
        return data.tests.map(test => {
          const firstProb = test.probabilities[0];
          const lastProb = test.probabilities[test.probabilities.length - 1];
          const midProb = test.probabilities[Math.floor(test.probabilities.length / 2)];

          return {
            city: test.city,
            result: test.result,
            totalPoints: test.probabilities.length,
            firstPoint: {
              over: firstProb.over,
              pWin: firstProb.pWin,
              pDraw: firstProb.pDraw,
              pLoss: firstProb.pLoss
            },
            midPoint: {
              over: midProb.over,
              pWin: midProb.pWin,
              pDraw: midProb.pDraw,
              pLoss: midProb.pLoss
            },
            lastPoint: {
              over: lastProb.over,
              pWin: lastProb.pWin,
              pDraw: lastProb.pDraw,
              pLoss: lastProb.pLoss
            }
          };
        });
      });
  });

  console.log('\nData loaded by app:');
  console.log(JSON.stringify(rawData, null, 2));

  // Save to file
  fs.writeFileSync('debug-data.json', JSON.stringify({ chartData, rawData }, null, 2));

  console.log('\n✓ Screenshot saved to visualization-debug.png');
  console.log('✓ Data saved to debug-data.json');

  await browser.close();
})();
