import type { GameState, ProbPoint } from '@wicketworm/shared-types';

console.log('WicketWorm UI starting...');

// TODO: Load model.json
// TODO: Set up offline replay data
// TODO: Initialize D3 chart
// TODO: Render worm chart

const statusEl = document.getElementById('status');
if (statusEl) {
  statusEl.textContent = 'Offline replay mode - UI skeleton ready';
}

// Placeholder for development
console.log('UI package structure created. Next steps:');
console.log('1. Implement model inference (src/inference/)');
console.log('2. Create D3 worm chart (src/chart/)');
console.log('3. Add sample match data (src/data/)');
console.log('4. Wire up offline replay');
