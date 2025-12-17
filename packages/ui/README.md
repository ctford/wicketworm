# ui

Interactive worm chart visualization for Test match Win/Draw/Loss probabilities.

## Features

- **Stacked area chart**: Win (green), Draw (gray), Loss (red)
- **X-axis**: Overs with innings boundaries
- **Y-axis**: Probability (0-1)
- **Annotations**: Wickets, declarations, innings endings
- **Offline replay**: Works without live data
- **Responsive**: Works on desktop and mobile

## Architecture

```
src/
  inference/       # Browser-based model inference
  chart/           # D3 worm chart rendering
  data/            # Sample match data for offline mode
  main.ts          # Entry point
```

## Development

```bash
pnpm install
pnpm dev
```

Open `http://localhost:5173`

## Offline replay mode

The UI works in offline mode by default, loading sample match data from `src/data/sample-match.json`.

To connect to live proxy:
```typescript
// In src/main.ts
const API_URL = 'http://localhost:3000/api';
```

## Model inference

The browser loads `model.json` (trained in `model-train` package) and runs inference in pure TypeScript:

1. Load model.json
2. Compute features from GameState
3. Standardize features
4. Compute logits: `W·x + b`
5. Apply softmax
6. Return {pWin, pDraw, pLoss}

No ML library needed!
