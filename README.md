# WicketWorm

[![CI](https://github.com/ctford/wicketworm/actions/workflows/ci.yml/badge.svg)](https://github.com/ctford/wicketworm/actions/workflows/ci.yml)

> Test cricket match probability visualizer

WicketWorm is an open-source tool that visualizes **Win / Draw / Loss probabilities** for Test cricket matches as an interactive worm chart. It uses only **anonymized scorecard state** (no team strength modeling) to provide insights into match dynamics.

**[View Demo →](https://ctford.github.io/wicketworm/)**

## How it works

1. **Model training** (Python): Train hybrid XGBoost + Monte Carlo model on historical Test match data from Cricsheet
2. **Probability generation** (Python): Generate predictions for match states using the hybrid model
3. **Visualization** (D3): Render an interactive worm chart showing probability evolution over overs with smooth curves and wicket markers

## Architecture

```
wicketworm/
  packages/
    shared-types/    # Shared TypeScript types
    model-train/     # Python: XGBoost + Monte Carlo model training
    ui/              # Vite + TS + D3: Interactive worm chart visualization
```

## Key principles

- Probabilities always sum to 1.0
- No team strength, venue effects, or player ratings
- Simple, explainable model
- Works offline-first
- Test matches only

## Getting started

### Prerequisites

- Node.js 18+ (or Bun)
- Python 3.10+
- pnpm 8+

### Installation

```bash
# Clone the repository
git clone https://github.com/ctford/wicketworm.git
cd wicketworm

# Install dependencies
pnpm install

# Start the visualization
cd packages/ui
pnpm dev
```

The visualization will show pre-generated probability data for The Ashes 2025-26 series.

## Data sources

- **Training data**: [Cricsheet](https://cricsheet.org/) (Test match ball-by-ball data)

## License

MIT License - see [LICENSE](LICENSE) for details

## Contributing

Contributions welcome! This is an early-stage project. Please open an issue before starting major work.
