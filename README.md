# WicketWorm

[![CI](https://github.com/ctford/wicketworm/actions/workflows/ci.yml/badge.svg)](https://github.com/ctford/wicketworm/actions/workflows/ci.yml)

> Test cricket match probability visualizer

WicketWorm is an open-source tool that visualizes **Win / Draw / Loss probabilities** for Test cricket matches as an interactive worm chart. It uses a hybrid XGBoost + Monte Carlo model trained on historical match data from Cricsheet.

**[View Demo →](https://ctford.github.io/wicketworm/)**

## How it works

1. **Model training** (Python): Train hybrid XGBoost + Monte Carlo model on 866 historical Test matches from Cricsheet
2. **Probability generation** (Python): Generate predictions for match states using the hybrid model
3. **Visualization** (D3): Render an interactive worm chart showing probability evolution over overs with smooth curves and wicket markers

## Model features

The XGBoost model uses **10 features** to predict match outcomes (Win/Draw/Loss) with **83.7% accuracy**:

### Core match state (50.8%)
- **Wickets remaining** (22.2%): First and second team wickets left (20 → 0 each)
- **First team lead** (7.2%): Run lead/deficit
- **Overs left** (9.7%): Match time remaining (450 → 0)
- **Home advantage** (11.8%): Whether first batting team plays at home

### Team context (51.3%)
- **Team ratings** (29.8%): ELO ratings for both teams built from match history
- **Toss advantage** (10.9%): Whether first batting team won the toss

### Chase dynamics (8.3%)
- **Chase ease** (4.2%): Inverse of runs required per wicket (4th innings only)
- **Required run rate** (4.1%): Runs per over needed to win (4th innings only)

## Architecture

```
wicketworm/
  packages/
    shared-types/    # Shared TypeScript types
    model-train/     # Python: XGBoost + Monte Carlo model training
    ui/              # Vite + TS + D3: Interactive worm chart visualization
```

## Model performance

- **Training data**: 277,544 game states from 866 Test matches (1970s-2025)
- **Test accuracy**: 83.7%
- **Hybrid approach**: XGBoost for time pressure scenarios, Monte Carlo for tight finishes
- **Recency weighting**: Recent matches weighted higher (10-year exponential decay)

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
