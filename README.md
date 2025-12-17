# WicketWorm

[![CI](https://github.com/ctford/wicketworm/actions/workflows/ci.yml/badge.svg)](https://github.com/ctford/wicketworm/actions/workflows/ci.yml)

> Test cricket match probability visualizer

WicketWorm is an open-source tool that visualizes **Win / Draw / Loss probabilities** for Test cricket matches as an interactive worm chart. It uses an XGBoost model trained on historical match data from Cricsheet.

**[View Demo →](https://ctford.github.io/wicketworm/)**

## How it works

1. **Model training** (Python): Train XGBoost model on 865 historical Test matches from Cricsheet (277,401 game states)
2. **Probability generation** (Python): Generate predictions for match states using the trained model
3. **Visualization** (D3): Render an interactive worm chart showing probability evolution over overs with smooth curves and wicket markers

## Ashes 2025-26 Match Reports

Detailed scorecards and day-by-day summaries for the current Ashes series:

- **[PERTH.md](PERTH.md)** - 1st Test: Australia won by 8 wickets (Nov 21-22, 2025) - Travis Head's 69-ball century in a two-day thriller
- **[BRISBANE.md](BRISBANE.md)** - 2nd Test: Australia won by 8 wickets (Dec 4-7, 2025) - Joe Root's maiden Ashes century in Australia
- **[ADELAIDE.md](ADELAIDE.md)** - 3rd Test: In Progress (Dec 17-21, 2025) - Alex Carey's maiden Ashes century

**Series Status:** Australia leads 2-0 with 3 Tests remaining

## Model features

The XGBoost model uses **8 features** to predict match outcomes (Win/Draw/Loss) with **83.5% accuracy**:

### Core match state
- **Wickets remaining**: First and second team wickets left (20 → 0 each)
- **First team lead**: Run lead/deficit
- **Overs left**: Match time remaining (450 → 0)
- **Home advantage**: Whether first batting team plays at home

### Team context
- **Team ratings**: ELO ratings for both teams built from match history
- **Toss advantage**: Whether first batting team won the toss

**→ See [FEATURES.md](FEATURES.md) for detailed explanations of all 8 features, value ranges, and examples.**

## Architecture

```
wicketworm/
  packages/
    shared-types/    # Shared TypeScript types
    model-train/     # Python: XGBoost model training and prediction generation
    ui/              # Vite + TS + D3: Interactive worm chart visualization
```

## Model performance

- **Training data**: 277,401 game states from 865 Test matches (1970s-2025)
- **Test accuracy**: 83.5%
- **Recency weighting**: Recent matches weighted higher (10-year exponential decay)
- **Out-of-sample validation**: Ashes 2025-26 excluded from training for genuine predictions

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

## Documentation

- **[FEATURES.md](FEATURES.md)** - Detailed explanation of all 10 model features with examples and value ranges
- **[CLAUDE.md](CLAUDE.md)** - Development guide for updating match data and maintaining the model

## License

MIT License - see [LICENSE](LICENSE) for details

## Contributing

Contributions welcome! This is an early-stage project. Please open an issue before starting major work.
