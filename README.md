# WicketWorm

> Live Test cricket match probability visualizer

WicketWorm is an open-source tool that visualizes **Win / Draw / Loss probabilities** for live Test cricket matches as an interactive worm chart. It uses only **anonymized scorecard state** (no team strength modeling) to provide real-time insights into match dynamics.

## Features

- **Team-agnostic predictions**: Based purely on match state, not team reputation
- **Over-based visualization**: X-axis tracks overs, not real time
- **Browser-based inference**: ML model runs entirely in your browser
- **Offline replay mode**: Review historical matches without live data
- **Open source**: MIT licensed, built with modern web standards

## How it works

1. **Model training** (Python): Train a multinomial logistic regression model on historical Test match data from Cricsheet
2. **Live proxy** (Node/Bun): Normalize live match data from various providers into a canonical format
3. **Browser inference** (TypeScript): Run the trained model in-browser to compute probabilities
4. **Visualization** (D3): Render an interactive worm chart showing probability evolution over overs

## Architecture

```
wicketworm/
  packages/
    shared-types/    # Shared TypeScript types
    model-train/     # Python: train + export model.json
    live-proxy/      # Node/Bun: live match data normalization
    ui/              # Vite + TS + D3: HTML worm chart
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
git clone https://github.com/yourusername/wicketworm.git
cd wicketworm

# Install dependencies
pnpm install

# Train the model
cd packages/model-train
python train.py

# Start the UI in offline mode
cd ../ui
pnpm dev
```

## Roadmap

- [x] Project structure and planning
- [ ] Offline replay mode
- [ ] Model training and validation
- [ ] Browser inference engine
- [ ] Live proxy server
- [ ] Live updates
- [ ] Annotations and milestones
- [ ] Documentation and polish

## Data sources

- **Training data**: [Cricsheet](https://cricsheet.org/) (Test match ball-by-ball data)
- **Live data**: Cricsheet recent matches, ESPN Cricinfo (with rate limiting)

## License

MIT License - see [LICENSE](LICENSE) for details

## Contributing

Contributions welcome! This is an early-stage project. Please open an issue before starting major work.

## Acknowledgments

- Data provided by [Cricsheet](https://cricsheet.org/)
- Built with modern web standards
