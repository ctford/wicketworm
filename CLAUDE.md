# WicketWorm Development Guide

This document describes how to maintain and update the WicketWorm cricket match prediction visualization.

## Updating Test Match Data

The visualization data for Ashes tests is manually curated in `packages/model-train/src/generate_ashes_series.py`.

### How to Update In-Progress Tests

When a test match is in progress (like Adelaide), update the match data by editing the relevant function in `generate_ashes_series.py`:

#### 1. Locate the Test Function

Find the test you want to update (e.g., `generate_adelaide_test()`, `generate_brisbane_test()`, `generate_perth_test()`).

#### 2. Add New Over Data

Each innings has a loop that generates states at 5-over intervals. Add data points for the current match state:

```python
# Innings 1: Australia batting
for i, over in enumerate(range(0, 85, 5)):  # Extend range as match progresses
    if over == 0:
        runs, wickets = 0, 0
    elif over == 15:
        runs, wickets = 48, 0
    elif over == 30:
        runs, wickets = 102, 2
    # Add new overs here as match progresses
    elif over == 80:
        runs, wickets = 268, 9
    else:
        # Interpolate for intermediate overs
        runs = int(over * 3.35)
        wickets = min(9, over // 10)
```

#### 3. Update Match Metadata

Update the match result and days completed:

```python
return {
    'matchId': 'adelaide-ashes-2025',
    'city': 'Adelaide',
    'dates': 'Dec 17-21, 2025',
    'result': 'In progress (Day 2)',  # Update this
    'days': 2,  # Update this
    'states': states
}
```

#### 4. Regenerate Visualization Data

After updating the match data, regenerate the JSON file:

```bash
cd packages/model-train
source venv/bin/activate
python src/generate_ashes_series.py
```

This will:
- Calculate probabilities for each game state using the hybrid XGBoost + Monte Carlo model
- Save updated data to `packages/ui/src/data/ashes-series-2025.json`
- The dev server will automatically reload with new data

#### 5. Verify in Browser

Check http://localhost:5175/ to see the updated visualization.

### Data Structure

Each test match has:

**States**: Game state at each over
```javascript
{
  matchId: string,
  innings: 1-4,
  over: number,
  runsFor: number,
  wicketsDown: number,
  ballsBowled: number,
  lead: number,  // Positive = first team ahead
  matchOversLimit: 450,
  ballsRemaining: number,
  completedInnings: 0-3,
  isChasing: boolean,
  battingTeam: 'England' | 'Australia'
}
```

**Probabilities**: Calculated for each state
```javascript
{
  xOver: number,  // Cumulative over number (0-450)
  innings: 1-4,
  over: number,
  score: string,  // "123/4" or "Match Complete"
  pWin: number,   // Australia win probability (0-1)
  pDraw: number,  // Draw probability (0-1)
  pLoss: number   // Australia loss probability (0-1)
}
```

### Hybrid Prediction Model

The system uses a hybrid XGBoost + Monte Carlo model:

- **XGBoost**: Used for most situations, learns from historical Test match data
- **Monte Carlo**: Used for close chases when:
  - 4th innings AND
  - (≤3 wickets remaining OR ≤80 runs needed) AND
  - >30 overs remaining

Monte Carlo uses partnership distributions learned from historical data:
- 1st wicket: avg 38.5 runs in 11.0 overs
- 10th wicket: avg 17.0 runs in 3.7 overs

This gives more optimistic/realistic probabilities for easy chases while maintaining the same overall accuracy (57.4%) as pure XGBoost.

### Key Files

- `packages/model-train/src/generate_ashes_series.py` - Match data and probability generation
- `packages/model-train/src/train.py` - XGBoost model training
- `packages/model-train/src/monte_carlo_model.py` - Monte Carlo partnership simulator
- `packages/ui/src/data/ashes-series-2025.json` - Generated visualization data
- `packages/ui/src/chart/worm.ts` - D3.js visualization logic

### Adding New Tests

To add a new test match:

1. Create a new `generate_CITY_test()` function in `generate_ashes_series.py`
2. Follow the same structure as existing tests
3. Add to the main generation:
```python
tests = [
    generate_perth_test(),
    generate_brisbane_test(),
    generate_adelaide_test(),
    generate_YOUR_NEW_test(),  # Add here
]
```
4. Regenerate: `python src/generate_ashes_series.py`

### Model Retraining

If you want to retrain the models with updated historical data:

```bash
cd packages/model-train
source venv/bin/activate

# Download latest Cricsheet Test data to data/ directory
# Visit: https://cricsheet.org/downloads/

# Train XGBoost model
python src/train.py

# Train Monte Carlo model
python src/monte_carlo_model.py
```

This will update:
- `output/model.pkl` - XGBoost model
- `output/model_metadata.json` - Feature names and labels
- `output/monte_carlo_model.pkl` - Partnership distributions
