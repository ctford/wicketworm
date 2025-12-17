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
- Calculate probabilities for each game state using the XGBoost model
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

### XGBoost Prediction Model

The system uses an XGBoost model with **83.5% test accuracy** on 277,401 game states.

#### Model Features (10 features)

Features with importance percentages:

1. **first_team_rating** (15.2%) - ELO rating of first batting team
2. **second_team_rating** (14.6%) - ELO rating of second batting team
3. **first_team_is_home** (11.8%) - Home advantage
4. **first_team_wickets_remaining** (11.1%) - Wickets left (20 → 0)
5. **second_team_wickets_remaining** (11.1%) - Wickets left (20 → 0)
6. **first_team_won_toss** (10.9%) - Toss advantage
7. **overs_left** (9.7%) - Match time remaining (450 → 0)
8. **first_team_lead** (7.2%) - Run lead/deficit
9. **chase_ease** (4.2%) - Inverse of runs per wicket (4th innings only)
10. **required_run_rate** (4.1%) - Runs per over needed (4th innings only)

Training data: 277,401 game states from 865 Test matches (1970s-2025)
- Ashes 2025-26 excluded from training (used for ELO only) to ensure out-of-sample predictions
- Recency weighting: 10-year exponential decay (recent matches weighted higher)

### Key Files

- `packages/model-train/src/generate_ashes_series.py` - Match data and probability generation
- `packages/model-train/src/train.py` - XGBoost model training
- `packages/model-train/src/parse_cricsheet.py` - Cricsheet data parser with ELO system
- `packages/model-train/src/team_ratings.py` - ELO rating system implementation
- `packages/ui/src/data/ashes-series-2025.json` - Generated visualization data
- `packages/ui/src/chart/worm.ts` - D3.js visualization logic

### Adding New Tests (e.g., Melbourne Test)

To add a new test match to the series:

#### 1. Gather Match Information

Before coding, collect:
- **Toss winner**: Who won the toss and which team batted first
- **Match scorecard**: Runs and wickets for each innings at regular intervals
- **Match venue**: City (e.g., Melbourne)
- **Match dates**: e.g., "Dec 26-30, 2025"

**Important**: The team that bats first determines how ratings are assigned:
- If England bats first: `first_team_rating=1593.1`, `second_team_rating=1757.3`
- If Australia bats first: `first_team_rating=1757.3`, `second_team_rating=1593.1`

The prediction function will automatically handle this in `generate_ashes_series.py` at line 549-554.

#### 2. Create Generation Function

Add a new function in `packages/model-train/src/generate_ashes_series.py`:

```python
def generate_melbourne_test():
    """
    Melbourne Test (Dec 26-30, 2025)
    [Add final result when known]
    England: XXX & XXX
    Australia: XXX & XXX
    """
    states = []

    # Innings 1: [TEAM] batting (estimate overs based on run rate)
    for over in range(0, FINAL_OVER + 1, 5):
        if over == 0:
            runs, wickets = 0, 0
        elif over == 20:
            runs, wickets = 67, 1  # From actual scorecard
        elif over == 40:
            runs, wickets = 142, 3  # From actual scorecard
        # Add more data points as match progresses
        else:
            # Interpolate for intermediate overs
            runs = int(over * 3.0)  # Adjust run rate
            wickets = min(10, over // 15)  # Adjust wicket fall rate

        states.append({
            'matchId': 'melbourne-test-2025',
            'innings': 1,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': runs,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - over * 6,
            'completedInnings': 0,
            'isChasing': False
        })

    # Add innings 2, 3, 4 following the same pattern...

    # Add batting teams (specify which team batted first)
    states = add_batting_teams(states, first_batting='England')  # or 'Australia'

    return {
        'matchId': 'melbourne-test-2025',
        'city': 'Melbourne',
        'dates': 'Dec 26-30, 2025',
        'result': 'In progress (Day 1)',  # Update as match progresses
        'days': 1,
        'states': states
    }
```

#### 3. Add to Main Generation

In the `main()` function around line 446, add the new test:

```python
def main():
    # Generate data for all tests
    perth = generate_perth_test()
    brisbane = generate_brisbane_test()
    adelaide = generate_adelaide_test()
    melbourne = generate_melbourne_test()  # Add new test

    # Add to output around line 660
    output = {
        'series': 'The Ashes 2025-26',
        'tests': [perth, brisbane, adelaide, melbourne]  # Add here
    }
```

#### 4. Regenerate Predictions

```bash
cd packages/model-train
source venv/bin/activate
python src/generate_ashes_series.py
```

This automatically:
- Determines which team bats first from the `first_batting` parameter
- Assigns correct team ratings (England: 1593.1, Australia: 1757.3)
- Sets home team (all Ashes 2025-26 matches in Australia)
- Sets toss winner (assumes team batting first won toss)
- Calculates probabilities using the 10-feature model

#### 5. Verify Output

```bash
cd ../ui
pnpm dev
# Visit http://localhost:5173/
```

Check that:
- Melbourne test appears in the visualization
- Probabilities look reasonable at the start (home team with toss should be favored)
- All four tests display correctly

### Model Retraining

If you want to retrain the model with updated historical data:

```bash
cd packages/model-train
source venv/bin/activate

# Download latest Cricsheet Test data to data/ directory
# Visit: https://cricsheet.org/downloads/

# Train XGBoost model
python src/train.py
```

This will update:
- `output/model.pkl` - XGBoost model
- `output/model_metadata.json` - Feature names and labels

#### IMPORTANT: Excluding Ashes 2025-26 from Training

To avoid data leakage, Ashes 2025-26 matches must be excluded from training but included in ELO calculation.

**When new Ashes tests appear in Cricsheet**, add them to the exclusion list in `src/train.py`:

```python
# Around line 207 in train.py
exclude_matches = [
    '1455611',  # Perth Test (Nov 21-22, 2025)
    '1455612',  # Brisbane Test (Dec 4-7, 2025) - ADD WHEN AVAILABLE
    '1455613',  # Adelaide Test (Dec 17-21, 2025) - ADD WHEN AVAILABLE
    # Add Melbourne, Sydney when they appear in data/
]
```

**How to find match IDs**:
```bash
python3 -c "
import json
from pathlib import Path
for file in Path('data').glob('*.json'):
    with open(file) as f:
        data = json.load(f)
    dates = data.get('info', {}).get('dates', [])
    teams = list(data.get('info', {}).get('players', {}).keys())
    if dates and 'England' in teams and 'Australia' in teams and dates[0] >= '2025-11-21':
        print(f'{dates[0]}  {file.stem}')
"
```

**Why this matters**:
- ✓ ELO ratings stay current (uses all matches including Ashes 2025-26)
- ✓ Model predictions are out-of-sample (hasn't seen game states)
- ✓ True validation of model performance

Current exclusions: Perth (1455611)
