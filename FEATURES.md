# WicketWorm Model Features

The XGBoost model uses **8 features** to predict Test match outcomes (Win/Draw/Loss) with **83.5% accuracy**.

## Feature Categories

### 1. Team Strength (29.8% importance)

#### `first_team_rating` (15.2%)
- **Description**: ELO rating of the team that bats innings 1 and 3
- **Range**: Typically 1400-1800 (1500 = starting rating)
- **Source**: Built from 866 historical Test matches using ELO system (K-factor=40)
- **Why it matters**: Captures team quality. Higher-rated teams win more often.
- **Current Ashes ratings**: Australia 1757.3, England 1593.1

#### `second_team_rating` (14.6%)
- **Description**: ELO rating of the team that bats innings 2 and 4
- **Range**: Typically 1400-1800
- **Source**: Same ELO system as first_team_rating
- **Why it matters**: Strength gap between teams is crucial. 164-point gap → 72% win probability on neutral ground.

**Note**: The model uses both absolute ratings, not just the difference. Matches between highly-rated teams behave differently than matches between lower-rated teams with the same gap.

### 2. Match Resources (22.2% importance)

#### `first_team_wickets_remaining` (11.1%)
- **Description**: Cumulative wickets available across both innings (1 and 3)
- **Range**: 0-20 (starts at 20, decreases as wickets fall)
- **Calculation**: `20 - (innings_1_wickets + innings_3_wickets)`
- **Why it matters**: Core cricket resource. When 0, team is all out and can't score more.
- **Example**: After innings 1: 172 all out → 10 wickets remaining for innings 3

#### `second_team_wickets_remaining` (11.1%)
- **Description**: Cumulative wickets available across both innings (2 and 4)
- **Range**: 0-20 (starts at 20, decreases as wickets fall)
- **Calculation**: `20 - (innings_2_wickets + innings_4_wickets)`
- **Why it matters**: Equally important as first team wickets. Determines capacity to chase or defend.

### 3. Home Advantage (11.8% importance)

#### `first_team_is_home` (11.8%)
- **Description**: Binary indicator of whether first batting team plays at home
- **Range**: 0 or 1 (1 = playing at home)
- **Determination**: Based on city-to-country mapping covering 58 venues across 12 Test nations
- **Why it matters**: Home teams have significant advantage from:
  - Familiarity with conditions
  - Crowd support
  - No travel fatigue
- **Example**: All Ashes 2025-26 matches in Australia → Australia always 1 when batting first

### 4. Toss Advantage (10.9% importance)

#### `first_team_won_toss` (10.9%)
- **Description**: Binary indicator of whether first batting team won the toss
- **Range**: 0 or 1 (1 = won toss)
- **Why it matters**: Winning toss allows choice of:
  - Batting first on a good pitch
  - Bowling first if conditions favor bowlers
  - Strategic advantage in match planning
- **Ashes 2025-26**:
  - Perth: England won toss, chose to bat (lost anyway)
  - Brisbane: England won toss, chose to bat (lost anyway)
  - Adelaide: Australia won toss, chose to bat

### 5. Time Pressure (9.7% importance)

#### `overs_left` (9.7%)
- **Description**: Total match overs remaining
- **Range**: 450 → 0 (5-day Test = 90 overs/day × 5 days)
- **Calculation**: `450 - total_overs_bowled`
- **Why it matters**: Creates draw pressure as time runs out. Teams can "bat for the draw" when behind with little time left.
- **Example**: At 50 overs left, even trailing by 100 runs, draw becomes more likely than loss.

### 6. Match Position (7.2% importance)

#### `first_team_lead` (7.2%)
- **Description**: First team's run lead (positive) or deficit (negative)
- **Range**: Typically -500 to +500 (can be larger in exceptional matches)
- **Calculation**: `(innings_1_runs + innings_3_runs) - (innings_2_runs + innings_4_runs)`
- **Why it matters**: Core indicator of who's ahead. Large lead → high win probability.
- **Examples**:
  - +65 runs: Moderate lead, need wickets to defend
  - -177 runs: Significant deficit, difficult to recover
  - +300 runs: Dominant position, very high win probability

## Feature Importance Interpretation

Feature importance shows **how much the model relies on each feature** when making predictions:

- **High importance** (>10%): Critical to most predictions
- **Medium importance** (5-10%): Important context, especially in specific situations
- **Low importance** (<5%): Useful but not decisive on their own

## Why These Features Work

### Covers All Match Phases
- **Early match**: Team ratings, home advantage, toss predict starting probabilities
- **Mid-match**: Wickets remaining, lead, overs left track match development
- **Late match**: Combination of lead, wickets, and overs captures chase dynamics naturally

### Balances Different Aspects
- **Quality** (ratings): Who's better?
- **Resources** (wickets, overs): Who has more left?
- **Position** (lead): Who's ahead?
- **Context** (home, toss): Who has advantages?

### Cumulative Design
Features like wickets_remaining accumulate across **both innings** (20 → 0), which:
- Eliminates anomalous jumps at innings boundaries
- Naturally captures "all out" conditions (0 wickets = can't score more)
- Aligns with cricket intuition

## Model Training

- **Training data**: 277,401 game states from 865 Test matches (1970s-2025)
- **Excluded**: Ashes 2025-26 (used for ELO but not training → out-of-sample validation)
- **Recency weighting**: 10-year exponential decay (recent matches weighted higher)
- **Test accuracy**: 83.5%
- **Architecture**: XGBoost with 100 trees, max depth 6, learning rate 0.1

## Feature Evolution

The model has evolved through several iterations:
1. **v1**: 6 features (no team context) → 57.4% accuracy
2. **v2**: +1 home advantage → 57.9% accuracy (+0.5%)
3. **v3**: +2 team ratings → 83.3% accuracy (+25.4% 🚀)
4. **v4**: +1 toss advantage → 83.7% accuracy (+0.4%)
5. **v5**: Excluded Ashes 2025-26 → 83.5% accuracy (clean out-of-sample)
6. **v6**: Removed chase dynamics (redundant features) → 83.5% accuracy (maintained)

The massive jump from v2 to v3 shows that **team strength ratings are crucial** for accurate predictions. The v6 simplification confirms that chase_ease and required_run_rate were redundant—XGBoost learns these patterns from the base features (lead, overs, wickets) without explicit calculation.

## See Also

- [README.md](README.md) - Project overview
- [CLAUDE.md](CLAUDE.md) - Development guide with detailed feature calculations
- [packages/model-train/src/train.py](packages/model-train/src/train.py) - Feature extraction code
- [packages/model-train/src/team_ratings.py](packages/model-train/src/team_ratings.py) - ELO rating system
