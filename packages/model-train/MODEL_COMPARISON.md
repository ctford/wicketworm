# WicketWorm Model Comparison: Team-Aware vs Scorecard-Only

WicketWorm uses two XGBoost models to predict Test match outcomes (Win/Draw/Loss):

1. **Team-Aware Model** (8 features) - Default, 83.5% accuracy
2. **Scorecard-Only Model** (5 features) - Optional, 58.8% accuracy

## Why Two Models?

The **team-aware model** incorporates context about team strength (ELO ratings) and home advantage, making it more accurate but dependent on knowing which teams are playing.

The **scorecard-only model** uses only visible match state that anyone watching can see on a scorecard, making predictions "context-free" but less accurate. This lets users compare how much team strength matters vs pure match situation.

## Model Architecture

Both models use **XGBoost multi-class classification**:
- 3 output classes: Win, Draw, Loss
- Label encoding: {draw: 0, loss: 1, win: 2}
- Predictions from first batting team's perspective (flipped for Australia in UI)
- Trained on historical Test matches with exponential time decay (recent matches weighted higher)

## Team-Aware Model (8 features)

### Features
1. **overs_left** (450 → 0): Total match overs remaining
2. **first_team_wickets_remaining** (20 → 0): First batting team's wickets left across both innings
3. **second_team_wickets_remaining** (20 → 0): Second batting team's wickets left across both innings
4. **first_team_lead** (+/- runs): First team's lead (positive) or deficit (negative)
5. **first_team_is_home** (0/1): Whether first team is playing at home
6. **first_team_won_toss** (0/1): Whether first team won the toss
7. **first_team_rating** (1000-2000): First team's ELO rating
8. **second_team_rating** (1000-2000): Second team's ELO rating

### Performance
- **Test Accuracy**: 83.5%
- **Training**: 70% train, 15% validation, 15% test split

### Feature Importance (from XGBoost)
The most influential features for predictions:

1. **first_team_lead** - Most important by far
   - Dominant predictor: large leads strongly predict wins
   - Captures current match momentum

2. **first_team_rating & second_team_rating** - Second & third most important
   - Team strength is a major factor (explains the 25% accuracy gap)
   - Stronger teams win from similar positions more often

3. **wickets_remaining** (both teams) - Critical tactical state
   - Low wickets + deficit = very unlikely to win
   - Determines if team can build/defend a lead

4. **overs_left** - Time pressure
   - Low overs + close score = draw likely
   - High overs + lead = time to force result

5. **first_team_is_home** - Moderate importance
   - Home advantage exists but less significant than ratings
   - Worth ~50-100 rating points

6. **first_team_won_toss** - Minor importance
   - Toss advantage is real but small
   - More important in specific conditions (not captured here)

### Typical Prediction
**Adelaide, England 213/8 trailing by 158:**
- Features: AUS lead +158, ENG 13 wickets remaining, 285 overs left
- **+ Team context**: AUS rating 1757, ENG rating 1593, AUS home
- **Result**: AUS 85.4%, Draw 4.7%, ENG 10.0%
- **Why reasonable**: Strong team with big lead at home → dominant

## Scorecard-Only Model (5 features)

### Features
1. **overs_left** (450 → 0): Total match overs remaining
2. **first_team_wickets_remaining** (20 → 0): First batting team's wickets left
3. **second_team_wickets_remaining** (20 → 0): Second batting team's wickets left
4. **first_team_lead** (+/- runs): First team's lead/deficit
5. **first_team_won_toss** (0/1): Whether first team won the toss

### What's Missing?
- **No team ratings**: Doesn't know Australia > England
- **No home advantage**: Doesn't know Australia playing at home
- Relies entirely on match state features

### Performance
- **Test Accuracy**: 58.8%
- **Accuracy gap**: -24.7 percentage points vs team-aware model

### Feature Importance (from XGBoost)
1. **first_team_lead** - Still most important
   - Same as team-aware: lead dominates

2. **wickets_remaining** (both teams) - More important than in full model
   - Model must rely more heavily on tactical state
   - Tries to infer team quality from wickets taken

3. **overs_left** - Slightly more important
   - Without team context, time pressure matters more

4. **first_team_won_toss** - Slightly more important
   - Model tries to extract any signal it can find
   - Toss winner might correlate with better team

### Typical Prediction
**Same Adelaide state (England 213/8 trailing by 158):**
- Features: AUS lead +158, ENG 13 wickets remaining, 285 overs left
- **No team context**: Doesn't know AUS > ENG, doesn't know AUS home
- **Result**: AUS 35.0%, Draw 41.4%, ENG 23.6%
- **Why unreasonable**: Sees ENG has MORE total wickets remaining (13 vs 10)!

### Why Scorecard-Only Performs Poorly

**Adelaide example breakdown:**
- Australia: 371/10 (innings 1), 0/0 (innings 3) = **10 wickets remaining**
- England: 213/7 (innings 2), 0/0 (innings 4) = **13 wickets remaining**

The model sees England has MORE wickets remaining! It doesn't understand that:
1. England's 13 wickets are split: **3 left NOW** + 10 later
2. England must survive with just 3 wickets before trailing further
3. Australia is simply the better team (1757 vs 1593 rating)

This is a **fundamental limitation** of cumulative wickets features:
- Can't distinguish "3 wickets now + 10 later" vs "13 wickets now"
- Innings-specific features would help but add complexity
- Without team quality, model doesn't know Australia will likely bat well again

## When Each Model Is Useful

### Use Team-Aware Model When:
- You want the most accurate predictions (default)
- You're analyzing specific teams (Ashes, India vs Aus, etc.)
- You have team ratings available
- You care about home advantage

### Use Scorecard-Only Model When:
- You want to see "what does the match state alone suggest?"
- Comparing how much team strength matters
- Educational: understanding cricket tactics vs team quality
- You don't have/trust team ratings

## Key Insight: Team Quality Matters

The **24.7% accuracy difference** shows that knowing the teams matters enormously:
- Same match state, different teams = different likely outcomes
- Strong teams win from positions where weak teams draw
- Weak teams lose from positions where strong teams draw

**Example**: Leading by 150 with 10 wickets left and 200 overs remaining:
- If you're Australia (1757 rating): ~80% win chance
- If you're Bangladesh (1400 rating): ~60% win chance
- Same scorecard, different team quality = 20% swing

## Training Details

Both models trained on:
- **Historical Test matches** from Cricsheet data
- **Exponential time decay**: Recent matches weighted higher (10-year half-life)
- **Exclude outliers**: Ties removed (very rare, ~1%)
- **Stratified splits**: Balanced win/draw/loss across train/val/test
- **XGBoost hyperparameters**:
  - max_depth: 6
  - learning_rate: 0.1
  - n_estimators: 100
  - objective: multi:softprob

## Files

- `train.py` - Trains team-aware model (8 features)
- `train_scorecard_only.py` - Trains scorecard-only model (5 features)
- `output/model.pkl` - Team-aware model artifact
- `output/model_scorecard_only.pkl` - Scorecard-only model artifact
- `generate_ashes_series.py` - Generates predictions from both models

## References

For feature definitions and calculation logic, see:
- `parse_cricsheet.py` - GameState dataclass and feature extraction
- `generate_ashes_series.py` - predict_probabilities() function
