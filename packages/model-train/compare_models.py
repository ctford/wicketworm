#!/usr/bin/env python3
"""
Compare XGBoost model vs simple wicket-value heuristic on innings 4 chases
"""

import pickle
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from parse_cricsheet import load_all_matches

# Load XGBoost model
model_path = Path('output/model.pkl')
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

xgb_model = model_data['model']
label_encoder = model_data['label_encoder']

print("=" * 70)
print("MODEL COMPARISON: XGBoost vs Simple Wicket-Value Heuristic")
print("=" * 70)

# Load all innings 4 chase states
print("\n1. Loading innings 4 chase data...")
data_dir = Path('data')
all_states = load_all_matches(data_dir)

# Filter to innings 4 chases only
chase_states = [s for s in all_states if s.current_innings == 4 and s.runs_to_win > 0]
print(f"   Found {len(chase_states)} innings 4 chase states")


def simple_heuristic_model(runs_to_win, wickets_left, n_simulations=10000):
    """
    Simple model: each wicket worth random runs from distribution
    - Median: 30 runs
    - Min: 0 runs
    - Max: 200 runs

    Use gamma distribution to approximate this:
    - Shape k and scale θ such that median ≈ 30
    - Truncate at 200

    Returns: probability of winning (scoring enough runs)
    """
    if wickets_left <= 0:
        return 0.0

    if runs_to_win <= 0:
        return 1.0

    # Gamma distribution parameters for median ≈ 30
    # For gamma: median ≈ k*θ*(1-1/(3k)) for k>1
    # Use k=2, θ=17 gives median ≈ 30
    shape = 2.0
    scale = 17.0

    # Monte Carlo simulation
    wins = 0
    for _ in range(n_simulations):
        # Sample runs for each wicket
        runs_per_wicket = np.random.gamma(shape, scale, wickets_left)
        # Truncate at 200
        runs_per_wicket = np.minimum(runs_per_wicket, 200)
        total_runs = np.sum(runs_per_wicket)

        if total_runs >= runs_to_win:
            wins += 1

    return wins / n_simulations


# Compare models on a sample of chase states
print("\n2. Comparing models on chase states...")
print()
print("Sample predictions (runs_to_win, wickets_left):")
print("-" * 70)
print(f"{'Runs':>5} {'Wkts':>4} {'Actual':>8} | {'XGBoost':>8} {'Heuristic':>10} | {'XGB Correct':>12}")
print("-" * 70)

# Group states by outcome for balanced comparison
chase_by_outcome = {
    'win': [s for s in chase_states if s.outcome == 'loss'],  # second_team win = first_team loss
    'loss': [s for s in chase_states if s.outcome == 'win'],  # second_team loss = first_team win
}

# Sample 10 wins and 10 losses
sample_states = []
np.random.seed(42)
for outcome in ['win', 'loss']:
    states = chase_by_outcome[outcome]
    if len(states) > 10:
        sample_states.extend(np.random.choice(states, 10, replace=False))
    else:
        sample_states.extend(states)

xgb_correct = 0
heur_correct = 0

for state in sample_states:
    runs_to_win = state.runs_to_win
    wickets_left = 10 - state.second_team_wickets_inn4
    actual_won = (state.outcome == 'loss')  # first_team loss = second_team win

    # XGBoost prediction (from first_team perspective)
    features = np.array([[
        state.overs_left,
        state.current_innings,
        state.first_team_score_inn1,
        state.first_team_wickets_inn1,
        state.second_team_score_inn2,
        state.second_team_wickets_inn2,
        state.first_team_score_inn3,
        state.first_team_wickets_inn3,
        state.second_team_score_inn4,
        state.second_team_wickets_inn4,
        state.current_lead,
        state.runs_to_win
    ]])

    xgb_probs = xgb_model.predict_proba(features)[0]
    # Label mapping: {'draw': 0, 'loss': 1, 'win': 2}
    # p_loss (first_team) = p_win (second_team chasing)
    xgb_p_win_chase = float(xgb_probs[1])  # first_team loss = chase success

    # Heuristic prediction
    heur_p_win = simple_heuristic_model(runs_to_win, wickets_left, n_simulations=5000)

    # Check correctness (use 50% threshold)
    xgb_pred = xgb_p_win_chase > 0.5
    heur_pred = heur_p_win > 0.5

    xgb_correct += (xgb_pred == actual_won)
    heur_correct += (heur_pred == actual_won)

    actual_str = "WIN" if actual_won else "LOSS"
    xgb_correct_str = "✓" if (xgb_pred == actual_won) else "✗"

    print(f"{runs_to_win:5d} {wickets_left:4d} {actual_str:>8} | {xgb_p_win_chase:7.1%} {heur_p_win:9.1%} | {xgb_correct_str:>12}")

print("-" * 70)
print(f"\nAccuracy on sample (n={len(sample_states)}):")
print(f"  XGBoost:    {xgb_correct}/{len(sample_states)} = {xgb_correct/len(sample_states):.1%}")
print(f"  Heuristic:  {heur_correct}/{len(sample_states)} = {heur_correct/len(sample_states):.1%}")

# Full comparison on all chase states (using smaller sample for speed)
print("\n3. Full comparison on larger sample...")
np.random.seed(42)
sample_size = min(1000, len(chase_states))
full_sample = np.random.choice(chase_states, sample_size, replace=False)

xgb_total_correct = 0
heur_total_correct = 0

for i, state in enumerate(full_sample):
    if i % 100 == 0:
        print(f"   Processing {i}/{sample_size}...")

    runs_to_win = state.runs_to_win
    wickets_left = 10 - state.second_team_wickets_inn4
    actual_won = (state.outcome == 'loss')

    # XGBoost
    features = np.array([[
        state.overs_left, state.current_innings,
        state.first_team_score_inn1, state.first_team_wickets_inn1,
        state.second_team_score_inn2, state.second_team_wickets_inn2,
        state.first_team_score_inn3, state.first_team_wickets_inn3,
        state.second_team_score_inn4, state.second_team_wickets_inn4,
        state.current_lead, state.runs_to_win
    ]])
    xgb_probs = xgb_model.predict_proba(features)[0]
    xgb_p_win = float(xgb_probs[1])

    # Heuristic
    heur_p_win = simple_heuristic_model(runs_to_win, wickets_left, n_simulations=1000)

    xgb_total_correct += ((xgb_p_win > 0.5) == actual_won)
    heur_total_correct += ((heur_p_win > 0.5) == actual_won)

print()
print("=" * 70)
print(f"FINAL RESULTS (n={sample_size} chase states):")
print("=" * 70)
print(f"XGBoost Model:          {xgb_total_correct}/{sample_size} = {xgb_total_correct/sample_size:.1%} accuracy")
print(f"Simple Heuristic:       {heur_total_correct}/{sample_size} = {heur_total_correct/sample_size:.1%} accuracy")
print()

if xgb_total_correct > heur_total_correct:
    diff = xgb_total_correct - heur_total_correct
    print(f"✓ XGBoost wins by {diff} predictions ({diff/sample_size:.1%})")
else:
    diff = heur_total_correct - xgb_total_correct
    print(f"✗ Heuristic wins by {diff} predictions ({diff/sample_size:.1%})")

print("=" * 70)
