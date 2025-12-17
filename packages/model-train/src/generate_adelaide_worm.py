#!/usr/bin/env python3
"""
Generate worm chart data for Adelaide Ashes Test
Based on known data: Australia 326/8 in 90 overs
"""

import json
from pathlib import Path
import numpy as np

# Load model
model_path = Path(__file__).parent.parent / "output" / "model.json"
with open(model_path) as f:
    model_data = json.load(f)

means = np.array(model_data['featureMeans'])
stds = np.array(model_data['featureStds'])
coefficients = np.array(model_data['coefficients'])
intercepts = np.array(model_data['intercepts'])


def predict_probabilities(state):
    """Predict win/draw/loss probabilities"""
    run_rate = (state['runsFor'] / state['ballsBowled']) * 6 if state['ballsBowled'] > 0 else 0
    runs_per_wicket = state['runsFor'] / (state['wicketsDown'] + 1)
    required_run_rate = 0

    features = np.array([[
        state['innings'],
        state['wicketsDown'],
        run_rate,
        state['lead'],
        state['ballsRemaining'],
        runs_per_wicket,
        1 if state['isChasing'] else 0,
        required_run_rate
    ]])

    # Standardize
    features_scaled = (features - means) / stds

    # Compute logits
    logits = np.dot(features_scaled, coefficients.T) + intercepts

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    return {
        'pWin': float(probs[0][0]),
        'pDraw': float(probs[0][1]),
        'pLoss': float(probs[0][2])
    }


# Create realistic progression for Adelaide Test
# Based on: 326/8 in 90 overs, Khawaja 82, Carey 106, 94/4 after lunch

wicket_falls = [
    (0, 3, 0),      # Early wicket
    (20, 45, 1),    # Another early
    (35, 75, 2),    # Third wicket
    (38, 94, 3),    # Before lunch around 38 overs
    (40, 94, 4),    # 94/4 after lunch (known)
    (60, 180, 5),   # Recovery begins
    (70, 240, 6),   # Khawaja goes for 82
    (85, 300, 7),   # Another wicket
    (88, 320, 8),   # Eighth wicket, Carey on 106
]

states = []

# Generate states every 5 overs
for over in range(0, 91, 5):
    # Find wickets and runs at this point
    wickets = 0
    runs = 0

    for wkt_over, wkt_score, wkt_num in wicket_falls:
        if wkt_over <= over:
            wickets = wkt_num
            runs = wkt_score

    # Interpolate runs between wickets
    if over == 0:
        runs = 0
        wickets = 0
    elif over == 90:
        runs = 326
        wickets = 8
    else:
        # Find surrounding wickets
        prev_wkt = None
        next_wkt = None
        for i, (wkt_over, wkt_score, wkt_num) in enumerate(wicket_falls):
            if wkt_over <= over:
                prev_wkt = (wkt_over, wkt_score, wkt_num)
            if wkt_over > over and next_wkt is None:
                next_wkt = (wkt_over, wkt_score, wkt_num)
                break

        if prev_wkt and next_wkt:
            # Linear interpolation
            over_diff = next_wkt[0] - prev_wkt[0]
            score_diff = next_wkt[1] - prev_wkt[1]
            progress = (over - prev_wkt[0]) / over_diff if over_diff > 0 else 0
            runs = int(prev_wkt[1] + score_diff * progress)
            wickets = prev_wkt[2]
        elif prev_wkt:
            # Extrapolate from last wicket
            runs = int(prev_wkt[1] + (326 - prev_wkt[1]) * ((over - prev_wkt[0]) / (90 - prev_wkt[0])))
            wickets = prev_wkt[2]

    state = {
        'matchId': 'adelaide-ashes-2025',
        'innings': 1,
        'over': over,
        'runsFor': runs,
        'wicketsDown': wickets,
        'ballsBowled': over * 6,
        'lead': 0,
        'matchOversLimit': 450,
        'ballsRemaining': 450 * 6 - over * 6,
        'completedInnings': 0,
        'isChasing': False
    }

    states.append(state)

# Generate probability points
prob_points = []
for state in states:
    probs = predict_probabilities(state)
    prob_points.append({
        'xOver': state['over'],
        'innings': state['innings'],
        'over': state['over'],
        'score': f"{state['runsFor']}/{state['wicketsDown']}",
        **probs
    })

# Save to UI data directory
output = {
    'matchId': 'adelaide-ashes-2025',
    'description': 'Adelaide Ashes Test 2025 - Day 1',
    'states': states,
    'probabilities': prob_points
}

output_path = Path(__file__).parent.parent.parent / "ui" / "src" / "data" / "adelaide-test.json"
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Generated Adelaide Test worm data: {len(prob_points)} points")
print(f"  Saved to: {output_path}")

# Print some probabilities
print(f"\nKey moments:")
for i, point in enumerate(prob_points):
    if i % 4 == 0 or i == len(prob_points) - 1:
        print(f"  Over {point['over']:2d}: {point['score']:>7} | Win: {point['pWin']:5.1%} | Draw: {point['pDraw']:5.1%} | Loss: {point['pLoss']:5.1%}")
