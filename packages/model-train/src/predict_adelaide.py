#!/usr/bin/env python3
"""
Predict Win/Draw/Loss probabilities for Adelaide Ashes Test
"""

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Adelaide Ashes Test - Day 1 stumps
# Australia 326-8 in 90 overs (innings 1)
match_state = {
    'innings': 1,
    'wicketsDown': 8,
    'ballsBowled': 90 * 6,  # 540 balls
    'runsFor': 326,
    'lead': 0,  # First innings, no lead yet
    'target': None,
    'matchOversLimit': 450,  # Typical Test match estimate
    'ballsRemaining': 450 * 6 - 540,  # Rough estimate
    'isChasing': False
}

# Calculate derived features
run_rate = (match_state['runsFor'] / match_state['ballsBowled']) * 6 if match_state['ballsBowled'] > 0 else 0
runs_per_wicket = match_state['runsFor'] / (match_state['wicketsDown'] + 1)
required_run_rate = 0  # Not chasing

features = [
    match_state['innings'],
    match_state['wicketsDown'],
    run_rate,
    match_state['lead'],
    match_state['ballsRemaining'],
    runs_per_wicket,
    1 if match_state['isChasing'] else 0,
    required_run_rate
]

print("Adelaide Ashes Test - Day 1 Stumps")
print("=" * 60)
print(f"Australia: {match_state['runsFor']}/{match_state['wicketsDown']}")
print(f"Overs: {match_state['ballsBowled'] / 6:.1f}")
print(f"Run rate: {run_rate:.2f}")
print(f"\nFeatures:")
print(f"  innings: {match_state['innings']}")
print(f"  wicketsDown: {match_state['wicketsDown']}")
print(f"  runRate: {run_rate:.2f}")
print(f"  lead: {match_state['lead']}")
print(f"  ballsRemaining: {match_state['ballsRemaining']}")
print(f"  runsPerWicket: {runs_per_wicket:.2f}")
print(f"  isChasing: {0}")
print(f"  requiredRunRate: {required_run_rate:.2f}")

# Load model
model_path = Path(__file__).parent.parent / "output" / "model.json"
with open(model_path) as f:
    model_data = json.load(f)

# Standardize features
features_array = np.array([features])
means = np.array(model_data['featureMeans'])
stds = np.array(model_data['featureStds'])
features_scaled = (features_array - means) / stds

# Compute logits
coefficients = np.array(model_data['coefficients'])
intercepts = np.array(model_data['intercepts'])
logits = np.dot(features_scaled, coefficients.T) + intercepts

# Softmax
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits)

print(f"\n" + "=" * 60)
print("PREDICTIONS (from Australia's perspective)")
print("=" * 60)
print(f"Win:  {probs[0][0]:.1%}")
print(f"Draw: {probs[0][1]:.1%}")
print(f"Loss: {probs[0][2]:.1%}")

print(f"\n💡 Interpretation:")
if probs[0][0] > 0.5:
    print(f"   Australia is favored to win")
elif probs[0][2] > 0.5:
    print(f"   England is favored (Australia likely to lose)")
elif probs[0][1] > 0.5:
    print(f"   Match likely to end in a draw")
else:
    print(f"   Match is finely balanced")

print(f"\n📊 Context:")
print(f"   - Australia batted first and reached 326-8")
print(f"   - Good recovery from 94-4 after lunch")
print(f"   - Alex Carey's century (106) crucial")
print(f"   - Still 2 wickets remaining")
print(f"   - Tail can add more runs on Day 2")
