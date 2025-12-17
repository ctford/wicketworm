#!/usr/bin/env python3
"""
Compare Monte Carlo and XGBoost on specific game situations
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from parse_cricsheet import load_all_matches
from train import extract_features
from monte_carlo_model import MonteCarloPredictor


def evaluate_subset(X, y, xgb_model, mc_predictor, subset_name):
    """Evaluate both models on a subset of data"""
    print(f"\n{subset_name}")
    print("-" * 60)
    print(f"Subset size: {len(X)} samples")

    if len(X) == 0:
        print("  (no samples)")
        return

    # XGBoost predictions
    xgb_predictions = xgb_model.predict(X)
    xgb_accuracy = accuracy_score(y, xgb_predictions)

    # Monte Carlo predictions
    mc_predictions = []
    for features in X:
        overs_left = features[0]
        first_wickets = int(features[1])
        second_wickets = int(features[2])
        lead = int(features[3])

        p_win, p_draw, p_loss = mc_predictor.simulate_match(
            overs_left, first_wickets, second_wickets, lead,
            n_simulations=100
        )

        probs = [p_draw, p_loss, p_win]
        predicted_class = np.argmax(probs)
        mc_predictions.append(predicted_class)

    mc_accuracy = accuracy_score(y, mc_predictions)

    print(f"XGBoost Accuracy:     {xgb_accuracy:.3f}")
    print(f"Monte Carlo Accuracy: {mc_accuracy:.3f}")
    print(f"Difference:           {(mc_accuracy - xgb_accuracy):+.3f}")


def main():
    print("Model Comparison by Game Situation")
    print("=" * 60)

    # Paths
    data_dir = Path("data")
    output_dir = Path("output")

    # Load data
    print("\n1. Loading test data...")
    states = load_all_matches(data_dir)
    df = extract_features(states)

    # Filter out ties
    df = df[df['outcome'] != 'tie'].copy()

    # Features and labels
    feature_cols = [
        'overs_left',
        'first_team_wickets_remaining',
        'second_team_wickets_remaining',
        'first_team_lead',
        'chase_ease',
        'required_run_rate'
    ]

    X = df[feature_cols].values
    y = df['outcome'].values

    # Split: 70% train, 15% validation, 15% test
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )

    print(f"Total test set: {len(X_test)} samples")

    # Load models
    print("\n2. Loading models...")
    with open(output_dir / "model.pkl", 'rb') as f:
        xgb_data = pickle.load(f)
        xgb_model = xgb_data['model']

    mc_predictor = MonteCarloPredictor()
    mc_predictor.load(output_dir / "monte_carlo_model.pkl")

    print("\n3. Analyzing specific game situations...")

    # Fourth innings chases (chase_ease > 0)
    chase_mask = X_test[:, 4] > 0  # chase_ease > 0
    evaluate_subset(
        X_test[chase_mask],
        y_test[chase_mask],
        xgb_model,
        mc_predictor,
        "Fourth Innings Chases (chase_ease > 0)"
    )

    # Late game (< 50 overs left)
    late_mask = X_test[:, 0] < 50
    evaluate_subset(
        X_test[late_mask],
        y_test[late_mask],
        xgb_model,
        mc_predictor,
        "Late Game (< 50 overs remaining)"
    )

    # Very late game (< 20 overs left)
    very_late_mask = X_test[:, 0] < 20
    evaluate_subset(
        X_test[very_late_mask],
        y_test[very_late_mask],
        xgb_model,
        mc_predictor,
        "Very Late Game (< 20 overs remaining)"
    )

    # Endgame (< 10 overs left)
    endgame_mask = X_test[:, 0] < 10
    evaluate_subset(
        X_test[endgame_mask],
        y_test[endgame_mask],
        xgb_model,
        mc_predictor,
        "Endgame (< 10 overs remaining)"
    )

    # Few wickets remaining (either team has ≤ 3 wickets)
    few_wickets_mask = (X_test[:, 1] <= 3) | (X_test[:, 2] <= 3)
    evaluate_subset(
        X_test[few_wickets_mask],
        y_test[few_wickets_mask],
        xgb_model,
        mc_predictor,
        "Few Wickets (either team ≤ 3 wickets left)"
    )

    # Terminal states (both teams all out)
    terminal_mask = (X_test[:, 1] == 0) & (X_test[:, 2] == 0)
    evaluate_subset(
        X_test[terminal_mask],
        y_test[terminal_mask],
        xgb_model,
        mc_predictor,
        "Terminal States (both teams all out)"
    )

    # One team all out
    one_out_mask = ((X_test[:, 1] == 0) | (X_test[:, 2] == 0)) & ~terminal_mask
    evaluate_subset(
        X_test[one_out_mask],
        y_test[one_out_mask],
        xgb_model,
        mc_predictor,
        "One Team All Out (but not both)"
    )

    # Early game (> 300 overs left)
    early_mask = X_test[:, 0] > 300
    evaluate_subset(
        X_test[early_mask],
        y_test[early_mask],
        xgb_model,
        mc_predictor,
        "Early Game (> 300 overs remaining)"
    )

    # Mid game (100-300 overs left)
    mid_mask = (X_test[:, 0] >= 100) & (X_test[:, 0] <= 300)
    evaluate_subset(
        X_test[mid_mask],
        y_test[mid_mask],
        xgb_model,
        mc_predictor,
        "Mid Game (100-300 overs remaining)"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
