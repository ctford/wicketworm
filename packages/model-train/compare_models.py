#!/usr/bin/env python3
"""
Compare Monte Carlo and XGBoost models
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


def main():
    print("Model Comparison: Monte Carlo vs XGBoost")
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

    # Split: 70% train, 15% validation, 15% test (same as training)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )

    print(f"Test set: {len(X_test)} samples")

    # Load models
    print("\n2. Loading models...")

    # XGBoost
    with open(output_dir / "model.pkl", 'rb') as f:
        xgb_data = pickle.load(f)
        xgb_model = xgb_data['model']
        xgb_label_encoder = xgb_data['label_encoder']

    print("✓ Loaded XGBoost model")

    # Monte Carlo
    mc_predictor = MonteCarloPredictor()
    mc_predictor.load(output_dir / "monte_carlo_model.pkl")
    print("✓ Loaded Monte Carlo model")

    # Evaluate XGBoost
    print("\n3. Evaluating XGBoost...")
    xgb_predictions = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    print(f"XGBoost Test Accuracy: {xgb_accuracy:.3f}")

    # Evaluate Monte Carlo
    print("\n4. Evaluating Monte Carlo...")
    mc_predictions = []

    for i, features in enumerate(X_test):
        if i % 5000 == 0:
            print(f"  Predicting {i}/{len(X_test)}...")

        overs_left = features[0]
        first_wickets = int(features[1])
        second_wickets = int(features[2])
        lead = int(features[3])

        # Get probabilities from Monte Carlo
        p_win, p_draw, p_loss = mc_predictor.simulate_match(
            overs_left, first_wickets, second_wickets, lead,
            n_simulations=100  # Use fewer simulations for speed
        )

        # Predict class with highest probability
        probs = [p_draw, p_loss, p_win]  # Order: draw, loss, win (same as label encoding)
        predicted_class = np.argmax(probs)
        mc_predictions.append(predicted_class)

    mc_accuracy = accuracy_score(y_test, mc_predictions)
    print(f"Monte Carlo Test Accuracy: {mc_accuracy:.3f}")

    # Compare on terminal states
    print("\n5. Terminal State Comparison...")
    print("Testing: Both teams all out, first team +50 lead")

    # Test state: both all out, 50 run lead, 0 overs left
    test_features = np.array([[0, 0, 0, 50, 0, 0]])  # terminal state

    # XGBoost prediction
    xgb_probs = xgb_model.predict_proba(test_features)[0]
    xgb_pred_class = xgb_model.predict(test_features)[0]
    xgb_outcome = xgb_label_encoder.inverse_transform([xgb_pred_class])[0]

    print(f"\nXGBoost:")
    print(f"  Predicted: {xgb_outcome}")
    print(f"  Draw: {xgb_probs[0]*100:.1f}%, Loss: {xgb_probs[1]*100:.1f}%, Win: {xgb_probs[2]*100:.1f}%")

    # Monte Carlo prediction
    mc_p_win, mc_p_draw, mc_p_loss = mc_predictor.simulate_match(0, 0, 0, 50, n_simulations=1000)

    print(f"\nMonte Carlo:")
    print(f"  Predicted: {'win' if mc_p_win > 0.5 else 'draw' if mc_p_draw > 0.5 else 'loss'}")
    print(f"  Draw: {mc_p_draw*100:.1f}%, Loss: {mc_p_loss*100:.1f}%, Win: {mc_p_win*100:.1f}%")

    # Test chase scenario
    print("\n6. Chase Scenario Comparison...")
    print("Testing: Need 18 runs with 9 wickets, 20 overs left")

    # Chase state: 20 overs, 20 wickets, 9 wickets, -18 lead (chasing team needs 18)
    # From second team perspective: they have 9 wickets, need 18 runs
    # From first team perspective: lead = +18, second team has 9 wickets
    chase_ease = 1.0 / max((18 / 9), 0.5)  # = 1 / 2 = 0.5
    required_run_rate = 18 / 20  # = 0.9

    chase_features = np.array([[20, 10, 9, 18, chase_ease, required_run_rate]])

    # XGBoost prediction
    xgb_chase_probs = xgb_model.predict_proba(chase_features)[0]

    print(f"\nXGBoost (first team perspective, holding 18 run lead):")
    print(f"  Draw: {xgb_chase_probs[0]*100:.1f}%, Loss: {xgb_chase_probs[1]*100:.1f}%, Win: {xgb_chase_probs[2]*100:.1f}%")

    # Monte Carlo prediction
    mc_chase_p_win, mc_chase_p_draw, mc_chase_p_loss = mc_predictor.simulate_match(
        20, 10, 9, 18, n_simulations=1000
    )

    print(f"\nMonte Carlo (first team perspective, holding 18 run lead):")
    print(f"  Draw: {mc_chase_p_draw*100:.1f}%, Loss: {mc_chase_p_loss*100:.1f}%, Win: {mc_chase_p_win*100:.1f}%")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  XGBoost Accuracy:     {xgb_accuracy:.3f}")
    print(f"  Monte Carlo Accuracy: {mc_accuracy:.3f}")
    print(f"  Difference:           {(mc_accuracy - xgb_accuracy):.3f}")


if __name__ == "__main__":
    main()
