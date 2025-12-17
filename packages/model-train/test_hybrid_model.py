#!/usr/bin/env python3
"""
Test hybrid model: XGBoost + Monte Carlo with transition boundary
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


class HybridPredictor:
    """Hybrid model that switches from XGBoost to Monte Carlo at transition boundary"""

    def __init__(self, xgb_model, mc_predictor, label_encoder):
        self.xgb_model = xgb_model
        self.mc_predictor = mc_predictor
        self.label_encoder = label_encoder

    def predict_proba(self, features):
        """
        Predict probabilities, switching to Monte Carlo at boundary

        Transition to Monte Carlo when in 4th innings chase AND:
        - (Chasing team has ≤ 3 wickets remaining OR ≤ 50 runs needed)
        - AND overs_left > 15 (enough time, no draw pressure)

        Monte Carlo doesn't model "batting for the draw" so use XGBoost for time pressure.
        """
        overs_left = features[0]
        first_wickets = int(features[1])
        second_wickets = int(features[2])
        lead = int(features[3])
        chase_ease = features[4]
        required_run_rate = features[5]

        # Check if we're in a 4th innings chase
        in_chase = chase_ease > 0

        if in_chase:
            # Calculate runs needed (from chasing team's perspective)
            runs_needed = abs(lead)  # If first team has positive lead, second team needs that many

            # Determine chasing team's wickets
            # If lead > 0, second team is chasing
            chasing_wickets = second_wickets if lead > 0 else first_wickets

            # Check transition conditions: wicket/run pressure AND no time pressure
            use_mc = (chasing_wickets <= 3 or runs_needed <= 50) and overs_left > 30
        else:
            use_mc = False

        if use_mc:
            # Use Monte Carlo
            p_win, p_draw, p_loss = self.mc_predictor.simulate_match(
                overs_left, first_wickets, second_wickets, lead,
                n_simulations=200
            )
            return np.array([p_draw, p_loss, p_win])
        else:
            # Use XGBoost
            return self.xgb_model.predict_proba([features])[0]

    def predict(self, X):
        """Batch prediction"""
        predictions = []
        for features in X:
            probs = self.predict_proba(features)
            predictions.append(np.argmax(probs))
        return np.array(predictions)


def main():
    print("Hybrid Model Testing")
    print("=" * 60)

    # Paths
    data_dir = Path("data")
    output_dir = Path("output")

    # Load data
    print("\n1. Loading test data...")
    states = load_all_matches(data_dir)
    df = extract_features(states)
    df = df[df['outcome'] != 'tie'].copy()

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

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )

    print(f"Test set: {len(X_test)} samples")

    # Load models
    print("\n2. Loading models...")
    with open(output_dir / "model.pkl", 'rb') as f:
        xgb_data = pickle.load(f)
        xgb_model = xgb_data['model']
        xgb_label_encoder = xgb_data['label_encoder']

    mc_predictor = MonteCarloPredictor()
    mc_predictor.load(output_dir / "monte_carlo_model.pkl")

    # Create hybrid model
    print("\n3. Creating hybrid model...")
    print("Transition boundary (4th innings chase):")
    print("  - (Chasing team ≤ 3 wickets OR ≤ 50 runs)")
    print("  - AND overs_left > 30")
    print("  - (Monte Carlo for wicket/run pressure, NOT time pressure)")

    hybrid = HybridPredictor(xgb_model, mc_predictor, xgb_label_encoder)

    # Evaluate baseline models
    print("\n4. Evaluating baseline models...")
    xgb_predictions = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    print(f"XGBoost Accuracy:     {xgb_accuracy:.3f}")

    # Evaluate hybrid
    print("\n5. Evaluating hybrid model...")
    hybrid_predictions = []
    mc_count = 0  # Count how many predictions used Monte Carlo

    for i, features in enumerate(X_test):
        if i % 5000 == 0:
            print(f"  Predicting {i}/{len(X_test)}...")

        # Check if this will use MC (for counting)
        overs_left = features[0]
        first_wickets = int(features[1])
        second_wickets = int(features[2])
        lead = int(features[3])
        chase_ease = features[4]

        in_chase = chase_ease > 0
        if in_chase:
            runs_needed = abs(lead)
            chasing_wickets = second_wickets if lead > 0 else first_wickets
            use_mc = (chasing_wickets <= 3 or runs_needed <= 50) and overs_left > 30
            if use_mc:
                mc_count += 1

        probs = hybrid.predict_proba(features)
        hybrid_predictions.append(np.argmax(probs))

    hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)
    print(f"Hybrid Accuracy:      {hybrid_accuracy:.3f}")
    print(f"Monte Carlo used for: {mc_count}/{len(X_test)} predictions ({mc_count/len(X_test)*100:.1f}%)")

    # Analyze chase scenarios specifically
    print("\n6. Performance on 4th innings chases...")
    chase_mask = X_test[:, 4] > 0
    X_chase = X_test[chase_mask]
    y_chase = y_test[chase_mask]

    if len(X_chase) > 0:
        xgb_chase_pred = xgb_model.predict(X_chase)
        xgb_chase_acc = accuracy_score(y_chase, xgb_chase_pred)

        hybrid_chase_pred = []
        mc_chase_count = 0
        for features in X_chase:
            probs = hybrid.predict_proba(features)
            hybrid_chase_pred.append(np.argmax(probs))

            # Count MC usage
            overs_left = features[0]
            lead = int(features[3])
            runs_needed = abs(lead)
            chasing_wickets = int(features[2]) if lead > 0 else int(features[1])
            if (chasing_wickets <= 3 or runs_needed <= 50) and overs_left > 30:
                mc_chase_count += 1

        hybrid_chase_acc = accuracy_score(y_chase, hybrid_chase_pred)

        print(f"Chase scenarios: {len(X_chase)} samples")
        print(f"XGBoost Accuracy:  {xgb_chase_acc:.3f}")
        print(f"Hybrid Accuracy:   {hybrid_chase_acc:.3f}")
        print(f"Monte Carlo used:  {mc_chase_count}/{len(X_chase)} ({mc_chase_count/len(X_chase)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  XGBoost Only:  {xgb_accuracy:.3f}")
    print(f"  Hybrid Model:  {hybrid_accuracy:.3f}")
    print(f"  Improvement:   {(hybrid_accuracy - xgb_accuracy):+.3f}")


if __name__ == "__main__":
    main()
