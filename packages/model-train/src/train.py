#!/usr/bin/env python3
"""
Train multinomial logistic regression model for Test match W/D/L prediction
"""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from parse_cricsheet import load_all_matches, GameState


def extract_features(states: List[GameState], max_overs: int = 450) -> pd.DataFrame:
    """
    Extract features from game states

    Features:
    - innings (1-4)
    - wicketsDown
    - runRate
    - lead
    - ballsRemaining (estimated)
    - runsPerWicket
    - isChasing (innings 3 or 4)
    - requiredRunRate (if chasing)
    """
    records = []

    for state in states:
        # Basic stats
        run_rate = (state.runs_for / state.balls_bowled) * 6 if state.balls_bowled > 0 else 0
        runs_per_wicket = state.runs_for / (state.wickets_down + 1)

        # Estimate balls remaining (rough estimate: 450 overs = 2700 balls for a Test)
        # Adjust based on innings progression
        max_balls = max_overs * 6
        balls_used = (state.innings - 1) * 540 + state.balls_bowled  # Rough: 90 overs per innings
        balls_remaining = max(0, max_balls - balls_used)

        # Chasing indicator
        is_chasing = state.innings >= 3
        required_run_rate = 0
        if is_chasing and state.lead < 0 and balls_remaining > 0:
            runs_needed = abs(state.lead) + 1
            required_run_rate = (runs_needed / balls_remaining) * 6

        records.append({
            'innings': state.innings,
            'wicketsDown': state.wickets_down,
            'runRate': run_rate,
            'lead': state.lead,
            'ballsRemaining': balls_remaining,
            'runsPerWicket': runs_per_wicket,
            'isChasing': 1 if is_chasing else 0,
            'requiredRunRate': required_run_rate,
            'outcome': state.outcome
        })

    return pd.DataFrame(records)


def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train multinomial logistic regression model with train/val/test split

    Returns:
    - Trained model
    - Feature scaler
    """
    # Split: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 of 85% ≈ 15%
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        C=1.0
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate on all three sets
    train_score = model.score(X_train_scaled, y_train)
    val_score = model.score(X_val_scaled, y_val)
    test_score = model.score(X_test_scaled, y_test)

    print(f"\nModel performance:")
    print(f"  Train accuracy: {train_score:.3f}")
    print(f"  Validation accuracy: {val_score:.3f}")
    print(f"  Test accuracy: {test_score:.3f}")

    # Class distribution
    print(f"\nOutcome distribution in test set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for outcome, count in zip(unique, counts):
        print(f"  {outcome}: {count} ({count/len(y_test)*100:.1f}%)")

    return model, scaler


def export_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    feature_names: List[str],
    output_path: Path
) -> None:
    """
    Export model to JSON for browser inference
    """
    model_data = {
        "coefficients": model.coef_.tolist(),
        "intercepts": model.intercept_.tolist(),
        "featureMeans": scaler.mean_.tolist(),
        "featureStds": scaler.scale_.tolist(),
        "featureNames": feature_names
    }

    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"✓ Model exported to {output_path}")


def main():
    """Main training pipeline"""
    print("WicketWorm Model Training")
    print("=" * 50)

    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\n1. Loading Cricsheet data...")
    states = load_all_matches(data_dir)

    if not states:
        print("\n❌ No data loaded. Please download Cricsheet Test data to data/ directory")
        print("   Visit: https://cricsheet.org/downloads/")
        return

    # Extract features
    print("\n2. Extracting features...")
    df = extract_features(states)
    df['match_id'] = [s.match_id for s in states]
    print(f"Extracted features for {len(df)} game states")

    # Split at MATCH level to avoid data leakage
    print("\n3. Splitting data by match (to avoid leakage)...")
    unique_matches = df['match_id'].unique()
    print(f"Total matches: {len(unique_matches)}")

    # Shuffle and split matches
    np.random.seed(42)
    shuffled_matches = np.random.permutation(unique_matches)

    n_train = int(0.70 * len(shuffled_matches))
    n_val = int(0.15 * len(shuffled_matches))

    train_matches = shuffled_matches[:n_train]
    val_matches = shuffled_matches[n_train:n_train+n_val]
    test_matches = shuffled_matches[n_train+n_val:]

    print(f"Train matches: {len(train_matches)}")
    print(f"Validation matches: {len(val_matches)}")
    print(f"Test matches: {len(test_matches)}")

    # Split dataframe by matches
    train_df = df[df['match_id'].isin(train_matches)]
    val_df = df[df['match_id'].isin(val_matches)]
    test_df = df[df['match_id'].isin(test_matches)]

    # Prepare training data
    print("\n4. Preparing training data...")
    feature_cols = ['innings', 'wicketsDown', 'runRate', 'lead', 'ballsRemaining',
                    'runsPerWicket', 'isChasing', 'requiredRunRate']

    X_train = train_df[feature_cols].values
    y_train = train_df['outcome'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['outcome'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['outcome'].values

    print(f"Train states: {len(X_train)}")
    print(f"Validation states: {len(X_val)}")
    print(f"Test states: {len(X_test)}")

    print(f"\nOutcome distribution:")
    for name, y_set in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        unique, counts = np.unique(y_set, return_counts=True)
        dist = {outcome: f"{count} ({count/len(y_set)*100:.1f}%)" for outcome, count in zip(unique, counts)}
        print(f"  {name}: {dist}")

    # Train model
    print("\n5. Training model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        C=1.0
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    val_score = model.score(X_val_scaled, y_val)
    test_score = model.score(X_test_scaled, y_test)

    print(f"\nModel performance:")
    print(f"  Train accuracy: {train_score:.3f}")
    print(f"  Validation accuracy: {val_score:.3f}")
    print(f"  Test accuracy: {test_score:.3f}")

    # Export
    print("\n6. Exporting model...")
    export_model(model, scaler, feature_cols, output_dir / "model.json")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
