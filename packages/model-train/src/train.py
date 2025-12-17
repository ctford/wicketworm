#!/usr/bin/env python3
"""
Train XGBoost model for Test match W/D/L prediction
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from parse_cricsheet import load_all_matches, GameState


def extract_features(states: List[GameState]) -> pd.DataFrame:
    """
    Extract 4 features representing full match state

    Features:
    - overs_left: Total match overs remaining (450 → 0)
    - first_team_wickets_remaining: Wickets first team has left (20 → 0)
    - second_team_wickets_remaining: Wickets second team has left (20 → 0)
    - first_team_lead: First team's lead (positive) or deficit (negative)
    """
    records = []

    for state in states:
        records.append({
            'overs_left': state.overs_left,
            'first_team_wickets_remaining': state.first_team_wickets_remaining,
            'second_team_wickets_remaining': state.second_team_wickets_remaining,
            'first_team_lead': state.first_team_lead,
            'match_id': state.match_id,
            'outcome': state.outcome
        })

    return pd.DataFrame(records)


def extract_year_from_match_id(match_id: str) -> int:
    """Extract year from Cricsheet match ID (format: NNNNNNNN where first 4 digits are year)"""
    # Match IDs are typically like "20230201" (YYYYMMDD format)
    match = re.match(r'^(\d{4})', match_id)
    if match:
        return int(match.group(1))
    return 2000  # Default fallback


def calculate_sample_weights(match_ids: pd.Series, decay_years: float = 10.0) -> np.ndarray:
    """
    Calculate exponential decay weights based on match year

    Args:
        match_ids: Series of match IDs
        decay_years: Half-life for exponential decay (default 10 years)

    Returns:
        Array of sample weights (recent matches weighted higher)
    """
    current_year = 2025
    weights = []

    for match_id in match_ids:
        match_year = extract_year_from_match_id(match_id)
        years_ago = max(0, current_year - match_year)

        # Exponential decay: weight = exp(-years_ago / decay_years)
        # Cap years_ago at 50 to avoid numerical overflow
        years_ago = min(years_ago, 50)

        # Match from 2025: weight = 1.0
        # Match from 2015: weight ≈ 0.37
        # Match from 2005: weight ≈ 0.14
        # Match from 1975: weight ≈ 0.007
        weight = np.exp(-years_ago / decay_years)
        weights.append(weight)

    return np.array(weights)


def train_model(X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray) -> Tuple[xgb.XGBClassifier, LabelEncoder]:
    """
    Train XGBoost classifier with train/val/test split and sample weighting

    Returns:
    - Trained XGBoost model
    - Label encoder
    """
    # Encode outcomes: 0=draw, 1=loss, 2=win (alphabetical)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # Split: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
        X, y_encoded, sample_weights, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 of 85% ≈ 15%
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train XGBoost model
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(label_encoder.classes_),
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        eval_metric='mlogloss'
    )

    # Fit with sample weights
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight_eval_set=[w_train, w_val],
        verbose=False
    )

    # Evaluate on all three sets
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    test_score = model.score(X_test, y_test)

    print(f"\nModel performance:")
    print(f"  Train accuracy: {train_score:.3f}")
    print(f"  Validation accuracy: {val_score:.3f}")
    print(f"  Test accuracy: {test_score:.3f}")

    # Class distribution
    print(f"\nOutcome distribution in test set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for outcome_idx, count in zip(unique, counts):
        outcome_name = label_encoder.inverse_transform([outcome_idx])[0]
        print(f"  {outcome_name}: {count} ({count/len(y_test)*100:.1f}%)")

    return model, label_encoder


def export_model(
    model: xgb.XGBClassifier,
    label_encoder: LabelEncoder,
    feature_names: List[str],
    output_dir: Path
) -> None:
    """
    Export XGBoost model for server-side inference
    """
    # Save model and label encoder using pickle
    model_path = output_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
    print(f"✓ Model and label encoder saved to {model_path}")

    # Save feature names as metadata
    metadata = {
        "feature_names": feature_names,
        "label_classes": label_encoder.classes_.tolist(),
        "model_type": "xgboost"
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Model metadata saved to {metadata_path}")


def main():
    """Main training pipeline"""
    print("WicketWorm Model Training (XGBoost)")
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
    print(f"Extracted features for {len(df)} game states")

    # Filter out ties (user requested to exclude ties, not count as draws)
    print("\n3. Filtering out ties...")
    before_count = len(df)
    df = df[df['outcome'] != 'tie'].copy()
    after_count = len(df)
    print(f"Removed {before_count - after_count} tie states")
    print(f"Remaining: {after_count} states")

    # Calculate sample weights (based on recency)
    print("\n4. Calculating sample weights (recent matches weighted higher)...")
    sample_weights = calculate_sample_weights(df['match_id'])

    print(f"Weight range: {sample_weights.min():.3f} to {sample_weights.max():.3f}")

    # Prepare training data
    print("\n5. Preparing training data...")
    feature_cols = [
        'overs_left',
        'first_team_wickets_remaining',
        'second_team_wickets_remaining',
        'first_team_lead'
    ]

    X = df[feature_cols].values
    y = df['outcome'].values

    print(f"Total states: {len(X)}")
    print(f"\nOutcome distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for outcome, count in zip(unique, counts):
        print(f"  {outcome}: {count} ({count/len(y)*100:.1f}%)")

    # Train model
    print("\n6. Training XGBoost model...")
    model, label_encoder = train_model(X, y, sample_weights)

    # Export
    print("\n7. Exporting model...")
    export_model(model, label_encoder, feature_cols, output_dir)

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
