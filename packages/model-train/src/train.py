#!/usr/bin/env python3
"""
Train multinomial logistic regression model for Test match W/D/L prediction
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_cricsheet_data(data_dir: Path) -> pd.DataFrame:
    """
    Load and parse Cricsheet YAML files

    Returns DataFrame with columns:
    - matchId, innings, over, runsFor, wicketsDown, lead, etc.
    """
    # TODO: Implement Cricsheet YAML parsing
    # For now, return empty DataFrame
    print("⚠️  Cricsheet data loading not yet implemented")
    return pd.DataFrame()


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from game state

    Features:
    - innings (categorical, one-hot)
    - wicketsDown
    - runRate
    - lead
    - ballsRemaining
    - runsPerWicket
    - isChasing
    - requiredRunRate
    """
    # TODO: Implement feature extraction
    print("⚠️  Feature extraction not yet implemented")
    return pd.DataFrame()


def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train multinomial logistic regression model

    Returns:
    - Trained model
    - Feature scaler
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Train accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")

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
    df = load_cricsheet_data(data_dir)

    if df.empty:
        print("\n❌ No data loaded. Please download Cricsheet Test data to data/ directory")
        print("   Visit: https://cricsheet.org/downloads/")
        return

    # Extract features
    print("\n2. Extracting features...")
    features_df = extract_features(df)

    # Train model
    print("\n3. Training model...")
    # X, y, feature_names = prepare_training_data(features_df)
    # model, scaler = train_model(X, y)

    # Export
    # print("\n4. Exporting model...")
    # export_model(model, scaler, feature_names, output_dir / "model.json")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
