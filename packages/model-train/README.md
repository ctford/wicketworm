# model-train

Python package for training the WicketWorm probability model.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Data

Download Cricsheet Test match data:
1. Visit https://cricsheet.org/downloads/
2. Download Test match YAML files
3. Place in `data/` directory

## Usage

```bash
# Train model
python src/train.py

# Output will be in output/model.json
```

## Model features

- `innings` (1-4, one-hot encoded)
- `wicketsDown` (0-10)
- `runRate` (runs per over)
- `lead` (signed runs lead/deficit)
- `ballsRemaining` (in match)
- `runsPerWicket` (runs / (wickets + 1))
- `isChasing` (boolean)
- `requiredRunRate` (if chasing, else 0)

## Output format

```json
{
  "coefficients": [[...], [...], [...]],
  "intercepts": [...],
  "featureMeans": [...],
  "featureStds": [...],
  "featureNames": [...]
}
```
