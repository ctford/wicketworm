# How to Update Ashes 2025-26 Data

When new match data becomes available (e.g., Adelaide Day 3), follow these steps:

## Step 1: Update the generate function

Edit `src/generate_ashes_series.py` and update the `generate_adelaide_test()` function:

```python
def generate_adelaide_test():
    """
    Adelaide Test (Dec 17-21, 2025) - In progress (Day 3)  # <- Update status
    Australia: 371 all out
    England: 253 all out  # <- Update score
    Australia: 45/0  # <- Add new innings
    """
    # ... update the innings data and scoring patterns ...
```

### Key things to update:

1. **Docstring**: Update day number and current score
2. **Result/Days**: At bottom of function, update:
   ```python
   'result': 'In progress (Day 3)',  # <- Change day
   'days': 3,  # <- Change count
   ```
3. **Innings data**: Add new innings loops following the 5-over bucket pattern

### Example: Adding England's complete 1st innings

If England finished at 253 all out in 85 overs:

```python
# Innings 2: England 253 all out (~85 overs)
for over in range(0, 86, 5):  # Note: 0, 5, 10, ... 85
    if over == 0:
        runs, wickets = 0, 0
    elif over <= 10:
        # Early collapse: 42/3 at over 10
        runs = int(over * 4.2)
        wickets = min(3, over // 4)
    # ... continue pattern matching actual match progression ...
    elif over >= 85:
        runs, wickets = 253, 10  # Final score
```

### Example: Adding Australia's 2nd innings

```python
# Innings 3: Australia 2nd innings (in progress: 45/0)
aus_inn1_overs = 95
eng_inn1_overs = 85
for over in range(0, 16, 5):  # Currently at 15 overs
    if over == 0:
        runs, wickets = 0, 0
    else:
        # Calculate based on actual scoring pattern
        runs = int(over * 3.0)  # Adjust to match actual run rate
        wickets = 0  # No wickets yet
    
    lead = (371 + runs) - 253  # Total lead
    states.append({
        'matchId': 'adelaide-test-2025',
        'innings': 3,
        'over': over,
        'runsFor': runs,
        'wicketsDown': wickets,
        'ballsBowled': over * 6,
        'lead': lead,
        'matchOversLimit': 450,
        'ballsRemaining': 450 * 6 - (aus_inn1_overs + eng_inn1_overs + over) * 6,
        'completedInnings': 2,  # <- Two innings complete
        'isChasing': False
    })
```

## Step 2: Regenerate predictions

```bash
cd packages/model-train
source venv/bin/activate
python src/generate_ashes_series.py
```

This will:
- Generate new data using your updated function
- Run predictions through the trained model (no retraining needed!)
- Save updated probabilities to `packages/ui/src/data/ashes-series-2025.json`

## Step 3: Verify in the UI

```bash
cd packages/ui
pnpm dev
# Open http://localhost:5173/wicketworm/
```

Check that:
- Adelaide now shows the correct day count
- The worm chart has smooth curves (no gaps or jumps)
- Probabilities look reasonable for the match state

## Step 4: Commit

```bash
git add packages/model-train/src/generate_ashes_series.py
git add packages/ui/src/data/ashes-series-2025.json
git commit -m "Update Adelaide Test with Day 3 data"
```

## Important Rules

### ✅ DO:
- Use **consistent 5-over intervals** (0, 5, 10, 15, ...)
- Update wicket counts at each bucket to reflect current state
- Match actual run totals at key milestones (50, 100, final score)
- Update `result` and `days` fields
- Follow the same pattern as Perth and Brisbane

### ❌ DON'T:
- Create irregular intervals at wicket falls (8, 9, 24, etc.)
- Retrain the model (it's already trained on historical data)
- Edit `adelaide-test.json` directly (this file doesn't exist anymore!)
- Forget to update the docstring and result fields

## Why This Approach?

1. **Consistent with Perth/Brisbane**: All three tests use the same generation pattern
2. **Easy updates**: Just modify the function parameters and regenerate
3. **Version controlled**: Changes are tracked in the Python file
4. **No retraining**: Model stays fixed, only new match states are predicted

## Questions?

See `CLAUDE.md` for more details on the model and prediction system.
