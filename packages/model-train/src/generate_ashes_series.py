#!/usr/bin/env python3
"""
Generate worm chart data for all three Ashes tests
Perth, Brisbane, and Adelaide (in progress)
"""

import json
import pickle
from pathlib import Path
import numpy as np

# Import OVER_OFFSET from shared types
OVER_OFFSET = 500

# Load XGBoost model
model_path = Path(__file__).parent.parent / "output" / "model.pkl"
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
label_encoder = model_data['label_encoder']

print(f"Loaded XGBoost model with label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")


def predict_probabilities(state, overs_left, batting_team='Australia'):
    """
    Predict win/draw/loss probabilities from Australia's perspective using XGBoost

    Args:
        state: Game state dict with innings, runsFor, wicketsDown, lead
        overs_left: Total match overs remaining (450 - cumulative_overs_bowled)
        batting_team: Which team is batting ('Australia' or 'England')

    Model predicts from batting team's perspective, so we flip when England is batting.
    """
    # 5 features: overs_left, innings, runs, wickets, lead
    features = np.array([[
        overs_left,
        state['innings'],
        state['runsFor'],
        state['wicketsDown'],
        state['lead']
    ]])

    # XGBoost prediction
    probs = model.predict_proba(features)[0]  # Shape: (3,) for [draw, loss, win]

    # Label mapping: {'draw': 0, 'loss': 1, 'win': 2}
    p_draw = float(probs[0])
    p_loss = float(probs[1])
    p_win = float(probs[2])

    # Model predicts from batting team's perspective
    # If England is batting, flip win/loss to show Australia's perspective
    if batting_team == 'England':
        return {
            'pWin': p_loss,  # England's loss = Australia's win
            'pDraw': p_draw,
            'pLoss': p_win   # England's win = Australia's loss
        }
    else:
        return {
            'pWin': p_win,   # Australia's win
            'pDraw': p_draw,
            'pLoss': p_loss  # Australia's loss
        }


def add_batting_teams(states, first_batting='England'):
    """Add batting team to each state based on innings"""
    for state in states:
        innings = state['innings']
        if first_batting == 'England':
            # England: innings 1, 3; Australia: innings 2, 4
            state['battingTeam'] = 'England' if innings in [1, 3] else 'Australia'
        else:
            # Australia: innings 1, 3; England: innings 2, 4
            state['battingTeam'] = 'Australia' if innings in [1, 3] else 'England'
    return states


def generate_perth_test():
    """
    Perth Test (Nov 21-22, 2025) - Finished in 2 days
    Australia won by 8 wickets
    England: 172 & 164
    Australia: 132 & 205/2
    """
    states = []

    # Innings 1: England 172 all out (estimate ~65 overs)
    # Slower scoring, early collapse: 172/65 = 2.65 run rate
    for over in range(0, 66, 5):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 20:
            # Decent start then wickets
            runs = int(over * 3.0)
            wickets = min(3, over // 8)
        elif over <= 40:
            # Middle collapse, slow scoring
            runs = 60 + int((over - 20) * 2.2)
            wickets = min(7, 3 + (over - 20) // 7)
        else:
            # Tail wagging slightly
            runs = int(104 + (172 - 104) * (over - 40) / 25)
            wickets = min(10, 7 + (over - 40) // 5)

        states.append({
            'matchId': 'perth-test-2025',
            'innings': 1,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': runs,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - over * 6,
            'completedInnings': 0,
            'isChasing': False
        })

    # Innings 2: Australia 132 all out (~50 overs)
    aus_inn1_overs = 50
    for i, over in enumerate(range(0, 51, 5)):
        if over == 0:
            runs, wickets = 0, 0
        else:
            # Australia collapsed - 123/9 then all out
            if over <= 35:
                runs = int(over * 3.0)
                wickets = min(7, over // 5)
            else:
                runs = int(105 + (132 - 105) * (over - 35) / 15)
                wickets = min(10, 7 + (over - 35) // 3)

        states.append({
            'matchId': 'perth-test-2025',
            'innings': 2,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': runs - 172,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (65 + over) * 6,
            'completedInnings': 1,
            'isChasing': False
        })

    # Innings 3: England 164 all out (~55 overs)
    for i, over in enumerate(range(0, 56, 5)):
        if over == 0:
            runs, wickets = 0, 0
        else:
            runs = int(over * 2.9)
            wickets = min(10, over // 6)

        lead = 172 - 132 + runs
        states.append({
            'matchId': 'perth-test-2025',
            'innings': 3,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': lead,
            'target': 172 - 132 + 164 + 1,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (65 + 50 + over) * 6,
            'completedInnings': 2,
            'isChasing': True
        })

    # Innings 4: Australia 205/2 (Travis Head 123 off 83 balls - rapid chase)
    # ~35 overs
    for i, over in enumerate(range(0, 36, 5)):
        if over == 0:
            runs, wickets = 0, 0
        elif over == 35:
            runs, wickets = 205, 2
        else:
            runs = int(over * 5.9)
            wickets = min(2, over // 18)

        lead = runs - (172 - 132 + 164)
        states.append({
            'matchId': 'perth-test-2025',
            'innings': 4,
            'over': over,
            'runsFor': runs,
            'wicketsDown': wickets,
            'ballsBowled': over * 6,
            'lead': lead,
            'target': 205,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (65 + 50 + 55 + over) * 6,
            'completedInnings': 3,
            'isChasing': True
        })

    # Add batting teams (England batted first)
    states = add_batting_teams(states, first_batting='England')

    return {
        'matchId': 'perth-test-2025',
        'city': 'Perth',
        'dates': 'Nov 21-22, 2025',
        'result': 'Australia won by 8 wickets',
        'days': 2,
        'states': states
    }


def generate_brisbane_test():
    """
    Brisbane Test (Dec 4-7, 2025) - 4 days
    Australia won by 8 wickets
    England: 334 & 241
    Australia: 511 & 69/2
    """
    states = []

    # Innings 1: England 334 (~105 overs, Root 138*)
    # Use realistic run rate: 334/105 = 3.18 runs per over
    for over in range(0, 106, 5):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 50:
            # Early phase: 3.2 run rate, steady batting
            runs = int(over * 3.2)
            wickets = min(3, over // 20)
        elif over <= 90:
            # Middle phase: Root building, 3.3 run rate
            runs = int(160 + (over - 50) * 3.3)
            wickets = min(7, 3 + (over - 50) // 15)
        else:
            # Late phase: acceleration to 334
            runs = int(292 + (334 - 292) * (over - 90) / 15)
            wickets = min(10, 7 + (over - 90) // 5)

        states.append({
            'matchId': 'brisbane-test-2025',
            'innings': 1,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': runs,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - over * 6,
            'completedInnings': 0,
            'isChasing': False
        })

    # Innings 2: Australia 511 (~145 overs, dominant)
    for i, over in enumerate(range(0, 146, 5)):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 60:
            runs = int(over * 2.8)
            wickets = min(3, over // 25)
        elif over <= 120:
            runs = int(168 + (over - 60) * 4.2)
            wickets = min(7, 3 + (over - 60) // 20)
        else:
            runs = int(420 + (511 - 420) * (over - 120) / 25)
            wickets = min(10, 7 + (over - 120) // 8)

        states.append({
            'matchId': 'brisbane-test-2025',
            'innings': 2,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': runs - 334,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (105 + over) * 6,
            'completedInnings': 1,
            'isChasing': False
        })

    # Innings 3: England 241 all out (~80 overs)
    for i, over in enumerate(range(0, 81, 5)):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 50:
            runs = int(over * 2.5)
            wickets = min(5, over // 12)
        else:
            runs = int(125 + (241 - 125) * (over - 50) / 30)
            wickets = min(10, 5 + (over - 50) // 6)

        # Lead from England's perspective (batting team)
        # England: 334 (inn1) + runs (inn3), Australia: 511 (inn2)
        lead = 334 + runs - 511
        states.append({
            'matchId': 'brisbane-test-2025',
            'innings': 3,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': lead,
            'target': 511 - 334 + 241 + 1,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (105 + 145 + over) * 6,
            'completedInnings': 2,
            'isChasing': True
        })

    # Innings 4: Australia 69/2 (easy chase, ~15 overs)
    for i, over in enumerate(range(0, 16, 5)):
        if over == 0:
            runs, wickets = 0, 0
        elif over == 15:
            runs, wickets = 69, 2
        else:
            runs = int(over * 4.6)
            wickets = min(2, over // 8)

        lead = runs - (241 - (511 - 334))
        target = 334 - 511 + 241 + 1
        states.append({
            'matchId': 'brisbane-test-2025',
            'innings': 4,
            'over': over,
            'runsFor': runs,
            'wicketsDown': wickets,
            'ballsBowled': over * 6,
            'lead': lead,
            'target': 65,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (105 + 145 + 80 + over) * 6,
            'completedInnings': 3,
            'isChasing': True
        })

    # Add batting teams (England batted first)
    states = add_batting_teams(states, first_batting='England')

    return {
        'matchId': 'brisbane-test-2025',
        'city': 'Brisbane',
        'dates': 'Dec 4-7, 2025',
        'result': 'Australia won by 8 wickets',
        'days': 4,
        'states': states
    }


def main():
    # Generate data for all three tests
    perth = generate_perth_test()
    brisbane = generate_brisbane_test()

    # Load Adelaide (already generated)
    adelaide_path = Path(__file__).parent.parent.parent / "ui" / "src" / "data" / "adelaide-test.json"
    with open(adelaide_path) as f:
        adelaide = json.load(f)

    # Add batting teams to Adelaide (Australia batted first)
    adelaide_states = add_batting_teams(adelaide['states'], first_batting='Australia')

    adelaide_match = {
        'matchId': 'adelaide-test-2025',
        'city': 'Adelaide',
        'dates': 'Dec 17-21, 2025',
        'result': 'In progress (Day 1)',
        'days': 1,
        'states': adelaide_states
    }

    # Calculate probabilities for all states
    for test_data in [perth, brisbane, adelaide_match]:
        prob_points = []
        cumulative_overs = 0
        innings_boundaries = [0]  # Track where each innings starts

        # First pass: calculate cumulative overs for each innings
        prev_innings = 1
        for state in test_data['states']:
            if state['innings'] != prev_innings:
                innings_boundaries.append(cumulative_overs)
                prev_innings = state['innings']
            cumulative_overs = max(cumulative_overs, state['over'])

        # Reset for second pass
        cumulative_overs = 0
        prev_innings = 1

        for state in test_data['states']:
            # When innings changes, add the previous innings' total overs
            if state['innings'] != prev_innings:
                cumulative_overs += max(s['over'] for s in test_data['states']
                                       if s['innings'] == prev_innings)
                prev_innings = state['innings']

            # Calculate overs_left (match-level time remaining)
            total_overs_bowled = cumulative_overs + state['over']
            overs_left = max(0, 450 - total_overs_bowled)

            # Get batting team from state
            batting_team = state.get('battingTeam', 'Australia')
            probs = predict_probabilities(state, overs_left, batting_team=batting_team)

            # xOver = cumulative overs from previous innings + current over
            xOver = cumulative_overs + state['over']
            prob_points.append({
                'xOver': xOver,
                'innings': state['innings'],
                'over': state['over'],
                'score': f"{state['runsFor']}/{state['wicketsDown']}",
                **probs
            })

        last_over = cumulative_overs + max(s['over'] for s in test_data['states']
                                           if s['innings'] == prev_innings)

        # If match is complete, extend to 450 overs showing final result
        if 'Australia won' in test_data['result']:
            # Australia won - show green at 100%, red at 0%
            final_probs = {'pWin': 1.0, 'pDraw': 0.0, 'pLoss': 0.0}
        elif 'England won' in test_data['result']:
            # England won - show red at 100%, green at 0%
            final_probs = {'pWin': 0.0, 'pDraw': 0.0, 'pLoss': 1.0}
        elif 'draw' in test_data['result'].lower():
            # Draw - show grey at 100%
            final_probs = {'pWin': 0.0, 'pDraw': 1.0, 'pLoss': 0.0}
        else:
            # Match in progress - don't extend
            final_probs = None

        # Extend visualization to 450 overs if match is complete
        if final_probs and last_over < 450:
            final_innings = test_data['states'][-1]['innings']
            for xover in range(last_over + 10, 451, 10):
                prob_points.append({
                    'xOver': xover,
                    'innings': final_innings,
                    'over': xover - cumulative_overs,
                    'score': 'Match Complete',
                    **final_probs
                })

        test_data['probabilities'] = prob_points

        # Add innings boundaries for x-axis labels
        boundaries = []
        cumulative = 0
        for i in range(1, 5):
            innings_states = [s for s in test_data['states'] if s['innings'] == i]
            if innings_states:
                batting_team = innings_states[0].get('battingTeam', 'Unknown')
                boundaries.append({
                    'innings': i,
                    'xOver': cumulative,
                    'battingTeam': batting_team
                })
                cumulative += max(s['over'] for s in innings_states)
        test_data['inningsBoundaries'] = boundaries

        # Track fall of wickets
        wicket_falls = []
        cumulative = 0
        prev_innings = 1
        prev_wickets = 0

        for state in test_data['states']:
            # When innings changes, reset
            if state['innings'] != prev_innings:
                cumulative += max(s['over'] for s in test_data['states']
                                 if s['innings'] == prev_innings)
                prev_wickets = 0
                prev_innings = state['innings']

            # Check if wicket fell
            if state['wicketsDown'] > prev_wickets:
                xOver = cumulative + state['over']
                wicket_falls.append({
                    'innings': state['innings'],
                    'xOver': xOver,
                    'wickets': state['wicketsDown'],
                    'score': f"{state['runsFor']}/{state['wicketsDown']}"
                })
                prev_wickets = state['wicketsDown']

        test_data['wicketFalls'] = wicket_falls

    # Save all three tests
    output = {
        'series': 'The Ashes 2025-26',
        'tests': [adelaide_match, brisbane, perth]  # Adelaide first (in progress)
    }

    output_path = Path(__file__).parent.parent.parent / "ui" / "src" / "data" / "ashes-series-2025.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Generated Ashes series worm data")
    print(f"  Perth: {len(perth['probabilities'])} points ({perth['days']} days)")
    print(f"  Brisbane: {len(brisbane['probabilities'])} points ({brisbane['days']} days)")
    print(f"  Adelaide: {len(adelaide_match['probabilities'])} points ({adelaide_match['days']} day)")
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
