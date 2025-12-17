#!/usr/bin/env python3
"""
Generate worm chart data for all three Ashes tests
Perth, Brisbane, and Adelaide (in progress)
"""

import json
from pathlib import Path
import numpy as np

# Load model
model_path = Path(__file__).parent.parent / "output" / "model.json"
with open(model_path) as f:
    model_data = json.load(f)

means = np.array(model_data['featureMeans'])
stds = np.array(model_data['featureStds'])
coefficients = np.array(model_data['coefficients'])
intercepts = np.array(model_data['intercepts'])


def predict_probabilities(state, batting_team='Australia'):
    """
    Predict win/draw/loss probabilities from Australia's perspective

    Model predicts from batting team's perspective, so we flip when England is batting.
    """
    run_rate = (state['runsFor'] / state['ballsBowled']) * 6 if state['ballsBowled'] > 0 else 0
    runs_per_wicket = state['runsFor'] / (state['wicketsDown'] + 1)

    # Calculate lead properly based on innings
    lead = state['lead']

    # Required run rate for chasing
    required_run_rate = 0
    if state['isChasing'] and lead < 0 and state['ballsRemaining'] > 0:
        runs_needed = abs(lead) + 1
        required_run_rate = (runs_needed / state['ballsRemaining']) * 6

    features = np.array([[
        state['innings'],
        state['wicketsDown'],
        run_rate,
        lead,
        state['ballsRemaining'],
        runs_per_wicket,
        1 if state['isChasing'] else 0,
        required_run_rate
    ]])

    features_scaled = (features - means) / stds
    logits = np.dot(features_scaled, coefficients.T) + intercepts
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # Model predicts from batting team's perspective
    # If England is batting, flip win/loss to show Australia's perspective
    if batting_team == 'England':
        return {
            'pWin': float(probs[0][2]),  # England's loss = Australia's win
            'pDraw': float(probs[0][1]),
            'pLoss': float(probs[0][0])  # England's win = Australia's loss
        }
    else:
        return {
            'pWin': float(probs[0][0]),  # Australia's win
            'pDraw': float(probs[0][1]),
            'pLoss': float(probs[0][2])  # Australia's loss
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
    for over in range(0, 66, 5):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 20:
            runs = int(over * 2.5)
            wickets = min(2, over // 10)
        elif over <= 40:
            runs = 50 + int((over - 20) * 2.0)
            wickets = min(5, 2 + (over - 20) // 10)
        else:
            runs = int(90 + (172 - 90) * (over - 40) / 25)
            wickets = min(10, 5 + (over - 40) // 5)

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
    for over in range(0, 106, 5):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 50:
            runs = int(over * 2.5)
            wickets = min(4, over // 15)
        elif over <= 90:
            runs = int(125 + (over - 50) * 3.2)
            wickets = min(7, 4 + (over - 50) // 15)
        else:
            runs = int(253 + (334 - 253) * (over - 90) / 15)
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

        lead = 511 - 334 - runs
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
        last_over = 0

        for state in test_data['states']:
            # Get batting team from state, or determine from match
            batting_team = state.get('battingTeam', 'Australia')
            probs = predict_probabilities(state, batting_team=batting_team)
            prob_points.append({
                'xOver': state['over'],
                'innings': state['innings'],
                'over': state['over'],
                'score': f"{state['runsFor']}/{state['wicketsDown']}",
                **probs
            })
            last_over = max(last_over, state['over'])

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
            for over in range(last_over + 5, 451, 10):
                prob_points.append({
                    'xOver': over,
                    'innings': test_data['states'][-1]['innings'],
                    'over': over,
                    'score': 'Match Complete',
                    **final_probs
                })

        test_data['probabilities'] = prob_points

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
