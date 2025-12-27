#!/usr/bin/env python3
"""
Generate worm chart data for all three Ashes tests
Perth, Brisbane, and Adelaide (in progress)

Generates predictions from TWO models:
1. Full model (8 features): includes team ratings and home advantage
2. Scorecard-only model (5 features): match state + toss only
"""

import json
import pickle
from pathlib import Path
import numpy as np

# Load full XGBoost model (8 features)
model_path = Path(__file__).parent.parent / "output" / "model.pkl"
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

xgb_model_full = model_data['model']
label_encoder_full = model_data['label_encoder']

print(f"Loaded full XGBoost model (8 features) with label mapping: {dict(zip(label_encoder_full.classes_, range(len(label_encoder_full.classes_))))}")

# Load scorecard-only XGBoost model (5 features)
model_path_scorecard = Path(__file__).parent.parent / "output" / "model_scorecard_only.pkl"
with open(model_path_scorecard, 'rb') as f:
    model_data_scorecard = pickle.load(f)

xgb_model_scorecard = model_data_scorecard['model']
label_encoder_scorecard = model_data_scorecard['label_encoder']

print(f"Loaded scorecard-only XGBoost model (5 features) with label mapping: {dict(zip(label_encoder_scorecard.classes_, range(len(label_encoder_scorecard.classes_))))}")


def predict_probabilities(full_match_state, overs_left, first_team='England', home_team=None,
                         first_team_rating=1500.0, second_team_rating=1500.0, first_team_won_toss=0,
                         use_scorecard_only=False):
    """
    Predict win/draw/loss probabilities from first team's perspective using XGBoost

    Args:
        full_match_state: Dict with all innings states:
            - first_team_score_inn1, first_team_wickets_inn1
            - second_team_score_inn2, second_team_wickets_inn2
            - first_team_score_inn3, first_team_wickets_inn3
            - second_team_score_inn4, second_team_wickets_inn4
        overs_left: Total match overs remaining
        first_team: Which team bats innings 1 and 3 ('England' or 'Australia')
        home_team: Which team is playing at home (e.g., 'Australia' for Ashes in Australia)
        first_team_rating: ELO rating of first team
        second_team_rating: ELO rating of second team
        first_team_won_toss: 1 if first team won toss, 0 otherwise
        use_scorecard_only: If True, use 5-feature model (no ratings/home), else use 8-feature model

    Returns probabilities from Australia's perspective (flips if needed)
    """
    # Calculate wickets remaining (cumulative across both innings)
    first_team_wickets_remaining = 20 - (
        full_match_state['first_team_wickets_inn1'] +
        full_match_state['first_team_wickets_inn3']
    )
    second_team_wickets_remaining = 20 - (
        full_match_state['second_team_wickets_inn2'] +
        full_match_state['second_team_wickets_inn4']
    )

    # Calculate first team lead
    first_team_lead = (
        full_match_state['first_team_score_inn1'] +
        full_match_state['first_team_score_inn3']
    ) - (
        full_match_state['second_team_score_inn2'] +
        full_match_state['second_team_score_inn4']
    )

    if use_scorecard_only:
        # 5 features: overs_left, wickets_remaining x2, lead, won_toss
        features = np.array([[
            overs_left,
            first_team_wickets_remaining,
            second_team_wickets_remaining,
            first_team_lead,
            first_team_won_toss
        ]])
        probs = xgb_model_scorecard.predict_proba(features)[0]
    else:
        # Determine if first team is home team
        first_team_is_home = 1 if (home_team and first_team == home_team) else 0

        # 8 features: overs_left, wickets_remaining x2, lead, is_home, won_toss, ratings x2
        features = np.array([[
            overs_left,
            first_team_wickets_remaining,
            second_team_wickets_remaining,
            first_team_lead,
            first_team_is_home,
            first_team_won_toss,
            first_team_rating,
            second_team_rating
        ]])
        probs = xgb_model_full.predict_proba(features)[0]

    use_mc = False  # No longer using Monte Carlo

    # Label mapping: {'draw': 0, 'loss': 1, 'win': 2}
    p_draw = float(probs[0])
    p_loss = float(probs[1])  # First team loss
    p_win = float(probs[2])   # First team win

    # Convert to Australia's perspective
    if first_team == 'England':
        # England is first team, so flip for Australia's perspective
        return {
            'pWin': p_loss,  # England's loss = Australia's win
            'pDraw': p_draw,
            'pLoss': p_win,  # England's win = Australia's loss
            'usedMonteCarlo': use_mc
        }
    else:
        # Australia is first team, use directly
        return {
            'pWin': p_win,   # Australia's win
            'pDraw': p_draw,
            'pLoss': p_loss,  # Australia's loss
            'usedMonteCarlo': use_mc
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

    # Manually specify wicket fall overs for realistic clumpy visualization
    wicket_falls = [
        # Innings 1: England - early wobble, then collapse
        {'innings': 1, 'xOver': 8, 'wickets': 1},
        {'innings': 1, 'xOver': 16, 'wickets': 2},
        {'innings': 1, 'xOver': 19, 'wickets': 3},
        {'innings': 1, 'xOver': 24, 'wickets': 4},
        {'innings': 1, 'xOver': 28, 'wickets': 5},
        {'innings': 1, 'xOver': 31, 'wickets': 6},
        {'innings': 1, 'xOver': 35, 'wickets': 7},
        {'innings': 1, 'xOver': 48, 'wickets': 8},
        {'innings': 1, 'xOver': 53, 'wickets': 9},
        {'innings': 1, 'xOver': 60, 'wickets': 10},

        # Innings 2: Australia - steady fall then late collapse
        {'innings': 2, 'xOver': 70, 'wickets': 1},  # cumulative xOver
        {'innings': 2, 'xOver': 75, 'wickets': 2},
        {'innings': 2, 'xOver': 80, 'wickets': 3},
        {'innings': 2, 'xOver': 85, 'wickets': 4},
        {'innings': 2, 'xOver': 90, 'wickets': 5},
        {'innings': 2, 'xOver': 95, 'wickets': 6},
        {'innings': 2, 'xOver': 100, 'wickets': 7},
        {'innings': 2, 'xOver': 103, 'wickets': 8},
        {'innings': 2, 'xOver': 107, 'wickets': 9},
        {'innings': 2, 'xOver': 113, 'wickets': 10},

        # Innings 3: England - regular fall
        {'innings': 3, 'xOver': 121, 'wickets': 1},
        {'innings': 3, 'xOver': 127, 'wickets': 2},
        {'innings': 3, 'xOver': 133, 'wickets': 3},
        {'innings': 3, 'xOver': 139, 'wickets': 4},
        {'innings': 3, 'xOver': 145, 'wickets': 5},
        {'innings': 3, 'xOver': 151, 'wickets': 6},
        {'innings': 3, 'xOver': 157, 'wickets': 7},
        {'innings': 3, 'xOver': 163, 'wickets': 8},
        {'innings': 3, 'xOver': 167, 'wickets': 9},
        {'innings': 3, 'xOver': 170, 'wickets': 10},

        # Innings 4: Australia - only 2 wickets
        {'innings': 4, 'xOver': 188, 'wickets': 1},
        {'innings': 4, 'xOver': 200, 'wickets': 2},
    ]

    return {
        'matchId': 'perth-test-2025',
        'city': 'Perth',
        'dates': 'Nov 21-22, 2025',
        'result': 'Australia won by 8 wickets',
        'days': 2,
        'states': states,
        'wicket_falls_manual': wicket_falls  # Separate from auto-generated
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

    # Manually specify wicket fall overs
    wicket_falls = [
        # Innings 1: England 334 (~105 overs) - steady accumulation with Root century
        {'innings': 1, 'xOver': 18, 'wickets': 1},
        {'innings': 1, 'xOver': 42, 'wickets': 2},
        {'innings': 1, 'xOver': 56, 'wickets': 3},
        {'innings': 1, 'xOver': 72, 'wickets': 4},
        {'innings': 1, 'xOver': 78, 'wickets': 5},
        {'innings': 1, 'xOver': 84, 'wickets': 6},
        {'innings': 1, 'xOver': 91, 'wickets': 7},
        {'innings': 1, 'xOver': 97, 'wickets': 8},
        {'innings': 1, 'xOver': 101, 'wickets': 9},
        {'innings': 1, 'xOver': 104, 'wickets': 10},

        # Innings 2: Australia 511 (~145 overs) - dominant
        {'innings': 2, 'xOver': 130, 'wickets': 1},
        {'innings': 2, 'xOver': 155, 'wickets': 2},
        {'innings': 2, 'xOver': 178, 'wickets': 3},
        {'innings': 2, 'xOver': 205, 'wickets': 4},
        {'innings': 2, 'xOver': 215, 'wickets': 5},
        {'innings': 2, 'xOver': 224, 'wickets': 6},
        {'innings': 2, 'xOver': 232, 'wickets': 7},
        {'innings': 2, 'xOver': 240, 'wickets': 8},
        {'innings': 2, 'xOver': 245, 'wickets': 9},
        {'innings': 2, 'xOver': 249, 'wickets': 10},

        # Innings 3: England 241 (~80 overs) - steady collapse
        {'innings': 3, 'xOver': 262, 'wickets': 1},
        {'innings': 3, 'xOver': 274, 'wickets': 2},
        {'innings': 3, 'xOver': 283, 'wickets': 3},
        {'innings': 3, 'xOver': 293, 'wickets': 4},
        {'innings': 3, 'xOver': 301, 'wickets': 5},
        {'innings': 3, 'xOver': 310, 'wickets': 6},
        {'innings': 3, 'xOver': 317, 'wickets': 7},
        {'innings': 3, 'xOver': 323, 'wickets': 8},
        {'innings': 3, 'xOver': 327, 'wickets': 9},
        {'innings': 3, 'xOver': 329, 'wickets': 10},

        # Innings 4: Australia 69/2 (~15 overs) - easy chase
        {'innings': 4, 'xOver': 337, 'wickets': 1},
        {'innings': 4, 'xOver': 343, 'wickets': 2},
    ]

    return {
        'matchId': 'brisbane-test-2025',
        'city': 'Brisbane',
        'dates': 'Dec 4-7, 2025',
        'result': 'Australia won by 8 wickets',
        'days': 4,
        'states': states,
        'wicket_falls_manual': wicket_falls
    }


def generate_adelaide_test():
    """
    Adelaide Test (Dec 17-21, 2025) - In progress (Day 2)
    Australia: 371 all out
    England: 213/8 (trail by 158)
    """
    states = []

    # Innings 1: Australia 371 all out (~95 overs, Carey 106, Khawaja 82, Starc 54)
    # Early collapse 33/2, then 94/4, recovery to 371
    for over in range(0, 96, 5):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 10:
            # Early wickets: 33/2 at over 9
            runs = int(over * 3.3)
            wickets = min(2, over // 5)
        elif over <= 25:
            # Collapse: 94/4 at over 24
            runs = int(33 + (94 - 33) * (over - 10) / 15)
            wickets = min(4, 2 + (over - 10) // 8)
        elif over <= 50:
            # Recovery: Khawaja-Carey partnership
            runs = int(94 + (197 - 94) * (over - 25) / 25)
            wickets = min(5, 4 + (over - 47) // 10) if over > 47 else 4
        elif over <= 80:
            # Middle order contributions
            runs = int(197 + (335 - 197) * (over - 50) / 30)
            wickets = min(8, 5 + (over - 60) // 10)
        else:
            # Tail wags: Starc 54
            runs = int(335 + (371 - 335) * (over - 80) / 15)
            wickets = min(10, 8 + (over - 90) // 3)

        states.append({
            'matchId': 'adelaide-test-2025',
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

    # Innings 2: England 286 all out (~80 overs, trail by 85)
    # Collapse: 42/3 at over 10, then fightback with Stokes
    aus_inn1_overs = 95
    for over in range(0, 81, 5):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 10:
            # Collapse: 42/3 at over 10
            runs = int(over * 4.2)
            wickets = min(3, over // 4)
        elif over <= 20:
            # Root dismissed: 80/4 at over 20
            runs = int(42 + (80 - 42) * (over - 10) / 10)
            wickets = 4
        elif over <= 40:
            # Brook resistance: 141/5 at over 40
            runs = int(80 + (141 - 80) * (over - 20) / 20)
            wickets = 5
        elif over <= 45:
            # Another wicket: 159/6 at over 45
            runs = int(141 + (159 - 141) * (over - 40) / 5)
            wickets = 6
        elif over <= 70:
            # Stokes-Archer partnership: 240/8 at over 70
            runs = int(159 + (240 - 159) * (over - 45) / 25)
            wickets = min(8, 6 + (over - 65) // 3)
        else:
            # Tail ends: 286 all out at over 80
            runs = int(240 + (286 - 240) * (over - 70) / 10)
            wickets = min(10, 8 + (over - 75) // 3)

        lead = runs - 371
        states.append({
            'matchId': 'adelaide-test-2025',
            'innings': 2,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': lead,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (aus_inn1_overs + over) * 6,
            'completedInnings': 1,
            'isChasing': False
        })

    # Innings 3: Australia 2nd innings - 349 all out (Head 170, Carey 72)
    # Building on 85-run first innings lead
    eng_inn2_overs = 80
    for over in range(0, 86, 5):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 5:
            # Early wicket: 8/1 (Weatherald)
            runs = int(over * 1.6)
            wickets = 1 if over >= 2 else 0
        elif over <= 20:
            # 53/2 (Labuschagne out at over 16)
            runs = int(8 + (53 - 8) * (over - 5) / 15)
            wickets = 2 if over >= 17 else 1
        elif over <= 35:
            # Head-Khawaja partnership: 139/3 at over 35
            runs = int(53 + (139 - 53) * (over - 20) / 15)
            wickets = 3 if over >= 35 else 2
        elif over <= 40:
            # Green out: 149/4 at over 37
            runs = int(139 + (149 - 139) * (over - 35) / 5)
            wickets = 4 if over >= 37 else 3
        elif over <= 75:
            # Head-Carey partnership: 311/5 at over 74 (Head 170)
            runs = int(149 + (311 - 149) * (over - 40) / 35)
            wickets = 5 if over >= 74 else 4
        elif over <= 80:
            # Collapse: 335/7 at over 80
            runs = int(311 + (335 - 311) * (over - 75) / 5)
            wickets = min(7, 5 + (over - 78) // 2)
        else:
            # Tail ends: 349 all out at over 85
            runs = int(335 + (349 - 335) * (over - 80) / 5)
            wickets = min(10, 7 + (over - 83) // 1)

        # Lead = first innings lead (85) + second innings runs
        lead = 85 + runs
        states.append({
            'matchId': 'adelaide-test-2025',
            'innings': 3,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': lead,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (aus_inn1_overs + eng_inn2_overs + over) * 6,
            'completedInnings': 2,
            'isChasing': False
        })

    # Innings 4: England 2nd innings - 352 all out (102.5 overs)
    # Lost by 82 runs chasing 435
    aus_inn3_overs = 85
    # Include key overs: every 5 overs + stumps (63) + all out (103)
    overs_to_include = list(range(0, 105, 5)) + [63, 83, 97, 101, 103]
    for over in sorted(set(overs_to_include)):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 10:
            # Early wickets: Duckett out early (1.2 ov)
            runs = int(over * 2.5)
            wickets = 1 if over >= 3 else 0
        elif over <= 30:
            # Root building: 109/3 at over 29
            runs = int(25 + (109 - 25) * (over - 10) / 19)
            wickets = min(3, 1 + (over - 10) // 10)
        elif over <= 50:
            # Crawley 85, middle order: 177/4 at over 47, 189/5 at over 52
            runs = int(109 + (189 - 109) * (over - 30) / 22)
            wickets = min(5, 3 + (over - 45) // 5)
        elif over == 63:
            # Stumps Day 4: 207/6
            runs, wickets = 207, 6
        elif over <= 83:
            # Day 5: Smith resistance, 285/7 at over 83
            runs = int(207 + (285 - 207) * (over - 63) / 20)
            wickets = 7 if over >= 83 else 6
        elif over <= 97:
            # Jacks resistance: 337/8 at over 97
            runs = int(285 + (337 - 285) * (over - 83) / 14)
            wickets = 8 if over >= 97 else 7
        elif over <= 103:
            # Final wickets: 349/9 at over 101, 352 all out at over 103
            if over == 103:
                runs, wickets = 352, 10
            elif over >= 101:
                runs = int(337 + (352 - 337) * (over - 97) / 6)
                wickets = 9 if over >= 101 else 8
            else:
                runs = int(337 + (349 - 337) * (over - 97) / 4)
                wickets = 8

        # Lead is negative (England trailing)
        lead = runs - 434  # Australia's total lead is 434
        states.append({
            'matchId': 'adelaide-test-2025',
            'innings': 4,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': lead,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (aus_inn1_overs + eng_inn2_overs + aus_inn3_overs + over) * 6,
            'completedInnings': 3,
            'isChasing': True
        })

    # Add batting teams (Australia batted first)
    states = add_batting_teams(states, first_batting='Australia')

    # Manually specify wicket fall overs
    wicket_falls = [
        # Innings 1: Australia 371 (~95 overs) - early collapse then recovery
        {'innings': 1, 'xOver': 6, 'wickets': 1},  # Early wicket
        {'innings': 1, 'xOver': 9, 'wickets': 2},  # 33/2
        {'innings': 1, 'xOver': 17, 'wickets': 3},  # Collapse starts
        {'innings': 1, 'xOver': 24, 'wickets': 4},  # 94/4
        {'innings': 1, 'xOver': 48, 'wickets': 5},  # Partnership broken
        {'innings': 1, 'xOver': 64, 'wickets': 6},  # Middle order
        {'innings': 1, 'xOver': 72, 'wickets': 7},
        {'innings': 1, 'xOver': 80, 'wickets': 8},  # Lower order
        {'innings': 1, 'xOver': 91, 'wickets': 9},  # Tail starts
        {'innings': 1, 'xOver': 94, 'wickets': 10}, # All out

        # Innings 2: England 286 all out (~80 overs) - early collapse then Stokes fightback
        {'innings': 2, 'xOver': 98, 'wickets': 1},  # Early wicket
        {'innings': 2, 'xOver': 101, 'wickets': 2},
        {'innings': 2, 'xOver': 105, 'wickets': 3}, # 42/3 collapse
        {'innings': 2, 'xOver': 115, 'wickets': 4}, # Root dismissed
        {'innings': 2, 'xOver': 135, 'wickets': 5}, # Brook resistance ends
        {'innings': 2, 'xOver': 140, 'wickets': 6}, # 159/6
        {'innings': 2, 'xOver': 158, 'wickets': 7}, # Lower order
        {'innings': 2, 'xOver': 164, 'wickets': 8}, # 213/8
        {'innings': 2, 'xOver': 171, 'wickets': 9}, # Stokes out
        {'innings': 2, 'xOver': 175, 'wickets': 10}, # All out 286

        # Innings 3: Australia 2nd innings - 349 all out (Head 170, Carey 72)
        {'innings': 3, 'xOver': 177, 'wickets': 1}, # Weatherald 8/1
        {'innings': 3, 'xOver': 192, 'wickets': 2}, # Labuschagne 53/2
        {'innings': 3, 'xOver': 211, 'wickets': 3}, # Khawaja 139/3
        {'innings': 3, 'xOver': 213, 'wickets': 4}, # Green 149/4
        {'innings': 3, 'xOver': 249, 'wickets': 5}, # Head 170, 311/5
        {'innings': 3, 'xOver': 254, 'wickets': 6}, # Carey 72, 329/6
        {'innings': 3, 'xOver': 256, 'wickets': 7}, # Inglis 335/7
        {'innings': 3, 'xOver': 259, 'wickets': 8}, # Cummins 344/8
        {'innings': 3, 'xOver': 259, 'wickets': 9}, # Lyon 344/9
        {'innings': 3, 'xOver': 260, 'wickets': 10}, # Boland 349 all out

        # Innings 4: England 2nd innings - 352 all out (lost by 82 runs)
        # Duckett 4 (1.2), Pope 31 (9.4), Root 109 (28.6), Brook 177 (47.2)
        # Stokes 189 (51.6), Crawley 194 (53.3), Smith 285 (82.5), Jacks 337 (97.2)
        # Archer 349 (101.3), Tongue 352 (102.5)
        {'innings': 4, 'xOver': 262, 'wickets': 1}, # Duckett 4 (1.2 ov)
        {'innings': 4, 'xOver': 270, 'wickets': 2}, # Pope 31 (9.4 ov)
        {'innings': 4, 'xOver': 289, 'wickets': 3}, # Root 109 (28.6 ov)
        {'innings': 4, 'xOver': 308, 'wickets': 4}, # Brook 177 (47.2 ov)
        {'innings': 4, 'xOver': 312, 'wickets': 5}, # Stokes 189 (51.6 ov)
        {'innings': 4, 'xOver': 314, 'wickets': 6}, # Crawley 194 (53.3 ov)
        {'innings': 4, 'xOver': 343, 'wickets': 7}, # Smith 285 (82.5 ov)
        {'innings': 4, 'xOver': 357, 'wickets': 8}, # Jacks 337 (97.2 ov)
        {'innings': 4, 'xOver': 362, 'wickets': 9}, # Archer 349 (101.3 ov)
        {'innings': 4, 'xOver': 363, 'wickets': 10}, # Tongue 352 all out (102.5 ov)
    ]

    return {
        'matchId': 'adelaide-test-2025',
        'city': 'Adelaide',
        'dates': 'Dec 17-21, 2025',
        'result': 'Australia won by 82 runs',
        'days': 5,
        'states': states,
        'wicket_falls_manual': wicket_falls
    }


def generate_melbourne_test():
    """Generate Melbourne Boxing Day Test - Day 1 (20-wicket day)"""
    states = []

    # Innings 1: Australia 152 all out (45.2 overs) - Josh Tongue 5-45
    # 72/4 at lunch (25 overs), then collapsed to 152 all out
    for over in range(0, 46, 5):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 10:
            # Early wickets: Weatherald and Head
            runs = int(over * 2.8)
            wickets = 1 if over >= 6 else 0
        elif over == 25:
            # Lunch: 72/4
            runs, wickets = 72, 4
        elif over <= 20:
            # Building to lunch
            runs = int(28 + (72 - 28) * (over - 10) / 15)
            wickets = min(3, 1 + (over - 10) // 6)
        elif over <= 40:
            # Post-lunch collapse: 130/8
            runs = int(72 + (130 - 72) * (over - 25) / 15)
            wickets = min(8, 4 + (over - 27) // 4)
        else:
            # Tail wagging: Neser 35, finished 152 all out
            runs = int(130 + (152 - 130) * (over - 40) / 5)
            wickets = 10 if over >= 45 else min(9, 8 + (over - 40) // 3)

        # Lead from Australia's perspective (batting team)
        lead = runs
        states.append({
            'matchId': 'melbourne-test-2025',
            'innings': 1,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': lead,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - over * 6,
            'completedInnings': 0,
            'isChasing': False
        })

    # Innings 2: England 110 all out (29.5 overs) - Neser 4-45, Boland 3-30
    # Rapid collapse, only Brook 41 resisted
    aus_inn1_overs = 45
    overs_to_include = list(range(0, 31, 5)) + [30]
    for over in sorted(set(overs_to_include)):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 10:
            # Early wickets: Duckett and Crawley fall
            runs = int(over * 2.5)
            wickets = 1 if over >= 6 else 0
        elif over <= 20:
            # Brook counter-attack: 41 off 34 balls, 70/5 at over 20
            runs = int(25 + (70 - 25) * (over - 10) / 10)
            wickets = min(5, 1 + (over - 10) // 3)
        elif over == 30:
            # All out for 110
            runs, wickets = 110, 10
        else:
            # Final collapse: towards 110 all out
            runs = int(70 + (110 - 70) * (over - 20) / 10)
            wickets = min(9, 5 + (over - 20) // 2)

        # Lead from England's perspective (batting team)
        # England trailing by (152 - runs)
        lead = runs - 152
        states.append({
            'matchId': 'melbourne-test-2025',
            'innings': 2,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': lead,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (aus_inn1_overs + over) * 6,
            'completedInnings': 1,
            'isChasing': True
        })

    # Innings 3: Australia 2nd innings - 132 all out (34.3 overs)
    # Head 46, Smith 24*, Carse 4-34, Stokes 3-24
    eng_inn2_overs = 30
    overs_to_include = list(range(0, 36, 5)) + [1, 21, 31, 35]
    for over in sorted(set(overs_to_include)):
        if over == 0:
            runs, wickets = 0, 0
        elif over == 1:
            # Stumps Day 1
            runs, wickets = 4, 0
        elif over <= 10:
            # Early wickets: Boland 22/1
            runs = int(over * 4)
            wickets = 1 if over >= 6 else 0
        elif over <= 21:
            # Head building: 82/4 at over 21
            runs = int(40 + (82 - 40) * (over - 10) / 11)
            wickets = min(4, 1 + (over - 10) // 4)
        elif over == 31:
            # Collapse: 120/8
            runs, wickets = 120, 8
        elif over <= 30:
            # Middle order: Khawaja, Carey fall
            runs = int(82 + (119 - 82) * (over - 21) / 9)
            wickets = min(7, 4 + (over - 22) // 3)
        elif over == 35:
            # All out 132
            runs, wickets = 132, 10
        else:
            # Final wickets
            runs = int(120 + (132 - 120) * (over - 31) / 4)
            wickets = min(10, 8 + (over - 31) // 2)

        # Lead from Australia's perspective
        # Australia lead by (152 - 110) + runs = 42 + runs
        lead = 42 + runs
        states.append({
            'matchId': 'melbourne-test-2025',
            'innings': 3,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': lead,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (aus_inn1_overs + eng_inn2_overs + over) * 6,
            'completedInnings': 2,
            'isChasing': False
        })

    # Innings 4: England 2nd innings - 178/6 chasing 175 (won by 4 wickets)
    # Bethell 40, Brook 18, Root 15
    aus_inn3_overs = 35
    overs_to_include = list(range(0, 46, 5)) + [45]
    for over in sorted(set(overs_to_include)):
        if over == 0:
            runs, wickets = 0, 0
        elif over <= 15:
            # Early scoring: Crawley and Bethell
            runs = int(over * 3.5)
            wickets = 1 if over >= 10 else 0
        elif over <= 30:
            # Middle order: Bethell 40, Brook 18, Root 15
            runs = int(52 + (130 - 52) * (over - 15) / 15)
            wickets = min(5, 1 + (over - 15) // 5)
        elif over == 45:
            # Victory: 178/6
            runs, wickets = 178, 6
        else:
            # Final runs: Brook and Smith see England home
            runs = int(130 + (178 - 130) * (over - 30) / 15)
            wickets = 6

        # Lead from England's perspective (chasing)
        # Target is 175, England needs (175 - runs) more
        lead = runs - 174  # Australia set 175
        target = 175
        states.append({
            'matchId': 'melbourne-test-2025',
            'innings': 4,
            'over': over,
            'runsFor': runs,
            'wicketsDown': min(wickets, 10),
            'ballsBowled': over * 6,
            'lead': lead,
            'target': target,
            'matchOversLimit': 450,
            'ballsRemaining': 450 * 6 - (aus_inn1_overs + eng_inn2_overs + aus_inn3_overs + over) * 6,
            'completedInnings': 3,
            'isChasing': True
        })

    # Add batting teams (Australia batted first)
    states = add_batting_teams(states, first_batting='Australia')

    # Manually specify wicket fall overs
    wicket_falls = [
        # Innings 1: Australia 152 all out - early wickets then collapse
        {'innings': 1, 'xOver': 6, 'wickets': 1},   # Weatherald 10
        {'innings': 1, 'xOver': 12, 'wickets': 2},  # Head 12
        {'innings': 1, 'xOver': 18, 'wickets': 3},  # Labuschagne 9
        {'innings': 1, 'xOver': 24, 'wickets': 4},  # Smith 6, lunch 72/4
        {'innings': 1, 'xOver': 30, 'wickets': 5},  # Khawaja 29
        {'innings': 1, 'xOver': 34, 'wickets': 6},  # Green 17
        {'innings': 1, 'xOver': 37, 'wickets': 7},  # Carey 20
        {'innings': 1, 'xOver': 42, 'wickets': 8},  # Starc
        {'innings': 1, 'xOver': 44, 'wickets': 9},  # Boland
        {'innings': 1, 'xOver': 45, 'wickets': 10}, # All out 152

        # Innings 2: England 110 all out - rapid collapse
        {'innings': 2, 'xOver': 51, 'wickets': 1},  # Duckett
        {'innings': 2, 'xOver': 56, 'wickets': 2},  # Crawley
        {'innings': 2, 'xOver': 59, 'wickets': 3},  # Pope
        {'innings': 2, 'xOver': 63, 'wickets': 4},  # Root
        {'innings': 2, 'xOver': 66, 'wickets': 5},  # Brook 41
        {'innings': 2, 'xOver': 68, 'wickets': 6},  # Stokes
        {'innings': 2, 'xOver': 71, 'wickets': 7},  # Bethell
        {'innings': 2, 'xOver': 73, 'wickets': 8},  # Atkinson 28
        {'innings': 2, 'xOver': 74, 'wickets': 9},  # Carse
        {'innings': 2, 'xOver': 75, 'wickets': 10}, # All out 110

        # Innings 3: Australia 2nd innings - 132 all out (Head 46, Carse 4-34)
        {'innings': 3, 'xOver': 82, 'wickets': 1},  # Boland 22/1 (6.1 ov)
        {'innings': 3, 'xOver': 86, 'wickets': 2},  # Weatherald 40/2 (10.5 ov)
        {'innings': 3, 'xOver': 92, 'wickets': 3},  # Labuschagne 61/3 (17.1 ov)
        {'innings': 3, 'xOver': 96, 'wickets': 4},  # Head 46, 82/4 (20.6 ov)
        {'innings': 3, 'xOver': 97, 'wickets': 5},  # Khawaja 83/5 (21.3 ov)
        {'innings': 3, 'xOver': 98, 'wickets': 6},  # Carey 88/6 (22.5 ov)
        {'innings': 3, 'xOver': 106, 'wickets': 7}, # Green 119/7 (30.5 ov)
        {'innings': 3, 'xOver': 107, 'wickets': 8}, # Neser 120/8 (31.5 ov)
        {'innings': 3, 'xOver': 107, 'wickets': 9}, # Starc 121/9 (31.6 ov)
        {'innings': 3, 'xOver': 110, 'wickets': 10}, # Richardson 132 all out (34.3 ov)

        # Innings 4: England 2nd innings - 178/6 (won by 4 wickets chasing 175)
        # Bethell 40, Brook 18, Root 15
        {'innings': 4, 'xOver': 120, 'wickets': 1}, # Crawley lbw Boland
        {'innings': 4, 'xOver': 128, 'wickets': 2}, # Bethell c Khawaja b Boland
        {'innings': 4, 'xOver': 133, 'wickets': 3}, # Root 15 lbw Richardson
        {'innings': 4, 'xOver': 138, 'wickets': 4}, # Pope
        {'innings': 4, 'xOver': 143, 'wickets': 5}, # Stokes 2
        {'innings': 4, 'xOver': 150, 'wickets': 6}, # (lost 6 wickets total)
    ]

    return {
        'matchId': 'melbourne-test-2025',
        'city': 'Melbourne',
        'dates': 'Dec 26-27, 2025',
        'result': 'England won by 4 wickets',
        'days': 2,
        'states': states,
        'wicket_falls_manual': wicket_falls
    }


def cumulative_overs_at_innings_start(states, target_innings):
    """Calculate cumulative overs at the start of a given innings"""
    cumulative = 0
    for i in range(1, target_innings):
        innings_states = [s for s in states if s['innings'] == i]
        if innings_states:
            cumulative += max(s['over'] for s in innings_states)
    return cumulative


def main():
    # Generate data for all four tests
    perth = generate_perth_test()
    brisbane = generate_brisbane_test()
    adelaide = generate_adelaide_test()
    melbourne = generate_melbourne_test()

    # Calculate probabilities for all states
    for test_data in [perth, brisbane, adelaide, melbourne]:
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

        # Track final scores/wickets for completed innings
        innings_final_states = {}  # {innings_num: {'score': X, 'wickets': Y}}

        for state in test_data['states']:
            # When innings changes, record final state of previous innings
            if state['innings'] != prev_innings:
                # Get final state of previous innings
                prev_innings_states = [s for s in test_data['states'] if s['innings'] == prev_innings]
                if prev_innings_states:
                    final = prev_innings_states[-1]
                    innings_final_states[prev_innings] = {
                        'score': final['runsFor'],
                        'wickets': final['wicketsDown']
                    }

                cumulative_overs += max(s['over'] for s in test_data['states']
                                       if s['innings'] == prev_innings)
                prev_innings = state['innings']

            # Calculate overs_left (match-level time remaining)
            total_overs_bowled = cumulative_overs + state['over']
            overs_left = max(0, 450 - total_overs_bowled)

            # Build full match state
            # Determine first team (who bats innings 1 and 3)
            first_team_name = test_data['states'][0].get('battingTeam', 'England')

            # Helper to get innings state
            def get_innings_state(inn_num):
                if inn_num in innings_final_states:
                    # Completed innings - use final state
                    return innings_final_states[inn_num]['score'], innings_final_states[inn_num]['wickets']
                elif inn_num == state['innings']:
                    # Current innings - use current state
                    return state['runsFor'], state['wicketsDown']
                else:
                    # Future innings - hasn't happened yet
                    return 0, 0

            score_inn1, wickets_inn1 = get_innings_state(1)
            score_inn2, wickets_inn2 = get_innings_state(2)
            score_inn3, wickets_inn3 = get_innings_state(3)
            score_inn4, wickets_inn4 = get_innings_state(4)

            full_match_state = {
                'current_innings': state['innings'],
                'first_team_score_inn1': score_inn1,
                'first_team_wickets_inn1': wickets_inn1,
                'second_team_score_inn2': score_inn2,
                'second_team_wickets_inn2': wickets_inn2,
                'first_team_score_inn3': score_inn3,
                'first_team_wickets_inn3': wickets_inn3,
                'second_team_score_inn4': score_inn4,
                'second_team_wickets_inn4': wickets_inn4
            }

            # Determine toss winner and ratings for this match
            # Perth: England won toss (batted first), Brisbane: England won toss (batted first)
            # Adelaide: Australia won toss (batted first)
            # In all three matches, the team that batted first won the toss
            first_team_won_toss = 1

            # Set ratings based on which team bats first
            # England rating: 1593.1, Australia rating: 1757.3
            if first_team_name == 'England':
                first_team_rating = 1593.1
                second_team_rating = 1757.3
            else:  # Australia bats first
                first_team_rating = 1757.3
                second_team_rating = 1593.1

            # Check if target has been reached in innings 4
            target_reached = False
            if state['innings'] == 4 and score_inn4 > 0:
                target = (score_inn1 + score_inn3) - score_inn2 + 1
                runs_to_win = target - score_inn4
                if runs_to_win <= 0:
                    # Target reached, match is over - use final probabilities
                    target_reached = True

            if target_reached:
                # Match ended - use final result probabilities
                if 'Australia won' in test_data['result']:
                    probs_full = {'pWin': 1.0, 'pDraw': 0.0, 'pLoss': 0.0}
                    probs_scorecard = {'pWin': 1.0, 'pDraw': 0.0, 'pLoss': 0.0}
                elif 'England won' in test_data['result']:
                    probs_full = {'pWin': 0.0, 'pDraw': 0.0, 'pLoss': 1.0}
                    probs_scorecard = {'pWin': 0.0, 'pDraw': 0.0, 'pLoss': 1.0}
                else:
                    # Shouldn't happen, but use actual prediction
                    probs_full = predict_probabilities(full_match_state, overs_left, first_team=first_team_name,
                                                      home_team='Australia', first_team_rating=first_team_rating,
                                                      second_team_rating=second_team_rating,
                                                      first_team_won_toss=first_team_won_toss,
                                                      use_scorecard_only=False)
                    probs_scorecard = predict_probabilities(full_match_state, overs_left, first_team=first_team_name,
                                                           home_team='Australia', first_team_rating=first_team_rating,
                                                           second_team_rating=second_team_rating,
                                                           first_team_won_toss=first_team_won_toss,
                                                           use_scorecard_only=True)
            else:
                # Match still in progress - predict from both models
                probs_full = predict_probabilities(full_match_state, overs_left, first_team=first_team_name,
                                                  home_team='Australia', first_team_rating=first_team_rating,
                                                  second_team_rating=second_team_rating,
                                                  first_team_won_toss=first_team_won_toss,
                                                  use_scorecard_only=False)
                probs_scorecard = predict_probabilities(full_match_state, overs_left, first_team=first_team_name,
                                                       home_team='Australia', first_team_rating=first_team_rating,
                                                       second_team_rating=second_team_rating,
                                                       first_team_won_toss=first_team_won_toss,
                                                       use_scorecard_only=True)

            # xOver = cumulative overs from previous innings + current over
            xOver = cumulative_overs + state['over']
            prob_points.append({
                'xOver': xOver,
                'innings': state['innings'],
                'over': state['over'],
                'score': f"{state['runsFor']}/{state['wicketsDown']}",
                **probs_full,  # Full model predictions (default)
                'pWin_scorecard': probs_scorecard['pWin'],
                'pDraw_scorecard': probs_scorecard['pDraw'],
                'pLoss_scorecard': probs_scorecard['pLoss']
            })

        last_over = cumulative_overs + max(s['over'] for s in test_data['states']
                                           if s['innings'] == prev_innings)

        # If match is complete, extend to 450 overs showing final result
        if 'Australia won' in test_data['result']:
            # Australia won - show green at 100%, red at 0%
            final_probs = {'pWin': 1.0, 'pDraw': 0.0, 'pLoss': 0.0,
                          'pWin_scorecard': 1.0, 'pDraw_scorecard': 0.0, 'pLoss_scorecard': 0.0}
        elif 'England won' in test_data['result']:
            # England won - show red at 100%, green at 0%
            final_probs = {'pWin': 0.0, 'pDraw': 0.0, 'pLoss': 1.0,
                          'pWin_scorecard': 0.0, 'pDraw_scorecard': 0.0, 'pLoss_scorecard': 1.0}
        elif 'draw' in test_data['result'].lower():
            # Draw - show grey at 100%
            final_probs = {'pWin': 0.0, 'pDraw': 1.0, 'pLoss': 0.0,
                          'pWin_scorecard': 0.0, 'pDraw_scorecard': 1.0, 'pLoss_scorecard': 0.0}
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

        # Use manually specified wicket falls (more realistic clumping than auto-generated from 5-over buckets)
        wicket_falls = test_data.get('wicket_falls_manual', [])

        # Add score labels by finding nearest state
        for wicket in wicket_falls:
            # Find the state closest to this wicket's xOver (cumulative over already set)
            innings_start_over = cumulative_overs_at_innings_start(test_data['states'], wicket['innings'])
            closest_state = min(
                [s for s in test_data['states'] if s['innings'] == wicket['innings']],
                key=lambda s: abs((innings_start_over + s['over']) - wicket['xOver']),
                default=None
            )
            if closest_state:
                # Estimate runs at wicket fall (interpolate if needed)
                wicket['score'] = f"~{closest_state['runsFor']}/{wicket['wickets']}"
            else:
                wicket['score'] = f"?/{wicket['wickets']}"

        # Find match end over (last real state before "Match Complete" extension)
        # Only set for completed matches, not matches in progress
        if 'won' in test_data['result'].lower():
            real_states = [p for p in prob_points if p['score'] != 'Match Complete']
            if real_states:
                match_end_over = real_states[-1]['xOver']
                test_data['matchEndOver'] = match_end_over
                # Filter out wickets that occurred after match ended
                wicket_falls = [w for w in wicket_falls if w['xOver'] <= match_end_over]

        test_data['wicketFalls'] = wicket_falls

    # Save all four tests
    output = {
        'series': 'The Ashes 2025-26',
        'tests': [melbourne, adelaide, brisbane, perth]  # Melbourne first (in progress)
    }

    output_path = Path(__file__).parent.parent.parent / "ui" / "src" / "data" / "ashes-series-2025.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Generated Ashes series worm data")
    print(f"  Perth: {len(perth['probabilities'])} points ({perth['days']} days)")
    print(f"  Brisbane: {len(brisbane['probabilities'])} points ({brisbane['days']} days)")
    print(f"  Adelaide: {len(adelaide['probabilities'])} points ({adelaide['days']} days)")
    print(f"  Melbourne: {len(melbourne['probabilities'])} points ({melbourne['days']} day)")
    print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
