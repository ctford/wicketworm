#!/usr/bin/env python3
"""
Parse Cricsheet JSON files and extract game states
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class GameState:
    """Represents the full state of a Test match at a specific point"""
    match_id: str
    current_innings: int
    over: int
    overs_left: float  # Total match overs remaining (450 - total_overs_bowled)

    # First team (bats innings 1 and 3)
    first_team_score_inn1: int
    first_team_wickets_inn1: int
    first_team_score_inn3: int
    first_team_wickets_inn3: int

    # Second team (bats innings 2 and 4)
    second_team_score_inn2: int
    second_team_wickets_inn2: int
    second_team_score_inn4: int
    second_team_wickets_inn4: int

    current_lead: int  # First team's lead (positive) or deficit (negative)
    runs_to_win: int   # In innings 4 chase: runs still needed to win (0 if not chasing)

    outcome: str  # "win", "draw", "loss" (from first team's perspective)


def parse_match(file_path: Path, max_overs: int = 450) -> List[GameState]:
    """
    Parse a single Cricsheet JSON file and extract game states per over

    Args:
        file_path: Path to Cricsheet JSON file
        max_overs: Maximum overs for match (default 450 for 5-day Test)
    """
    with open(file_path) as f:
        data = json.load(f)

    # Get outcome
    outcome_info = data['info'].get('outcome', {})
    winner = outcome_info.get('winner')

    # Determine if match was drawn
    is_draw = winner is None

    # Get teams - first team in list bats innings 1 and 3
    teams = list(data['info']['players'].keys())
    if len(teams) != 2:
        return []  # Invalid match

    first_team = teams[0]

    # Track all innings states
    innings_data_by_num = {}
    innings_overs = []  # Track overs bowled in each completed innings

    states = []
    match_id = file_path.stem

    # First pass: collect all innings data
    for innings_idx, innings_info in enumerate(data.get('innings', [])):
        innings_num = innings_idx + 1
        innings_data_by_num[innings_num] = {
            'team': innings_info.get('team'),
            'states': []  # Will store (over, runs, wickets, balls) tuples
        }

        runs = 0
        wickets = 0
        balls = 0

        for over_data in innings_info.get('overs', []):
            over_num = over_data['over']

            for delivery in over_data.get('deliveries', []):
                balls += 1
                runs += delivery['runs']['total']

                if 'wickets' in delivery:
                    wickets += len(delivery['wickets'])

            innings_data_by_num[innings_num]['states'].append((over_num, runs, wickets, balls))

            # Stop if all out
            if wickets >= 10:
                break

        # Record final overs for this innings
        final_overs = balls / 6.0 if balls > 0 else 0
        innings_overs.append(final_overs)

    # Determine outcome from first team's perspective
    if is_draw:
        outcome_first_team = "draw"
    elif winner == first_team:
        outcome_first_team = "win"
    else:
        outcome_first_team = "loss"

    # Second pass: create GameState for each over with full match context
    cumulative_overs = 0

    for innings_num in sorted(innings_data_by_num.keys()):
        innings_info = innings_data_by_num[innings_num]

        for state_idx, (over_num, runs, wickets, balls) in enumerate(innings_info['states']):
            # Calculate overs_left
            total_overs_bowled = cumulative_overs + (over_num + 1)
            overs_left = max(0, max_overs - total_overs_bowled)

            # Helper function to get innings state
            def get_innings_state(inn_num):
                """Get score/wickets for an innings at this point in time"""
                if inn_num not in innings_data_by_num or not innings_data_by_num[inn_num]['states']:
                    return 0, 0  # Innings hasn't happened

                if inn_num < innings_num:
                    # Past innings - use final state
                    final_state = innings_data_by_num[inn_num]['states'][-1]
                    return final_state[1], final_state[2]  # runs, wickets
                elif inn_num == innings_num:
                    # Current innings - use current state
                    return runs, wickets
                else:
                    # Future innings - hasn't happened yet
                    return 0, 0

            # Get state for all 4 innings
            score_inn1, wickets_inn1 = get_innings_state(1)
            score_inn2, wickets_inn2 = get_innings_state(2)
            score_inn3, wickets_inn3 = get_innings_state(3)
            score_inn4, wickets_inn4 = get_innings_state(4)

            # Calculate current lead: first team total - second team total
            # Positive = first team ahead, Negative = first team behind
            current_lead = (score_inn1 + score_inn3) - (score_inn2 + score_inn4)

            # Calculate runs_to_win for innings 4 chase
            # Standard match: innings 1=first_team, 2=second_team, 3=first_team, 4=second_team chases
            if innings_num == 4 and score_inn4 > 0:
                # second_team needs to overcome the deficit from innings 2
                # Target = (first_team's total) - (second_team's inn2) + 1
                target = (score_inn1 + score_inn3) - score_inn2 + 1
                runs_to_win = target - score_inn4
            else:
                # Not in a chase situation
                runs_to_win = 0

            state = GameState(
                match_id=match_id,
                current_innings=innings_num,
                over=over_num,
                overs_left=overs_left,
                first_team_score_inn1=score_inn1,
                first_team_wickets_inn1=wickets_inn1,
                second_team_score_inn2=score_inn2,
                second_team_wickets_inn2=wickets_inn2,
                first_team_score_inn3=score_inn3,
                first_team_wickets_inn3=wickets_inn3,
                second_team_score_inn4=score_inn4,
                second_team_wickets_inn4=wickets_inn4,
                current_lead=current_lead,
                runs_to_win=runs_to_win,
                outcome=outcome_first_team
            )

            states.append(state)

        cumulative_overs += innings_overs[innings_num - 1]

    return states


def load_all_matches(data_dir: Path, max_matches: int = None) -> List[GameState]:
    """
    Load and parse all Cricsheet JSON files
    """
    json_files = sorted(data_dir.glob('*.json'))

    if max_matches:
        json_files = json_files[:max_matches]

    all_states = []

    for i, file_path in enumerate(json_files):
        if i % 100 == 0:
            print(f"Processing match {i+1}/{len(json_files)}...")

        try:
            states = parse_match(file_path)
            all_states.extend(states)
        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")
            continue

    print(f"\n✓ Loaded {len(all_states)} game states from {len(json_files)} matches")
    return all_states


if __name__ == "__main__":
    # Test parsing
    data_dir = Path(__file__).parent.parent / "data"
    states = load_all_matches(data_dir, max_matches=10)

    print(f"\nSample states:")
    for state in states[:5]:
        print(f"  Match {state.match_id}, Innings {state.innings}, Over {state.over}: "
              f"{state.runs_for}/{state.wickets_down}, Lead: {state.lead}, Outcome: {state.outcome}")
