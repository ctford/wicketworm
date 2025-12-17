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
    overs_left: float  # Total match overs remaining (450 - total_overs_bowled)

    # Cumulative team resources (20 wickets each across both innings)
    first_team_wickets_remaining: int   # 20 - (inn1_wickets + inn3_wickets)
    second_team_wickets_remaining: int  # 20 - (inn2_wickets + inn4_wickets)

    # Match position
    first_team_lead: int  # First team's lead (positive) or deficit (negative)

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

            # Calculate wickets remaining (cumulative across both innings)
            first_team_wickets_remaining = 20 - (wickets_inn1 + wickets_inn3)
            second_team_wickets_remaining = 20 - (wickets_inn2 + wickets_inn4)

            # Calculate first team lead
            # Positive = first team ahead, Negative = first team behind
            first_team_lead = (score_inn1 + score_inn3) - (score_inn2 + score_inn4)

            state = GameState(
                match_id=match_id,
                overs_left=overs_left,
                first_team_wickets_remaining=first_team_wickets_remaining,
                second_team_wickets_remaining=second_team_wickets_remaining,
                first_team_lead=first_team_lead,
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
        print(f"  Match {state.match_id}: "
              f"Wickets: {state.first_team_wickets_remaining}/{state.second_team_wickets_remaining}, "
              f"Lead: {state.first_team_lead}, Overs left: {state.overs_left:.1f}, "
              f"Outcome: {state.outcome}")
