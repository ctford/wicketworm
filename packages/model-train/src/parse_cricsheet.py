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
    """Represents the state of a Test match at a specific point"""
    match_id: str
    innings: int
    over: int
    balls_bowled: int
    runs_for: int
    wickets_down: int
    lead: int
    overs_left: float  # Total match overs remaining (450 - total_overs_bowled)
    outcome: str  # "win", "draw", "loss" (from batting team perspective)


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

    # Get teams
    teams = list(data['info']['players'].keys())
    if len(teams) != 2:
        return []  # Invalid match

    team_a, team_b = teams[0], teams[1]

    # Track innings scores and overs
    innings_scores = []
    innings_overs = []  # Track overs bowled in each completed innings

    states = []
    match_id = file_path.stem

    for innings_idx, innings_data in enumerate(data.get('innings', [])):
        innings_num = innings_idx + 1
        batting_team = innings_data.get('team')

        # Determine which team is batting
        is_team_a = batting_team == team_a

        runs = 0
        wickets = 0
        balls = 0

        for over_data in innings_data.get('overs', []):
            over_num = over_data['over']

            for delivery in over_data.get('deliveries', []):
                balls += 1
                runs += delivery['runs']['total']

                if 'wickets' in delivery:
                    wickets += len(delivery['wickets'])

            # Calculate lead after this over
            lead = runs
            for i, prev_score in enumerate(innings_scores):
                if i % 2 == 0:  # Team A innings
                    lead += prev_score if is_team_a else -prev_score
                else:  # Team B innings
                    lead += -prev_score if is_team_a else prev_score

            # Calculate overs_left (match-level time remaining)
            # Total overs = completed innings overs + current over (over_num is 0-indexed, so +1)
            total_overs_bowled = sum(innings_overs) + (over_num + 1)
            overs_left = max(0, max_overs - total_overs_bowled)

            # Determine outcome from this batting team's perspective
            if is_draw:
                outcome = "draw"
            elif winner == batting_team:
                outcome = "win"
            else:
                outcome = "loss"

            # Create game state for this over
            state = GameState(
                match_id=match_id,
                innings=innings_num,
                over=over_num,
                balls_bowled=balls,
                runs_for=runs,
                wickets_down=wickets,
                lead=lead,
                overs_left=overs_left,
                outcome=outcome
            )
            states.append(state)

            # Stop if all out
            if wickets >= 10:
                break

        # Record final innings score and overs
        innings_scores.append(runs)
        # Calculate actual overs bowled (convert balls to overs)
        final_overs = balls / 6.0 if balls > 0 else 0
        innings_overs.append(final_overs)

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
