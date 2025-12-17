#!/usr/bin/env python3
"""
Calculate team ratings from historical match results using ELO-style system
"""

from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict
import json


class TeamRatingSystem:
    """
    Simple ELO-style rating system for Test cricket teams

    Starting rating: 1500
    K-factor: 40 (how much ratings change per match)
    """

    def __init__(self, k_factor: float = 40, starting_rating: float = 1500):
        self.k_factor = k_factor
        self.starting_rating = starting_rating
        self.ratings: Dict[str, float] = defaultdict(lambda: starting_rating)
        # Track rating history: {match_id: {team: rating}}
        self.rating_history: Dict[str, Dict[str, float]] = {}

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for team A against team B"""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, team_a: str, team_b: str, result: str, match_id: str) -> Tuple[float, float]:
        """
        Update ratings after a match

        Args:
            team_a: First team name
            team_b: Second team name
            result: 'team_a_win', 'team_b_win', or 'draw'
            match_id: Unique match identifier

        Returns:
            Tuple of (team_a_rating, team_b_rating) before the match
        """
        # Get current ratings (before match)
        rating_a = self.ratings[team_a]
        rating_b = self.ratings[team_b]

        # Store pre-match ratings
        self.rating_history[match_id] = {
            team_a: rating_a,
            team_b: rating_b
        }

        # Calculate expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        # Determine actual scores
        if result == 'team_a_win':
            actual_a, actual_b = 1.0, 0.0
        elif result == 'team_b_win':
            actual_a, actual_b = 0.0, 1.0
        else:  # draw
            actual_a, actual_b = 0.5, 0.5

        # Update ratings
        self.ratings[team_a] = rating_a + self.k_factor * (actual_a - expected_a)
        self.ratings[team_b] = rating_b + self.k_factor * (actual_b - expected_b)

        return rating_a, rating_b

    def get_match_ratings(self, match_id: str, team_a: str, team_b: str) -> Tuple[float, float]:
        """
        Get ratings for a match (before it was played)

        Returns:
            Tuple of (team_a_rating, team_b_rating)
        """
        if match_id in self.rating_history:
            history = self.rating_history[match_id]
            return history.get(team_a, self.starting_rating), history.get(team_b, self.starting_rating)
        else:
            # Match not in history, return current ratings
            return self.ratings[team_a], self.ratings[team_b]


def build_rating_system(data_dir: Path) -> TeamRatingSystem:
    """
    Build rating system from all historical matches in chronological order
    """
    rating_system = TeamRatingSystem()

    # Load all matches sorted by date
    json_files = sorted(data_dir.glob('*.json'))
    matches = []

    for file_path in json_files:
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Extract match info
            info = data.get('info', {})
            dates = info.get('dates', [])
            if not dates:
                continue

            match_date = dates[0]  # First day of match
            teams = list(info.get('players', {}).keys())

            if len(teams) != 2:
                continue

            team_a, team_b = teams[0], teams[1]

            # Determine outcome
            outcome_info = info.get('outcome', {})
            winner = outcome_info.get('winner')

            if winner is None:
                result = 'draw'
            elif winner == team_a:
                result = 'team_a_win'
            elif winner == team_b:
                result = 'team_b_win'
            else:
                continue  # Tie or no result

            matches.append({
                'date': match_date,
                'match_id': file_path.stem,
                'team_a': team_a,
                'team_b': team_b,
                'result': result
            })

        except Exception:
            continue

    # Sort matches by date
    matches.sort(key=lambda x: x['date'])

    # Process matches in chronological order
    for match in matches:
        rating_system.update_ratings(
            match['team_a'],
            match['team_b'],
            match['result'],
            match['match_id']
        )

    return rating_system


if __name__ == "__main__":
    # Test the rating system
    data_dir = Path(__file__).parent.parent / "data"

    print("Building team rating system from historical matches...")
    rating_system = build_rating_system(data_dir)

    print(f"\n✓ Processed {len(rating_system.rating_history)} matches")
    print(f"\nCurrent ratings (top 10):")

    sorted_teams = sorted(rating_system.ratings.items(), key=lambda x: x[1], reverse=True)
    for i, (team, rating) in enumerate(sorted_teams[:10], 1):
        print(f"  {i}. {team}: {rating:.1f}")
