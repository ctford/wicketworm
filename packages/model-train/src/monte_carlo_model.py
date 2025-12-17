#!/usr/bin/env python3
"""
Monte Carlo simulation model for cricket match prediction
Based on probability distributions of wicket partnerships
"""

import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass

import json
from parse_cricsheet import load_all_matches


@dataclass
class PartnershipStatistics:
    """Statistics for a wicket partnership (e.g., 1st wicket, 2nd wicket, ...)"""
    overs_distribution: List[float]  # Distribution of partnership duration in overs (0-100)
    runs_distribution: List[float]  # Distribution of runs scored in partnership (0-300)


class MonteCarloPredictor:
    """Predicts match outcomes using Monte Carlo simulation of wicket partnerships"""

    def __init__(self):
        # Statistics indexed by wicket number (1-10)
        # wicket 1 = opening partnership, wicket 10 = last wicket
        self.partnership_stats: Dict[int, PartnershipStatistics] = {}

    def learn_from_data(self, data_dir: Path):
        """Learn partnership distributions from historical Cricsheet data"""
        print("Learning partnership distributions from historical Cricsheet data...")

        # partnership_data[wicket_num] = {'runs': [...], 'overs': [...]}
        partnership_data = defaultdict(lambda: {'runs': [], 'overs': []})

        json_files = sorted(data_dir.glob('*.json'))
        print(f"Analyzing {len(json_files)} match files...")

        for file_idx, json_file in enumerate(json_files):
            if file_idx % 100 == 0:
                print(f"Processing match {file_idx+1}/{len(json_files)}...")

            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Process each innings
                for innings_info in data.get('innings', []):
                    # Track partnerships: wicket_num -> (runs, overs)
                    wicket_falls = []  # List of (wicket_num, runs, balls)

                    total_runs = 0
                    total_balls = 0
                    wickets_down = 0

                    for over_data in innings_info.get('overs', []):
                        for delivery in over_data.get('deliveries', []):
                            total_balls += 1
                            total_runs += delivery['runs']['total']

                            if 'wickets' in delivery:
                                wickets_down += len(delivery['wickets'])
                                # Record wicket fall
                                wicket_falls.append((wickets_down, total_runs, total_balls))

                                if wickets_down >= 10:
                                    break  # Innings over

                        if wickets_down >= 10:
                            break

                    # Calculate partnership statistics from wicket falls
                    prev_runs = 0
                    prev_balls = 0

                    for wicket_num, runs_at_fall, balls_at_fall in wicket_falls:
                        partnership_runs = runs_at_fall - prev_runs
                        partnership_balls = balls_at_fall - prev_balls
                        partnership_overs = partnership_balls / 6.0

                        if 1 <= wicket_num <= 10:
                            partnership_data[wicket_num]['runs'].append(partnership_runs)
                            partnership_data[wicket_num]['overs'].append(partnership_overs)

                        prev_runs = runs_at_fall
                        prev_balls = balls_at_fall

            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                continue

        # Build probability distributions from collected data
        print("\nBuilding partnership probability distributions...")
        for wicket_num in range(1, 11):
            if wicket_num not in partnership_data or len(partnership_data[wicket_num]['runs']) == 0:
                # No data - use defaults
                overs_dist = [1.0 / 101] * 101  # 0-100 overs
                runs_dist = [1.0 / 301] * 301   # 0-300 runs
                print(f"  Wicket {wicket_num}: No data, using uniform distribution")
            else:
                # Build distributions from data
                runs_data = partnership_data[wicket_num]['runs']
                overs_data = partnership_data[wicket_num]['overs']

                # Create histograms
                # Overs: 0-100 (cap at 100)
                overs_counts = defaultdict(int)
                for overs in overs_data:
                    overs_capped = int(min(100, max(0, overs)))
                    overs_counts[overs_capped] += 1

                total_overs = sum(overs_counts.values())
                overs_dist = []
                for i in range(101):
                    prob = overs_counts[i] / total_overs if total_overs > 0 else 0
                    overs_dist.append(prob)

                # Smooth the distribution (add small epsilon to avoid zero probabilities)
                overs_dist = [(p + 0.0001) for p in overs_dist]
                overs_sum = sum(overs_dist)
                overs_dist = [p / overs_sum for p in overs_dist]

                # Runs: 0-300 (cap at 300)
                runs_counts = defaultdict(int)
                for runs in runs_data:
                    runs_capped = int(min(300, max(0, runs)))
                    runs_counts[runs_capped] += 1

                total_runs = sum(runs_counts.values())
                runs_dist = []
                for i in range(301):
                    prob = runs_counts[i] / total_runs if total_runs > 0 else 0
                    runs_dist.append(prob)

                # Smooth the distribution
                runs_dist = [(p + 0.0001) for p in runs_dist]
                runs_sum = sum(runs_dist)
                runs_dist = [p / runs_sum for p in runs_dist]

                self.partnership_stats[wicket_num] = PartnershipStatistics(
                    overs_distribution=overs_dist,
                    runs_distribution=runs_dist
                )

                avg_overs = np.dot(overs_dist, range(101))
                avg_runs = np.dot(runs_dist, range(301))
                print(f"  Wicket {wicket_num}: avg {avg_runs:.1f} runs in {avg_overs:.1f} overs "
                      f"({len(runs_data)} samples)")

    def simulate_innings(self, wickets_start: int, overs_available: float) -> Tuple[int, float]:
        """
        Simulate a single innings

        Args:
            wickets_start: Number of wickets available at start
            overs_available: Overs available for this innings

        Returns:
            (total_runs, overs_used)
        """
        total_runs = 0
        overs_used = 0.0
        wickets_remaining = wickets_start

        # Simulate each wicket partnership
        for wicket_num in range(1, wickets_start + 1):
            if overs_used >= overs_available:
                break  # Out of time

            if wicket_num in self.partnership_stats:
                stats = self.partnership_stats[wicket_num]
                # Sample partnership duration and runs
                partnership_overs = np.random.choice(101, p=stats.overs_distribution)
                partnership_runs = np.random.choice(301, p=stats.runs_distribution)
            else:
                # Defaults if no data
                partnership_overs = np.random.exponential(10)  # avg 10 overs
                partnership_runs = np.random.poisson(30)  # avg 30 runs

            # Cap by available overs
            actual_overs = min(partnership_overs, overs_available - overs_used)

            # Scale runs proportionally if we run out of time
            if actual_overs < partnership_overs and partnership_overs > 0:
                actual_runs = partnership_runs * (actual_overs / partnership_overs)
            else:
                actual_runs = partnership_runs

            total_runs += int(actual_runs)
            overs_used += actual_overs

            if overs_used >= overs_available:
                break  # Innings ended due to time

        return total_runs, overs_used

    def simulate_match(self, overs_left: float, first_wickets: int, second_wickets: int,
                      lead: int, n_simulations: int = 1000) -> Tuple[float, float, float]:
        """
        Simulate match from current state

        Returns: (p_win, p_draw, p_loss) from first team's perspective
        """
        # Handle terminal states deterministically
        if first_wickets == 0 and second_wickets == 0:
            # Both all out, outcome determined by lead
            if lead > 0:
                return (1.0, 0.0, 0.0)  # First team wins
            elif lead < 0:
                return (0.0, 0.0, 1.0)  # First team loses
            else:
                return (0.0, 1.0, 0.0)  # Draw

        outcomes = {'win': 0, 'draw': 0, 'loss': 0}

        for _ in range(n_simulations):
            # Simulate remaining match
            # This is simplified - we simulate as if both teams get to bat with remaining wickets
            # In reality, Test cricket structure is more complex

            sim_overs_left = overs_left
            sim_lead = lead

            # Determine who bats next based on current state
            # Heuristic: team with more wickets hasn't batted yet or is currently batting
            if first_wickets > second_wickets:
                # First team likely batting or yet to bat
                first_runs, first_overs = self.simulate_innings(first_wickets, sim_overs_left / 2)
                sim_lead += first_runs
                sim_overs_left -= first_overs

                if sim_overs_left > 0 and second_wickets > 0:
                    second_runs, second_overs = self.simulate_innings(second_wickets, sim_overs_left)
                    sim_lead -= second_runs
            else:
                # Second team likely batting or yet to bat
                second_runs, second_overs = self.simulate_innings(second_wickets, sim_overs_left / 2)
                sim_lead -= second_runs
                sim_overs_left -= second_overs

                if sim_overs_left > 0 and first_wickets > 0:
                    first_runs, first_overs = self.simulate_innings(first_wickets, sim_overs_left)
                    sim_lead += first_runs

            # Determine outcome
            if sim_lead > 0:
                outcomes['win'] += 1
            elif sim_lead < 0:
                outcomes['loss'] += 1
            else:
                outcomes['draw'] += 1

        # Convert to probabilities
        total = sum(outcomes.values())
        return (
            outcomes['win'] / total,
            outcomes['draw'] / total,
            outcomes['loss'] / total
        )

    def save(self, path: Path):
        """Save learned distributions"""
        # Convert dataclasses to dictionaries for safe pickling
        data_to_save = {}
        for wicket_num, stats in self.partnership_stats.items():
            data_to_save[wicket_num] = {
                'overs_distribution': stats.overs_distribution,
                'runs_distribution': stats.runs_distribution
            }

        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"✓ Saved Monte Carlo model to {path}")

    def load(self, path: Path):
        """Load learned distributions"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Convert dictionaries back to dataclasses
        self.partnership_stats = {}
        for wicket_num, stats_dict in data.items():
            self.partnership_stats[wicket_num] = PartnershipStatistics(
                overs_distribution=stats_dict['overs_distribution'],
                runs_distribution=stats_dict['runs_distribution']
            )
        print(f"✓ Loaded Monte Carlo model from {path}")


if __name__ == "__main__":
    # Build and save Monte Carlo model
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    predictor = MonteCarloPredictor()
    predictor.learn_from_data(data_dir)

    # Save model
    predictor.save(output_dir / "monte_carlo_model.pkl")

    print("\n✓ Monte Carlo model training complete!")
