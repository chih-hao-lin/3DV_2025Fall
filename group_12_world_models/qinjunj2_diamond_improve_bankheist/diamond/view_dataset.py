#!/usr/bin/env python3
"""
Script to view collected dataset episodes in human-readable format.
Saves episode data to a text file.

Usage:
    python3 view_dataset.py --summary
    python3 view_dataset.py --dataset outputs/2025-11-19/00-45-27/dataset/train/ --max-steps 30
    python3 view_dataset.py --episodes 0 1 2 --output results/episodes.txt
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from data import Dataset, Episode
except ImportError:
    print("Error: Could not import Diamond modules. Make sure you're in the correct directory.")
    print("Run this script from the diamond project root: python3 view_dataset.py")
    sys.exit(1)

import torch


def view_episode(episode: Episode, episode_id: int, max_steps: int = None) -> str:
    """Convert an episode to human-readable format."""
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"Episode {episode_id}")
    output.append(f"{'='*80}")
    output.append(f"Length: {len(episode)} steps")
    output.append(f"Return: {episode.rew.sum().item():.2f}")
    output.append(f"Info: {episode.info}\n")
    
    # Determine how many steps to display
    num_steps = len(episode) if max_steps == -1 or max_steps is None else min(max_steps, len(episode))
    
    output.append(f"First {num_steps} steps:")
    output.append(f"{'-'*80}")
    output.append(f"{'Step':<6} {'Obs Mean':<12} {'Obs Std':<12} {'Action':<8} {'Reward':<8} {'End':<5} {'Trunc':<5}")
    output.append(f"{'-'*80}")
    
    for step in range(num_steps):
        obs_mean = episode.obs[step].mean().item()
        obs_std = episode.obs[step].std().item()
        action = episode.act[step].item()
        reward = episode.rew[step].item()
        end = episode.end[step].item()
        trunc = episode.trunc[step].item()
        
        output.append(
            f"{step:<6} {obs_mean:<12.4f} {obs_std:<12.4f} {action:<8} {reward:<8.2f} {end:<5} {trunc:<5}"
        )
    
    if num_steps < len(episode):
        output.append(f"... ({len(episode) - num_steps} more steps)")
    
    output.append(f"{'-'*80}\n")
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="View collected dataset episodes in human-readable format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/train",
        help="Path to dataset directory (default: dataset/train)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Specific episode IDs to view (e.g., --episodes 0 5 10 or --episodes 5 6 7)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start episode ID (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End episode ID (inclusive)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max steps to display per episode (default: 20, use -1 for all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print all steps instead of truncating (equivalent to --max-steps -1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save to (default: print to console only)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Only show dataset summary without episode details",
    )
    
    args = parser.parse_args()
    
    # Handle verbose flag
    if args.verbose:
        args.max_steps = -1  # -1 means all steps
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    # Load dataset
    dataset = Dataset(dataset_path, cache_in_ram=False)
    dataset.load_from_default_path()
    
    output_lines = []
    
    # Print summary
    summary = f"\nDataset Summary: {dataset}\n"
    summary += f"Episodes: {dataset.num_episodes}\n"
    summary += f"Total Steps: {dataset.num_steps}\n"
    summary += f"Reward Counts: -1={dataset.counts_rew[0]}, 0={dataset.counts_rew[1]}, +1={dataset.counts_rew[2]}\n"
    summary += f"End Counts: not_ended={dataset.counts_end[0]}, ended={dataset.counts_end[1]}\n"
    print(summary)
    output_lines.append(summary)
    
    if not args.summary:
        # Determine which episodes to display
        if args.episodes:
            episode_ids = args.episodes
        elif args.start is not None or args.end is not None:
            start = args.start if args.start is not None else 0
            end = args.end if args.end is not None else dataset.num_episodes - 1
            episode_ids = range(start, end + 1)
        else:
            episode_ids = range(dataset.num_episodes)
            if dataset.num_episodes > 10:
                print(f"Found {dataset.num_episodes} episodes.")
                print(f"Displaying all episodes. (Use --start and --end to limit range)\n")
        
        # Load and display episodes
        for ep_id in episode_ids:
            if ep_id >= dataset.num_episodes:
                print(f"Warning: Episode {ep_id} does not exist (max: {dataset.num_episodes - 1})")
                continue
            
            episode = dataset.load_episode(ep_id)
            ep_text = view_episode(episode, ep_id, args.max_steps)
            print(ep_text)
            output_lines.append(ep_text)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("".join(output_lines))
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
