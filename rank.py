#!/usr/bin/env python3
"""
Main ranking script for Redrob Intelligent Candidate Discovery Challenge.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runs feature extraction, scoring, and generates ranked submission CSV.
All computation happens on CPU without any API calls.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import time
import csv

from src.feature_extractor import FeatureExtractor
from src.scorer import RankingEngine


def load_candidates(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load candidates from JSONL file."""
    candidates = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    candidate = json.loads(line)
                    candidates.append(candidate)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)
    
    return candidates


def extract_all_features(
    candidates: List[Dict[str, Any]],
    extractor: FeatureExtractor
) -> Dict[str, Dict[str, float]]:
    """
    Extract features for all candidates.
    """
    features_dict = {}
    
    print(f"Extracting features for {len(candidates)} candidates...", file=sys.stderr)
    
    for i, candidate in enumerate(candidates):
        candidate_id = candidate.get('candidate_id')
        if not candidate_id:
            continue
        
        try:
            features = extractor.extract_candidate_features(candidate)
            features_dict[candidate_id] = features
        except Exception as e:
            print(f"Warning: Error processing candidate {candidate_id}: {e}", file=sys.stderr)
            continue
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(candidates)}...", file=sys.stderr)
    
    print(f"Successfully extracted features for {len(features_dict)} candidates", file=sys.stderr)
    return features_dict


def rank_all_candidates(
    candidates: List[Dict[str, Any]],
    features_dict: Dict[str, Dict[str, float]],
    engine: RankingEngine
) -> List[Dict[str, Any]]:
    """
    Score and rank all candidates.
    """
    print("Ranking candidates...", file=sys.stderr)
    
    ranked = engine.rank_candidates(candidates, features_dict)
    
    print(f"Ranked {len(ranked)} candidates", file=sys.stderr)
    
    return ranked


def write_submission_csv(
    ranked_candidates: List[Dict[str, Any]],
    output_path: str,
    top_n: int = 100
) -> None:
    """
    Write top N candidates to submission CSV.
    """
    top_candidates = ranked_candidates[:top_n]
    
    print(f"Writing top {len(top_candidates)} candidates to {output_path}...", file=sys.stderr)
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
            
            # Write data rows
            for candidate in top_candidates:
                writer.writerow([
                    candidate['candidate_id'],
                    candidate['rank'],
                    f"{candidate['score']:.4f}",
                    candidate['reasoning'],
                ])
        
        print(f"Successfully wrote submission to {output_path}", file=sys.stderr)
        
        # Print summary stats
        scores = [c['score'] for c in top_candidates]
        print(f"\nSubmission Summary:", file=sys.stderr)
        print(f"  Total candidates: {len(top_candidates)}", file=sys.stderr)
        print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}", file=sys.stderr)
        print(f"  Mean score: {sum(scores)/len(scores):.4f}", file=sys.stderr)
        
    except IOError as e:
        print(f"Error writing to {output_path}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Rank candidates for Senior AI Engineer role'
    )
    parser.add_argument(
        '--candidates',
        required=True,
        help='Path to candidates.jsonl file'
    )
    parser.add_argument(
        '--out',
        default='submission.csv',
        help='Output CSV path (default: submission.csv)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=100,
        help='Number of top candidates to rank (default: 100)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80, file=sys.stderr)
    print("Redrob Intelligent Candidate Discovery & Ranking", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    
    start_time = time.time()
    
    # Load candidates
    print(f"\nLoading candidates from {args.candidates}...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates", file=sys.stderr)
    
    if not candidates:
        print("Error: No candidates loaded", file=sys.stderr)
        sys.exit(1)
    
    # Extract features
    print("\nExtracting features...", file=sys.stderr)
    extractor = FeatureExtractor()
    features_dict = extract_all_features(candidates, extractor)
    
    if not features_dict:
        print("Error: No features extracted", file=sys.stderr)
        sys.exit(1)
    
    # Rank candidates
    print("\nScoring and ranking candidates...", file=sys.stderr)
    engine = RankingEngine()
    ranked = rank_all_candidates(candidates, features_dict, engine)
    
    # Write submission
    print("\nWriting submission...", file=sys.stderr)
    write_submission_csv(ranked, args.out, args.top_n)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds", file=sys.stderr)
    print("=" * 80, file=sys.stderr)


if __name__ == '__main__':
    main()