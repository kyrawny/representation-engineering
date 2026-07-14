"""
Gram–Schmidt Orthogonalisation for EPA Direction Vectors.

Applies per-layer Gram–Schmidt orthogonalisation to make the E, P, A
direction vectors mutually orthogonal.  This reduces cross-dimension
interference during steering (e.g. Activity steering inadvertently
shifting Evaluation).

Usage::

    python -m examples.act_three.orthogonalise_directions
    python -m examples.act_three.orthogonalise_directions --input custom.pkl --output custom_ortho.pkl

The priority order is E → P → A: Evaluation is kept as-is, Potency is
projected orthogonal to E, and Activity is projected orthogonal to both
E and P.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# =========================================================================
# Gram–Schmidt helpers
# =========================================================================

def gram_schmidt_pair(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Project *v* orthogonal to *u* (unit vector not required)."""
    dot = np.dot(v, u)
    norm_sq = np.dot(u, u)
    if norm_sq < 1e-12:
        return v
    return v - (dot / norm_sq) * u


def orthogonalise_layer(
    e_dir: np.ndarray,
    p_dir: np.ndarray,
    a_dir: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthogonalise three direction vectors using Gram–Schmidt.

    Priority order: E (unchanged) → P ⊥ E → A ⊥ {E, P}.
    """
    e_orth = e_dir.copy()
    p_orth = gram_schmidt_pair(p_dir.copy(), e_orth)
    a_orth = gram_schmidt_pair(a_dir.copy(), e_orth)
    a_orth = gram_schmidt_pair(a_orth, p_orth)
    return e_orth, p_orth, a_orth


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =========================================================================
# Main
# =========================================================================

def orthogonalise_directions(
    rep_readers: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Apply Gram–Schmidt orthogonalisation to EPA directions.

    Modifies the direction arrays in-place and returns the same dict.

    Args:
        rep_readers: Dict mapping dimension name → RepReader.
        verbose: Print per-layer cosine similarity before/after.

    Returns:
        The same ``rep_readers`` dict with orthogonalised directions.
    """
    dim_names = ["evaluation", "potency", "activity"]
    for dim in dim_names:
        if dim not in rep_readers:
            raise KeyError(f"Missing dimension '{dim}' in rep_readers")

    layers = sorted(rep_readers["evaluation"].directions.keys())

    if verbose:
        print(f"Orthogonalising {len(layers)} layers (E → P → A priority)")
        print(f"{'Layer':>6} | {'E·P before':>10} | {'E·A before':>10} | "
              f"{'P·A before':>10} | {'E·P after':>10} | {'E·A after':>10} | "
              f"{'P·A after':>10}")
        print("-" * 82)

    for layer in layers:
        e_dir = np.array(rep_readers["evaluation"].directions[layer]).flatten()
        p_dir = np.array(rep_readers["potency"].directions[layer]).flatten()
        a_dir = np.array(rep_readers["activity"].directions[layer]).flatten()

        # Before
        cos_ep_before = cosine_similarity(e_dir, p_dir)
        cos_ea_before = cosine_similarity(e_dir, a_dir)
        cos_pa_before = cosine_similarity(p_dir, a_dir)

        # Orthogonalise
        e_orth, p_orth, a_orth = orthogonalise_layer(e_dir, p_dir, a_dir)

        # After
        cos_ep_after = cosine_similarity(e_orth, p_orth)
        cos_ea_after = cosine_similarity(e_orth, a_orth)
        cos_pa_after = cosine_similarity(p_orth, a_orth)

        if verbose:
            print(f"{layer:>6} | {cos_ep_before:>10.4f} | {cos_ea_before:>10.4f} | "
                  f"{cos_pa_before:>10.4f} | {cos_ep_after:>10.4f} | "
                  f"{cos_ea_after:>10.4f} | {cos_pa_after:>10.4f}")

        # Write back (reshape to original shape)
        orig_shape = rep_readers["evaluation"].directions[layer].shape
        rep_readers["evaluation"].directions[layer] = e_orth.reshape(orig_shape)
        rep_readers["potency"].directions[layer] = p_orth.reshape(orig_shape)
        rep_readers["activity"].directions[layer] = a_orth.reshape(orig_shape)

    if verbose:
        # Summary statistics
        all_cos = []
        for layer in layers:
            e = np.array(rep_readers["evaluation"].directions[layer]).flatten()
            p = np.array(rep_readers["potency"].directions[layer]).flatten()
            a = np.array(rep_readers["activity"].directions[layer]).flatten()
            all_cos.extend([
                abs(cosine_similarity(e, p)),
                abs(cosine_similarity(e, a)),
                abs(cosine_similarity(p, a)),
            ])
        print(f"\nMean |cosine similarity| after orthogonalisation: "
              f"{np.mean(all_cos):.6f}")
        print(f"Max  |cosine similarity| after orthogonalisation: "
              f"{np.max(all_cos):.6f}")

    return rep_readers


def main():
    parser = argparse.ArgumentParser(
        description="Orthogonalise EPA direction vectors (Gram–Schmidt)")
    parser.add_argument(
        "--input", default=None,
        help="Input directions pickle (default: epa_directions.pkl in act_three/)")
    parser.add_argument(
        "--output", default=None,
        help="Output directions pickle (default: epa_directions_ortho.pkl in act_three/)")
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-layer output")
    args = parser.parse_args()

    # Resolve paths
    act_three_dir = Path(__file__).resolve().parent
    input_path = args.input or str(act_three_dir / "epa_directions.pkl")
    output_path = args.output or str(act_three_dir / "epa_directions_ortho.pkl")

    # Load
    print(f"Loading directions from: {input_path}")
    with open(input_path, "rb") as f:
        saved = pickle.load(f)

    rep_readers = saved["rep_readers"]
    hidden_layers = saved["hidden_layers"]
    model_name = saved.get("model_name", "unknown")

    # Orthogonalise
    orthogonalise_directions(rep_readers, verbose=not args.quiet)

    # Save
    with open(output_path, "wb") as f:
        pickle.dump({
            "rep_readers": rep_readers,
            "hidden_layers": hidden_layers,
            "model_name": model_name,
            "orthogonalised": True,
        }, f)

    print(f"\nOrthogonalised directions saved to: {output_path}")


if __name__ == "__main__":
    main()
