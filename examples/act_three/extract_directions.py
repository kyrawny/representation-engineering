"""
Standalone EPA Direction Extraction CLI.

Wraps the ``direction_extraction.extract_epa_directions()`` function so
that the full direction extraction pipeline can be run from the command
line instead of requiring the Jupyter notebook.

Usage::

    python -m examples.act_three.extract_directions
    python -m examples.act_three.extract_directions --model meta-llama/Llama-3.1-8B-Instruct \\
        --output epa_directions.pkl --batch-size 4
"""

import argparse
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Extract EPA direction vectors via contrastive PCA")
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path (default: Llama-3.1-8B-Instruct)")
    parser.add_argument(
        "--output", default=None,
        help="Output pickle path (default: epa_directions.pkl in act_three/)")
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for the rep-reading pipeline (default: 8)")
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Maximum sequence length for tokenisation (default: 512)")
    parser.add_argument(
        "--n-train", type=int, default=None,
        help="Number of training pairs per dimension (default: all)")
    parser.add_argument(
        "--orthogonalise", action="store_true",
        help="Apply Gram–Schmidt orthogonalisation after extraction")
    args = parser.parse_args()

    # Resolve output path
    act_three_dir = Path(__file__).resolve().parent
    repo_root = act_three_dir.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    output_path = args.output
    if output_path is None:
        if args.orthogonalise:
            output_path = str(act_three_dir / "epa_directions_ortho.pkl")
        else:
            output_path = str(act_three_dir / "epa_directions.pkl")

    # ---- Load model ----
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # ---- Create extraction datasets ----
    print("Creating EPA extraction datasets...")
    from examples.act_three.dataset import create_all_epa_datasets

    # Resolve data directory (data/act/ relative to repo root)
    data_dir = str(repo_root / "data" / "act")
    n_train = args.n_train or 256

    datasets = create_all_epa_datasets(
        data_dir=data_dir,
        n_train=n_train,
    )

    for dim, data in datasets.items():
        n = len(data["train"]["data"])
        print(f"  {dim}: {n} training pairs")

    # ---- Extract directions ----
    print("Extracting EPA directions via contrastive PCA...")
    t0 = time.time()

    from examples.act_three.direction_extraction import (
        extract_epa_directions,
        save_directions,
    )

    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))

    rep_readers = extract_epa_directions(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        hidden_layers=hidden_layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    elapsed = time.time() - t0
    print(f"Direction extraction complete in {elapsed:.1f}s")

    # ---- Optional orthogonalisation ----
    if args.orthogonalise:
        print("\nApplying Gram–Schmidt orthogonalisation...")
        from examples.act_three.orthogonalise_directions import (
            orthogonalise_directions,
        )
        orthogonalise_directions(rep_readers, verbose=True)

    # ---- Save ----
    from examples.act_three.direction_extraction import save_directions
    # If orthogonalised, save with the flag
    if args.orthogonalise:
        import pickle
        with open(output_path, "wb") as f:
            pickle.dump({
                "rep_readers": rep_readers,
                "hidden_layers": hidden_layers,
                "model_name": args.model,
                "orthogonalised": True,
            }, f)
    else:
        save_directions(rep_readers, hidden_layers, args.model, output_path)
    print(f"\nDirections saved to: {output_path}")

    # ---- Quick validation ----
    print("\n--- Quick Validation ---")
    for dim in ["evaluation", "potency", "activity"]:
        reader = rep_readers[dim]
        n_layers = len(reader.directions)
        sample_layer = list(reader.directions.keys())[0]
        shape = reader.directions[sample_layer].shape
        print(f"  {dim}: {n_layers} layers, direction shape {shape}")


if __name__ == "__main__":
    main()
