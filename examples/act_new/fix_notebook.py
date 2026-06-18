"""
Fix layer index mismatch in hyperparameter_tuning.ipynb.

The stored EPA directions use NEGATIVE layer indices (-1..-31),
but the grid search was using POSITIVE indices (6, 8, 10, ...),
causing make_epa_activations to produce zero vectors (no matching keys).

This script patches cells 7 and 11 to:
1. Define LAYER_CANDIDATES with negative indices matching stored directions
2. Convert negative→positive for wrap_block/set_controller calls
"""

import json

with open("hyperparameter_tuning.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Fix Cell 7: Grid Search ──────────────────────────────────────────────
cell7_source = [
    "# === Cell 7: Grid Search over Layers & Coefficients ===\n",
    "#\n",
    "# For each EPA dimension, sweep over layer subsets and coefficient values.\n",
    "# At each setting, apply steering and measure:\n",
    "#   - On-target Spearman correlation (higher = better)\n",
    "#   - Cross-dimension Spearman correlation change (lower = less interference)\n",
    "\n",
    "# --- Layer index conversion ---\n",
    "# Directions are stored with NEGATIVE indices (e.g. -1..-31).\n",
    "# wrap_block/set_controller need POSITIVE indices into model.model.layers.\n",
    "# For an N-layer model, negative index -k = positive index N-k.\n",
    "n_layers = len(model.model.layers)\n",
    "print(f\"Model has {n_layers} layers\")\n",
    "print(f\"Direction keys sample: {sorted(hidden_layers)[:3]}...{sorted(hidden_layers)[-3:]}\")\n",
    "\n",
    "def neg_to_pos(neg_layers):\n",
    "    \"\"\"Convert negative layer indices to positive for wrap_block.\"\"\"\n",
    "    return [n_layers + k for k in neg_layers]\n",
    "\n",
    "# --- Candidate hyperparameters ---\n",
    "# Defined using NEGATIVE indices to match the stored directions.\n",
    "LAYER_CANDIDATES = {\n",
    "    \"middle_every2\":     [k for k in hidden_layers if -26 <= k <= -4 and k % 2 == 0],\n",
    "    \"paper_7b_pattern\":  [k for k in hidden_layers if -24 <= k <= -2 and k % 3 == 0],\n",
    "    \"narrow_middle\":     [k for k in hidden_layers if -22 <= k <= -4 and k % 2 == 0],\n",
    "    \"early_to_mid\":      [k for k in hidden_layers if -26 <= k <= -8 and k % 3 == 0],\n",
    "    \"mid_to_late\":       [k for k in hidden_layers if -20 <= k <= -1 and k % 3 == 0],\n",
    "}\n",
    "COEFF_CANDIDATES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0]\n",
    "\n",
    "# Print the layer sets for verification\n",
    "for name, lays in LAYER_CANDIDATES.items():\n",
    "    pos_equiv = neg_to_pos(lays)\n",
    "    print(f\"  {name:20s}: neg={lays} -> pos={pos_equiv}\")\n",
    "\n",
    "# Storage for results\n",
    "grid_results = {dim: [] for dim in DIMENSIONS}\n",
    "\n",
    "# Wrap the model once\n",
    "wrapped_model = WrappedReadingVecModel(model, tokenizer)\n",
    "\n",
    "total_combos = len(LAYER_CANDIDATES) * len(COEFF_CANDIDATES) * len(DIMENSIONS)\n",
    "print(f\"\\nGrid search: {total_combos} combinations \"\n",
    "      f\"({len(LAYER_CANDIDATES)} layer sets × {len(COEFF_CANDIDATES)} coefficients × {len(DIMENSIONS)} dimensions)\")\n",
    "print(f\"Each combination evaluates {len(utterances)} utterances across 3 dimensions.\")\n",
    "print(\"\")\n",
    "\n",
    "for steer_dim in DIMENSIONS:\n",
    "    print(f\"\\n{'='*60}\")\n",
    "    print(f\"Steering dimension: {steer_dim.upper()}\")\n",
    "    print(f\"{'='*60}\")\n",
    "    \n",
    "    for layer_name, neg_layers in LAYER_CANDIDATES.items():\n",
    "        pos_layers = neg_to_pos(neg_layers)\n",
    "        \n",
    "        for coeff in COEFF_CANDIDATES:\n",
    "            # Build steering activations using NEGATIVE keys\n",
    "            # (these match the keys in rep_readers[dim].directions)\n",
    "            kwargs = {f\"{d[0]}_coeff\": 0.0 for d in DIMENSIONS}\n",
    "            kwargs[f\"{steer_dim[0]}_coeff\"] = coeff\n",
    "            \n",
    "            activations_neg = make_epa_activations(\n",
    "                rep_readers=rep_readers,\n",
    "                layers=neg_layers,\n",
    "                device=model.device,\n",
    "                dtype=model.dtype,\n",
    "                normalize=True,\n",
    "                **kwargs,\n",
    "            )\n",
    "            \n",
    "            # Convert activation dict keys: negative -> positive\n",
    "            activations_pos = {\n",
    "                n_layers + k: v for k, v in activations_neg.items()\n",
    "            }\n",
    "            \n",
    "            # Apply steering using POSITIVE indices\n",
    "            wrapped_model.unwrap()\n",
    "            wrapped_model.wrap_block(pos_layers, block_name=\"decoder_block\")\n",
    "            wrapped_model.set_controller(pos_layers, activations_pos, \"decoder_block\")\n",
    "            \n",
    "            # Compute expected Likert values under steering\n",
    "            steered_expected = {dim: [] for dim in DIMENSIONS}\n",
    "            \n",
    "            for u in utterances:\n",
    "                for dim in DIMENSIONS:\n",
    "                    prompt = make_likert_prompt(u['text'], dim)\n",
    "                    result = compute_likert_distribution(model, tokenizer, prompt, dim)\n",
    "                    steered_expected[dim].append(result['expected_value'])\n",
    "            \n",
    "            # Reset model\n",
    "            wrapped_model.reset()\n",
    "            wrapped_model.unwrap()\n",
    "            \n",
    "            # Compute correlations\n",
    "            result_entry = {\n",
    "                \"steer_dim\": steer_dim,\n",
    "                \"layer_name\": layer_name,\n",
    "                \"layers_neg\": neg_layers,\n",
    "                \"layers_pos\": pos_layers,\n",
    "                \"coefficient\": coeff,\n",
    "                \"correlations\": {},\n",
    "                \"correlation_deltas\": {},\n",
    "            }\n",
    "            \n",
    "            for dim in DIMENSIONS:\n",
    "                rho, pval = spearmanr(steered_expected[dim], ground_truth[dim])\n",
    "                rho_baseline, _ = spearmanr(baseline_expected[dim], ground_truth[dim])\n",
    "                result_entry[\"correlations\"][dim] = float(rho)\n",
    "                result_entry[\"correlation_deltas\"][dim] = float(rho - rho_baseline)\n",
    "            \n",
    "            grid_results[steer_dim].append(result_entry)\n",
    "            \n",
    "            on_target_rho = result_entry['correlations'][steer_dim]\n",
    "            on_target_delta = result_entry['correlation_deltas'][steer_dim]\n",
    "            print(f\"  {layer_name:20s} coeff={coeff:.2f}  \"\n",
    "                  f\"ρ({steer_dim[:4]})={on_target_rho:+.3f} (Δ={on_target_delta:+.3f})\")\n",
    "\n",
    "print(\"\\nGrid search complete.\")",
]

# Find Cell 7 by id
for cell in nb["cells"]:
    if cell.get("id") == "cell-grid-search":
        cell["source"] = cell7_source
        cell["execution_count"] = None
        cell["outputs"] = []
        break

# ── Fix Cell 8: Best params — update layers key references ─────────────
cell8_source = [
    "# === Cell 8: Select Best Hyperparameters ===\n",
    "#\n",
    "# For each dimension, select the (layers, coeff) that maximizes on-target\n",
    "# correlation while penalizing cross-dimension interference.\n",
    "#\n",
    "# Score = on_target_rho - λ * mean(|cross_dim_delta|)\n",
    "# where λ controls the interference penalty weight.\n",
    "\n",
    "INTERFERENCE_PENALTY = 0.5  # weight for cross-dimension interference\n",
    "\n",
    "best_params = {}\n",
    "\n",
    "for steer_dim in DIMENSIONS:\n",
    "    other_dims = [d for d in DIMENSIONS if d != steer_dim]\n",
    "    \n",
    "    best_score = -float('inf')\n",
    "    best_entry = None\n",
    "    \n",
    "    for entry in grid_results[steer_dim]:\n",
    "        on_target = entry['correlations'][steer_dim]\n",
    "        cross_interference = np.mean([\n",
    "            abs(entry['correlation_deltas'][d]) for d in other_dims\n",
    "        ])\n",
    "        score = on_target - INTERFERENCE_PENALTY * cross_interference\n",
    "        \n",
    "        if score > best_score:\n",
    "            best_score = score\n",
    "            best_entry = entry\n",
    "    \n",
    "    best_params[steer_dim] = best_entry\n",
    "    \n",
    "    print(f\"\\n{'='*60}\")\n",
    "    print(f\"Best for {steer_dim.upper()}:\")\n",
    "    print(f\"  Layers: {best_entry['layer_name']} = neg{best_entry['layers_neg']} / pos{best_entry['layers_pos']}\")\n",
    "    print(f\"  Coefficient: {best_entry['coefficient']}\")\n",
    "    print(f\"  On-target ρ: {best_entry['correlations'][steer_dim]:+.3f} \"\n",
    "          f\"(Δ from baseline: {best_entry['correlation_deltas'][steer_dim]:+.3f})\")\n",
    "    print(f\"  Cross-dimension effects:\")\n",
    "    for d in other_dims:\n",
    "        print(f\"    {d:12s}: ρ = {best_entry['correlations'][d]:+.3f} \"\n",
    "              f\"(Δ = {best_entry['correlation_deltas'][d]:+.3f})\")",
]

for cell in nb["cells"]:
    if cell.get("id") == "cell-best-params":
        cell["source"] = cell8_source
        cell["execution_count"] = None
        cell["outputs"] = []
        break

# ── Fix Cell 11: Scatter Plots ────────────────────────────────────────────
cell11_source = [
    "# === Cell 11: Scatter Plots — Best Hyperparameters ===\n",
    "#\n",
    "# For the best hyperparameters, re-run steering and plot\n",
    "# expected Likert value vs ground-truth for each dimension.\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "for idx, steer_dim in enumerate(DIMENSIONS):\n",
    "    best = best_params[steer_dim]\n",
    "    neg_layers = best['layers_neg']\n",
    "    pos_layers = best['layers_pos']\n",
    "    coeff = best['coefficient']\n",
    "    \n",
    "    # Apply steering — activations use NEGATIVE keys, wrap uses POSITIVE\n",
    "    kwargs = {f\"{d[0]}_coeff\": 0.0 for d in DIMENSIONS}\n",
    "    kwargs[f\"{steer_dim[0]}_coeff\"] = coeff\n",
    "    \n",
    "    activations_neg = make_epa_activations(\n",
    "        rep_readers=rep_readers,\n",
    "        layers=neg_layers,\n",
    "        device=model.device,\n",
    "        dtype=model.dtype,\n",
    "        normalize=True,\n",
    "        **kwargs,\n",
    "    )\n",
    "    activations_pos = {\n",
    "        n_layers + k: v for k, v in activations_neg.items()\n",
    "    }\n",
    "    \n",
    "    wrapped_model.unwrap()\n",
    "    wrapped_model.wrap_block(pos_layers, block_name=\"decoder_block\")\n",
    "    wrapped_model.set_controller(pos_layers, activations_pos, \"decoder_block\")\n",
    "    \n",
    "    steered_vals = []\n",
    "    gt_vals = []\n",
    "    for u in tqdm(utterances, desc=f\"Scatter {steer_dim[:4]}\", leave=False):\n",
    "        prompt = make_likert_prompt(u['text'], steer_dim)\n",
    "        result = compute_likert_distribution(model, tokenizer, prompt, steer_dim)\n",
    "        steered_vals.append(result['expected_value'])\n",
    "        gt_vals.append(LIKERT_VALUES[u['likert_targets'][steer_dim]])\n",
    "    \n",
    "    wrapped_model.reset()\n",
    "    wrapped_model.unwrap()\n",
    "    \n",
    "    # Plot\n",
    "    rho, _ = spearmanr(steered_vals, gt_vals)\n",
    "    axes[idx].scatter(gt_vals, steered_vals, alpha=0.5, s=30, c='#2196F3')\n",
    "    axes[idx].plot([1, 7], [1, 7], 'k--', alpha=0.3, label='Perfect')\n",
    "    axes[idx].set_xlabel(\"Ground-Truth Likert\")\n",
    "    axes[idx].set_ylabel(\"Expected Likert (Steered)\")\n",
    "    axes[idx].set_title(\n",
    "        f\"{steer_dim.capitalize()}\\n\"\n",
    "        f\"layers={best['layer_name']}, coeff={coeff}\\n\"\n",
    "        f\"ρ = {rho:+.3f}\"\n",
    "    )\n",
    "    axes[idx].set_xlim(0.5, 7.5)\n",
    "    axes[idx].set_ylim(0.5, 7.5)\n",
    "    axes[idx].set_xticks(range(1, 8))\n",
    "    axes[idx].legend()\n",
    "\n",
    "plt.suptitle(\"Expected Likert Value (Steered) vs Ground-Truth\", fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"scatter_best_params.png\", dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print(\"Scatter plots saved to scatter_best_params.png\")",
]

for cell in nb["cells"]:
    if cell.get("id") == "cell-scatter":
        cell["source"] = cell11_source
        cell["execution_count"] = None
        cell["outputs"] = []
        break

# ── Fix Cell 12: Save Results — update layers key references ─────────
cell12_source = [
    "# === Cell 12: Save Results ===\n",
    "\n",
    "results_output = {\n",
    "    \"metadata\": {\n",
    "        \"model_name\": model_name,\n",
    "        \"tuned_at\": datetime.now().isoformat(),\n",
    "        \"dataset_path\": DATASET_PATH,\n",
    "        \"n_utterances\": len(utterances),\n",
    "        \"interference_penalty\": INTERFERENCE_PENALTY,\n",
    "    },\n",
    "    \"layer_candidates\": {k: v for k, v in LAYER_CANDIDATES.items()},\n",
    "    \"coeff_candidates\": COEFF_CANDIDATES,\n",
    "    \"best_hyperparameters\": {},\n",
    "    \"baseline_correlations\": {},\n",
    "}\n",
    "\n",
    "for dim in DIMENSIONS:\n",
    "    rho_base, _ = spearmanr(baseline_expected[dim], ground_truth[dim])\n",
    "    results_output[\"baseline_correlations\"][dim] = float(rho_base)\n",
    "    \n",
    "    best = best_params[dim]\n",
    "    results_output[\"best_hyperparameters\"][dim] = {\n",
    "        \"layer_name\": best['layer_name'],\n",
    "        \"layers_neg\": best['layers_neg'],\n",
    "        \"layers_pos\": best['layers_pos'],\n",
    "        \"coefficient\": best['coefficient'],\n",
    "        \"on_target_correlation\": best['correlations'][dim],\n",
    "        \"on_target_delta\": best['correlation_deltas'][dim],\n",
    "        \"cross_dim_correlations\": {\n",
    "            d: best['correlations'][d] for d in DIMENSIONS if d != dim\n",
    "        },\n",
    "        \"cross_dim_deltas\": {\n",
    "            d: best['correlation_deltas'][d] for d in DIMENSIONS if d != dim\n",
    "        },\n",
    "    }\n",
    "\n",
    "# Also save full grid results for later analysis\n",
    "results_output[\"full_grid_results\"] = grid_results\n",
    "\n",
    "results_path = \"epa_tuning_results.json\"\n",
    "with open(results_path, 'w') as f:\n",
    "    json.dump(results_output, f, indent=2)\n",
    "\n",
    "print(f\"Results saved to {results_path}\")\n",
    "print(\"\\n\" + \"=\"*60)\n",
    "print(\"SUMMARY — Best Hyperparameters\")\n",
    "print(\"=\"*60)\n",
    "for dim in DIMENSIONS:\n",
    "    b = results_output['best_hyperparameters'][dim]\n",
    "    print(f\"\\n{dim.upper()}:\")\n",
    "    print(f\"  Layers:      {b['layer_name']} = neg{b['layers_neg']} / pos{b['layers_pos']}\")\n",
    "    print(f\"  Coefficient: {b['coefficient']}\")\n",
    "    print(f\"  ρ (target):  {b['on_target_correlation']:+.3f} (Δ={b['on_target_delta']:+.3f})\")",
]

for cell in nb["cells"]:
    if cell.get("id") == "cell-save-results":
        cell["source"] = cell12_source
        cell["execution_count"] = None
        cell["outputs"] = []
        break

# Also clear outputs from cells 1-6 since they have stale data
for cell in nb["cells"]:
    if cell.get("id") in ("cell-setup", "cell-load-model", "cell-generate-dataset",
                           "cell-load-dataset", "cell-likert-logprob", "cell-baseline"):
        cell["execution_count"] = None
        cell["outputs"] = []

with open("hyperparameter_tuning.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("OK - Patched cells 7, 8, 11, 12 with negative-to-positive layer index conversion.")
print("  Cleared all cell outputs. Ready to re-run from Cell 1.")
