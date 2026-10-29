"""
Direction geometry (WS1.5): raw E/P/A cosine similarities per layer, the
sensitivity of Gram–Schmidt to dimension order (all 3! = 6 orders), and the
order-independent symmetric (Löwdin) alternative.

CPU-only; reads the direction pickles.

Usage::

    python -m examples.act_three.experiments.analysis.direction_geometry
"""

import itertools
import pickle

import numpy as np

from examples.act_three.orthogonalise_directions import lowdin_layer, gram_schmidt_ordered

from .common import ACT_THREE_DIR, DIMS, FIG_DIR, MODEL_LABEL, ensure_dirs, save_json

DIRECTION_FILES = {
    "llama3": ACT_THREE_DIR / "models" / "llama3" / "epa_directions.pkl",
    "ministral": ACT_THREE_DIR / "models" / "ministral" / "epa_directions.pkl",
}


def _unit(v):
    return v / np.linalg.norm(v)


def analyse(model: str) -> dict:
    with open(DIRECTION_FILES[model], "rb") as f:
        saved = pickle.load(f)
    readers = saved["rep_readers"]
    layers = sorted(readers["evaluation"].directions.keys())
    per_layer = []
    for layer in layers:
        V = [np.asarray(readers[d].directions[layer], float).ravel() for d in DIMS]
        U = [_unit(v) for v in V]
        cos = {"EP": float(U[0] @ U[1]), "EA": float(U[0] @ U[2]), "PA": float(U[1] @ U[2])}
        L = lowdin_layer(*V)
        # Gram–Schmidt under all orders; compare each resulting direction with Löwdin's
        gs_dev = {d: [] for d in DIMS}
        for order in itertools.permutations(range(3)):
            G = gram_schmidt_ordered(V, order)
            for k, d in enumerate(DIMS):
                gs_dev[d].append(float(_unit(G[k]) @ _unit(L[k])))
        per_layer.append({
            "layer": int(layer),
            "cos_raw": cos,
            "lowdin_vs_raw": {d: float(_unit(L[k]) @ U[k]) for k, d in enumerate(DIMS)},
            "gs_orders_min_cos_to_lowdin": {d: float(min(v)) for d, v in gs_dev.items()},
            "gs_orders_spread": {d: float(max(v) - min(v)) for d, v in gs_dev.items()},
        })
    abs_cos = np.array([[abs(p["cos_raw"][k]) for k in ("EP", "EA", "PA")] for p in per_layer])
    return {
        "model": model,
        "n_layers": len(layers),
        "mean_abs_cos_raw": {k: float(abs_cos[:, j].mean()) for j, k in enumerate(("EP", "EA", "PA"))},
        "max_abs_cos_raw": {k: float(abs_cos[:, j].max()) for j, k in enumerate(("EP", "EA", "PA"))},
        "min_lowdin_vs_raw": {d: float(min(p["lowdin_vs_raw"][d] for p in per_layer)) for d in DIMS},
        "max_gs_order_spread": {d: float(max(p["gs_orders_spread"][d] for p in per_layer)) for d in DIMS},
        "per_layer": per_layer,
    }


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dirs()
    fig, axes = plt.subplots(1, len(res), figsize=(7.0, 2.2), sharey=True)
    for ax, (m, r) in zip(np.atleast_1d(axes), res.items()):
        L = [p["layer"] for p in r["per_layer"]]
        for k in ("EP", "EA", "PA"):
            ax.plot(L, [p["cos_raw"][k] for p in r["per_layer"]], label=f"cos({k[0]},{k[1]})")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(MODEL_LABEL[m], fontsize=9)
        ax.set_xlabel("Layer (negative index)", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("Raw cosine", fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_direction_cosines.{ext}", dpi=200)
    plt.close(fig)


def run() -> dict:
    res = {m: analyse(m) for m in DIRECTION_FILES if DIRECTION_FILES[m].exists()}
    make_figure(res)
    save_json(res, "direction_geometry.json")
    return res


if __name__ == "__main__":
    for m, r in run().items():
        print(m, {k: v for k, v in r.items() if k != "per_layer"})
