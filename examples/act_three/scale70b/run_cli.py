"""
Run one of the paper's existing CLIs (direction extraction, orthogonalisation,
reader tuning) unchanged, but with the model loaded according to the
ACT70B_* environment variables (dtype, quantisation, per-GPU memory).

The existing CLIs hard-code ``torch_dtype=float16, device_map="auto"``.  This
wrapper replaces only those loading keyword arguments, so a 70B model can be
spread across several GPUs (or quantised) in exactly the same way in every
stage.  Nothing else in the CLI changes.

Usage::

    python -m examples.act_three.scale70b.run_cli examples.act_three.extract_directions \
        --model meta-llama/Llama-3.1-70B-Instruct
"""

import runpy
import sys

from . import common  # noqa: F401  (sets sys.path, imports torch)
from .common import model_load_kwargs


def _patch_from_pretrained():
    from transformers import AutoModelForCausalLM
    original = AutoModelForCausalLM.from_pretrained.__func__

    def patched(cls, name, *args, **kwargs):
        kwargs.update(model_load_kwargs())
        print(f"[run_cli] loading {name} with "
              f"{ {k: (str(v) if k != 'quantization_config' else 'bnb') for k, v in model_load_kwargs().items()} }")
        return original(cls, name, *args, **kwargs)

    AutoModelForCausalLM.from_pretrained = classmethod(patched)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    _patch_from_pretrained()
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
