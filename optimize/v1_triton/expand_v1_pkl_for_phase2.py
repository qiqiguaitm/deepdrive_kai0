"""Expand V1 pkl's language_embeds slot for Phase 2 state encoding.

V1 (discrete_state_input=False) sizes max_prompt_len = len(ckpt['language_embeds']).
The default convert_kai0_to_v1.py bakes a 7-token prompt; Phase 2 needs ~50-60
tokens (task + 14 state ints + separators).

This helper resizes language_embeds in-place to a chosen N (default 200) by
padding with zeros. Content of those slots doesn't matter because Phase 2's
`v1_forward_with_state` overwrites encoder_x[start:start+plen] directly with
state-conditioned embeds (see serve_policy_v1.py:v1_forward_with_state).

Side effect: pkl size grows by (N - 7) × 2048 × 2 bytes ≈ 800 KB per 200 rows.
For 6.3 GB pkl this is negligible.

Usage:
    python optimize/v1_triton/expand_v1_pkl_for_phase2.py \\
        --in  optimize/results/task_a_mix_b6000_p1200_v1.pkl \\
        --out optimize/results/task_a_mix_b6000_p1200_v1_p200.pkl \\
        --max-prompt-len 200
"""
import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", required=True, help="input V1 pkl")
    parser.add_argument("--out", required=True, help="output V1 pkl (with expanded language_embeds)")
    parser.add_argument(
        "--max-prompt-len", type=int, default=200,
        help="target language_embeds row count (default 200, plenty for kai0 14-DoF state)",
    )
    parser.add_argument("--inplace", action="store_true",
                        help="overwrite input file instead of writing new (uses --out as temp)")
    args = parser.parse_args()

    inp = Path(args.inp).resolve()
    out = Path(args.out).resolve()
    if not inp.exists():
        print(f"[FAIL] Input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {inp} ({inp.stat().st_size/1e9:.2f} GB) ...")
    with open(inp, "rb") as f:
        ckpt = pickle.load(f)

    le = ckpt.get("language_embeds")
    if le is None:
        print("[FAIL] pkl missing 'language_embeds' key", file=sys.stderr)
        sys.exit(1)
    cur_len, dim = le.shape
    print(f"  current language_embeds: {tuple(le.shape)} ({le.dtype})")

    if cur_len >= args.max_prompt_len:
        print(f"  already ≥ {args.max_prompt_len}; nothing to do.")
        if not args.inplace and out != inp:
            print(f"  copying {inp} → {out}")
            shutil.copy2(inp, out)
        return

    # Pad to max_prompt_len with zeros. Phase 2 overwrites encoder_x directly,
    # so the content doesn't matter — only the size determines max_prompt_len.
    pad = torch.zeros(args.max_prompt_len - cur_len, dim, dtype=le.dtype)
    new_le = torch.cat([le, pad], dim=0)
    ckpt["language_embeds"] = new_le
    print(f"  new language_embeds:     {tuple(new_le.shape)} (added {args.max_prompt_len-cur_len} zero rows)")

    # Write (use temp + rename to avoid corrupting on crash)
    tmp = out.with_suffix(out.suffix + ".tmp")
    print(f"Saving to {tmp} ...")
    os.makedirs(out.parent, exist_ok=True)
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp, out)
    size_gb = out.stat().st_size / 1e9
    print(f"OK: {out} ({size_gb:.2f} GB)")

    if args.inplace and out != inp:
        print(f"  --inplace: replacing {inp} ...")
        os.replace(out, inp)
        print(f"OK: {inp} updated in-place")


if __name__ == "__main__":
    main()
