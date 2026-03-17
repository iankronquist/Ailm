#!/usr/bin/env python3
# Written by Claude Opus 4.6
"""Print a table of optimizer moment statistics per layer from a safetensors optimizer state."""

import sys
import mlx.core as mx


def load_opt_state(path: str) -> dict[str, mx.array]:
    return dict(mx.load(path))


def moment_std(tensors: dict[str, mx.array], key: str) -> float | None:
    if key not in tensors:
        return None
    t = tensors[key].astype(mx.float32)
    return t.std().item()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <optimizer.safetensors> [layer_indices...]")
        sys.exit(1)

    path = sys.argv[1]
    tensors = load_opt_state(path)

    # Figure out how many layers exist
    all_layers = set()
    for k in tensors:
        if k.startswith("transformer.h."):
            parts = k.split(".")
            all_layers.add(int(parts[2]))

    if len(sys.argv) > 2:
        layers = [int(x) for x in sys.argv[2:]]
    else:
        layers = sorted(all_layers)

    # Collect per-layer stats
    sublayer_keys = [
        ("attn.c_attn", "attn c_attn"),
        ("attn.c_proj", "attn c_proj"),
        ("mlp.c_fc", "mlp c_fc"),
        ("mlp.c_proj", "mlp c_proj"),
        ("attn_residual_projection", "attn_res_proj"),
        ("mlp_residual_projection", "mlp_res_proj"),
    ]

    # Header
    col_w = 14
    header = f"{'layer':>5}"
    for _, label in sublayer_keys:
        header += f" | {label + ' m':>{col_w}} {label + ' v':>{col_w}}"
    print(header)
    print("-" * len(header))

    for li in layers:
        row = f"{li:>5}"
        for subkey, label in sublayer_keys:
            m_key = f"transformer.h.{li}.{subkey}.weight.m"
            v_key = f"transformer.h.{li}.{subkey}.weight.v"
            m_std = moment_std(tensors, m_key)
            v_std = moment_std(tensors, v_key)
            m_str = f"{m_std:.2e}" if m_std is not None else "N/A"
            v_str = f"{v_std:.2e}" if v_std is not None else "N/A"
            row += f" | {m_str:>{col_w}} {v_str:>{col_w}}"
        print(row)

    # Also show non-layer params
    print()
    print("Non-layer parameters:")
    non_layer = {}
    for k in sorted(tensors):
        if not k.startswith("transformer.h.") and k.endswith(".m"):
            base = k[:-2]
            m_std = moment_std(tensors, k)
            v_std = moment_std(tensors, base + ".v")
            m_str = f"{m_std:.2e}" if m_std is not None else "N/A"
            v_str = f"{v_std:.2e}" if v_std is not None else "N/A"
            print(f"  {base:>45}  m_std={m_str}  v_std={v_str}")

    # Print step and learning rate if present
    if "step" in tensors:
        print(f"\n  optimizer step: {tensors['step'].item()}")
    if "learning_rate" in tensors:
        print(f"  learning rate:  {tensors['learning_rate'].item():.6g}")


if __name__ == "__main__":
    main()
