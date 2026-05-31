#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


POLICIES = {
    "UD-Q8_K_XL_APPROX": {
        "base": "Q8_0",
        "target_gb": 9.34,
        "rules": [
            {"match": "token_embd", "type": "F16", "reason": "embedding/tool-token sensitivity"},
            {"match": "output", "type": "F16", "reason": "logit head sensitivity"},
            {"match": "attn", "type": "Q8_0", "reason": "attention path retained high precision"},
            {"match": "gate", "type": "Q8_0", "reason": "router/gate routing sensitivity"},
            {"match": "router", "type": "Q8_0", "reason": "router sensitivity"},
            {"match": "norm", "type": "F16", "reason": "normalization stability"},
            {"match": "ffn", "type": "Q8_0", "reason": "expert/base MLP storage"},
        ],
    },
    "UD-Q6_K_XL_APPROX": {
        "base": "Q6_K",
        "target_gb": 7.74,
        "rules": [
            {"match": "token_embd", "type": "Q8_0", "reason": "embedding/tool-token sensitivity"},
            {"match": "output", "type": "Q8_0", "reason": "logit head sensitivity"},
            {"match": "attn_q", "type": "Q8_0", "reason": "tool-token attention sensitivity"},
            {"match": "attn_k", "type": "Q8_0", "reason": "tool-token attention sensitivity"},
            {"match": "attn_v", "type": "Q8_0", "reason": "tool-token attention sensitivity"},
            {"match": "attn_output", "type": "Q8_0", "reason": "attention output sensitivity"},
            {"match": "gate", "type": "Q8_0", "reason": "router/gate routing sensitivity"},
            {"match": "router", "type": "Q8_0", "reason": "router sensitivity"},
            {"match": "norm", "type": "F16", "reason": "normalization stability"},
            {"match": "ffn", "type": "Q6_K", "reason": "default expert/base MLP target"},
        ],
    },
    "UD-Q5_K_XL_APPROX": {
        "base": "Q5_K_M",
        "target_gb": 6.39,
        "rules": [
            {"match": "token_embd", "type": "Q8_0", "reason": "embedding/tool-token sensitivity"},
            {"match": "output", "type": "Q8_0", "reason": "logit head sensitivity"},
            {"match": "attn", "type": "Q6_K", "reason": "attention kept above base"},
            {"match": "gate", "type": "Q8_0", "reason": "router/gate routing sensitivity"},
            {"match": "router", "type": "Q8_0", "reason": "router sensitivity"},
            {"match": "norm", "type": "F16", "reason": "normalization stability"},
            {"match": "ffn", "type": "Q5_K_M", "reason": "default expert/base MLP target"},
        ],
    },
    "UD-Q4_K_XL_APPROX": {
        "base": "Q4_K_M",
        "target_gb": 5.35,
        "rules": [
            {"match": "token_embd", "type": "Q6_K", "reason": "embedding/tool-token sensitivity"},
            {"match": "output", "type": "Q6_K", "reason": "logit head sensitivity"},
            {"match": "attn_q", "type": "Q6_K", "reason": "attention query sensitivity"},
            {"match": "attn_k", "type": "Q6_K", "reason": "attention key sensitivity"},
            {"match": "attn_v", "type": "Q5_K_M", "reason": "attention value above base"},
            {"match": "attn_output", "type": "Q5_K_M", "reason": "attention output above base"},
            {"match": "gate", "type": "Q8_0", "reason": "router/gate routing sensitivity"},
            {"match": "router", "type": "Q8_0", "reason": "router sensitivity"},
            {"match": "norm", "type": "F16", "reason": "normalization stability"},
            {"match": "ffn", "type": "Q4_K_M", "reason": "default expert/base MLP target"},
        ],
    },
}


TENSOR_TYPE_ALIASES = {
    "Q5_K_M": "q5_k",
    "Q4_K_M": "q4_k",
}


def write_tensor_type_file(path: Path, policy: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rule in policy["rules"]:
            # llama.cpp parses this file as whitespace-separated tokens only,
            # so comments cannot be included inline.
            ggml_type = TENSOR_TYPE_ALIASES.get(rule["type"], rule["type"].lower())
            f.write(f"{rule['match']}={ggml_type}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bf16-gguf", type=Path, default=Path("artifacts/gguf/bf16/LFM-2.5-8B-1B-hermes-ft-BF16.gguf"))
    parser.add_argument("--imatrix", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for name, policy in POLICIES.items():
        tensor_file = args.out_dir / f"{name}.tensor-types.txt"
        write_tensor_type_file(tensor_file, policy)
        out_name = f"LFM-2.5-8B-1B-hermes-ft-{name.replace('UD-', '').replace('_APPROX', '_approx')}.gguf"
        cmd = [
            "llama-quantize",
            "--tensor-type-file",
            str(tensor_file),
        ]
        if args.imatrix:
            cmd.extend(["--imatrix", str(args.imatrix)])
        cmd.extend([str(args.bf16_gguf), str(args.out_dir / out_name), policy["base"]])
        commands.append(
            {
                "name": name,
                "base_quant": policy["base"],
                "target_gb": policy["target_gb"],
                "tensor_type_file": str(tensor_file),
                "output": str(args.out_dir / out_name),
                "command": cmd,
                "policy": policy,
            }
        )

    manifest = {
        "note": "Approximate Unsloth Dynamic XL policies for stock llama.cpp. These are not byte-identical Unsloth Dynamic 2.0 quants.",
        "bf16_parent": str(args.bf16_gguf),
        "imatrix": str(args.imatrix) if args.imatrix else None,
        "commands": commands,
    }
    (args.out_dir / "xl_quant_plan.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out_dir / "xl_quant_commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n\n".join(" ".join(item["command"]) for item in commands)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"policies": list(POLICIES), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
