#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


STOCK_QUANTS = {
    "Q8_0": "LFM-2.5-8B-1B-hermes-ft-Q8_0.gguf",
    "Q6_K": "LFM-2.5-8B-1B-hermes-ft-Q6_K.gguf",
    "Q5_K_M": "LFM-2.5-8B-1B-hermes-ft-Q5_K_M.gguf",
    "Q4_K_M": "LFM-2.5-8B-1B-hermes-ft-Q4_K_M.gguf",
}


def run(cmd: list[str], log: Path, cwd: Path | None = None, dry_run: bool = False) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        if dry_run:
            return
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
        if proc.returncode:
            raise SystemExit(proc.returncode)


def require(path: Path, label: str) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", type=Path, required=True, help="Fused HF/safetensors source checkpoint.")
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--xl-plan", type=Path, default=None)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-stock", action="store_true")
    parser.add_argument("--skip-xl", action="store_true")
    args = parser.parse_args()

    llama_cpp = args.llama_cpp.resolve()
    converter = require(llama_cpp / "convert_hf_to_gguf.py", "llama.cpp HF converter")
    quantize = llama_cpp / "build" / "bin" / "llama-quantize"
    imatrix = llama_cpp / "build" / "bin" / "llama-imatrix"
    require(quantize, "llama-quantize")
    if args.calibration:
        require(imatrix, "llama-imatrix")
        require(args.calibration, "calibration text")

    out_dir = args.out_dir.resolve()
    bf16_dir = out_dir / "bf16"
    quant_dir = out_dir / "quants"
    log_dir = out_dir / "logs"
    bf16_dir.mkdir(parents=True, exist_ok=True)
    quant_dir.mkdir(parents=True, exist_ok=True)
    bf16_gguf = bf16_dir / "LFM-2.5-8B-1B-hermes-ft-BF16.gguf"

    tokenizer_config = args.hf_model / "tokenizer_config.json"
    tokenizer_meta = {}
    if tokenizer_config.exists():
        tokenizer_meta = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    if tokenizer_meta.get("tool_parser_type") != "pythonic":
        raise SystemExit("Refusing GGUF export: tokenizer_config.json does not preserve tool_parser_type='pythonic'.")

    convert_cmd = [
        "python3",
        str(converter),
        str(args.hf_model),
        "--outfile",
        str(bf16_gguf),
        "--outtype",
        "bf16",
    ]
    run(convert_cmd, log_dir / "convert_bf16.log", dry_run=args.dry_run)

    imatrix_out = out_dir / "calibration" / "hermes_tool_router_imatrix.gguf"
    if args.calibration:
        run(
            [
                str(imatrix),
                "-m",
                str(bf16_gguf),
                "-f",
                str(args.calibration),
                "-o",
                str(imatrix_out),
                "--output-format",
                "gguf",
                "--ctx-size",
                str(args.ctx_size),
                "--chunks",
                "32",
                "--no-ppl",
            ],
            log_dir / "imatrix.log",
            dry_run=args.dry_run,
        )

    stock_commands = []
    if not args.skip_stock:
        for qtype, filename in STOCK_QUANTS.items():
            output = quant_dir / filename
            cmd = [str(quantize), str(bf16_gguf), str(output), qtype]
            stock_commands.append({"qtype": qtype, "output": str(output), "command": cmd})
            run(cmd, log_dir / f"quant_stock_{qtype}.log", dry_run=args.dry_run)

    xl_commands = []
    if args.xl_plan and not args.skip_xl:
        plan = json.loads(args.xl_plan.read_text(encoding="utf-8"))
        for item in plan["commands"]:
            cmd = [str(quantize) if token == "llama-quantize" else token for token in item["command"]]
            cmd = [str(imatrix_out) if token == str(plan.get("imatrix")) else token for token in cmd]
            xl_commands.append({"name": item["name"], "output": item["output"], "command": cmd})
            run(cmd, log_dir / f"quant_{item['name']}.log", dry_run=args.dry_run)

    manifest = {
        "hf_model": str(args.hf_model),
        "llama_cpp": str(llama_cpp),
        "bf16_gguf": str(bf16_gguf),
        "tokenizer_tool_parser_type": tokenizer_meta.get("tool_parser_type"),
        "calibration": str(args.calibration) if args.calibration else None,
        "imatrix": str(imatrix_out) if args.calibration else None,
        "stock_commands": stock_commands,
        "xl_commands": xl_commands,
        "dry_run": args.dry_run,
    }
    (out_dir / "gguf_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run and bf16_gguf.exists():
        shutil.rmtree(bf16_dir, ignore_errors=True)
    print(json.dumps({"manifest": str(out_dir / "gguf_export_manifest.json"), "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
