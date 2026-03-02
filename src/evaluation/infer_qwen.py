"""Run Qwen-Image-Edit inference over an evaluation JSONL file.

Usage:
    python scripts/infer_qwen.py \
        --input  data/eval_sets/color_test.jsonl \
        --output-dir outputs/qwen_color_test \
        --config configs/qwen_eval.yaml

    # With fine-tuned LoRA:
    python scripts/infer_qwen.py \
        --input  data/eval_sets/color_test.jsonl \
        --output-dir outputs/qwen_finetuned_color \
        --config configs/qwen_eval.yaml \
        --lora checkpoints/qwen_lora_ep10.safetensors

Produces:
    outputs/qwen_color_test/
        0000.png
        0001.png
        ...
        manifest.jsonl
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image
from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from qflux.data.config import load_config_from_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"Skipping malformed line {i}: {e}")
    return records


def resolve_image_path(path_str: str, data_root: Path | None) -> Path:
    path_str = path_str.replace("\\", "/")
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    if data_root:
        resolved = data_root / p
        if resolved.exists():
            return resolved
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not resolve image path: {path_str}")


def main():
    p = argparse.ArgumentParser(description="Qwen-Image-Edit inference")
    p.add_argument("--input", type=Path, required=True, help="JSONL eval file")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="qflux YAML config")
    p.add_argument("--lora", type=Path, default=None, help="LoRA weights (.safetensors)")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-inference-steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=4.0, help="true_cfg_scale")
    args = p.parse_args()

    # ---- load dataset ----
    records = load_jsonl(args.input)
    log.info(f"Loaded {len(records)} samples from {args.input}")

    # ---- load model ----
    config = load_config_from_yaml(str(args.config))
    if args.lora is not None:
        config.model.lora.pretrained_weight = str(args.lora)
        log.info(f"Using LoRA weights: {args.lora}")

    trainer = QwenImageEditTrainer(config)
    trainer.setup_predict()
    log.info("Model ready")

    # ---- prepare output ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"

    # ---- inference loop ----
    total_time = 0.0
    with open(manifest_path, "w", encoding="utf-8") as manifest_f:
        for idx, record in enumerate(records):
            pair_id = record.get("pair_id", f"sample_{idx:04d}")
            prompt = record["instruction"]
            src_path = resolve_image_path(record["source_image"], args.data_root)
            source_img = Image.open(src_path).convert("RGB")

            log.info(f"[{idx + 1}/{len(records)}] {pair_id}: {prompt[:80]}...")

            torch.manual_seed(args.seed)

            t0 = time.time()
            results = trainer.predict(
                prompt_image=source_img,
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.cfg,
            )
            elapsed = time.time() - t0
            total_time += elapsed

            out_filename = f"{idx:04d}.png"
            out_path = args.output_dir / out_filename
            results[0].save(out_path)

            # ---- manifest entry (shared schema with VAREdit) ----
            entry = {
                "index": idx,
                "pair_id": pair_id,
                "instruction": prompt,
                "source_image": str(src_path),
                "output_image": str(out_path),
                "elapsed_seconds": round(elapsed, 2),
                "edit_type": record.get("edit_type"),
                "metadata": record.get("metadata", {}),
            }
            if "target_image" in record:
                try:
                    gt_path = resolve_image_path(record["target_image"], args.data_root)
                    entry["ground_truth"] = str(gt_path)
                except FileNotFoundError:
                    log.warning(f"Ground truth not found for {pair_id}")

            manifest_f.write(json.dumps(entry) + "\n")
            manifest_f.flush()

            log.info(f"  → {out_path} ({elapsed:.1f}s)")

    avg = total_time / len(records) if records else 0
    log.info(f"Done. {len(records)} images in {total_time:.1f}s (avg {avg:.1f}s)")
    log.info(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()