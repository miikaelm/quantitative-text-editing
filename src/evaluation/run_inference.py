#!/usr/bin/env python3
"""Run inference over an evaluation JSONL file and save output images.

CLI usage:
    python -m eval.run_inference \
        --input  data/eval_sets/color_test.jsonl \
        --backend qwen \
        --output-dir outputs/qwen_color_test

Programmatic usage:
    from eval.run_inference import run_inference
    from models import get_model

    # Option A: let run_inference load the model
    run_inference("data/color_test.jsonl", "qwen", "outputs/qwen_run1")

    # Option B: pass a pre-loaded model (avoids reloading between runs)
    model = get_model("qwen")(model_dir="/workspace/models")
    model.load()
    run_inference("data/color_test.jsonl", model=model, output_dir="outputs/run1")
    run_inference("data/spatial_test.jsonl", model=model, output_dir="outputs/run2")

Produces:
    outputs/qwen_color_test/
        0000.png
        0001.png
        ...
        manifest.jsonl   <-- maps index → original metadata + paths
"""

import argparse
import json
import logging
import time
from pathlib import Path

from PIL import Image

from models import get_model
from models.base import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> list[dict]:
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
    """Resolve an image path from the JSONL.

    Handles both absolute paths and paths relative to a data root.
    Also normalises Windows-style backslashes.
    """
    path_str = path_str.replace("\\", "/")
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    if data_root:
        resolved = data_root / p
        if resolved.exists():
            return resolved
    # Fall back: try relative to cwd
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Could not resolve image path: {path_str} "
        f"(data_root={data_root})"
    )


def run_inference(
    input_path: str | Path,
    backend: str | BaseModel = "qwen",
    output_dir: str | Path = "outputs",
    *,
    model_dir: str = "/workspace/models",
    device: str = "cuda",
    seed: int = 0,
    data_root: str | Path | None = None,
    **gen_kwargs,
) -> Path:
    """Run inference on a JSONL eval set. Returns path to the manifest.

    Args:
        input_path: Path to JSONL eval file.
        backend: Either a model name string ("qwen", "varedit") or a
                 pre-loaded BaseModel instance. Passing a pre-loaded model
                 avoids reloading weights between multiple calls.
        output_dir: Where to save output images and manifest.
        model_dir: Where cached model weights live (ignored if backend
                   is a pre-loaded model).
        device: Torch device (ignored if backend is pre-loaded).
        seed: Random seed for generation.
        data_root: Root dir for resolving relative image paths in JSONL.
        **gen_kwargs: Passed through to model.generate()
                      (e.g. num_inference_steps, cfg, tau).

    Returns:
        Path to the generated manifest.jsonl.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    data_root = Path(data_root) if data_root else None

    # ---- load dataset ----
    records = load_jsonl(input_path)
    log.info(f"Loaded {len(records)} samples from {input_path}")

    # ---- resolve model ----
    if isinstance(backend, BaseModel):
        model = backend
        model.ensure_loaded()
    else:
        ModelClass = get_model(backend)
        model = ModelClass(model_dir=model_dir, device=device)
        log.info(f"Loading {backend} model...")
        model.load()
        log.info("Model loaded ✓")

    # ---- prepare output dir ----
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_f = open(manifest_path, "w", encoding="utf-8")

    # ---- inference loop ----
    total_time = 0.0
    for idx, record in enumerate(records):
        pair_id = record.get("pair_id", f"sample_{idx:04d}")
        prompt = record["instruction"]
        src_path = resolve_image_path(record["source_image"], data_root)
        source_img = Image.open(src_path).convert("RGB")

        log.info(f"[{idx+1}/{len(records)}] {pair_id}: {prompt[:80]}...")

        t0 = time.time()
        output_img = model.generate(
            image=source_img,
            prompt=prompt,
            seed=seed,
            **gen_kwargs,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        # Save output
        out_filename = f"{idx:04d}.png"
        out_path = output_dir / out_filename
        output_img.save(out_path)

        # Write manifest line (carries forward all original metadata)
        manifest_entry = {
            "index": idx,
            "pair_id": pair_id,
            "output_image": str(out_path),
            "source_image": str(src_path),
            "elapsed_seconds": round(elapsed, 2),
            "edit_type": record.get("edit_type"),
            "instruction": prompt,
            "metadata": record.get("metadata", {}),
        }
        if "target_image" in record:
            try:
                gt_path = resolve_image_path(record["target_image"], data_root)
                manifest_entry["ground_truth"] = str(gt_path)
            except FileNotFoundError:
                log.warning(f"Ground truth not found for {pair_id}")

        manifest_f.write(json.dumps(manifest_entry) + "\n")
        manifest_f.flush()

        log.info(f"  → saved {out_path} ({elapsed:.1f}s)")

    manifest_f.close()

    avg = total_time / len(records) if records else 0
    log.info(
        f"Done. {len(records)} images generated in {total_time:.1f}s "
        f"(avg {avg:.1f}s/image)"
    )
    log.info(f"Manifest: {manifest_path}")
    return manifest_path


def main():
    p = argparse.ArgumentParser(description="Run model inference on eval set")
    p.add_argument("--input", type=Path, required=True, help="JSONL eval file")
    p.add_argument("--backend", type=str, required=True, choices=["qwen", "varedit"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=str, default="/workspace/models")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--cfg", type=float, default=None)
    p.add_argument("--tau", type=float, default=None)
    args = p.parse_args()

    gen_kwargs = {}
    if args.num_inference_steps is not None:
        gen_kwargs["num_inference_steps"] = args.num_inference_steps
    if args.cfg is not None:
        gen_kwargs["cfg"] = args.cfg
    if args.tau is not None:
        gen_kwargs["tau"] = args.tau

    run_inference(
        input_path=args.input,
        backend=args.backend,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        device=args.device,
        seed=args.seed,
        data_root=args.data_root,
        **gen_kwargs,
    )


if __name__ == "__main__":
    main()