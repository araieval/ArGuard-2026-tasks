"""
Download ArGuard 2026 Task 1 / Track A data from the Hugging Face Hub.

Source repo : https://huggingface.co/datasets/QCRI/ArGuard-Task1
Default output: ``task1/data/``

Output layout
-------------
    task1/data/
    ├── splits/
    │   ├── train.jsonl            # if available on the Hub
    │   ├── dev.jsonl
    │   ├── dev_test.jsonl         # released during the development phase (no labels)
    │   └── test.jsonl             # released during the final evaluation phase
    └── img/
        └── <id>                   # one file per meme (id is the original filename)

Each JSONL record matches the schema used by the baselines, the format
checker, and the scorer:

    {
        "id":                 str,
        "image_path":         "img/<id>",                # relative to this dir
        "text":               str,                       # OCR-extracted Arabic text
        "label":              "Hateful" | "Not Hateful" | null,
        "fine_grained_label": list[str],
        "annotations":        []                         # not distributed publicly
    }

Notes
-----
- The repo on the Hub is private/gated during the development phase.
  Run ``huggingface-cli login`` (or set ``HF_TOKEN``) before invoking
  this script.
- ``dev_test`` is unlabelled (``label = null``, ``fine_grained_label = []``).
  Submit predictions on ``dev_test`` to the development-phase leaderboard.
- ``test`` will be released later for the final evaluation phase.

Usage
-----
    python data/download_data.py                              # all available splits
    python data/download_data.py --splits train dev           # subset of splits
    python data/download_data.py --output-dir /tmp/arguard1   # custom destination
    python data/download_data.py --skip-images                # jsonl only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    from datasets import get_dataset_split_names, load_dataset
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "ERROR: the `datasets` package is required.\n"
        "       pip install datasets huggingface_hub Pillow\n"
    )
    sys.exit(2)

log = logging.getLogger("download_data")

DEFAULT_REPO_ID = "QCRI/ArGuard-Task1"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
CANDIDATE_SPLITS = ("train", "dev", "dev_test", "test")


def save_image(image_obj, dest: Path) -> None:
    """Persist a PIL/HF image to ``dest``; preserve the original encoding when possible."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fmt = (getattr(image_obj, "format", None) or "").upper()
    if fmt in ("JPEG", "JPG"):
        image_obj.save(dest, format="JPEG", quality=95)
    elif fmt == "PNG":
        image_obj.save(dest, format="PNG")
    else:
        suffix = dest.suffix.lower()
        if suffix == ".png":
            image_obj.convert("RGBA" if image_obj.mode == "RGBA" else "RGB").save(dest, format="PNG")
        else:
            image_obj.convert("RGB").save(dest, format="JPEG", quality=95)


def normalize_label(value):
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id")
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="where to write splits/ and img/ (default: this script's directory)",
    )
    p.add_argument(
        "--splits", nargs="+", default=None, choices=list(CANDIDATE_SPLITS),
        help="restrict to a subset of splits (default: every split present on the Hub)",
    )
    p.add_argument("--token", default=None, help="HF token; falls back to HF_TOKEN / cached login")
    p.add_argument("--revision", default=None, help="optional git revision / branch / tag to pin")
    p.add_argument(
        "--skip-images", action="store_true",
        help="do not write image files (jsonl only). image_path is still emitted.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Quiet the noisy probing that the `datasets` / `huggingface_hub` libraries
    # emit at INFO level (404s on legacy loader scripts, 501 from the datasets
    # server, redirect HEAD requests, etc.). These are normal and not errors.
    for noisy in (
        "datasets", "huggingface_hub", "filelock", "urllib3", "fsspec",
        "httpx", "httpcore", "hf_xet",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args()

    splits_dir = args.output_dir / "splits"
    img_dir = args.output_dir / "img"
    splits_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    try:
        available = set(
            get_dataset_split_names(args.repo_id, token=args.token, revision=args.revision)
        )
    except Exception as exc:
        log.error("could not list splits on %s: %s", args.repo_id, exc)
        log.error("If the dataset is gated/private, run `huggingface-cli login` first.")
        return 2

    wanted = list(args.splits) if args.splits else list(CANDIDATE_SPLITS)
    splits_to_pull = [s for s in wanted if s in available]
    missing = [s for s in wanted if s not in available]
    if missing:
        log.warning("requested but not on the Hub: %s", missing)
    if not splits_to_pull:
        log.error("no requested splits are present on %s (available: %s)",
                  args.repo_id, sorted(available))
        return 2

    log.info("pulling splits %s from %s", splits_to_pull, args.repo_id)

    for split in splits_to_pull:
        ds = load_dataset(
            args.repo_id, split=split, token=args.token, revision=args.revision,
        )
        log.info("[%s] %d records", split, len(ds))

        out_path = splits_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for ex in ds:
                rec_id = ex["id"]
                if not args.skip_images:
                    save_image(ex["image"], img_dir / rec_id)

                record = {
                    "id": rec_id,
                    "image_path": f"img/{rec_id}",
                    "text": ex.get("text") or "",
                    "label": normalize_label(ex.get("label")),
                    "fine_grained_label": list(ex.get("fine_grained_label") or []),
                    "annotations": [],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        log.info("[%s] wrote %s", split, out_path)

    log.info("done. images: %s   splits: %s", img_dir, splits_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
