"""
Image-only baseline for ArGuard 2026 Task A (Task 1) / Track A (both subtasks).

Reads JSONL splits produced by ``data/download_data.py``. Images live in
``data/img/<id>``; the relative ``image_path`` field of each record is
resolved against the ``--data-dir`` argument.

Usage
-----
    python baselines/train_image.py --subtask a1 \
        --model google/vit-base-patch16-224 \
        --data-dir data --target dev_test \
        --out predictions/image_a1.tsv --run-id vit_image

    python baselines/train_image.py --subtask a2 \
        --model microsoft/beit-base-patch16-224 \
        --data-dir data --target dev_test \
        --out predictions/image_a2.jsonl --run-id beit_image
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import (  # noqa: E402
    read_jsonl,
    write_multilabel_jsonl,
    write_subtask_a1_tsv,
)
from labels import TaskSpec, get_task  # noqa: E402

log = logging.getLogger("train_image")


def resolve_image(record: dict, data_dir: Path) -> Path:
    rel = record.get("image_path") or ""
    p = Path(rel)
    if p.is_absolute():
        return p
    return data_dir / p


class ImageDataset(Dataset):
    def __init__(self, records: list[dict], processor, spec: TaskSpec, data_dir: Path):
        self.records = records
        self.processor = processor
        self.spec = spec
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        path = resolve_image(rec, self.data_dir)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            log.warning("could not open %s: %s", path, e)
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        px = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        item = {"pixel_values": px}
        if self.spec.is_multilabel:
            target = torch.zeros(self.spec.num_labels, dtype=torch.float32)
            for l in (rec.get("fine_grained_label") or []):
                if l in self.spec.label2id:
                    target[self.spec.label2id[l]] = 1.0
            item["labels"] = target
        else:
            label = rec.get("label")
            item["labels"] = int(-100) if label is None else int(self.spec.label2id[label])
        return item


def collate_single(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


def collate_multi(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


class MultiLabelTrainer(Trainer):
    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        if self._pos_weight is not None:
            pw = self._pos_weight.to(outputs.logits.device)
            loss = nn.BCEWithLogitsLoss(pos_weight=pw)(outputs.logits, labels)
        else:
            loss = nn.BCEWithLogitsLoss()(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_pos_weight(records: list[dict], spec: TaskSpec) -> torch.Tensor:
    counts = np.zeros(spec.num_labels, dtype=np.float64)
    N = max(len(records), 1)
    for r in records:
        for l in (r.get("fine_grained_label") or []):
            if l in spec.label2id:
                counts[spec.label2id[l]] += 1
    return torch.tensor((N - counts) / np.clip(counts, 1.0, None), dtype=torch.float32)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtask", required=True,
                    choices=["a1", "a2", "A1", "A2", "subtask_a1", "subtask_a2"])
    ap.add_argument("--model", required=True, help="HF image classification model id")
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--target", default="dev_test")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", default="image_baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--no-pos-weight", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("results/image"))
    ap.add_argument("--bf16", action="store_true", default=False,
                    help="enable bf16 mixed precision (requires a GPU + driver that supports it)")
    ap.add_argument("--fp16", action="store_true",
                    help="enable fp16 mixed precision")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    set_seed(args.seed)

    spec = get_task(args.subtask)
    splits_dir = args.data_dir / "splits"
    train_records = read_jsonl(splits_dir / "train.jsonl")
    dev_records = read_jsonl(splits_dir / "dev.jsonl")
    target_records = read_jsonl(splits_dir / f"{args.target}.jsonl")
    log.info(
        "records: train=%d  dev=%d  target=%d",
        len(train_records), len(dev_records), len(target_records),
    )

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=spec.num_labels,
        id2label={int(i): n for i, n in spec.id2label.items()},
        label2id=dict(spec.label2id),
        problem_type=spec.problem_type,
        ignore_mismatched_sizes=True,
    )
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False

    train_ds = ImageDataset(train_records, processor, spec, args.data_dir)
    dev_ds = ImageDataset(dev_records, processor, spec, args.data_dir)
    target_ds = ImageDataset(target_records, processor, spec, args.data_dir)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / args.model.replace("/", "__")),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_total_limit=1,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=2,
    )

    if spec.is_multilabel:
        pos_w = None if args.no_pos_weight else compute_pos_weight(train_records, spec)
        trainer = MultiLabelTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=dev_ds,
            data_collator=collate_multi,
            processing_class=processor, pos_weight=pos_w,
        )
    else:
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=dev_ds,
            data_collator=collate_single,
            processing_class=processor,
        )

    trainer.train()

    p = trainer.predict(target_ds)
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    if spec.is_multilabel:
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= args.threshold).astype(int)
        rows = []
        for r, row in zip(target_records, preds):
            rows.append((r["id"], [spec.id2label[i] for i, v in enumerate(row) if v]))
        write_multilabel_jsonl(rows, args.out)
    else:
        pred_ids = np.argmax(logits, axis=1)
        rows = [(r["id"], spec.id2label[int(p)]) for r, p in zip(target_records, pred_ids)]
        write_subtask_a1_tsv(rows, args.out, run_id=args.run_id)

    log.info("wrote predictions -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
