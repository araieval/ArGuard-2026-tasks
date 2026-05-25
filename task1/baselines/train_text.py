"""
Text-only baseline for ArGuard 2026 Task 1 / Track A (both subtasks).

Reads JSONL splits produced by ``data/download_data.py``:

    {
        "id": ...,
        "image_path": "img/<id>",
        "text": "...",
        "label": "Hateful" | "Not Hateful" | null,
        "fine_grained_label": [...],
        "annotations": []
    }

Subtask A1 is binary single-label; Subtask A2 is multi-label over the
unified fine-grained taxonomy. Both subtasks train on the full
train/dev splits — no subset filtering.

Usage
-----
    python baselines/train_text.py --subtask a1 \
        --model aubmindlab/bert-base-arabertv02 \
        --data-dir data --target dev_test \
        --out predictions/text_a1.tsv --run-id arabert_text

    python baselines/train_text.py --subtask a2 \
        --model UBC-NLP/MARBERTv2 \
        --data-dir data --target dev_test \
        --out predictions/text_a2.jsonl --run-id marbert_text
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
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
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

log = logging.getLogger("train_text")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, spec: TaskSpec, max_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.spec = spec
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        text = (rec.get("text") or "").strip()
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.spec.is_multilabel:
            target = torch.zeros(self.spec.num_labels, dtype=torch.float32)
            for l in (rec.get("fine_grained_label") or []):
                if l in self.spec.label2id:
                    target[self.spec.label2id[l]] = 1.0
            item["labels"] = target
        else:
            label = rec.get("label")
            if label is None:
                # Allow inference on unlabelled records — store -100 so HF Trainer ignores.
                item["labels"] = int(-100)
            else:
                item["labels"] = int(self.spec.label2id[label])
        return item


def collate_single(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


def collate_multi(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# --------------------------------------------------------------------------
# multilabel trainer with optional pos_weight
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtask", required=True, choices=["a1", "a2", "A1", "A2"])
    ap.add_argument("--model", required=True, help="HF text classification model id")
    ap.add_argument("--data-dir", default="data", type=Path,
                    help="directory containing splits/{train,dev,...}.jsonl")
    ap.add_argument("--target", default="dev_test",
                    help="which split under data/splits/ to predict on")
    ap.add_argument("--out", required=True, type=Path, help="output predictions file")
    ap.add_argument("--run-id", default="text_baseline", help="run identifier (Subtask A1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--no-pos-weight", action="store_true",
                    help="disable BCEWithLogits pos_weight (multi-label only)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="sigmoid decision threshold for multi-label (default 0.5)")
    ap.add_argument("--output-dir", type=Path, default=Path("results/text"),
                    help="HF Trainer scratch dir (checkpoints + logs)")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=spec.num_labels,
        id2label={int(i): n for i, n in spec.id2label.items()},
        label2id=dict(spec.label2id),
        problem_type=spec.problem_type,
        ignore_mismatched_sizes=True,
    )

    train_ds = TextDataset(train_records, tokenizer, spec, args.max_seq_length)
    dev_ds = TextDataset(dev_records, tokenizer, spec, args.max_seq_length)
    target_ds = TextDataset(target_records, tokenizer, spec, args.max_seq_length)

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
            processing_class=tokenizer, pos_weight=pos_w,
        )
    else:
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=dev_ds,
            data_collator=collate_single,
            processing_class=tokenizer,
        )

    trainer.train()

    # ---- inference on target split ---------------------------------------
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
