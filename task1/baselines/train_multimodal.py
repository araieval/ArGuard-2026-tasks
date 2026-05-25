"""
Lightweight multimodal baseline for ArGuard 2026 Task 1 / Track A
(both subtasks).

Architecture (late fusion): a text encoder (any HF BERT-family model)
produces a sentence embedding; an image encoder (any HF ViT-family
model) produces a pooled image embedding; the two are concatenated and
fed through a small MLP head. Trained end-to-end.

This is the smallest reasonable multimodal baseline — easy to read,
fits on a single mid-range GPU, no LoRA / adapter / VLM machinery
required. For a stronger VLM-based baseline (Qwen-VL LoRA fine-tune
via ms-swift), see the ArHate5k research repo linked in the
project README.

Usage
-----
    python baselines/train_multimodal.py --subtask a1 \
        --text-model aubmindlab/bert-base-arabertv02 \
        --image-model google/vit-base-patch16-224 \
        --data-dir data --target dev_test \
        --out predictions/mm_a1.tsv --run-id arabert_vit
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
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_utils import (  # noqa: E402
    read_jsonl,
    write_multilabel_jsonl,
    write_subtask_a1_tsv,
)
from labels import TaskSpec, get_task  # noqa: E402

log = logging.getLogger("train_multimodal")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def resolve_image(record: dict, data_dir: Path) -> Path:
    rel = record.get("image_path") or ""
    p = Path(rel)
    return p if p.is_absolute() else data_dir / p


class MultimodalDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        tokenizer,
        processor,
        spec: TaskSpec,
        data_dir: Path,
        max_len: int,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.processor = processor
        self.spec = spec
        self.data_dir = data_dir
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
        try:
            image = Image.open(resolve_image(rec, self.data_dir)).convert("RGB")
        except Exception as e:
            log.warning("could not open image for %s: %s", rec.get("id"), e)
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        px = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": px,
        }
        if self.spec.is_multilabel:
            target = torch.zeros(self.spec.num_labels, dtype=torch.float32)
            for l in (rec.get("fine_grained_label") or []):
                if l in self.spec.label2id:
                    target[self.spec.label2id[l]] = 1.0
            item["labels"] = target
        else:
            label = rec.get("label")
            item["labels"] = torch.tensor(
                -100 if label is None else self.spec.label2id[label], dtype=torch.long,
            )
        return item


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class LateFusionClassifier(nn.Module):
    """Text encoder + image encoder → concat → MLP head."""

    def __init__(
        self,
        text_model_name: str,
        image_model_name: str,
        num_labels: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = AutoModel.from_pretrained(image_model_name)

        td = self.text_encoder.config.hidden_size
        idim = self.image_encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(td + idim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def _text_pool(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0, :]
        return pooled

    def _image_pool(self, pixel_values):
        out = self.image_encoder(pixel_values=pixel_values)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0, :]
        return pooled

    def forward(self, input_ids, attention_mask, pixel_values):
        t = self._text_pool(input_ids, attention_mask)
        i = self._image_pool(pixel_values)
        return self.head(torch.cat([t, i], dim=-1))


# --------------------------------------------------------------------------
# train loop
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(records: list[dict], spec: TaskSpec) -> torch.Tensor:
    counts = np.zeros(spec.num_labels, dtype=np.float64)
    N = max(len(records), 1)
    for r in records:
        for l in (r.get("fine_grained_label") or []):
            if l in spec.label2id:
                counts[spec.label2id[l]] += 1
    return torch.tensor((N - counts) / np.clip(counts, 1.0, None), dtype=torch.float32)


def run_epoch(model, loader, optimizer, scheduler, loss_fn, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
            )
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss = loss_fn(logits, labels.long())
            else:
                loss = loss_fn(logits, labels.float())
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += float(loss) * labels.size(0)
            n += labels.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, is_multilabel: bool, threshold: float):
    model.eval()
    all_logits = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("labels", None)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
        )
        all_logits.append(logits.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    if is_multilabel:
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (probs >= threshold).astype(int)
    return np.argmax(logits, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtask", required=True,
                    choices=["a1", "a2", "A1", "A2", "subtask_a1", "subtask_a2"])
    ap.add_argument("--text-model", required=True)
    ap.add_argument("--image-model", required=True)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--target", default="dev_test")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", default="multimodal_baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4,
                    help="learning rate for the fusion head (encoders use --lr)")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-seq-length", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-pos-weight", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    set_seed(args.seed)

    device = torch.device(args.device)
    spec = get_task(args.subtask)
    splits_dir = args.data_dir / "splits"
    train_records = read_jsonl(splits_dir / "train.jsonl")
    dev_records = read_jsonl(splits_dir / "dev.jsonl")
    target_records = read_jsonl(splits_dir / f"{args.target}.jsonl")
    log.info(
        "records: train=%d  dev=%d  target=%d",
        len(train_records), len(dev_records), len(target_records),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    processor = AutoImageProcessor.from_pretrained(args.image_model)
    model = LateFusionClassifier(
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        num_labels=spec.num_labels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_ds = MultimodalDataset(
        train_records, tokenizer, processor, spec, args.data_dir, args.max_seq_length,
    )
    dev_ds = MultimodalDataset(
        dev_records, tokenizer, processor, spec, args.data_dir, args.max_seq_length,
    )
    target_ds = MultimodalDataset(
        target_records, tokenizer, processor, spec, args.data_dir, args.max_seq_length,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate)
    target_loader = DataLoader(target_ds, batch_size=args.eval_batch_size, shuffle=False,
                               num_workers=2, collate_fn=collate)

    encoder_params = list(model.text_encoder.parameters()) + list(model.image_encoder.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
        ]
    )
    num_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_steps * args.warmup_ratio),
        num_training_steps=num_steps,
    )

    if spec.is_multilabel:
        pos_w = None if args.no_pos_weight else compute_pos_weight(train_records, spec).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w) if pos_w is not None else nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_dev = float("inf")
    best_state = None
    for ep in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, train=True)
        dv_loss = run_epoch(model, dev_loader, None, None, loss_fn, device, train=False)
        log.info("[epoch %d/%d] train_loss=%.4f  dev_loss=%.4f", ep, args.epochs, tr_loss, dv_loss)
        if dv_loss < best_dev:
            best_dev = dv_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = predict(model, target_loader, device, spec.is_multilabel, args.threshold)

    if spec.is_multilabel:
        rows = []
        for r, row in zip(target_records, preds):
            rows.append((r["id"], [spec.id2label[i] for i, v in enumerate(row) if v]))
        write_multilabel_jsonl(rows, args.out)
    else:
        rows = [(r["id"], spec.id2label[int(p)]) for r, p in zip(target_records, preds)]
        write_subtask_a1_tsv(rows, args.out, run_id=args.run_id)

    log.info("wrote predictions -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
