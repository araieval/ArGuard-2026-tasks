# Task A (Task 1) / Track A — Data

The official dataset lives on the Hugging Face Hub at
**[QCRI/ArGuard-Task1](https://huggingface.co/datasets/QCRI/ArGuard-Task1)**.

## Quick start

```bash
# from the taskA/ directory
python data/download_data.py
```

This writes JSONL splits to `splits/` and images to `img/` (one file per
meme, named after the meme `id`). By default every split published on the
Hub is pulled — `train`, `dev`, `dev_test`, and, now that the
final-evaluation phase is open, the blind **`test`** split.

To fetch only the blind test split:

```bash
python data/download_data.py --splits test
```

## Splits

| Split        | Size  | Labels   | Released                   | Purpose                                    |
|--------------|------:|----------|----------------------------|--------------------------------------------|
| `train`      | 3,500 | full     | development phase          | model training                             |
| `dev`        |   500 | full     | development phase          | hyperparameter selection, local evaluation |
| `dev_test`   |   500 | full     | development phase          | dev-phase leaderboard target; **labels now released** |
| `test`       |   500 | **none** | final-evaluation phase     | blind final ranking (triple-annotated gold) |

> **`test` is the only unlabelled split** (`label = null`,
> `fine_grained_label = []`). Its gold labels are never published;
> scoring happens on CodaBench.
>
> **The final-evaluation phase is open: submit your predictions on
> `test`.** The `dev_test` split is no longer a submission target — the
> development phase is closed and its gold labels have been released, so
> you may now use `dev_test` as **additional labelled training data**. It
> is disjoint from `test`.

## Record schema

```json
{
    "id": "f9a8…b1.jpg",
    "image_path": "img/f9a8…b1.jpg",
    "text": "OCR-extracted Arabic meme text…",
    "label": "Hateful" | "Not Hateful" | null,
    "fine_grained_label": ["Mocking", "Incitement"],
    "annotations": []
}
```

- `id` — original meme filename; used to match predictions to gold.
- `image_path` — relative to this directory (`img/<id>`).
- `text` — OCR-extracted overlaid text. Empty string is allowed.
- `label` — binary label for Subtask A1. `null` only on the blind `test` split.
- `fine_grained_label` — multi-label sub-types from the unified A2 vocab.
  Hateful memes use hateful sub-types; non-hateful memes use
  `{Humor, Sarcasm, Other}`. Empty list only on the blind `test` split.
- `annotations` — placeholder kept for backwards compatibility with the
  research repo. Per-annotator records are not distributed publicly.

## Download options

```bash
python data/download_data.py --help

# Subset of splits
python data/download_data.py --splits train dev

# Different destination
python data/download_data.py --output-dir /path/to/out

# JSONL only, do not write image files (much faster)
python data/download_data.py --skip-images

# Pin to a specific dataset revision (e.g. a tag)
python data/download_data.py --revision v1.0
```

If `huggingface-cli login` is not enough, you can also pass `--token <hf_token>`
(or set `HF_TOKEN` in the environment).

## Sample submissions

[`sample_submissions/`](sample_submissions/) contains tiny example files in the
exact submission format for each subtask. They are intended to be inspected
with your favourite editor and to confirm that the format checker accepts the
expected shape:

```bash
python ../format_checker/format_checker.py --subtask a1 \
    --predictions sample_submissions/subtask_a1_sample.tsv

python ../format_checker/format_checker.py --subtask a2 \
    --predictions sample_submissions/subtask_a2_sample.jsonl
```

## Notes on storage

- The released dataset has embedded images (parquet shards on the Hub). The
  download script decodes them back to standalone `.jpg`/`.png` files so the
  image baseline can read them with `PIL.Image.open`.
- Total on-disk size after download is roughly **40 MB** for the development
  splits (train + dev + dev_test), plus about **5 MB** for `test`.

## License

Released under **CC BY-NC 4.0** for non-commercial research use only. See
the dataset card on the Hub for details.
