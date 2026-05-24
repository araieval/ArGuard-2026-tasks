# Task 1 / Track A — Format Checker

A single script, `format_checker.py`, validates the **structural shape**
of a submission file. Run it before submitting:

```bash
python format_checker/format_checker.py --subtask a1 --predictions <file.tsv>
python format_checker/format_checker.py --subtask a2 --predictions <file.jsonl>
```

Exit code is **0** when all checks pass, **1** on a format error, and
**2** on a missing input file.

## What is checked

**Subtask A1** (TSV with header `id<TAB>label<TAB>run_id`):
- The header line is exactly `id\tlabel\trun_id`.
- Each row has 3 tab-separated columns.
- `label` ∈ `{"Hateful", "Not Hateful"}` (exact match, case sensitive).
- `run_id` is a single non-empty string and the same on every row.
- IDs are non-empty and unique.

**Subtask A2** (JSONL with `{"id": str, "labels": [str, ...]}`):
- Every line is valid JSON with required keys `id` and `labels`.
- `id` is a non-empty string; IDs are unique.
- `labels` is a list of strings drawn from the unified fine-grained
  taxonomy (hateful sub-types + non-hateful sub-types + shared `Other`).
  An empty list is allowed.

## Optional ID consistency check

If you pass `--gold <gold_file>` the checker additionally verifies that
the **set of prediction IDs equals the set of gold IDs** (missing
predictions and extra predictions both fail the check):

```bash
python format_checker/format_checker.py --subtask a1 \
    --predictions preds_a1.tsv --gold data/splits/dev.jsonl
```

For local development this is most useful with `data/splits/dev.jsonl` as
the gold file; for blind-test submissions the scorer (server side) runs
the same comparison against the hidden gold IDs.

## Example error output

```
$ python format_checker/format_checker.py --subtask a1 --predictions bad.tsv
FORMAT ERROR: bad.tsv: 1 rows have an invalid label
              (expected one of ['Hateful', 'Not Hateful']).
              Examples: x.jpg='BADLABEL'
$ echo $?
1
```

## Notes on the A2 label space

The format checker accepts the **full taxonomy** for Subtask A2 — the 10
active labels plus 5 zero-support hateful classes (`Extremism`,
`Historical`, `Insults`, `Stereotyping`, `Threat`). When a prediction
contains a zero-support label the checker logs a warning but does not
fail; the scorer simply ignores those labels because they have no
positive support in the data.
