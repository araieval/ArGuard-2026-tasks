# [ArGuard](https://araieval.github.io/ArGuard2026/) at ArabicNLP 2026

ArGuard is an ArAIEval shared task on harmful content detection in Arabic memes and LLM prompts. This repository follows the structure of the ArAIEval ArabicNLP 2024 task repository and will contain the datasets, format checkers, scorers, baselines, and starter-kit material for each task.

- [Task A (Task 1): Multimodal Hateful Meme Detection](taskA)

  Given an Arabic meme image with extracted text, systems classify whether the meme is hateful and identify fine-grained hateful content categories.

- [Task B (Task 2): Textual Harmful Prompt Detection](taskB)

  Given an Arabic prompt directed at an LLM, systems classify whether the prompt is safe or unsafe and identify the unsafe prompt category.

## Repository Structure

```text
.
├── bibtex/
│   └── bibliography.bib
├── taskA/
│   ├── baselines/
│   ├── data/
│   ├── format_checker/
│   ├── scorer/
│   └── README.md
├── taskB/
│   ├── baselines/
│   ├── data/
│   ├── format_checker/
│   ├── scorer/
│   └── README.md
├── README.md
└── requirements.txt
```

## Timeline

The official schedule is maintained on the task website:

- Task website, training and development data, and evaluation scripts: May 25, 2026
- Development phase (`dev_test` leaderboard): May 25 – July 31, 2026 — **closed**
- Blind test set release: August 1, 2026 — **released**, unlabelled, for both tasks:
  - Task A — [QCRI/ArGuard-Task1](https://huggingface.co/datasets/QCRI/ArGuard-Task1) on the Hugging Face Hub, split `test` (500 memes)
  - Task B — [`taskB/data/`](taskB/data) in this repository
- Final submission deadline: **August 6, 2026, 23:59 AoE** (= August 7, 12:00 UTC) — final-evaluation phase **open now**, same deadline for both tasks
- Camera-ready system description papers: August 22, 2026

## Licensing

Please check the task-specific directory for licensing information for the respective dataset. Unless otherwise stated in the released files, dataset material is intended for research use under the task terms.

## Contact

- Website: <https://araieval.github.io/ArGuard2026/>
- Slack Channel: <https://join.slack.com/t/araieval/shared_invite/zt-20rzypxs7-LuHUsw8ltj7ylae9c4I7XQ>
- Email: <araieval@googlegroups.com>

## Citation

The task overview paper should be cited once available. A provisional BibTeX entry is provided in [bibtex/bibliography.bib](bibtex/bibliography.bib).

```bibtex
@inproceedings{alam-etal-2026-arguard,
   title = {{ArGuard Shared Task}: Harmful Content Detection in {A}rabic {M}emes and {LLM} {P}rompts},
    author = "
      Alam, Firoj and
      Biswas, Md. Rafiul  and
      Kmainasi, Mohamed Bayan  and
      Shahroor, Ali Ezzat  and
      Mubarak, Hamdy and
      Mikros, Georgios  and
      Hasnat, Abul  and
      Zaghouani, Wajdi",    
    booktitle = "Proceedings of The Fourth Arabic Natural Language Processing Conference: Shared Tasks",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
}
```

## Related Resources

- ArGuard task website: <https://araieval.github.io/ArGuard2026/>
- ArAIEval shared task archive: <https://araieval.github.io/>
