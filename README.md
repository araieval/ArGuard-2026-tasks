# [ArGuard](https://araieval.github.io/ArGuard2026/) at ArabicNLP 2026

ArGuard is an ArAIEval shared task on harmful content detection in Arabic memes and LLM prompts. This repository follows the structure of the ArAIEval ArabicNLP 2024 task repository and will contain the datasets, format checkers, scorers, baselines, and starter-kit material for each task.

- [Task 1: Multimodal Hateful Meme Detection](task1)

  Given an Arabic meme image with extracted text, systems classify whether the meme is hateful and identify fine-grained hateful content categories.

- [Task 2: Textual Harmful Prompt Detection](task2)

  Given an Arabic prompt directed at an LLM, systems classify whether the prompt is safe or unsafe and identify the unsafe prompt category.

## Repository Structure

```text
.
├── bibtex/
│   └── bibliography.bib
├── task1/
│   ├── baselines/
│   ├── data/
│   ├── format_checker/
│   ├── scorer/
│   └── README.md
├── task2/
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
- Registration deadline and blind test set release: July 25, 2026
- Final submission deadline and release of final results: July 30, 2026
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
@inproceedings{arguard:arabicnlp2026-overview,
  title = {{ArGuard Shared Task}: Harmful Content Detection in Arabic Memes and LLM Prompts},
  author = {Alam, Firoj and Biswas, Md. Rafiul and Kmainasi, Mohamed Bayan and Shahroor, Ali Ezzat and Zaghouani, Wajdi and Mikros, Georgios and Mubarak, Hamdy},
  booktitle = {Proceedings of the Arabic Natural Language Processing Conference},
  year = {2026},
  note = {To appear}
}
```

## Related Resources

- ArGuard task website: <https://araieval.github.io/ArGuard2026/>
- ArAIEval shared task archive: <https://araieval.github.io/>
