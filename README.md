# About the repository

The Gravitaitonal-wave Revolution Companion Code repository.

# Deadlines

- **May 15, 2026**: Outline of the chapter
- **August 15, 2026**: Completion of the first draft
- **September 15, 2026**: Review of first draft and comments conveyed to the authors
- **October 31, 2026**: Submission of the chapter draft to the Editor
- **December 1, 2026**: Editor shares the edited draft with the authors
- **January 1, 2027**: Submission of the draft to the publisher

# Contribution Instructions

## Recommended workflow

1. The `main` branch conatins the top-level chapter folders.
2. The `main` branch is protected so changes are merged through pull requests rather than pushed directly.
3. Before starting work, update your local copy from `main`.
4. Create a new branch from `main` for your work. Use a clear branch name such as: `chapter-03`, `chapter-03-revisions`, etc.
5. Add or update files only in your chapter folder.
6. Open a draft pull request early if you want feedback.
7. When the chapter materials are ready, mark the pull request ready for review.

## Notes for a code companion repo

- Keep each chapter's materials inside its own folder.
- Put Python scripts, notebooks, and supporting files for that chapter in the same folder.
- Avoid committing very large generated files or unnecessary notebook outputs.
- Include a short `README.md` in each chapter folder explaining what the examples do and how to run them.

## Suggested folder pattern

```text
repo-root/
├── chapter-01/
│   ├── README.md
│   ├── notebooks/
│   ├── scripts/
│   └── data/
├── chapter-02/
│   ├── README.md
│   ├── notebooks/
│   ├── scripts/
│   └── data/
└── requirements.txt
