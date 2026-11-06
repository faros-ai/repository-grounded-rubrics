# Runs

Every rubric and verdict in `data/` comes from one of the runs below. All runs used the prompts in
`prompts/`, which did not change between the first and the last run.

| Model | Provider model ID |
|---|---|
| Rubric generator | `claude-sonnet-4-6` (Anthropic API) |
| Judge | `gpt-5.4` (OpenAI API) |

No dated snapshot identifiers were recorded, so these IDs are the most specific versions on record.

## Rubric generation

The agent explored an isolated checkout of the repository at the task's base commit. Each run had a
limit of 50 steps and $15 per task.

| Generation | Date | Tasks |
|---|---|---|
| R1 | 2026-08-20 | 370: django (229), sympy, matplotlib, scikit-learn |
| R1 | 2026-09-07 | 86: the 84 tasks of astropy, xarray, pytest, pylint, requests, seaborn and flask, plus `django__django-7530` and `django__django-10097` |
| R1 | 2026-09-08 | 44: sphinx |
| R2 | 2026-09-17 | 100 repeatability tasks |
| R3 | 2026-09-18 | 100 repeatability tasks |

The generation strategy (`prompts/generation-strategy.txt`) was selected in an earlier optimization,
described in App. A of the paper, whose objective scored rubrics with `gpt-5.1`. Every judging run below
uses `gpt-5.4`.

## Judging

Each request carried only the messages and the model ID. The judge therefore ran at the provider's
defaults, temperature 1.0 and no reasoning effort, except in the medium-effort condition, which set
`reasoning_effort` to `medium`.

| Rubrics | Date | Patches | Judgments per patch | File |
|---|---|---|---|---|
| R1 | 2026-08-20 | 2,558 submissions and 370 reference patches on the tasks with August rubrics | 1 | `verdicts/single.csv.gz` |
| R1 | 2026-09-07 | 4,101 submissions and 86 reference patches: the rest outside sphinx | 1 | `verdicts/single.csv.gz` |
| R1 | 2026-09-08 | 632 submissions and 44 reference patches on sphinx | 1 | `verdicts/single.csv.gz` |
| R1 | 2026-09-14 | 400 repeatability patches, effort none | 5 | `verdicts/repeated.csv.gz` |
| R1 | 2026-09-14 | 400 repeatability patches, effort medium | 5 | `verdicts/repeated.csv.gz` |
| R2 | 2026-09-18 | labeled submissions and reference patches of the 100 repeatability tasks | 1 | `verdicts/single.csv.gz` |
| R2 | 2026-09-18 | 400 repeatability patches | 5 | `verdicts/repeated.csv.gz` |
| R3 | 2026-09-19 | labeled submissions and reference patches of the 100 repeatability tasks | 1 | `verdicts/single.csv.gz` |
| R3 | 2026-09-19 | 400 repeatability patches | 5 | `verdicts/repeated.csv.gz` |

Eight submissions exceed the judge's context window and have no verdicts under any rubric. Six of them
belong to the repeatability tasks.
