# Repository-grounded rubrics on SWE-bench Verified

Data and code accompanying the paper. They contain the rubrics, every judge verdict behind the reported
numbers, and the prompts that produced them. They also contain a script that recomputes the paper's
validation results from these files and a script that downloads and verifies the third-party benchmark
data.

## Contents

| Path | What it holds |
|---|---|
| `prompts/` | The rubric-generation and judge prompts |
| `data/rubrics/r1/` | One rubric per SWE-bench Verified task (500), behind the validation results of §4 |
| `data/rubrics/r2/`, `data/rubrics/r3/` | Two further independent rubrics for each of the 100 repeatability tasks, used in the rubric-generation repeatability study |
| `data/rubrics.csv` | Generator, date, criterion count and SHA-256 of every rubric |
| `data/verdicts/single.csv.gz` | Criterion verdicts from a single judgment of each patch against each rubric |
| `data/verdicts/repeated.csv.gz` | Criterion verdicts from five repeated judgments of the 400 repeatability patches |
| `data/submissions.csv` | The 7,299 leaderboard patches: task, system, resolved label, SHA-256 of the patch text |
| `data/unavailable.csv` | The 201 task-system cells with no patch, and why |
| `data/systems.csv` | The 15 leaderboard systems |
| `data/repeatability.csv` | The four patches selected per task for the repeatability study |
| `data/optimization.csv` | The 85 tasks used to optimize the rubric-generation strategy, with their split |
| `RUNS.md` | Models, settings and dates of every generation and judging run |
| `recompute.py` | Recomputes the validation results and compares each with the paper |
| `fetch.py` | Downloads the tasks and patches into `third_party/` and verifies them |

## Usage

Requires Python 3.11 or later.

```bash
pip install -r requirements.txt
python recompute.py
```

`recompute.py` reads only `data/`, prints each recomputed value, shows the paper's value beside any
that differs, and exits with status 1 if one does. It recomputes the population counts and the
population table of §4.2, the repeatability results of §4.3 and App. D, the reference-patch scores of
§4.4, the discrimination results of §4.5, the system ranking of §4.6, and the systems and AUROC tables
of App. C. It does not cover the sizes of the reference patches, the sizing comparison (App. B), the
comparisons with per-test harness records (§4.6), the similarity analysis (App. E) or the case
studies.

```bash
python fetch.py
```

`fetch.py` uses only the standard library. It downloads SWE-bench Verified from Hugging Face at a pinned
revision, and each leaderboard patch from the public SWE-bench submissions bucket. It checks every file
against the hashes in this repository and recomputes the labels from each system's `results.json` in
`SWE-bench/experiments` at commit `40f164d`. The patches are needed only to judge them again, so they
are not redistributed here.

## Data

Tasks are identified by their SWE-bench `instance_id`. Patches are identified by `patch_id`, which is a
leaderboard system name (the `system` column of `data/systems.csv` and `data/submissions.csv`) or `gold`
for the task's reference patch, the `patch` field of SWE-bench Verified. `systems.csv` gives each
system's display name in the paper and its submission date. `unavailable.csv` gives the reason a cell
has no patch: `no_generation` or `no_logs` as listed in the system's `results.json`, or `empty_patch`.
In `repeatability.csv`, `stratum` is `low`, `mid` or `high` for the task's lowest-, median- and
highest-scoring submission under R1, or `gold`.

A rubric is a YAML file with a `metadata.task_summary` and four lists of criteria under `axes`:
`file_change_rubrics`, `spec_alignment_rubrics`, `integrity_rubrics` and `runtime_rubrics`. Each
criterion has an `id`, a `description` and an integer `weight` from 1 to 3. Criterion IDs start with
the axis abbreviation: `FC`, `SA`, `I` or `R`.

Verdict files have one row per criterion and judgment, with `verdict` 1 if the judge found the
criterion satisfied and 0 otherwise. The `generation` column (`r1`, `r2`, `r3`) names the rubric judged
against. In `repeated.csv.gz`, `effort` is the judge's reasoning effort and `run` numbers the five
judgments. A patch's score is the weighted fraction of satisfied criteria:

    score = sum(weight * verdict) / sum(weight)

An axis score is the same fraction over the criteria of one axis.

`single.csv.gz` holds the R1 verdicts for 7,291 submissions and 500 reference patches. It also holds the
R2 and R3 verdicts for 1,451 submissions and 100 reference patches on the repeatability tasks. Eight
submissions exceed the judge's context window and have no verdicts.

## Prompts

| File | Role |
|---|---|
| `generation-system.jinja` | System message of the rubric-generation agent. `strategy` is filled with `generation-strategy.txt` |
| `generation-instance.jinja` | First user message. `cwd` is the repository checkout at the task's base commit, `task` the problem statement, `total_criteria` the size budget from §3.1, and `feedback` is empty |
| `generation-strategy.txt` | The deployed strategy (App. A) |
| `judge-system.txt` | System message of the judge |
| `judge-user.txt` | User message of the judge, filled with Python's `str.format` |

The judge's `pr_description` is the problem statement and `patch` the diff. `rubric` is a JSON object
mapping each criterion ID to its description, `json.dumps(criteria, indent=2)`. Weights and axes are not
shown to the judge. The judge replies with a JSON object of 0 or 1 per criterion ID, and a criterion
missing from the reply counts as 0. The verdict files do not mark which zeros came from omitted
criteria.

The agent harness that runs rubric generation is not part of this repository.

## License

Code is under the MIT License (`LICENSE`). Rubrics, verdicts and prompts are under CC BY 4.0
(`LICENSE-DATA`). SWE-bench Verified and the leaderboard submissions keep their own terms.
