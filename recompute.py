#!/usr/bin/env python3
"""Recompute the paper's validation numbers from data/ and compare each with the value the paper reports.

Scores are recomputed from the criterion verdicts and the rubric weights: a patch's score is the weighted
fraction of criteria it satisfies, and an axis score is the same fraction over that axis's criteria. Bootstrap
intervals resample tasks with replacement, 2,000 draws, seed 0. The intervals for mean per-task AUROC use Python's
random module; those for per-axis AUROC and the system ranking use NumPy's default generator, as in the analysis
behind the paper.

Usage: python recompute.py      (exits with status 1 if any recomputed value differs from the paper)
"""
import itertools
import pathlib
import random
import statistics
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

DATA = pathlib.Path(__file__).resolve().parent / "data"
AXES = {"file_change_rubrics": "FC", "spec_alignment_rubrics": "SA", "runtime_rubrics": "R", "integrity_rubrics": "I"}
AXIS_ORDER = ["FC", "SA", "I", "R"]
GENERATIONS = ["r1", "r2", "r3"]
PAIRS = list(itertools.combinations(GENERATIONS, 2))
NBOOT = 2000

results = []


def report(section, label, value, paper):
    results.append((section, label, value, paper))


def f3(x):
    return f"{x:.3f}"


def ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def pct(n, d):
    return f"{100 * n / d:.1f}%"


# ------------------------------------------------------------------ data

def repo(instance_id):
    owner, rest = instance_id.split("__")
    return f"{owner}/{rest.rsplit('-', 1)[0]}"


def load_criteria():
    rows = []
    for generation in GENERATIONS:
        for path in sorted((DATA / "rubrics" / generation).glob("*.yaml")):
            rubric = yaml.safe_load(path.read_text())
            for axis_key, items in rubric["axes"].items():
                for item in items:
                    rows.append((path.stem, generation, item["id"], AXES[axis_key], item["weight"]))
    return pd.DataFrame(rows, columns=["instance_id", "generation", "criterion", "axis", "weight"])


def score(verdicts, criteria, keys):
    v = verdicts.merge(criteria, on=["instance_id", "generation", "criterion"], how="left", validate="many_to_one")
    assert v.weight.notna().all(), "verdict for a criterion that is not in its rubric"
    v["earned"] = v.weight * v.verdict
    total = v.groupby(keys)[["earned", "weight"]].sum()
    out = (total.earned / total.weight).rename("score").to_frame()
    by_axis = v.groupby(keys + ["axis"])[["earned", "weight"]].sum()
    out = out.join((by_axis.earned / by_axis.weight).unstack("axis"))
    return out.reset_index()


criteria = load_criteria()
submissions = pd.read_csv(DATA / "submissions.csv")
submissions["resolved"] = submissions.resolved.astype(bool)
systems = pd.read_csv(DATA / "systems.csv")
optimization = set(pd.read_csv(DATA / "optimization.csv").instance_id)
single = pd.read_csv(DATA / "verdicts" / "single.csv.gz")
repeated = pd.read_csv(DATA / "verdicts" / "repeated.csv.gz")
cells = pd.read_csv(DATA / "repeatability.csv")
tasks = sorted(criteria[criteria.generation == "r1"].instance_id.unique())

scores = score(single, criteria, ["instance_id", "patch_id", "generation"])
r1 = scores[scores.generation == "r1"]
gold = r1[r1.patch_id == "gold"].set_index("instance_id")
S = r1[r1.patch_id != "gold"].rename(columns={"patch_id": "system"}).merge(submissions[["instance_id", "system", "resolved"]], on=["instance_id", "system"], validate="one_to_one")
S["repo"] = S.instance_id.map(repo)


def task_aurocs(df, col="score"):
    out = {}
    for iid, g in df.groupby("instance_id"):
        pos, neg = g[g.resolved][col].to_numpy(), g[~g.resolved][col].to_numpy()
        if len(pos) and len(neg):
            out[iid] = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    return pd.Series(out, dtype=float)


def boot_tasks(values, weights=None, seed=0):
    """Percentile interval of the (weighted) mean over tasks resampled with Python's random.choices."""
    rng = random.Random(seed)
    idx = range(len(values))
    if weights is None:
        draws = sorted(statistics.fmean(values[i] for i in rng.choices(idx, k=len(values))) for _ in range(NBOOT))
    else:
        draws = []
        for _ in range(NBOOT):
            pick = rng.choices(idx, k=len(values))
            draws.append(sum(values[i] * weights[i] for i in pick) / sum(weights[i] for i in pick))
        draws.sort()
    return draws[49], draws[1949]


def boot_numpy(values, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = [rng.choice(v, len(v)).mean() for _ in range(NBOOT)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


# ------------------------------------------------------------------ §4.2 population

sec = "§4.2 validation population"
report(sec, "labeled patches", f"{len(submissions):,}", "7,299")
report(sec, "resolved patches", f"{submissions.resolved.sum():,}", "3,688")
unavailable = pd.read_csv(DATA / "unavailable.csv")
assert len(unavailable) + len(submissions) == 500 * len(systems)
report(sec, "task-system cells without a patch", f"{len(unavailable)}", "201")
report(sec, "labeled patches without a score", f"{len(submissions) - len(S)}", "8")
report(sec, "labeled scored patches", f"{len(S):,}", "7,291")
labels = submissions.groupby("instance_id").resolved.agg(["any", "all"])
report(sec, "contested tasks", f"{(labels['any'] & ~labels['all']).sum()}", "400")
report(sec, "tasks resolved by every system", f"{labels['all'].sum()}", "35")
report(sec, "tasks resolved by none", f"{(~labels['any']).sum()}", "65")
splits = pd.read_csv(DATA / "optimization.csv").split.value_counts()
report(sec, "tasks in strategy optimization (train / validation)", f"{len(optimization)} ({splits['train']} / {splits['validation']})", "85 (66 / 19)")
n = criteria[criteria.generation == "r1"].groupby("instance_id").size()
report(sec, "criteria per rubric, median (min-max)", f"{n.median():.0f} ({n.min()}-{n.max()})", "10 (8-24)")

sec = "§4.2 population by repository: tasks, contested, patches, resolved"
paper_population = {
    "django/django": "231, 188, 3,370, 1,781", "sympy/sympy": "75, 59, 1,109, 524", "sphinx-doc/sphinx": "44, 35, 638, 228",
    "matplotlib/matplotlib": "34, 27, 461, 218", "scikit-learn/scikit-learn": "32, 27, 477, 345", "astropy/astropy": "22, 15, 327, 133",
    "pydata/xarray": "22, 17, 328, 176", "pytest-dev/pytest": "19, 17, 281, 154", "pylint-dev/pylint": "10, 5, 148, 41",
    "psf/requests": "8, 8, 116, 73", "mwaskom/seaborn": "2, 1, 30, 2", "pallets/flask": "1, 1, 14, 13"}
contested = set(labels[labels["any"] & ~labels["all"]].index)
by_repo_subs = submissions.groupby(submissions.instance_id.map(repo))
for name, paper in paper_population.items():
    g = by_repo_subs.get_group(name)
    ids = set(g.instance_id)
    report(sec, name, f"{len(ids)}, {len(ids & contested)}, {len(g):,}, {g.resolved.sum():,}", paper)

sec = "App. C systems: patches, resolved, resolve rate, cells without a patch"
paper_systems = {
    "20250627_agentless_MCTS-Refine-7B": "461, 116, 25.2%, 39", "20240728_sweagent_gpt4o": "447, 116, 26.0%, 53",
    "20240620_sweagent_claude3.5sonnet": "486, 168, 34.6%, 14", "20241028_agentless-1.5_gpt4o": "496, 194, 39.1%, 4",
    "20250226_swerl_llama3_70b": "499, 206, 41.3%, 1", "20240824_gru": "497, 226, 45.5%, 3", "20240820_honeycomb": "439, 203, 46.2%, 61",
    "20241022_tools_claude-3-5-sonnet-updated": "483, 245, 50.7%, 17", "20241202_agentless-1.5_claude-3.5-sonnet-20241022": "497, 254, 51.1%, 3",
    "20241108_devlo": "500, 271, 54.2%, 0", "20250203_openhands_4x_scaled": "498, 304, 61.0%, 2",
    "20250225_sweagent_claude-3-7-sonnet": "497, 312, 62.8%, 3", "20250316_augment_agent_v0": "500, 327, 65.4%, 0",
    "20250430_zencoder_ai": "499, 350, 70.1%, 1", "20251205_sonar-foundation-agent_claude-opus-4-5": "500, 396, 79.2%, 0"}
names = systems.set_index("system").name
for system, paper in paper_systems.items():
    g = submissions[submissions.system == system]
    report(sec, names[system], f"{len(g)}, {g.resolved.sum()}, {pct(g.resolved.sum(), len(g))}, {500 - len(g)}", paper)

# ------------------------------------------------------------------ 4.3 repeatability, fixed rubric

def run_scores(generation, effort):
    v = repeated[(repeated.generation == generation) & (repeated.effort == effort)]
    return score(v.drop(columns="effort"), criteria, ["instance_id", "patch_id", "generation", "run"])


def judge_repeatability(R, items):
    W = R.pivot_table(index=["instance_id", "patch_id"], columns="run", values="score")
    r = np.mean([pearsonr(W[a], W[b])[0] for a, b in itertools.combinations(W.columns, 2)])
    sd = W.std(axis=1, ddof=1)
    flips = items.groupby(["instance_id", "patch_id", "criterion"]).verdict.nunique().gt(1)
    taus = []
    for _, G in W.groupby(level=0):
        t = [kendalltau(G[a], G[b], variant="b")[0] for a, b in itertools.combinations(G.columns, 2)
             if G[a].nunique() > 1 and G[b].nunique() > 1]
        if t:
            taus.append(np.mean(t))
    taus = np.array(taus)
    return W, r, sd, flips, taus


original = r1.set_index(["instance_id", "patch_id"]).score
sec = "§4.3 and App. D judge repeatability with a fixed rubric"
means = {}
for effort, paper in (("none", ("0.985", "0.020", "4.8%", "0.960", "63/100")), ("medium", ("0.964", "0.030", "6.9%", "0.923", "57/100"))):
    R = run_scores("r1", effort)
    items = repeated[(repeated.generation == "r1") & (repeated.effort == effort)]
    W, r, sd, flips, taus = judge_repeatability(R, items)
    means[effort] = W.mean(axis=1)
    report(sec, f"effort {effort}: mean run-pair Pearson r", f3(r), paper[0])
    report(sec, f"effort {effort}: mean within-patch SD", f"{sd.mean():.3f}", paper[1])
    report(sec, f"effort {effort}: criterion verdicts not unanimous", f"{100 * flips.mean():.1f}%", paper[2])
    report(sec, f"effort {effort}: mean within-task Kendall tau-b", f3(taus.mean()), paper[3])
    report(sec, f"effort {effort}: tasks with the same ordering on every run pair, skipping all-tied runs", f"{(taus >= 1 - 1e-9).sum()}/{len(taus)}", paper[4])
    if effort == "none":
        orig = original.reindex(W.index)
        non_gold = W.index.get_level_values("patch_id") != "gold"
        report(sec, "effort none: mean SD, non-reference, original score in [0.8, 1.0]", f"{sd[non_gold & (orig >= 0.8)].mean():.3f}", "0.012")
        report(sec, "effort none: mean SD, non-reference, original score in [0.4, 0.6)", f"{sd[non_gold & (orig >= 0.4) & (orig < 0.6)].mean():.3f}", "0.034")
        report(sec, "effort none: patches with identical scores on all five runs", f"{(W.nunique(axis=1) == 1).sum()}/{len(W)}", "275/400")
d = (means["medium"] - means["none"]).abs()
report(sec, "five-run means differing by more than 0.10 between efforts", f"{(d > 0.10).sum()}/{len(d)}", "107/400")

# ------------------------------------------------------------------ 4.3 repeatability, rubric generation

sec = "§4.3 and App. D rubric-generation repeatability"
manifest = sorted(cells.instance_id.unique())
counts = criteria.groupby(["generation", "instance_id"]).size()
report(sec, "mean criteria per rubric, R1 / R2 / R3", " / ".join(f"{counts[g].reindex(manifest).mean():.2f}" for g in GENERATIONS), "10.42 / 10.47 / 10.44")

K1 = scores[scores.instance_id.isin(manifest) & (scores.patch_id != "gold")].pivot_table(index=["instance_id", "patch_id"], columns="generation", values="score").dropna()
K1 = K1.join(submissions.set_index(["instance_id", "system"]).resolved.rename_axis(["instance_id", "patch_id"]))
report(sec, "labeled submissions scored under every generation", f"{len(K1):,}", "1,451")


def tau(x, y):
    if len(set(x)) == 1 or len(set(y)) == 1:
        return np.nan
    return kendalltau(x, y, variant="b")[0]


five = pd.concat([run_scores(g, "none") for g in GENERATIONS])
five = five.groupby(["instance_id", "patch_id", "generation"]).score.agg(["mean", lambda x: x.std(ddof=1)])
five.columns = ["mean5", "sd5"]
five = five.reset_index()
M5 = five.pivot_table(index=["instance_id", "patch_id"], columns="generation", values="mean5")
paper_rows = {("r1", "r2"): ("0.866", "0.114", "0.677", "0.747", "33/100"),
              ("r1", "r3"): ("0.871", "0.112", "0.677", "0.732", "37/100"),
              ("r2", "r3"): ("0.873", "0.112", "0.669", "0.750", "34/100")}
for a, b in PAIRS:
    label = f"{a.upper()}-{b.upper()}"
    report(sec, f"{label}: score Pearson r", f3(pearsonr(K1[a], K1[b])[0]), paper_rows[(a, b)][0])
    report(sec, f"{label}: mean absolute score difference", f3((K1[a] - K1[b]).abs().mean()), paper_rows[(a, b)][1])
    t = [tau(G[a].to_numpy(), G[b].to_numpy()) for _, G in K1.groupby(level=0)]
    report(sec, f"{label}: mean within-task tau-b over submissions", f3(np.nanmean(t)), paper_rows[(a, b)][2])
    t4, same = [], []
    for _, G in M5.groupby(level=0):
        x, y = G[a].to_numpy(), G[b].to_numpy()
        v = tau(x, y)
        if not np.isnan(v):
            t4.append(v)
            same.append(bool(np.all(rankdata(x) == rankdata(y))))
    report(sec, f"{label}: mean four-patch tau-b of five-run means", f3(np.mean(t4)), paper_rows[(a, b)][3])
    report(sec, f"{label}: tasks with identical four-patch ordering", f"{sum(same)}/{len(same)}", paper_rows[(a, b)][4])

aurocs = {g: task_aurocs(K1.reset_index().rename(columns={g: "score_g"}), "score_g") for g in GENERATIONS}
report(sec, "contested tasks among the 100", f"{len(aurocs['r1'])}", "77")
report(sec, "mean per-task AUROC, R1 / R2 / R3", " / ".join(f3(aurocs[g].mean()) for g in GENERATIONS), "0.842 / 0.842 / 0.818")

within = five.groupby("instance_id").sd5.mean()
between = M5.std(axis=1, ddof=1).groupby(level=0).mean()
report(sec, "mean task-level SD within a rubric", f"{within.mean():.4f}", "0.0200")
report(sec, "mean task-level SD between rubric generations", f"{between.mean():.4f}", "0.0814")
report(sec, "ratio between / within", f"{between.mean() / within.mean():.2f}", "4.06")

# ------------------------------------------------------------------ 4.4 reference patches

sec = "§4.4 reference patches"
g = gold.score
report(sec, "reference patches scored", f"{len(g)}", "500")
report(sec, "mean / median", f"{f3(g.mean())} / {f3(g.median())}", "0.814 / 0.880")
report(sec, "first / third quartile", f"{f3(g.quantile(.25))} / {f3(g.quantile(.75))}", "0.739 / 0.957")
report(sec, "satisfy every criterion", f"{(g == 1).sum()} ({pct((g == 1).sum(), len(g))})", "59 (11.8%)")
report(sec, "score below 0.5", f"{(g < 0.5).sum()} ({pct((g < 0.5).sum(), len(g))})", "47 (9.4%)")
inside = g.index.isin(optimization)
report(sec, "mean outside / inside optimization", f"{f3(g[~inside].mean())} / {f3(g[inside].mean())}", "0.821 / 0.775")
report(sec, "axis means FC / SA / R / I", " / ".join(f3(gold[a].mean()) for a in ("FC", "SA", "R", "I")), "0.750 / 0.819 / 0.834 / 0.883")
report(sec, "reference patches whose lowest axis is FC, ties counted as FC", f"{(gold[AXIS_ORDER].idxmin(axis=1) == 'FC').sum()}", "265")

# ------------------------------------------------------------------ 4.5 discrimination

sec = "§4.5 within-task discrimination"
A = task_aurocs(S)
ids = [i for i in tasks if i in A.index]
values = [A[i] for i in ids]
report(sec, "contested tasks / patches in them", f"{len(ids)} / {S.instance_id.isin(ids).sum():,}", "400 / 5,834")
report(sec, "mean per-task AUROC [95% CI]", f"{f3(np.mean(values))} {ci(*boot_tasks(values))}", "0.826 [0.805, 0.847]")
pairs = S.groupby("instance_id").resolved.agg(lambda x: x.sum() * (~x).sum())
w = [pairs[i] for i in ids]
report(sec, "pair-weighted mean [95% CI]", f"{f3(np.average(values, weights=w))} {ci(*boot_tasks(values, w))}", "0.836 [0.818, 0.854]")
report(sec, "resolved x unresolved pairs per task, min / median / max", f"{min(w)} / {statistics.median(w):.0f} / {max(w)}", "12 / 36 / 56")
report(sec, "tasks with AUROC 1 / at or below 0.5", f"{(A == 1).sum()} / {(A <= 0.5).sum()}", "109 / 35")
report(sec, "median per-task AUROC", f3(A.median()), "0.901")

sec = "App. C per-task AUROC: tasks, contested, patches in contested tasks, mean [95% CI]"
populations = [("Verified, all tasks", tasks, "500, 400, 5,834, 0.826 [0.805, 0.847]"),
               ("not used in optimization", [i for i in tasks if i not in optimization], "415, 341, 4,974, 0.838 [0.817, 0.859]"),
               ("used in optimization", [i for i in tasks if i in optimization], "85, 59, 860, 0.760 [0.691, 0.822]")]
paper_repo = {"django/django": "231, 188, 2,747, 0.868 [0.844, 0.891]", "sympy/sympy": "75, 59, 872, 0.799 [0.734, 0.863]",
              "sphinx-doc/sphinx": "44, 35, 503, 0.822 [0.768, 0.869]", "matplotlib/matplotlib": "34, 27, 368, 0.765 [0.670, 0.854]",
              "scikit-learn/scikit-learn": "32, 27, 402, 0.767 [0.658, 0.863]", "pydata/xarray": "22, 17, 253, 0.797 [0.678, 0.892]",
              "astropy/astropy": "22, 15, 222, 0.789 [0.643, 0.912]", "pytest-dev/pytest": "19, 17, 250, 0.875 [0.790, 0.936]",
              "pylint-dev/pylint": "10, 5, 73, 0.793 [0.696, 0.891]", "psf/requests": "8, 8, 115, 0.578 [0.409, 0.742]",
              "mwaskom/seaborn": "2, 1, 15, 0.923 [0.923, 0.923]", "pallets/flask": "1, 1, 14, 0.346 [0.346, 0.346]"}
populations += [(name, [i for i in tasks if repo(i) == name], paper) for name, paper in paper_repo.items()]
for name, members, paper in populations:
    c = [i for i in members if i in A.index]
    v = [A[i] for i in c]
    report(sec, name, f"{len(members)}, {len(c)}, {S.instance_id.isin(c).sum():,}, {f3(np.mean(v))} {ci(*boot_tasks(v))}", paper)

sec = "§4.5 class distributions"
res, unr = S[S.resolved].score, S[~S.resolved].score
report(sec, "resolved mean (median, SD)", f"{f3(res.mean())} ({f3(res.median())}, {f3(res.std(ddof=0))})", "0.836 (0.885, 0.182)")
report(sec, "unresolved mean (median, SD)", f"{f3(unr.mean())} ({f3(unr.median())}, {f3(unr.std(ddof=0))})", "0.513 (0.500, 0.286)")
lenient, strict = (unr >= res.mean()).sum(), (res <= unr.mean()).sum()
report(sec, "leniency cases", f"{lenient} of {len(unr):,} ({pct(lenient, len(unr))})", "645 of 3,606 (17.9%)")
report(sec, "strictness cases", f"{strict} of {len(res):,} ({pct(strict, len(res))})", "289 of 3,685 (7.8%)")

sec = "§4.5 per-axis AUROC [95% CI]"
for axis, paper in (("FC", "0.773 [0.751, 0.793]"), ("SA", "0.766 [0.744, 0.786]"), ("R", "0.760 [0.740, 0.778]"), ("I", "0.583 [0.567, 0.598]")):
    a = task_aurocs(S, axis)
    report(sec, axis, f"{f3(a.mean())} {ci(*boot_numpy(a.to_numpy()))}", paper)

# ------------------------------------------------------------------ 4.6 ranking systems from failing patches

sec = "§4.6 system ranking from failing patches"
U = S[~S.resolved].copy()
resolve_rate = S.groupby("system").resolved.mean()


def rho(df, col="score"):
    m = df.groupby("system")[col].mean()
    return spearmanr(resolve_rate[m.index], m)[0]


rng = np.random.default_rng(0)
task_ids = U.instance_id.unique()
Ui = U.set_index("instance_id")
draws = [rho(Ui.loc[rng.choice(task_ids, len(task_ids))].reset_index()) for _ in range(NBOOT)]
report(sec, "failing patches", f"{len(U):,}", "3,606")
report(sec, "Spearman rho, resolve rate over scored patches vs mean failing-patch score [95% CI]", f"{f3(rho(U))} {ci(np.percentile(draws, 2.5), np.percentile(draws, 97.5))}", "0.968 [0.928, 0.971]")
U["centred"] = U.score - U.groupby("instance_id").score.transform("mean")
report(sec, "Spearman rho, scores centred within task", f3(rho(U, "centred")), "0.950")
every = U.groupby("instance_id").system.nunique()
every = every[every == len(resolve_rate)].index
report(sec, "tasks with a scored failing patch from every system / Spearman rho on them", f"{len(every)} / {f3(rho(U[U.instance_id.isin(every)]))}", "42 / 0.914")
by = S.groupby("system").apply(lambda g: pd.Series({"rate": g.resolved.mean(), "res": g[g.resolved].score.mean(), "unr": g[~g.resolved].score.mean()}), include_groups=False)
report(sec, "mean score of resolved patches across systems, min-max", f"{f3(by.res.min())}-{f3(by.res.max())}", "0.766-0.884")
gap = by.res - by.unr
report(sec, "resolved-failing gap, weakest / strongest system", f"{f3(gap[by.rate.idxmin()])} / {f3(gap[by.rate.idxmax()])}", "0.480 / 0.062")

# ------------------------------------------------------------------ output

mismatches = 0
current = None
width = max(len(label) for _, label, _, _ in results)
for section, label, value, paper in results:
    if section != current:
        print(f"\n{section}")
        current = section
    mismatches += value != paper
    print(f"  {label:<{width}}  {value}" + ("" if value == paper else f"   paper: {paper}"))
print(f"\n{len(results) - mismatches} of {len(results)} values match the paper" + (f"; {mismatches} differ" if mismatches else ""))
sys.exit(1 if mismatches else 0)
