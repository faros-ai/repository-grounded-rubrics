#!/usr/bin/env python3
"""Download the third-party inputs of the study into third_party/ and verify them.

third_party/verified.parquet
    SWE-bench Verified test split (500 tasks with problem statements and reference patches),
    Hugging Face dataset SWE-bench/SWE-bench_Verified at a pinned revision, checked by SHA-256.
third_party/submissions.jsonl
    The 7,299 leaderboard patches listed in data/submissions.csv, one JSON object per line with
    instance_id, system, resolved and patch. Patches come from the leaderboard's public S3 bucket,
    decoded as UTF-8 with undecodable bytes replaced by U+FFFD as in the study's own copy,
    and are checked against patch_sha256. Labels are recomputed from each system's results.json
    in SWE-bench/experiments at a pinned commit and checked against data/submissions.csv, and
    the cells in data/unavailable.csv are checked against the same files.

Standard library only. Usage: python fetch.py [--workers N]
"""
import argparse
import concurrent.futures
import csv
import hashlib
import http.client
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "third_party"

HF_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
HF_PARQUET = f"https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified/resolve/{HF_REVISION}/data/test-00000-of-00001.parquet"
HF_SHA256 = "030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25"
EXPERIMENTS_COMMIT = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
RESULTS = f"https://raw.githubusercontent.com/SWE-bench/experiments/{EXPERIMENTS_COMMIT}/evaluation/verified/{{system}}/results/results.json"
PATCH = "https://swe-bench-submissions.s3.amazonaws.com/verified/{system}/logs/{instance_id}/patch.diff"


def get(url, tries=4):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == tries - 1:
                raise
        except (OSError, http.client.HTTPException):
            if attempt == tries - 1:
                raise
        time.sleep(2 * (attempt + 1))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_verified(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def fetch_tasks():
    body = get(HF_PARQUET)
    if sha256(body) != HF_SHA256:
        sys.exit(f"verified.parquet: SHA-256 {sha256(body)} does not match {HF_SHA256}")
    write_verified(OUT / "verified.parquet", body)
    print("verified.parquet: SHA-256 verified")


def fetch_submissions(workers):
    expected = read_csv(DATA / "submissions.csv")
    unavailable = read_csv(DATA / "unavailable.csv")
    results = {row["system"]: json.loads(get(RESULTS.format(system=row["system"]))) for row in read_csv(DATA / "systems.csv")}

    problems = []
    for row in unavailable:
        if row["reason"] in ("no_generation", "no_logs") and row["instance_id"] not in results[row["system"]][row["reason"]]:
            problems.append(f"{row['instance_id']} {row['system']}: not listed as {row['reason']} in results.json")

    def one(row):
        return row, get(PATCH.format(system=row["system"], instance_id=row["instance_id"])).decode("utf-8", "replace")

    lines = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for n, (row, patch) in enumerate(pool.map(one, expected), 1):
            key = f"{row['instance_id']} {row['system']}"
            if sha256(patch.encode()) != row["patch_sha256"]:
                problems.append(f"{key}: patch SHA-256 differs")
            label = row["instance_id"] in results[row["system"]]["resolved"]
            if label != bool(int(row["resolved"])):
                problems.append(f"{key}: label differs from results.json")
            lines.append(json.dumps(dict(instance_id=row["instance_id"], system=row["system"], resolved=label, patch=patch)) + "\n")
            if n % 1000 == 0:
                print(f"  {n}/{len(expected)}", flush=True)
    if problems:
        sys.exit("\n".join(problems[:20]) + f"\n{len(problems)} mismatches; submissions.jsonl not written")
    write_verified(OUT / "submissions.jsonl", "".join(lines).encode())
    print(f"submissions.jsonl: {len(lines)} patches and labels verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    fetch_tasks()
    fetch_submissions(args.workers)
