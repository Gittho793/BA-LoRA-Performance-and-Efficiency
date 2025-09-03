#!/usr/bin/env python3
"""
Robust analyzer for GPU usage logs across a directory tree.

Fixes vs v1:
- Accepts *any* training file whose filename starts with "gpu_usage" (any extension) but
  does NOT contain "generation". (E.g., gpu_usage, gpu_usage.txt, gpu_usage.log)
- Accepts *any* generation file that contains both "generation" and "gpu_usage" in the filename.
- Supports decimal commas (e.g., "26,82 minutes") and flexible wording like
  "used in generation" or "inference". Also tolerates ":" vs "=" after labels,
  and "GB/GiB" units.
- Adds clear console diagnostics if no values were parsed for a folder.
- Saves a CSV summary next to the plots.

Usage remains the same:
    python analyze_gpu_usage.py --root /path/to/root [--vram-metric gb|pct|phase-gb] [--gen-choice max-minutes|newest]
"""
from src.eval.visualization.violin import pretty_run_label, sort_labels_numerically
import matplotlib.pyplot as plt
import os
import re
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
plt.rcParams['font.size'] = 16
# ---------- Filename matching ----------


def is_training_file(fn: str) -> bool:
    name = fn.lower()
    return name.startswith("gpu_usage") and ("generation" not in name)


def is_generation_file(fn: str) -> bool:
    name = fn.lower()
    return ("gpu_usage" in name) and ("generation" in name)

# ---------- Flexible number parsing ----------


def parse_num(s: str) -> float:
    # convert decimal comma -> dot, strip spaces
    s = s.strip().replace(",", ".")
    return float(s)


# Common numeric token (supports '.' or ',')
NUM = r"([0-9]+(?:[.,][0-9]+)?)"

# ---------- Regexes (more tolerant) ----------
RE_TRAIN_MIN = re.compile(
    rf"{NUM}\s*(?:min(?:ute)?s?)\s+used\s+(?:for|in)\s+training",
    re.IGNORECASE,
)
RE_GEN_MIN = re.compile(
    rf"{NUM}\s*(?:min(?:ute)?s?)\s+used\s+(?:for|in)\s+(?:generation|inference)",
    re.IGNORECASE,
)

RE_PEAK_GB = re.compile(
    rf"Peak\s*reserved\s*memory\s*[:=]\s*{NUM}\s*(?:G?i?B)",
    re.IGNORECASE,
)
RE_PEAK_PCT = re.compile(
    rf"Peak\s*reserved\s*memory\s*%?\s*of\s*max\s*memory\s*[:=]\s*{NUM}\s*%",
    re.IGNORECASE,
)

RE_PEAK_GEN_GB = re.compile(
    rf"Peak\s*reserved\s*memory\s*(?:for\s+generation|for\s+inference)\s*[:=]\s*{NUM}\s*(?:G?i?B)",
    re.IGNORECASE,
)
RE_PEAK_TRAIN_GB = re.compile(
    rf"Peak\s*reserved\s*memory\s*(?:for\s+training)\s*[:=]\s*{NUM}\s*(?:G?i?B)",
    re.IGNORECASE,
)


@dataclass
class FolderStats:
    folder: str
    train_minutes: Optional[float] = None
    gen_minutes: Optional[float] = None
    train_vram_gb: Optional[float] = None
    gen_vram_gb: Optional[float] = None
    train_vram_pct: Optional[float] = None
    gen_vram_pct: Optional[float] = None
    train_vram_for_phase_gb: Optional[float] = None
    gen_vram_for_phase_gb: Optional[float] = None


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_training_file(path: str) -> Dict[str, Optional[float]]:
    txt = read_text(path)
    out = dict(train_minutes=None, train_vram_gb=None,
               train_vram_pct=None, train_vram_for_phase_gb=None)
    m = RE_TRAIN_MIN.search(txt)
    if m:
        out["train_minutes"] = parse_num(m.group(1))
    m = RE_PEAK_GB.search(txt)
    if m:
        out["train_vram_gb"] = parse_num(m.group(1))
    m = RE_PEAK_PCT.search(txt)
    if m:
        out["train_vram_pct"] = parse_num(m.group(1))
    m = RE_PEAK_TRAIN_GB.search(txt)
    if m:
        out["train_vram_for_phase_gb"] = parse_num(m.group(1))
    return out


def parse_generation_file(path: str) -> Dict[str, Optional[float]]:
    txt = read_text(path)
    out = dict(gen_minutes=None, gen_vram_gb=None,
               gen_vram_pct=None, gen_vram_for_phase_gb=None)
    m = RE_GEN_MIN.search(txt)
    if m:
        out["gen_minutes"] = parse_num(m.group(1))
    m = RE_PEAK_GB.search(txt)
    if m:
        out["gen_vram_gb"] = parse_num(m.group(1))
    m = RE_PEAK_PCT.search(txt)
    if m:
        out["gen_vram_pct"] = parse_num(m.group(1))
    m = RE_PEAK_GEN_GB.search(txt)
    if m:
        out["gen_vram_for_phase_gb"] = parse_num(m.group(1))
    return out


def _choose_generation(entries: List[Tuple[str, Dict[str, Optional[float]]]], choice: str) -> Dict[str, Optional[float]]:
    if not entries:
        return {}
    if choice == "newest":
        return max(entries, key=lambda it: os.path.getmtime(it[0]))[1]
    # default: pick by largest gen minutes
    best = max(
        entries, key=lambda it: (-1 if it[1].get("gen_minutes") is None else it[1]["gen_minutes"]))
    return best[1]


def gather_stats(root_dir: str, gen_choice: str) -> List[FolderStats]:
    stats: List[FolderStats] = []
    for dirpath, _, filenames in os.walk(root_dir):
        train_files = [os.path.join(dirpath, fn)
                       for fn in filenames if is_training_file(fn)]
        gen_files = [os.path.join(dirpath, fn)
                     for fn in filenames if is_generation_file(fn)]
        if not train_files and not gen_files:
            continue

        rel_folder = os.path.relpath(dirpath, root_dir)
        if rel_folder == ".":
            rel_folder = os.path.basename(os.path.abspath(root_dir))

        fs = FolderStats(folder=rel_folder)

        is_rag = "rag" in rel_folder.lower()

        if train_files:
            tfile = max(train_files, key=os.path.getmtime)
            if is_rag:
                # For RAG: gpu_usage.txt → generation
                g = parse_generation_file(tfile)
                fs.gen_minutes = g["gen_minutes"]
                fs.gen_vram_gb = g["gen_vram_gb"]
                fs.gen_vram_pct = g["gen_vram_pct"]
                fs.gen_vram_for_phase_gb = g["gen_vram_for_phase_gb"]
            else:
                # Normal: gpu_usage.txt → training
                t = parse_training_file(tfile)
                fs.train_minutes = t["train_minutes"]
                fs.train_vram_gb = t["train_vram_gb"]
                fs.train_vram_pct = t["train_vram_pct"]
                fs.train_vram_for_phase_gb = t["train_vram_for_phase_gb"]

        if gen_files:
            parsed = [(gf, parse_generation_file(gf)) for gf in gen_files]
            best = _choose_generation(parsed, gen_choice)
            if best:
                fs.gen_minutes = best.get("gen_minutes")
                fs.gen_vram_gb = best.get("gen_vram_gb")
                fs.gen_vram_pct = best.get("gen_vram_pct")
                fs.gen_vram_for_phase_gb = best.get("gen_vram_for_phase_gb")

        # Diagnostics: warn if a folder had files but nothing parsed
        if (fs.train_minutes is None and fs.gen_minutes is None and
                fs.train_vram_gb is None and fs.gen_vram_gb is None):
            continue

        stats.append(fs)
    return stats


def make_dataframe(stats: List[FolderStats]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(s) for s in stats])
    cols = [
        "folder",
        "train_minutes", "gen_minutes",
        "train_vram_gb", "gen_vram_gb",
        "train_vram_pct", "gen_vram_pct",
        "train_vram_for_phase_gb", "gen_vram_for_phase_gb",
    ]
    df = df[[c for c in cols if c in df.columns]]
    return df


def plot_grouped_bars(x_labels, v1, v2, title, ylabel, legend_labels=("Training", "Generation"), outfile=None):
    vals1 = np.array([0 if v is None else v for v in v1], dtype=float)
    vals2 = np.array([0 if v is None else v for v in v2], dtype=float)
    missing1 = [v is None for v in v1]
    missing2 = [v is None for v in v2]

    n = len(x_labels)
    if n == 0:
        print("[WARN] No folders to plot.")
        return

    x = np.arange(n)
    width = 0.38
    plt.figure(figsize=(max(6, n * 1.2), 5))
    b1 = plt.bar(x - width/2, vals1, width, label=legend_labels[0])
    b2 = plt.bar(x + width/2, vals2, width, label=legend_labels[1])

    # Add hatch for missing entries; if both are missing we add a tiny epsilon so hatch is visible
    eps = 1e-6
    for i, rect in enumerate(b1):
        if missing1[i]:
            if rect.get_height() == 0:
                rect.set_height(eps)
            rect.set_hatch("//")
    for i, rect in enumerate(b2):
        if missing2[i]:
            if rect.get_height() == 0:
                rect.set_height(eps)
            rect.set_hatch("//")

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, x_labels, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")


def analyze(root_dir: str, gen_choice: str = "max-minutes", vram_metric: str = "gb",
            out_times: str = "training_generation_times.png",
            out_vram: str = "training_generation_vram.png",
            out_csv: str = "gpu_usage_summary.csv") -> None:
    stats = gather_stats(root_dir, gen_choice=gen_choice)
    if not stats:
        print(f"No matching files found under: {root_dir}")
        return

    df = make_dataframe(stats)
    df.to_csv(os.path.join(root_dir, out_csv), index=False)

    # ✨ INSERT HERE ✨
    raw_labels = df["folder"].tolist()
    pretty_labels = [pretty_run_label(lbl) for lbl in raw_labels]
    x_labels = sort_labels_numerically(pretty_labels)

    # align the dataframe rows to sorted labels
    df["_pretty_label"] = pretty_labels
    df = df.set_index("_pretty_label").loc[x_labels].reset_index()

    # now replace this old line:
    # x_labels = df["folder"].tolist()
    x_labels = df["_pretty_label"].tolist()

    # TIME
    plot_grouped_bars(
        x_labels,
        df["train_minutes"].tolist() if "train_minutes" in df else [
            None]*len(x_labels),
        df["gen_minutes"].tolist() if "gen_minutes" in df else [
            None]*len(x_labels),
        title="Training vs Generation Time",
        ylabel="Minutes",
        outfile=os.path.join(root_dir, out_times),
    )

    # VRAM
    if vram_metric == "pct":
        y1 = df["train_vram_pct"].tolist() if "train_vram_pct" in df else [
            None]*len(x_labels)
        y2 = df["gen_vram_pct"].tolist() if "gen_vram_pct" in df else [
            None]*len(x_labels)
        ylabel = "% of Max Memory"
        title = "Training vs Generation VRAM (% of Max)"
    elif vram_metric == "phase-gb":
        y1 = df["train_vram_for_phase_gb"].tolist(
        ) if "train_vram_for_phase_gb" in df else [None]*len(x_labels)
        y2 = df["gen_vram_for_phase_gb"].tolist() if "gen_vram_for_phase_gb" in df else [
            None]*len(x_labels)
        ylabel = "GB"
        title = "Training vs Generation VRAM (Phase-Specific Peak)"
    else:
        y1 = df["train_vram_gb"].tolist() if "train_vram_gb" in df else [
            None]*len(x_labels)
        y2 = df["gen_vram_gb"].tolist() if "gen_vram_gb" in df else [
            None]*len(x_labels)
        ylabel = "GB"
        title = "Training vs Generation VRAM (Peak Reserved)"

    plot_grouped_bars(
        x_labels, y1, y2,
        title=title, ylabel=ylabel,
        outfile=os.path.join(root_dir, out_vram),
    )

    # Diagnostics summary
    missing_time = int(sum(pd.isna(df.get("train_minutes", pd.Series([np.nan]*len(df)))))) + \
        int(sum(pd.isna(df.get("gen_minutes", pd.Series([np.nan]*len(df))))))
    if missing_time == 2*len(df):
        print("[INFO] All time values missing; check that regex matches 'minutes used for training/generation'.")

    print(f"Saved:\n - {os.path.join(root_dir, out_times)}\n - {os.path.join(root_dir, out_vram)}\n - {os.path.join(root_dir, out_csv)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".",
                    help="Root directory to search")
    ap.add_argument("--gen-choice", type=str, choices=["max-minutes", "newest"], default="max-minutes",
                    help="For multiple generation files in a folder")
    ap.add_argument("--vram-metric", type=str, choices=["gb", "pct", "phase-gb"], default="gb",
                    help="Which VRAM metric to plot")
    ap.add_argument("--out-times", type=str,
                    default="training_generation_times.png")
    ap.add_argument("--out-vram", type=str,
                    default="training_generation_vram.png")
    ap.add_argument("--out-csv", type=str, default="gpu_usage_summary.csv")
    args = ap.parse_args()
    analyze(args.root, gen_choice=args.gen_choice, vram_metric=args.vram_metric,
            out_times=args.out_times, out_vram=args.out_vram, out_csv=args.out_csv)


if __name__ == "__main__":
    main()
