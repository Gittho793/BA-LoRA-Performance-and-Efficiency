"""
Analyze results JSON and save summaries + violin/bar plots.

Usage:
  # Single file
  python analyze_results.py /path/to/<results>.json

  # Directory mode (loads all json files whose basename starts with "pdf")
  python analyze_results.py /path/to/dir

What it does:
- Loads JSON (expects a top-level key "detailed_results", but is tolerant).
- Flattens per-question metrics (both nested "existing_metrics.*.score" and any
  numeric top-level metrics like bleu*, rouge*, bert_*).

Single-file mode:
- Prints overall averages to stdout.
- Saves:
    - <two-levels-up>/results/pictures/<file_stem>/boxplots/
        - <metric>_violin.png            (for every numeric metric EXCEPT hallucination)
        - <hallucination_col>_bar.png    (bar counts of raw 0/1 values)
    - <two-levels-up>/results/pictures/<file_stem>/summary.csv
    - <two-levels-up>/results/pictures/<file_stem>/summary_by_source.csv

Directory mode (argument is a directory path):
- Finds all files in that directory matching: basename starts with "pdf" and ends with ".json".
- Loads and merges them, tagging rows by file stem.
- Prints combined overall averages to stdout.
- Saves (two-levels-up from the directory):
    - <two-levels-up>/results/pictures/<dir_basename>_comparative_pdf/summary_combined.csv
    - <two-levels-up>/results/pictures/<dir_basename>_comparative_pdf/summary_by_source_combined.csv
    - <two-levels-up>/results/pictures/<dir_basename>_comparative_pdf/comparative_boxplots/
        - <metric>_comparative_violin.png   (for every numeric metric EXCEPT hallucination)
        - <hallucination_col>_comparative_bar.png  (grouped bars per run: counts of 0 and 1)

Comparative plot labeling rules:
- If name contains "Meta-Llama" → show "Base"
- Else if name contains "RAG" → show "RAG"
- Else if the name ends with "r<digits>-a<digits>" → show that suffix
- Else → show the original stem

Dependencies: pandas, matplotlib, numpy
"""
from src.eval.visualization.boxplot_analyze_results import save_combined_summaries
import matplotlib.pyplot as plt
import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
plt.rcParams['font.size'] = 20


def flatten_records(data: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    detailed = data.get("detailed_results", {})

    if isinstance(detailed, list):
        detailed = {"__unknown_source__": detailed}

    for source_file, items in (detailed or {}).items():
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            row: Dict[str, Any] = {
                "source_file": source_file,
                "row_id": idx,
                "question": item.get("question"),
                "expected": item.get("expected"),
                "predicted": item.get("predicted"),
            }

            # Nested metrics: take the "score" if present
            em = item.get("existing_metrics", {})
            if isinstance(em, dict):
                for m_name, m_val in em.items():
                    if isinstance(m_val, dict) and "score" in m_val:
                        score = m_val.get("score")
                        if isinstance(score, (int, float)) and np.isfinite(score):
                            row[m_name] = float(score)

            # Top-level numeric metrics (e.g., bleu*, rouge*, bert_*)
            for k, v in item.items():
                if k in row or k in ("existing_metrics", "question", "expected", "predicted"):
                    continue
                if isinstance(v, (int, float)) and np.isfinite(v):
                    row[k] = float(v)

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def numeric_columns(df: pd.DataFrame) -> List[str]:
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    return [c for c in num_cols if c not in ("row_id",)]


# --- Metric locating helpers -------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def find_first_hallucination_col(df: pd.DataFrame) -> Optional[str]:
    targets = [_norm(s) for s in ["halluc", "hallucination", "hallucinations"]]
    for col in df.columns:
        ncol = _norm(col)
        if any(t in ncol for t in targets) and np.issubdtype(df[col].dtype, np.number):
            return col
    return None


# --- Output path helpers -----------------------------------------------------

def make_output_paths_for_file(input_json: str) -> Dict[str, str]:
    file_stem = os.path.splitext(os.path.basename(input_json))[0]
    two_up = os.path.abspath(os.path.join(
        os.path.dirname(input_json), "..", "..", ".."))
    out_dir = os.path.join(two_up, "results", "pictures", file_stem)
    os.makedirs(out_dir, exist_ok=True)
    return {
        "dir": out_dir,
        # kept name for compatibility
        "boxplot": os.path.join(out_dir, "boxplots"),
        "summary_csv": os.path.join(out_dir, "summary.csv"),
        "summary_by_source_csv": os.path.join(out_dir, "summary_by_source.csv"),
    }


def make_output_paths_for_dir(input_dir: str) -> Dict[str, str]:
    dir_base = os.path.basename(os.path.normpath(input_dir))
    one_up = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(input_dir)), ".."))
    out_dir = os.path.join(one_up, "results", "pictures",
                           f"{dir_base}_comparative_pdf")
    os.makedirs(out_dir, exist_ok=True)
    return {
        "dir": out_dir,
        # kept name
        "comparative_boxplots": os.path.join(out_dir, "comparative_boxplots"),
        "summary_combined_csv": os.path.join(out_dir, "summary_combined.csv"),
        "summary_by_source_combined_csv": os.path.join(out_dir, "summary_by_source_combined.csv"),
    }


# --- Summaries ---------------------------------------------------------------

def save_summaries(df: pd.DataFrame, out_paths: Dict[str, str]) -> None:
    metrics = numeric_columns(df)
    if not metrics:
        print("No numeric metrics found to summarize.")
        return

    overall = df[metrics].describe().T.sort_index()
    overall.to_csv(out_paths["summary_csv"])

    by_src = (
        df.groupby("source_file", dropna=False)[metrics]
        .agg(["count", "mean", "std", "min", "max"])
        .sort_index()
    )
    by_src.to_csv(out_paths["summary_by_source_csv"])

    print("\n=== Overall metric means ===")
    print(df[metrics].mean(numeric_only=True).sort_index().round(4))


# --- Plotting ----------------------------------------------------------------

def _violin(vals: np.ndarray, title: str, out_path: str) -> None:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        print(f"Skipping {title} (no valid data)")
        return
    plt.figure(figsize=(4.5, 4.5))
    plt.violinplot(vals, showmedians=True)
    plt.title(title)
    plt.xticks([])
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _bar(categories: List[str], values: List[float], title: str, out_path: str, rot: int = 0) -> None:
    if not values:
        print(f"Skipping {title} (no valid data)")
        return
    plt.figure(figsize=(max(4.5, 1 + 0.6 * len(categories)), 4.5))
    x = np.arange(len(categories))
    plt.bar(x, values)
    plt.title(title)
    plt.xticks(x, categories, rotation=rot, ha="right" if rot else "center")
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _grouped_bar(labels: List[str], zeros: List[float], ones: List[float], title: str, out_path: str) -> None:
    if not labels:
        print(f"Skipping {title} (no valid data)")
        return
    width = 0.4
    x = np.arange(len(labels))
    plt.figure(figsize=(max(6, 1 + 1.0 * len(labels)), 5))
    plt.bar(x - width/2, zeros, width, label="0")
    plt.bar(x + width/2, ones, width, label="1")
    plt.title(title)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.legend(title="Value")
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_violins_all_metrics_except_halluc(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    metrics = numeric_columns(df)
    if not metrics:
        print("No numeric metrics to plot.")
        return

    halluc_col = find_first_hallucination_col(df)
    for col in metrics:
        if halluc_col and col == halluc_col:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").values
        _violin(vals, f"{col} (violin)", os.path.join(
            out_dir, f"{col}_violin.png"))


def save_hallucination_bar_raw(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    halluc_col = find_first_hallucination_col(df)
    if not halluc_col:
        print("Hallucination column not found.")
        return
    vals = pd.to_numeric(df[halluc_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).dropna()
    # Keep raw 0/1 values; count each exactly
    counts = vals.value_counts().sort_index()
    zeros = int(counts.get(0, 0))
    ones = int(counts.get(1, 0))
    out_path = os.path.join(out_dir, f"{halluc_col}_bar.png")
    _bar(["0", "1"], [zeros, ones],
         f"{halluc_col} (counts of raw 0/1)", out_path)


def pretty_run_label(name: str) -> str:
    s = str(name or "")
    if re.search(r"meta[-\s]?llama", s, flags=re.IGNORECASE):
        return "Base"
    if re.search(r"\brag\b", s, flags=re.IGNORECASE) or re.search(r"rag", s, flags=re.IGNORECASE):
        return "RAG"
    m = re.search(r"(r\d+-a\d+)(?:_results)?$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    last_token = re.split(r"[_\s\-]+", s)[-1]
    if re.fullmatch(r"a\d+-r\d+", last_token, flags=re.IGNORECASE):
        return last_token
    return s


def sort_labels_numerically(labels: List[str]) -> List[str]:
    def sort_key(label: str):
        nums = [int(x) for x in re.findall(r"\d+", label)]
        return nums if nums else [float("inf"), label]
    return sorted(labels, key=sort_key)


def save_comparative_violins_all_metrics_except_halluc(df_all: pd.DataFrame, out_dir: str, run_col: str = "run") -> None:
    os.makedirs(out_dir, exist_ok=True)
    metrics = numeric_columns(df_all)
    if not metrics:
        print("No numeric metrics to plot (combined).")
        return

    halluc_col = find_first_hallucination_col(df_all)
    runs = list(df_all[run_col].dropna().unique())
    if not runs:
        print("No runs found for comparative plots.")
        return

    run_to_label = {r: pretty_run_label(r) for r in runs}
    unique_labels = list(set(run_to_label.values()))
    sorted_labels = sort_labels_numerically(unique_labels)

    for metric in metrics:
        if halluc_col and metric == halluc_col:
            continue

        # Collect values for each pretty label
        data = []
        labels = []
        for label in sorted_labels:
            matching_runs = [
                r for r, pl in run_to_label.items() if pl == label]
            vals = pd.to_numeric(
                df_all.loc[df_all[run_col].isin(matching_runs), metric],
                errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna().values
            if vals.size > 0:
                data.append(vals)
                labels.append(label)

        if not data:
            print(f"Skipping {metric} (no valid data across runs)")
            continue

        fig_width = max(6, 1 + 0.8 * len(labels))
        plt.figure(figsize=(fig_width, 5))
        plt.violinplot(data, showmedians=True)
        plt.title(f"{metric} — comparative (violin)")
        plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
        plt.grid(axis="y", linestyle=":", alpha=0.5)
        out_path = os.path.join(out_dir, f"{metric}_comparative_violin.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved comparative: {out_path}")


def save_comparative_hallucination_bar_raw(df_all: pd.DataFrame, out_dir: str, run_col: str = "run") -> None:
    os.makedirs(out_dir, exist_ok=True)
    halluc_col = find_first_hallucination_col(df_all)
    if not halluc_col:
        print("Hallucination column not found for comparative plots.")
        return

    runs = list(df_all[run_col].dropna().unique())
    if not runs:
        print("No runs found for comparative plots.")
        return

    run_to_label = {r: pretty_run_label(r) for r in runs}
    unique_labels = list(set(run_to_label.values()))
    sorted_labels = sort_labels_numerically(unique_labels)

    zeros, ones, labels = [], [], []
    for label in sorted_labels:
        matching_runs = [r for r, pl in run_to_label.items() if pl == label]
        vals = pd.to_numeric(
            df_all.loc[df_all[run_col].isin(matching_runs), halluc_col],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if not vals.empty:
            vc = vals.value_counts().sort_index()
            zeros.append(int(vc.get(0, 0)))
            ones.append(int(vc.get(1, 0)))
            labels.append(label)

    if not labels:
        print(f"Skipping {halluc_col} (no valid data across runs)")
        return

    out_path = os.path.join(out_dir, f"{halluc_col}_comparative_bar.png")
    _grouped_bar(labels, zeros, ones,
                 f"{halluc_col} — comparative (counts of raw 0/1)", out_path)


# --- I/O orchestration -------------------------------------------------------

def load_json_safely(path: str) -> Tuple[str, pd.DataFrame]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Skipping {path}: failed to read/parse JSON ({e})")
        return (os.path.splitext(os.path.basename(path))[0], pd.DataFrame())

    df = flatten_records(data)
    return (os.path.splitext(os.path.basename(path))[0], df)


def process_single_file(input_json: str) -> None:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = flatten_records(data)
    if df.empty:
        print("No rows found in JSON. Nothing to analyze.")
        return

    out_paths = make_output_paths_for_file(input_json)
    save_summaries(df, out_paths)
    save_violins_all_metrics_except_halluc(df, out_paths["boxplot"])
    save_hallucination_bar_raw(df, out_paths["boxplot"])
    print("\nDone.")


def process_directory(input_dir: str) -> None:
    # Find pdf*.json files (non-recursive)
    candidates = []
    for name in os.listdir(input_dir):
        if not name.lower().endswith(".json"):
            continue
        if not os.path.basename(name).startswith("pdf"):
            continue
        candidates.append(os.path.join(input_dir, name))

    if not candidates:
        print(
            f"No files starting with 'pdf' and ending with '.json' found in {input_dir}")
        return

    frames = []
    for path in sorted(candidates):
        run, df = load_json_safely(path)
        if df.empty:
            print(f"Warning: {path} produced no rows; skipping.")
            continue
        df = df.copy()
        df["run"] = run
        frames.append(df)

    if not frames:
        print("No valid data loaded from directory.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    out_paths = make_output_paths_for_dir(input_dir)

    save_combined_summaries(df_all, out_paths, run_col="run")
    save_comparative_violins_all_metrics_except_halluc(
        df_all, out_paths["comparative_boxplots"], run_col="run")
    save_comparative_hallucination_bar_raw(
        df_all, out_paths["comparative_boxplots"], run_col="run")
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", help="Path to *_results.json or a directory containing pdf*.json")
    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        process_directory(args.input_path)
    elif os.path.isfile(args.input_path):
        process_single_file(args.input_path)
    else:
        print(f"Path not found: {args.input_path}")


if __name__ == "__main__":
    main()
