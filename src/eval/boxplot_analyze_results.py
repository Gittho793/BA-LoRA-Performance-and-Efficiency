"""
Analyze results JSON and save summaries + boxplots.

Usage:
  # Single file (unchanged behavior)
  python analyze_results.py /path/to/<results>.json

  # NEW: Directory mode (loads all json files whose basename starts with "pdf")
  python analyze_results.py /path/to/dir

What it does:
- Loads JSON (expects a top-level key "detailed_results", but is tolerant).
- Flattens per-question metrics (both nested "existing_metrics.*.score" and any
  numeric top-level metrics like bleu*, rouge*, bert_*).

Single-file mode:
- Prints overall averages to stdout.
- Saves:
    - <two-levels-up>/results/pictures/<file_stem>/boxplots/<metric>_boxplot.png
    - <two-levels-up>/results/pictures/<file_stem>/summary.csv
    - <two-levels-up>/results/pictures/<file_stem>/summary_by_source.csv

Directory mode (argument is a directory path):
- Finds all files in that directory matching: basename starts with "pdf" and ends with ".json".
- Loads and merges them, tagging rows by file stem.
- Prints combined overall averages to stdout.
- Saves (two-levels-up from the directory):
    - <two-levels-up>/results/pictures/<dir_basename>_comparative_pdf/summary_combined.csv
    - <two-levels-up>/results/pictures/<dir_basename>_comparative_pdf/summary_by_source_combined.csv
    - <two-levels-up>/results/pictures/<dir_basename>_comparative_pdf/comparative_boxplots/<metric>_comparative.png
      (each image = one metric; boxplots grouped by file)

Comparative plot labeling rules:
- If name contains "Meta-Llama" → show "Base"
- Else if name contains "RAG" → show "RAG"
- Else if the name ends with a pattern like a<digits>-r<digits> (e.g., "a3-r7") → show that suffix
- Else → show the original stem

Dependencies: pandas, matplotlib, numpy
"""
import matplotlib.pyplot as plt
import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless


def flatten_records(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Turn the JSON structure into a flat table of rows (one per Q/A item).
    Handles:
      - data['detailed_results'] = { source_file: [ { ... item ... }, ... ], ... }
      - 'existing_metrics': { metric_name: {'score': float, ...}, ... }
      - any top-level numeric fields in each item (e.g., bleu1, rouge1, bert_f1, ...)
    """
    rows: List[Dict[str, Any]] = []
    detailed = data.get("detailed_results", {})

    # If the file doesn't follow the expected structure, try to treat it as a list
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
    # Drop obvious non-metric numeric IDs if present
    return [c for c in num_cols if c not in ("row_id",)]


def make_output_paths_for_file(input_json: str) -> Dict[str, str]:
    """
    Build the output directory two levels up from the input file path:
        ../../results/pictures/{file_stem}/
    and the files within it.
    """
    file_stem = os.path.splitext(os.path.basename(input_json))[0]
    two_up = os.path.abspath(os.path.join(
        os.path.dirname(input_json), "..", ".."))
    out_dir = os.path.join(two_up, "results", "pictures", file_stem)
    os.makedirs(out_dir, exist_ok=True)
    return {
        "dir": out_dir,
        "boxplot": os.path.join(out_dir, "boxplots"),
        "summary_csv": os.path.join(out_dir, "summary.csv"),
        "summary_by_source_csv": os.path.join(out_dir, "summary_by_source.csv"),
    }


def make_output_paths_for_dir(input_dir: str) -> Dict[str, str]:
    """
    Build the output directory two levels up from the directory path:
        ../../results/pictures/{dir_basename}_comparative_pdf/
    """
    dir_base = os.path.basename(os.path.normpath(input_dir))
    one_up = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(input_dir)), ".."))
    out_dir = os.path.join(one_up, "results", "pictures",
                           f"{dir_base}_comparative_pdf")
    os.makedirs(out_dir, exist_ok=True)
    return {
        "dir": out_dir,
        "comparative_boxplots": os.path.join(out_dir, "comparative_boxplots"),
        "summary_combined_csv": os.path.join(out_dir, "summary_combined.csv"),
        "summary_by_source_combined_csv": os.path.join(out_dir, "summary_by_source_combined.csv"),
    }


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

    # Also print a quick mean row to stdout for convenience
    print("\n=== Overall metric means ===")
    print(df[metrics].mean(numeric_only=True).sort_index().round(4))


def save_boxplots(df: pd.DataFrame, out_dir: str) -> None:
    """
    Save one boxplot per metric into individual PNG files in out_dir.
    """
    metrics = numeric_columns(df)
    if not metrics:
        print("No numeric metrics to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    for col in metrics:
        vals = df[col].values
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            print(f"Skipping {col} (no valid data)")
            continue

        plt.figure(figsize=(4, 4))
        plt.boxplot(vals, vert=True, widths=0.6)
        plt.title(col)
        plt.xticks([])
        plt.grid(axis="y", linestyle=":", alpha=0.5)

        out_path = os.path.join(out_dir, f"{col}_boxplot.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def pretty_run_label(name: str) -> str:
    """
    Normalize run labels for comparative plots:
    - If contains 'Meta-Llama' -> 'Base'
    - Else if contains 'RAG' -> 'RAG'
    - Else if ends with a<digits>-r<digits> (optionally followed by _results) -> that suffix
    - Else -> original name
    """
    s = str(name or "")
    # Meta-Llama takes precedence over RAG if both appear (unlikely)
    if re.search(r"meta[-\s]?llama", s, flags=re.IGNORECASE):
        return "Base"
    if re.search(r"rag", s, flags=re.IGNORECASE):
        return "RAG"

    # Match aN-rM at end, optionally followed by _results
    m = re.search(r"(r\d+-a\d+)(?:_results)?$", s)
    if m:
        return m.group(1)

    # (Optional) if last token matches the pattern, use it
    last_token = re.split(r"[_\s\-]+", s)[-1]
    if re.fullmatch(r"a\d+-r\d+", last_token):
        return last_token

    return s


def sort_labels_numerically(labels: List[str]) -> List[str]:
    """
    Sorts run labels in a human-friendly numeric order.
    E.g., "a8-r2" will come before "a128-r2".
    Falls back to plain string comparison if no numbers are present.
    """
    def sort_key(label: str):
        # Extract all integers in the label
        nums = [int(x) for x in re.findall(r"\d+", label)]
        # If no numbers, use the label itself as fallback
        return nums if nums else [float("inf"), label]

    return sorted(labels, key=sort_key)


def save_combined_summaries(df_all: pd.DataFrame, out_paths: Dict[str, str], run_col: str = "run") -> None:
    metrics = numeric_columns(df_all)
    if not metrics:
        print("No numeric metrics found to summarize (combined).")
        return

    overall = df_all[metrics].describe().T.sort_index()
    overall.to_csv(out_paths["summary_combined_csv"])

    by_src = (
        df_all.groupby([run_col, "source_file"], dropna=False)[metrics]
        .agg(["count", "mean", "std", "min", "max"])
        .sort_index()
    )
    by_src.to_csv(out_paths["summary_by_source_combined_csv"])

    print("\n=== Combined overall metric means across all runs ===")
    print(df_all[metrics].mean(numeric_only=True).sort_index().round(4))


def save_comparative_boxplots(df_all: pd.DataFrame, out_dir: str, run_col: str = "run") -> None:
    metrics = numeric_columns(df_all)
    if not metrics:
        print("No numeric metrics to plot (combined).")
        return

    os.makedirs(out_dir, exist_ok=True)

    runs = list(df_all[run_col].dropna().unique())
    if not runs:
        print("No runs found for comparative plots.")
        return

    # Map original run → pretty label
    run_to_label = {r: pretty_run_label(r) for r in runs}

    # Deduplicate labels before sorting
    unique_labels = list(set(run_to_label.values()))
    sorted_labels = sort_labels_numerically(unique_labels)

    for metric in metrics:
        data = []
        labels = []
        for label in sorted_labels:
            # Gather all runs that share this pretty label
            matching_runs = [
                r for r, pl in run_to_label.items() if pl == label]
            vals = df_all.loc[df_all[run_col].isin(
                matching_runs), metric].astype(float)
            vals = vals[np.isfinite(vals.values)]
            if vals.size > 0:
                data.append(vals.values)
                labels.append(label)

        if not data:
            print(f"Skipping {metric} (no valid data across runs)")
            continue

        fig_width = max(5, 1 + 0.6 * len(labels))
        plt.figure(figsize=(fig_width, 5))
        plt.boxplot(data, vert=True, widths=0.6)
        plt.title(f"{metric} — comparative")
        plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
        plt.grid(axis="y", linestyle=":", alpha=0.5)

        out_path = os.path.join(out_dir, f"{metric}_comparative.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved comparative: {out_path}")


def load_json_safely(path: str) -> Tuple[str, pd.DataFrame]:
    """Load one results JSON into a DataFrame with error handling. Returns (run_label, df)."""
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
    save_boxplots(df, out_paths["boxplot"])
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
    run_labels = []
    for path in sorted(candidates):
        run, df = load_json_safely(path)
        if df.empty:
            print(f"Warning: {path} produced no rows; skipping.")
            continue
        df = df.copy()
        df["run"] = run
        frames.append(df)
        run_labels.append(run)

    if not frames:
        print("No valid data loaded from directory.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    out_paths = make_output_paths_for_dir(input_dir)

    # Save combined summaries
    save_combined_summaries(df_all, out_paths, run_col="run")

    # Save comparative boxplots (one image per metric, with runs side-by-side)
    save_comparative_boxplots(
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
