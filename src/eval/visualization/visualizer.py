import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
import os
import sys
# Add import for pretty label helpers

load_dotenv("../../../.env")  # necessary for local imports on cluster

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.eval.visualization.violin import pretty_run_label, sort_labels_numerically

plt.rcParams['font.size'] = 16


class EvaluationVisualizer:
    """
    Visualization tool for model evaluation results.
    Handles MMLU / SuperGLUE / TruthfulQA plus detailed evaluation metrics.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.data: Dict[str, Dict] = {}
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))

    # --------------------------
    # Helpers & data ingestion
    # --------------------------
    def extract_model_name(self, filename: str) -> str:
        name = filename.replace(".json", "")
        name = name.replace("mmlu_suglue_tqa_", "").replace("_results", "")
        return name

    def parse_mmlu_stdout(self, stdout_text: str) -> Dict[str, float]:
        """Parse MMLU results from stdout."""
        results: Dict[str, float] = {}

        # Overall
        overall_match = re.search(
            r"\|mmlu\s+\|.*?\|acc\s+\|.*?\|([0-9.]+)\|", stdout_text)
        if overall_match:
            results["overall"] = float(overall_match.group(1))

        # Categories
        for category in ["humanities", "other", "social sciences", "stem"]:
            m = re.search(
                rf"\|\s*-\s*{re.escape(category)}\s+\|.*?\|acc\s+\|.*?\|([0-9.]+)\|", stdout_text)
            if m:
                results[category] = float(m.group(1))

        return results

    def load_benchmark_data(self) -> None:
        bench_dir = self.results_dir / "mmlu_superglue_truthfulqa"
        if not bench_dir.exists():
            print(f"Warning: {bench_dir} does not exist")
            return

        for file_path in bench_dir.glob("*.json"):
            model_name = self.extract_model_name(file_path.name)
            with open(file_path, "r") as f:
                raw = json.load(f)

            model_data: Dict[str, Dict] = {}

            # MMLU
            if "mmlu" in raw and isinstance(raw["mmlu"], dict) and "stdout" in raw["mmlu"]:
                model_data["mmlu"] = self.parse_mmlu_stdout(
                    raw["mmlu"]["stdout"])

            # SuperGLUE: metric keys already include task (e.g., "cb_acc", "record_f1")
            if "superglue" in raw and isinstance(raw["superglue"], dict):
                sg: Dict[str, float] = {}
                for _, metrics in raw["superglue"].items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            # Keep the metric name as-is (no double prefixing)
                            sg[metric_name] = value
                if sg:
                    model_data["superglue"] = sg

            # TruthfulQA: keys are like "truthfulqa_mc1": {"truthfulqa_mc1_acc": ...}
            if "truthfulqa" in raw and isinstance(raw["truthfulqa"], dict):
                tqa: Dict[str, float] = {}
                for _, metrics in raw["truthfulqa"].items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            # Normalize to short names: "truthfulqa_mc1_acc" -> "mc1_acc"
                            short = re.sub(r"^truthfulqa_", "", metric_name)
                            tqa[short] = value
                if tqa:
                    model_data["truthfulqa"] = tqa

            self.data[model_name] = model_data

    def load_detailed_metrics_data(self) -> None:
        metrics_dir = self.results_dir / "metrics_deepeval"
        if not metrics_dir.exists():
            print(f"Warning: {metrics_dir} does not exist")
            return

        for file_path in metrics_dir.glob("*.json"):
            model_name = self.extract_model_name(file_path.name)
            with open(file_path, "r") as f:
                data = json.load(f)

            if model_name not in self.data:
                self.data[model_name] = {}

            # Aggregated metrics (BLEU/ROUGE/BERT*)
            if "aggregated_metrics" in data and "overall" in data["aggregated_metrics"]:
                overall = data["aggregated_metrics"]["overall"]
                overall_metrics = {
                    k.replace("_mean", ""): v for k, v in overall.items() if k.endswith("_mean")}
                if overall_metrics:
                    self.data[model_name]["detailed_metrics"] = overall_metrics

            # Existing metrics (answer_relevancy, faithfulness, etc.)
            if "detailed_results" in data and isinstance(data["detailed_results"], dict):
                acc = {"answer_relevancy": [], "faithfulness": [], "contextual_relevancy": [],
                       "hallucination": [], "factual_correctness": []}
                for doc_results in data["detailed_results"].values():
                    for item in doc_results:
                        em = item.get("existing_metrics", {})
                        for metric in acc.keys():
                            if metric in em and isinstance(em[metric], dict) and "score" in em[metric]:
                                acc[metric].append(em[metric]["score"])
                means = {k: float(np.mean(v)) for k, v in acc.items() if v}
                if means:
                    self.data[model_name]["existing_metrics"] = means

    def load_all_data(self) -> None:
        self.load_benchmark_data()
        self.load_detailed_metrics_data()
        if not self.data:
            print("Warning: no evaluation data found.")

    def filter_models_by_type(self, data: Dict, model_type: str) -> Dict:
        """Return only models whose name starts with `model_type` (e.g., 'txt', 'pdf')."""
        return {m: d for m, d in data.items() if m.lower().startswith(model_type.lower())}

    # --------------------------
    # Plotting
    # --------------------------
    def plot_mmlu_comparison(self, data_subset: Dict, title_suffix: str, filename_suffix: str) -> None:
        mmlu_data = {m: d["mmlu"]
                     for m, d in data_subset.items() if "mmlu" in d and d["mmlu"]}
        if not mmlu_data:
            print(f"No MMLU data for {title_suffix} models")
            return

        # Use pretty labels and aggregate duplicates by mean
        rows = [{"Label": pretty_run_label(model), "Category": cat, "Score": sc}
                for model, metrics in mmlu_data.items()
                for cat, sc in metrics.items()]
        df = pd.DataFrame(rows)

        # Determine sorted label order
        labels_sorted = sort_labels_numerically(sorted(df["Label"].unique().tolist()))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Overall bar chart (mean across duplicates)
        overall = (df[df["Category"] == "overall"]
                   .groupby("Label", as_index=False)["Score"].mean())
        # Reindex to sorted label order
        overall = overall.set_index("Label").reindex(labels_sorted).dropna().reset_index()

        x = np.arange(len(overall))
        bars = ax1.bar(x, overall["Score"].values, color=self.colors[:len(overall)])
        ax1.set_title(
            f"MMLU Overall Performance ({title_suffix} models)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(overall["Label"].tolist(), rotation=45, ha="right")
        for i, (bar, score) in enumerate(zip(bars, overall["Score"].values)):
            ax1.text(i, min(score + 0.01, 0.99), f"{score:.3f}", ha="center", va="bottom")

        # Category heatmap (mean across duplicates)
        cats = (df[df["Category"] != "overall"]
                .groupby(["Label", "Category"], as_index=False)["Score"].mean())
        if not cats.empty:
            cat_df = cats.pivot(index="Label", columns="Category", values="Score")
            cat_df = cat_df.reindex(labels_sorted)
            sns.heatmap(cat_df, annot=True, fmt=".3f", cmap="RdYlBu_r",
                        ax=ax2, cbar_kws={"label": "Accuracy"})
            ax2.set_title(
                f"MMLU Category Breakdown ({title_suffix} models)", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Category")
            ax2.set_ylabel("Model")

        plt.tight_layout()
        out = self.results_dir / "pictures" / f"mmlu_comparison_{filename_suffix}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_superglue_comparison(self, data_subset: Dict, title_suffix: str, filename_suffix: str) -> None:
        sg_data = {m: d["superglue"] for m, d in data_subset.items()
                   if "superglue" in d and d["superglue"]}
        if not sg_data:
            print(f"No SuperGLUE data for {title_suffix} models")
            return

        rows = [{"Label": pretty_run_label(model), "Task_Metric": k, "Score": v}
                for model, metrics in sg_data.items() for k, v in metrics.items()]
        df = pd.DataFrame(rows)

        # Aggregate duplicates by mean and pivot
        agg = df.groupby(["Label", "Task_Metric"], as_index=False)["Score"].mean()
        pivot_df = agg.pivot(index="Label", columns="Task_Metric", values="Score")

        # Stable column ordering
        preferred = ["boolq_acc", "cb_acc", "cb_f1", "copa_acc", "multirc_acc",
                     "record_em", "record_f1", "rte_acc", "wic_acc", "wsc_acc"]
        cols = [c for c in preferred if c in pivot_df.columns] + \
               [c for c in pivot_df.columns if c not in preferred]
        pivot_df = pivot_df[cols]

        # Order labels using sort_labels_numerically
        labels_sorted = sort_labels_numerically(sorted(pivot_df.index.tolist()))
        pivot_df = pivot_df.reindex(labels_sorted)

        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".3f",
                    cmap="RdYlBu_r", cbar_kws={"label": "Score"})
        plt.title(
            f"SuperGLUE Performance Comparison ({title_suffix} models)", fontsize=16, fontweight="bold")
        plt.xlabel("Task_Metric")
        plt.ylabel("Model")
        plt.xticks(rotation=45, fontsize=16)
        plt.tight_layout()
        out = self.results_dir / "pictures" / f"superglue_comparison_{filename_suffix}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_truthfulqa_comparison(self, data_subset: Dict, title_suffix: str, filename_suffix: str) -> None:
        tqa_data = {m: d["truthfulqa"] for m, d in data_subset.items()
                    if "truthfulqa" in d and d["truthfulqa"]}
        if not tqa_data:
            print(f"No TruthfulQA data for {title_suffix} models")
            return

        rows = [{"Label": pretty_run_label(m), "Metric": k, "Score": v}
                for m, metrics in tqa_data.items() for k, v in metrics.items()]
        df = pd.DataFrame(rows)

        agg = df.groupby(["Label", "Metric"], as_index=False)["Score"].mean()
        pivot_df = agg.pivot(index="Label", columns="Metric", values="Score")

        keep = [c for c in ["mc1_acc", "mc2_acc"] if c in pivot_df.columns]
        pivot_df = pivot_df[keep]

        labels_sorted = sort_labels_numerically(sorted(pivot_df.index.tolist()))
        pivot_df = pivot_df.reindex(labels_sorted)

        color_map = {"mc1_acc": "#1f77b4", "mc2_acc": "#ff7f0e"}
        bar_colors = [color_map.get(c, self.colors[i % len(self.colors)])
                      for i, c in enumerate(pivot_df.columns)]

        ax = pivot_df.plot(kind="bar", figsize=(12, 6), color=bar_colors)
        ax.set_title(
            f"TruthfulQA Performance Comparison ({title_suffix} models)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.legend(title="Metric")
        ax.set_xticklabels(pivot_df.index.tolist(), rotation=45, ha="right", fontsize=16)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = self.results_dir / "pictures" / f"truthfulqa_comparison_{filename_suffix}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_detailed_metrics_comparison(self, data_subset: Dict, title_suffix: str, filename_suffix: str) -> None:
        detailed = {pretty_run_label(m): d["detailed_metrics"]
                    for m, d in data_subset.items() if "detailed_metrics" in d}
        existing = {pretty_run_label(m): d["existing_metrics"]
                    for m, d in data_subset.items() if "existing_metrics" in d}

        if not detailed and not existing:
            print(f"No detailed metrics for {title_suffix} models")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (1) BLEU/ROUGE heatmap (aggregate by label)
        if detailed:
            blu_rou = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge1", "rouge2", "rougeL"]
            rows = [{"Label": lbl, "Metric": k, "Score": v}
                    for lbl, ms in detailed.items() for k, v in ms.items() if k in blu_rou]
            if rows:
                df = pd.DataFrame(rows)
                pv = (df.groupby(["Label", "Metric"], as_index=False)["Score"].mean()
                      .pivot(index="Label", columns="Metric", values="Score"))
                pv = pv.reindex(sort_labels_numerically(sorted(pv.index.tolist())))
                sns.heatmap(pv, annot=True, fmt=".3f", cmap="Blues", ax=axes[0, 0])
                axes[0, 0].set_title(
                    f"BLEU & ROUGE Scores ({title_suffix} models)", fontweight="bold")

        # (2) BERT scores bar chart (aggregate by label)
        if detailed:
            bert = ["bert_precision", "bert_recall", "bert_f1"]
            rows = [{"Label": lbl, "Metric": k, "Score": v}
                    for lbl, ms in detailed.items() for k, v in ms.items() if k in bert]
            if rows:
                df = pd.DataFrame(rows)
                pv = (df.groupby(["Label", "Metric"], as_index=False)["Score"].mean()
                      .pivot(index="Label", columns="Metric", values="Score"))
                pv = pv.reindex(sort_labels_numerically(sorted(pv.index.tolist())))
                pv.plot(kind="bar", ax=axes[0, 1], color=self.colors[:len(pv.columns)])
                axes[0, 1].set_title(
                    f"BERT Scores ({title_suffix} models)", fontweight="bold")
                axes[0, 1].set_ylabel("Score")
                axes[0, 1].legend()
                axes[0, 1].set_xticklabels(pv.index.tolist(), rotation=45, ha="right")

        # (3) Existing metrics heatmap (aggregate by label)
        if existing:
            rows = [{"Label": lbl, "Metric": k, "Score": v}
                    for lbl, ms in existing.items() for k, v in ms.items()]
            df = pd.DataFrame(rows)
            pv = (df.groupby(["Label", "Metric"], as_index=False)["Score"].mean()
                  .pivot(index="Label", columns="Metric", values="Score"))
            pv = pv.reindex(sort_labels_numerically(sorted(pv.index.tolist())))
            sns.heatmap(pv, annot=True, fmt=".3f", cmap="RdYlBu_r", ax=axes[1, 0])
            axes[1, 0].set_title(
                f"Quality Metrics ({title_suffix} models)", fontweight="bold")

        # (4) Key metric scatter (aggregate by label)
        if existing:
            key = ["answer_relevancy", "faithfulness", "factual_correctness", "hallucination"]
            labels_sorted = sort_labels_numerically(sorted(existing.keys()))
            angles = np.linspace(0, 2 * np.pi, len(key), endpoint=False)
            for i, lbl in enumerate(labels_sorted):
                vals = [existing.get(lbl, {}).get(k, 0.0) for k in key]
                axes[1, 1].scatter(angles, vals, s=40, label=lbl, color=self.colors[i % len(self.colors)])
            axes[1, 1].set_xticks(angles)
            axes[1, 1].set_xticklabels(key)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title(
                f"Key Quality Metrics Comparison ({title_suffix} models)", fontweight="bold")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        out = self.results_dir / "pictures" / f"detailed_metrics_comparison_{filename_suffix}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_comprehensive_comparison(self, data_subset: Dict, title_suffix: str, filename_suffix: str) -> None:
        # Collect per-model metrics, labeled by pretty label
        rows = []
        for model, data in data_subset.items():
            label = pretty_run_label(model)
            if "mmlu" in data and "overall" in data["mmlu"]:
                rows.append({"Label": label, "Metric": "MMLU_Overall", "Score": data["mmlu"]["overall"]})
            if "superglue" in data:
                acc_scores = [v for k, v in data["superglue"].items() if "acc" in k]
                if acc_scores:
                    rows.append({"Label": label, "Metric": "SuperGLUE_OverallAccMean", "Score": float(np.mean(acc_scores))})
            if "truthfulqa" in data:
                for metric, score in data["truthfulqa"].items():
                    rows.append({"Label": label, "Metric": f"TQA_{metric}", "Score": score})
            if "existing_metrics" in data:
                for metric, score in data["existing_metrics"].items():
                    rows.append({"Label": label, "Metric": f"Quality_{metric}", "Score": score})

        if not rows:
            print(f"No data for comprehensive comparison ({title_suffix})")
            return

        df = pd.DataFrame(rows)
        # Aggregate duplicates by mean and pivot
        pv = (df.groupby(["Label", "Metric"], as_index=False)["Score"].mean()
              .pivot(index="Label", columns="Metric", values="Score")).fillna(0.0)
        pv = pv.reindex(sort_labels_numerically(sorted(pv.index.tolist())))

        plt.figure(figsize=(16, 10))
        sns.heatmap(pv, annot=True, fmt=".3f", cmap="RdYlBu_r",
                    cbar_kws={"label": "Score"})
        plt.title(f"Comprehensive Model Performance Comparison ({title_suffix} models)",
                  fontsize=18, fontweight="bold")
        plt.xlabel("Metrics")
        plt.ylabel("Models")
        plt.xticks(rotation=45)
        plt.tight_layout()
        out = self.results_dir / "pictures" / f"comprehensive_comparison_{filename_suffix}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show()

    def generate_summary_report(self, data_subset: Dict, title_suffix: str) -> None:
        print("\n" + "=" * 80)
        print(f"MODEL EVALUATION SUMMARY REPORT ({title_suffix} models)")
        print("=" * 80)

        # Group models by pretty label
        grouped: Dict[str, List[str]] = {}
        for model_name in data_subset.keys():
            lbl = pretty_run_label(model_name)
            grouped.setdefault(lbl, []).append(model_name)

        labels_sorted = sort_labels_numerically(sorted(grouped.keys()))

        for label in labels_sorted:
            models = grouped[label]
            # Merge metrics across models mapping to same label by mean
            merged = {}

            # Prepare accumulators
            mmlu_overall = []
            mmlu_cats = {"humanities": [], "other": [], "social sciences": [], "stem": []}
            superglue_accs = []
            tqa_scores: Dict[str, List[float]] = {}
            qual_scores: Dict[str, List[float]] = {}

            for model_name in models:
                data = data_subset[model_name]
                if "mmlu" in data and data["mmlu"]:
                    if "overall" in data["mmlu"]:
                        mmlu_overall.append(data["mmlu"]["overall"])
                    for k in mmlu_cats.keys():
                        if k in data["mmlu"]:
                            mmlu_cats[k].append(data["mmlu"][k])
                if "superglue" in data and data["superglue"]:
                    superglue_accs.extend([v for k, v in data["superglue"].items() if "acc" in k])
                if "truthfulqa" in data and data["truthfulqa"]:
                    for k, v in data["truthfulqa"].items():
                        tqa_scores.setdefault(k, []).append(v)
                if "existing_metrics" in data and data["existing_metrics"]:
                    for k, v in data["existing_metrics"].items():
                        qual_scores.setdefault(k, []).append(v)

            print(f"\n{label}")
            print("-" * len(label))
            if mmlu_overall:
                print(f"MMLU Overall: {np.mean(mmlu_overall):.3f}")
            for k, vals in mmlu_cats.items():
                if vals:
                    print(f"  {k.title()}: {np.mean(vals):.3f}")
            if superglue_accs:
                print(f"SuperGLUE mean(acc): {np.mean(superglue_accs):.3f}")
            for k, vals in tqa_scores.items():
                print(f"{k}: {np.mean(vals):.3f}")
            if qual_scores:
                print("Quality Metrics:")
                for k, vals in qual_scores.items():
                    print(f"  {k}: {np.mean(vals):.3f}")

        print("\n" + "=" * 80)

    # --------------------------
    # Pipeline
    # --------------------------
    def run_full_analysis(self) -> None:
        print("Loading data...")
        self.load_all_data()
        if not self.data:
            print("No models found. Exiting.")
            return

        # Optional style
        # plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Split by type
        txt_data = self.filter_models_by_type(self.data, "txt")
        pdf_data = self.filter_models_by_type(self.data, "pdf")

        # Print pretty labels
        txt_labels = sort_labels_numerically([pretty_run_label(m) for m in txt_data.keys()])
        pdf_labels = sort_labels_numerically([pretty_run_label(m) for m in pdf_data.keys()])
        print(f"\nTXT models: {txt_labels}")
        print(f"PDF models: {pdf_labels}")

        if txt_data:
            print("\nGenerating visualizations for TXT models...")
            self.plot_mmlu_comparison(txt_data, "txt", "txt")
            self.plot_superglue_comparison(txt_data, "txt", "txt")
            self.plot_truthfulqa_comparison(txt_data, "txt", "txt")
            self.plot_detailed_metrics_comparison(txt_data, "txt", "txt")
            self.plot_comprehensive_comparison(txt_data, "txt", "txt")
            self.generate_summary_report(txt_data, "txt")

        if pdf_data:
            print("\nGenerating visualizations for PDF models...")
            self.plot_mmlu_comparison(pdf_data, "pdf", "pdf")
            self.plot_superglue_comparison(pdf_data, "pdf", "pdf")
            self.plot_truthfulqa_comparison(pdf_data, "pdf", "pdf")
            self.plot_detailed_metrics_comparison(pdf_data, "pdf", "pdf")
            self.plot_comprehensive_comparison(pdf_data, "pdf", "pdf")
            self.generate_summary_report(pdf_data, "pdf")

        print("Analysis complete! Images saved under results/pictures/.")


def main():
    # Adjust this path to wherever your results live
    out_dir = Path("../../../results")
    (out_dir / "pictures").mkdir(parents=True, exist_ok=True)
    visualizer = EvaluationVisualizer(results_dir=str(out_dir))
    visualizer.run_full_analysis()


if __name__ == "__main__":
    main()
