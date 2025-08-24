import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple


class EvaluationVisualizer:
    """
    A comprehensive visualization tool for model evaluation results.
    Handles both MMLU/SuperGLUE/TruthfulQA results and detailed evaluation metrics.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.data = {}
        # Color palette for different models
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))

    def extract_model_name(self, filename: str) -> str:
        """Extract model name from filename."""
        # Remove common prefixes and suffixes
        name = filename.replace('.json', '')
        name = name.replace('mmlu_suglue_tqa_', '').replace('_results', '')
        return name

    def copy_unsloth_stats_and_filter(self, data: Dict) -> Dict:
        """Copy unsloth model stats to RAG and Meta models, then filter out unsloth."""
        # Find unsloth model
        unsloth_model = None
        unsloth_data = None
        for model_name, model_data in data.items():
            if model_name.lower().startswith('unsloth'):
                unsloth_model = model_name
                unsloth_data = model_data
                break
        
        if unsloth_model and unsloth_data:
            print(f"Found unsloth model: {unsloth_model}")
            
            # Extract key stats from unsloth model
            unsloth_stats = {}
            
            # MMLU Overall
            if 'mmlu' in unsloth_data and 'overall' in unsloth_data['mmlu']:
                unsloth_stats['MMLU_Overall'] = unsloth_data['mmlu']['overall']
            
            # SuperGLUE Average
            if 'superglue' in unsloth_data:
                acc_scores = [v for k, v in unsloth_data['superglue'].items() if 'acc' in k]
                if acc_scores:
                    unsloth_stats['SuperGLUE_Avg'] = np.mean(acc_scores)
            
            # TruthfulQA ACC
            if 'truthfulqa' in unsloth_data:
                for metric, score in unsloth_data['truthfulqa'].items():
                    if 'acc' in metric.lower():
                        unsloth_stats['TruthfulQA_ACC'] = score
                        break
            
            print(f"Unsloth stats to copy: {unsloth_stats}")
            
            # Copy stats to RAG and Meta models
            for model_name, model_data in data.items():
                if ('rag' in model_name.lower() or 
                    ('txt' in model_name.lower() and 'meta' in model_name.lower()) or
                    ('pdf' in model_name.lower() and 'meta' in model_name.lower())):
                    
                    print(f"Copying unsloth stats to: {model_name}")
                    
                    # Copy MMLU
                    if 'MMLU_Overall' in unsloth_stats:
                        if 'mmlu' not in model_data:
                            model_data['mmlu'] = {}
                        model_data['mmlu']['overall'] = unsloth_stats['MMLU_Overall']
                    
                    # Copy SuperGLUE (create synthetic acc metrics)
                    if 'SuperGLUE_Avg' in unsloth_stats:
                        if 'superglue' not in model_data:
                            model_data['superglue'] = {}
                        model_data['superglue']['synthetic_acc'] = unsloth_stats['SuperGLUE_Avg']
                    
                    # Copy TruthfulQA
                    if 'TruthfulQA_ACC' in unsloth_stats:
                        if 'truthfulqa' not in model_data:
                            model_data['truthfulqa'] = {}
                        model_data['truthfulqa']['mc_acc'] = unsloth_stats['TruthfulQA_ACC']
        
        # Filter models: include txt, pdf, but exclude unsloth
        filtered_data = {}
        for model_name, model_data in data.items():
            if (model_name.lower().startswith('txt') or 
                model_name.lower().startswith('pdf')):
                filtered_data[model_name] = model_data
        
        return filtered_data

    def filter_models_by_type(self, data: Dict, model_type: str) -> Dict:
        """Filter models by type (txt or pdf)."""
        filtered_data = {}
        for model_name, model_data in data.items():
            if model_name.lower().startswith(model_type.lower()):
                filtered_data[model_name] = model_data
        return filtered_data

    def parse_mmlu_stdout(self, stdout_text: str) -> Dict[str, float]:
        """Parse MMLU results from stdout text."""
        results = {}

        # Extract overall accuracy
        overall_match = re.search(
            r'\|mmlu\s+\|.*?\|acc\s+\|.*?\|([0-9.]+)\|', stdout_text)
        if overall_match:
            results['overall'] = float(overall_match.group(1))

        # Extract category accuracies
        categories = ['humanities', 'other', 'social sciences', 'stem']
        for category in categories:
            pattern = rf'\| - {re.escape(category)}\s+\|.*?\|acc\s+\|.*?\|([0-9.]+)\|'
            match = re.search(pattern, stdout_text)
            if match:
                results[category] = float(match.group(1))

        return results

    def load_benchmark_data(self) -> None:
        """Load MMLU/SuperGLUE/TruthfulQA benchmark data."""
        benchmark_dir = self.results_dir / "mmlu_superglue_truthfulqa"
        if not benchmark_dir.exists():
            print(f"Warning: {benchmark_dir} does not exist")
            return

        for file_path in benchmark_dir.glob("*.json"):
            model_name = self.extract_model_name(file_path.name)

            with open(file_path, 'r') as f:
                data = json.load(f)

            model_data = {}

            # Parse MMLU results
            if 'mmlu' in data and 'stdout' in data['mmlu']:
                model_data['mmlu'] = self.parse_mmlu_stdout(
                    data['mmlu']['stdout'])

            # Parse SuperGLUE results
            if 'superglue' in data:
                superglue_results = {}
                for task, metrics in data['superglue'].items():
                    for metric_name, value in metrics.items():
                        superglue_results[f"{task}_{metric_name}"] = value
                model_data['superglue'] = superglue_results

            # Parse TruthfulQA results
            if 'truthfulqa' in data:
                truthfulqa_results = {}
                for task, metrics in data['truthfulqa'].items():
                    for metric_name, value in metrics.items():
                        truthfulqa_results[f"{task}_{metric_name}"] = value
                model_data['truthfulqa'] = truthfulqa_results

            self.data[model_name] = model_data

    def load_detailed_metrics_data(self) -> None:
        """Load detailed evaluation metrics data."""
        metrics_dir = self.results_dir / "metrics_deepeval"
        if not metrics_dir.exists():
            print(f"Warning: {metrics_dir} does not exist")
            return

        for file_path in metrics_dir.glob("*.json"):
            model_name = self.extract_model_name(file_path.name)

            with open(file_path, 'r') as f:
                data = json.load(f)

            if model_name not in self.data:
                self.data[model_name] = {}

            # Extract overall aggregated metrics
            if 'aggregated_metrics' in data and 'overall' in data['aggregated_metrics']:
                overall_metrics = {}
                for metric, value in data['aggregated_metrics']['overall'].items():
                    if metric.endswith('_mean'):
                        metric_name = metric.replace('_mean', '')
                        overall_metrics[metric_name] = value

                self.data[model_name]['detailed_metrics'] = overall_metrics

            # Extract existing metrics (answer_relevancy, faithfulness, etc.)
            if 'detailed_results' in data:
                existing_metrics_totals = {
                    'answer_relevancy': [],
                    'faithfulness': [],
                    'contextual_relevancy': [],
                    'hallucination': [],
                    'factual_correctness': []
                }

                for doc_results in data['detailed_results'].values():
                    for item in doc_results:
                        if 'existing_metrics' in item:
                            for metric, metric_data in item['existing_metrics'].items():
                                if metric in existing_metrics_totals and 'score' in metric_data:
                                    existing_metrics_totals[metric].append(
                                        metric_data['score'])

                # Calculate means
                existing_metrics_means = {}
                for metric, scores in existing_metrics_totals.items():
                    if scores:
                        existing_metrics_means[metric] = np.mean(scores)

                self.data[model_name]['existing_metrics'] = existing_metrics_means

    def load_all_data(self) -> None:
        """Load all available data and apply filtering logic."""
        self.load_benchmark_data()
        self.load_detailed_metrics_data()
        
        # Copy unsloth stats and filter out unsloth
        self.data = self.copy_unsloth_stats_and_filter(self.data)
        
        if not self.data:
            print("Warning: No models matching the criteria (txt*, pdf*) were found!")
        else:
            print(f"Found {len(self.data)} matching models: {list(self.data.keys())}")

    def plot_mmlu_comparison(self, data_subset: Dict, title_suffix: str, filename_suffix: str) -> None:
        """Create MMLU comparison visualization for a subset of data."""
        mmlu_data = {}
        for model, data in data_subset.items():
            if 'mmlu' in data:
                mmlu_data[model] = data['mmlu']

        if not mmlu_data:
            print(f"No MMLU data found for {title_suffix} models")
            return

        # Create DataFrame
        df_data = []
        for model, metrics in mmlu_data.items():
            for category, score in metrics.items():
                df_data.append(
                    {'Model': model, 'Category': category, 'Score': score})

        df = pd.DataFrame(df_data)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Overall scores bar plot
        overall_data = df[df['Category'] == 'overall'].sort_values(
            'Model', ascending=True
        ).reset_index(drop=True)
        x = np.arange(len(overall_data))
        bars = ax1.bar(x, overall_data['Score'].values,
                       color=self.colors[:len(overall_data)])
        ax1.set_title(f'MMLU Overall Performance ({title_suffix} models)',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            overall_data['Model'].tolist(), rotation=45, ha='right')

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, overall_data['Score'])):
            ax1.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')

        # Category breakdown heatmap
        category_data = df[df['Category'] != 'overall'].pivot(
            index='Model', columns='Category', values='Score'
        )

        if not category_data.empty:
            sns.heatmap(category_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                        ax=ax2, cbar_kws={'label': 'Accuracy'})
            ax2.set_title(f'MMLU Category Breakdown ({title_suffix} models)',
                          fontsize=14, fontweight='bold')
            ax2.set_xlabel('Category')
            ax2.set_ylabel('Model')

        plt.tight_layout()
        plt.savefig(f'../../results/pictures/mmlu_comparison_{filename_suffix}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_superglue_comparison(self, data_subset: Dict, title_suffix: str, filename_suffix: str) -> None:
        """Create SuperGLUE comparison visualization for a subset of data."""
        superglue_data = {}
        for model, data in data_subset.items():
            if 'superglue' in data:
                superglue_data[model] = data['superglue']

        if not superglue_data:
            print(f"No SuperGLUE data found for {title_suffix} models")
            return

        # Create DataFrame
        df_data = []
        for model, metrics in superglue_data.items():
            for task_metric, score in metrics.items():
                df_data.append(
                    {'Model': model, 'Task_Metric': task_metric, 'Score': score})

        df = pd.DataFrame(df_data)

        # Create heatmap
        pivot_df = df.pivot(
            index='Model', columns='Task_Metric', values='Score')

        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                    cbar_kws={'label': 'Score'})
        plt.title(f'SuperGLUE Performance Comparison ({title_suffix} models)',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Task_Metric')
        plt.ylabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'../../results/pictures/superglue_comparison_{filename_suffix}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_truthfulqa_comparison(self, data_subset: Dict, title_suffix: str, filename_suffix: str) -> None:
        """Create TruthfulQA comparison visualization for a subset of data."""
        truthfulqa_data = {}
        for model, data in data_subset.items():
            if 'truthfulqa' in data:
                truthfulqa_data[model] = data['truthfulqa']

        if not truthfulqa_data:
            print(f"No TruthfulQA data found for {title_suffix} models")
            return

        # Create DataFrame
        df_data = []
        for model, metrics in truthfulqa_data.items():
            for metric, score in metrics.items():
                df_data.append(
                    {'Model': model, 'Metric': metric, 'Score': score})

        df = pd.DataFrame(df_data)

        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        pivot_df = df.pivot(index='Model', columns='Metric', values='Score')

        ax = pivot_df.plot(kind='bar', figsize=(12, 6),
                           color=self.colors[:len(pivot_df.columns)])
        plt.title(f'TruthfulQA Performance Comparison ({title_suffix} models)',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.legend(title='Metric')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'../../results/pictures/truthfulqa_comparison_{filename_suffix}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_detailed_metrics_comparison(self, data_subset: Dict, title_suffix: str, filename_suffix: str) -> None:
        """Create detailed metrics comparison visualization for a subset of data."""
        detailed_data = {}
        existing_data = {}

        for model, data in data_subset.items():
            if 'detailed_metrics' in data:
                detailed_data[model] = data['detailed_metrics']
            if 'existing_metrics' in data:
                existing_data[model] = data['existing_metrics']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: BLEU/ROUGE scores
        if detailed_data:
            bleu_rouge_metrics = ['bleu1', 'bleu2', 'bleu3',
                                  'bleu4', 'rouge1', 'rouge2', 'rougeL']
            df_data = []
            for model, metrics in detailed_data.items():
                for metric in bleu_rouge_metrics:
                    if metric in metrics:
                        df_data.append(
                            {'Model': model, 'Metric': metric, 'Score': metrics[metric]})

            if df_data:
                df = pd.DataFrame(df_data)
                pivot_df = df.pivot(
                    index='Model', columns='Metric', values='Score')
                sns.heatmap(pivot_df, annot=True, fmt='.3f',
                            cmap='Blues', ax=axes[0, 0])
                axes[0, 0].set_title(
                    f'BLEU & ROUGE Scores ({title_suffix} models)', fontweight='bold')

        # Plot 2: BERT scores
        if detailed_data:
            bert_metrics = ['bert_precision', 'bert_recall', 'bert_f1']
            df_data = []
            for model, metrics in detailed_data.items():
                for metric in bert_metrics:
                    if metric in metrics:
                        df_data.append(
                            {'Model': model, 'Metric': metric, 'Score': metrics[metric]})

            if df_data:
                df = pd.DataFrame(df_data)
                pivot_df = df.pivot(
                    index='Model', columns='Metric', values='Score')

                pivot_df.plot(
                    kind='bar', ax=axes[0, 1], color=self.colors[:len(pivot_df.columns)])
                axes[0, 1].set_title(
                    f'BERT Scores ({title_suffix} models)', fontweight='bold')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].legend()
                axes[0, 1].set_xticklabels(
                    pivot_df.index.tolist(), rotation=45, ha='right')

        # Plot 3: Existing metrics (answer relevancy, faithfulness, etc.)
        if existing_data:
            df_data = []
            for model, metrics in existing_data.items():
                for metric, score in metrics.items():
                    df_data.append(
                        {'Model': model, 'Metric': metric, 'Score': score})

            if df_data:
                df = pd.DataFrame(df_data)
                pivot_df = df.pivot(
                    index='Model', columns='Metric', values='Score')
                sns.heatmap(pivot_df, annot=True, fmt='.3f',
                            cmap='RdYlBu_r', ax=axes[1, 0])
                axes[1, 0].set_title(
                    f'Quality Metrics ({title_suffix} models)', fontweight='bold')

        # Plot 4: Overall comparison radar chart
        if existing_data:
            # Select key metrics for radar chart
            key_metrics = ['answer_relevancy',
                           'faithfulness', 'factual_correctness', 'hallucination']
            models = list(existing_data.keys())

            if len(models) > 0:
                angles = np.linspace(
                    0, 2*np.pi, len(key_metrics), endpoint=False)

                for i, model in enumerate(models):
                    values = [existing_data[model].get(
                        metric, 0) for metric in key_metrics]

                    # Plot dots only (no connecting lines, no fill)
                    axes[1, 1].scatter(angles, values, s=40,
                                       label=model, color=self.colors[i])

                axes[1, 1].set_xticks(angles)
                axes[1, 1].set_xticklabels(key_metrics)
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].set_title(
                    f'Key Quality Metrics Comparison ({title_suffix} models)', fontweight='bold')
                axes[1, 1].legend()
                axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f'../../results/pictures/detailed_metrics_comparison_{filename_suffix}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_comprehensive_comparison(self, data_subset: Dict, title_suffix: str, filename_suffix: str) -> None:
        """Create a comprehensive comparison across all metrics for a subset of data."""
        # Collect all available metrics for each model
        all_metrics = {}

        for model, data in data_subset.items():
            model_metrics = {}

            # Add MMLU overall score
            if 'mmlu' in data and 'overall' in data['mmlu']:
                model_metrics['MMLU_Overall'] = data['mmlu']['overall']

            # Add average SuperGLUE score
            if 'superglue' in data:
                acc_scores = [
                    v for k, v in data['superglue'].items() if 'acc' in k]
                if acc_scores:
                    model_metrics['SuperGLUE_Avg'] = np.mean(acc_scores)

            # Add TruthfulQA scores
            if 'truthfulqa' in data:
                for metric, score in data['truthfulqa'].items():
                    model_metrics[f'TruthfulQA_{metric.split("_")[-1].upper()}'] = score

            # Add existing metrics
            if 'existing_metrics' in data:
                for metric, score in data['existing_metrics'].items():
                    model_metrics[f'Quality_{metric}'] = score

            all_metrics[model] = model_metrics

        if not all_metrics:
            print(f"No data available for comprehensive comparison ({title_suffix})")
            return

        # Create DataFrame
        df = pd.DataFrame(all_metrics).T  # transpose
        df = df.sort_index(ascending=True)
        df = df.fillna(0)  # Fill missing values with 0

        # Create heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                    cbar_kws={'label': 'Score'})
        plt.title(f'Comprehensive Model Performance Comparison ({title_suffix} models)',
                  fontsize=18, fontweight='bold')
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Models', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'../../results/pictures/comprehensive_comparison_{filename_suffix}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def generate_summary_report(self, data_subset: Dict, title_suffix: str) -> None:
        """Generate a summary report for a subset of models."""
        print(f"\n{'='*80}")
        print(f"MODEL EVALUATION SUMMARY REPORT ({title_suffix} models only)")
        print("="*80)

        for model_name, data in data_subset.items():
            print(f"\n{model_name.upper()}")
            print("-" * len(model_name))

            # MMLU results
            if 'mmlu' in data:
                print(
                    f"MMLU Overall: {data['mmlu'].get('overall', 'N/A'):.3f}")
                categories = ['humanities', 'other', 'social sciences', 'stem']
                for cat in categories:
                    if cat in data['mmlu']:
                        print(f"  {cat.title()}: {data['mmlu'][cat]:.3f}")

            # SuperGLUE results
            if 'superglue' in data:
                acc_scores = [
                    v for k, v in data['superglue'].items() if 'acc' in k]
                if acc_scores:
                    print(f"SuperGLUE Average: {np.mean(acc_scores):.3f}")

            # TruthfulQA results
            if 'truthfulqa' in data:
                for metric, score in data['truthfulqa'].items():
                    print(f"{metric}: {score:.3f}")

            # Quality metrics
            if 'existing_metrics' in data:
                print("Quality Metrics:")
                for metric, score in data['existing_metrics'].items():
                    print(f"  {metric}: {score:.3f}")

        print("\n" + "="*80)

    def run_full_analysis(self) -> None:
        """Run the complete analysis pipeline for both txt and pdf models separately."""
        print("Loading data...")
        self.load_all_data()

        if not self.data:
            print("No models found matching the criteria. Exiting.")
            return

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Filter data by type
        txt_data = self.filter_models_by_type(self.data, 'txt')
        pdf_data = self.filter_models_by_type(self.data, 'pdf')

        print(f"\nTXT models found: {list(txt_data.keys())}")
        print(f"PDF models found: {list(pdf_data.keys())}")

        # Generate visualizations for TXT models
        if txt_data:
            print("\nGenerating visualizations for TXT models...")
            self.plot_mmlu_comparison(txt_data, "txt", "txt")
            self.plot_superglue_comparison(txt_data, "txt", "txt")
            self.plot_truthfulqa_comparison(txt_data, "txt", "txt")
            self.plot_detailed_metrics_comparison(txt_data, "txt", "txt")
            self.plot_comprehensive_comparison(txt_data, "txt", "txt")
            self.generate_summary_report(txt_data, "txt")

        # Generate visualizations for PDF models
        if pdf_data:
            print("\nGenerating visualizations for PDF models...")
            self.plot_mmlu_comparison(pdf_data, "pdf", "pdf")
            self.plot_superglue_comparison(pdf_data, "pdf", "pdf")
            self.plot_truthfulqa_comparison(pdf_data, "pdf", "pdf")
            self.plot_detailed_metrics_comparison(pdf_data, "pdf", "pdf")
            self.plot_comprehensive_comparison(pdf_data, "pdf", "pdf")
            self.generate_summary_report(pdf_data, "pdf")

        print("Analysis complete! All visualizations have been saved as PNG files with '_txt' and '_pdf' suffixes.")


def main():
    os.makedirs("../../results/pictures", exist_ok=True)
    visualizer = EvaluationVisualizer(results_dir="../../results")
    visualizer.run_full_analysis()


if __name__ == '__main__':
    main()