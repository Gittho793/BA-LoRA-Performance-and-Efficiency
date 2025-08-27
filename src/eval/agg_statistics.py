import json
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from pathlib import Path
from scipy import stats  # Add this import for confidence intervals


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute confidence interval for a dataset

    Args:
        data: numpy array or list of values
        confidence: confidence level (default 0.95 for 95% CI)

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)

    if n < 2:
        return (float(data[0]), float(data[0])) if n == 1 else (np.nan, np.nan)

    # Calculate mean and standard error
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean

    # Calculate confidence interval using t-distribution
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_error = t_value * std_err

    lower_bound = mean - margin_error
    upper_bound = mean + margin_error

    return (float(lower_bound), float(upper_bound))


def compute_metrics_statistics(json_file_path):
    """
    Compute mean and std for all metrics in the results JSON file
    """

    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Dictionary to store all metric values
    all_metrics = defaultdict(list)

    # Extract metrics from detailed_results
    detailed_results = data.get('detailed_results', {})

    for file_name, results_list in detailed_results.items():
        for result in results_list:
            # Extract existing_metrics scores
            existing_metrics = result.get('existing_metrics', {})
            for metric_name, metric_data in existing_metrics.items():
                if isinstance(metric_data, dict) and 'score' in metric_data:
                    all_metrics[metric_name].append(metric_data['score'])

            # Extract BLEU scores
            for bleu_metric in ['bleu1', 'bleu2', 'bleu3', 'bleu4']:
                if bleu_metric in result:
                    all_metrics[bleu_metric].append(result[bleu_metric])

            # Extract ROUGE scores
            for rouge_metric in ['rouge1', 'rouge2', 'rougeL']:
                if rouge_metric in result:
                    all_metrics[rouge_metric].append(result[rouge_metric])

            # Extract BERT scores
            for bert_metric in ['bert_precision', 'bert_recall', 'bert_f1']:
                if bert_metric in result:
                    all_metrics[bert_metric].append(result[bert_metric])

    # Compute statistics
    stats_results = {}

    for metric_name, values in all_metrics.items():
        if values:  # Only compute if we have values
            values_array = np.array(values)
            stats_results[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'variance': float(np.var(values_array)),
                'count': len(values),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array))
            }

    return stats_results


def numpy_mode(data):
    values, counts = np.unique(data, return_counts=True)
    max_count_index = np.argmax(counts)
    return values[max_count_index]


def add_existing_metrics_to_aggregated(json_file_path, confidence_level=0.95):
    """
    Extract only existing_metrics from detailed_results and add them to aggregated_metrics.overall
    """

    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Dictionary to store existing metric values
    existing_metrics_values = defaultdict(list)

    # Extract metrics from detailed_results
    detailed_results = data.get('detailed_results', {})

    for file_name, results_list in detailed_results.items():
        for result in results_list:
            # Only extract existing_metrics scores
            existing_metrics = result.get('existing_metrics', {})
            for metric_name, metric_data in existing_metrics.items():
                if isinstance(metric_data, dict) and 'score' in metric_data:
                    existing_metrics_values[metric_name].append(
                        metric_data['score'])

    # Compute statistics for existing metrics only
    existing_stats = {}

    for metric_name, values in existing_metrics_values.items():
        if values:  # Only compute if we have values
            values_array = np.array(values)
            # Create the metric name format: metric_name + "_mean", metric_name + "_std", metric_name + "_count"
            base_name = metric_name.lower()

            # Compute confidence interval
            ci_lower, ci_upper = compute_confidence_interval(values_array, confidence_level)

            existing_stats[f"{base_name}_mean"] = float(np.mean(values_array))
            existing_stats[f"{base_name}_std"] = float(np.std(values_array))
            existing_stats[f"{base_name}_var"] = float(np.var(values_array))
            existing_stats[f"{base_name}_median"] = float(np.median(values_array))
            existing_stats[f"{base_name}_mode"] = numpy_mode(values_array)
            existing_stats[f"{base_name}_count"] = len(values)
            existing_stats[f"{base_name}_ci_lower"] = ci_lower
            existing_stats[f"{base_name}_ci_upper"] = ci_upper
            existing_stats[f"{base_name}_ci_level"] = confidence_level

    # Add to aggregated_metrics.overall
    if 'aggregated_metrics' not in data:
        data['aggregated_metrics'] = {}
    if 'overall' not in data['aggregated_metrics']:
        data['aggregated_metrics']['overall'] = {}

    # Add the existing metrics statistics to the overall section
    data['aggregated_metrics']['overall'].update(existing_stats)

    # Save the updated JSON back to the file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return existing_stats


def process_directory(directory_path, output_dir=None):
    """
    Process all JSON files in a directory
    """
    directory_path = Path(directory_path)

    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = directory_path

    # Find all JSON files in the directory
    json_files = list(directory_path.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return

    print(f"Found {len(json_files)} JSON files in {directory_path}")
    print("=" * 80)

    # Process each JSON file
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        print("-" * 50)

        try:
            # Add existing metrics to aggregated
            added_stats = add_existing_metrics_to_aggregated(json_file)
            print(
                f"Added {len(added_stats)} existing metrics to aggregated_metrics.overall")

            # Compute all metrics statistics
            stats = compute_metrics_statistics(json_file)

            # Create output filename prefix
            file_stem = json_file.stem

            # Save statistics to CSV
            df_stats = pd.DataFrame.from_dict(stats, orient='index')
            df_stats = df_stats.round(4)
            csv_output_path = output_dir / f"{file_stem}_statistics.csv"
            df_stats.to_csv(csv_output_path)

            # Save statistics as JSON
            json_output_path = output_dir / f"{file_stem}_statistics.json"
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            # Create summary report
            report_path = output_dir / f"{file_stem}_summary_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(
                    f"EVALUATION METRICS SUMMARY REPORT: {json_file.name}\n")
                f.write("=" * 80 + "\n\n")

                if stats:
                    f.write(
                        f"Total number of evaluations: {next(iter(stats.values()))['count']}\n\n")

                    # Group metrics by category
                    categories = {
                        'Quality Metrics': ['answer_relevancy', 'faithfulness', 'contextual_relevancy', 'hallucination', 'factual_correctness'],
                        'BLEU Scores': ['bleu1', 'bleu2', 'bleu3', 'bleu4'],
                        'ROUGE Scores': ['rouge1', 'rouge2', 'rougeL'],
                        'BERT Scores': ['bert_precision', 'bert_recall', 'bert_f1']
                    }

                    for category, metrics in categories.items():
                        category_metrics = [m for m in metrics if m in stats]
                        if category_metrics:
                            f.write(f"{category}:\n")
                            f.write("-" * len(category) + "\n")

                            for metric in category_metrics:
                                s = stats[metric]
                                f.write(
                                    f"{metric.upper():20} | Mean: {s['mean']:6.4f} | Std: {s['std']:6.4f} | Range: [{s['min']:6.4f}, {s['max']:6.4f}]\n")
                            f.write("\n")

                    # Key insights
                    f.write("KEY INSIGHTS:\n")
                    f.write("-" * 13 + "\n")

                    # Best and worst performing metrics
                    mean_scores = {metric: data['mean'] for metric, data in stats.items(
                    ) if 'bleu' not in metric.lower()}
                    if mean_scores:
                        best_metric = max(mean_scores.items(),
                                          key=lambda x: x[1])
                        worst_metric = min(
                            mean_scores.items(), key=lambda x: x[1])

                        f.write(
                            f"• Highest mean score: {best_metric[0]} ({best_metric[1]:.4f})\n")
                        f.write(
                            f"• Lowest mean score: {worst_metric[0]} ({worst_metric[1]:.4f})\n")

                    # Variability insights
                    std_scores = {metric: data['std']
                                  for metric, data in stats.items()}
                    if std_scores:
                        most_variable = max(
                            std_scores.items(), key=lambda x: x[1])
                        least_variable = min(
                            std_scores.items(), key=lambda x: x[1])

                        f.write(
                            f"• Most variable metric: {most_variable[0]} (std: {most_variable[1]:.4f})\n")
                        f.write(
                            f"• Least variable metric: {least_variable[0]} (std: {least_variable[1]:.4f})\n")
                else:
                    f.write("No metrics found in this file.\n")

            print(f"✓ Generated outputs:")
            print(f"  - CSV: {csv_output_path}")
            print(f"  - JSON: {json_output_path}")
            print(f"  - Report: {report_path}")

        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {str(e)}")

    print("\n" + "=" * 80)
    print("Processing complete!")


# Main execution
if __name__ == "__main__":
    # Define directory containing JSON files
    results_directory = '../../results/metrics_deepeval/'

    # Optional: specify output directory (if None, outputs will be saved in the same directory as inputs)
    output_directory = '../../results/statistics_output/'

    # Process all JSON files in the directory
    process_directory(results_directory, output_directory)
