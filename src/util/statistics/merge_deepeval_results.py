"""
Merge DeepEval results into traditional metrics results without re-running DeepEval (for RAG pipeline)
"""
import json
import argparse
import numpy as np


def merge_deepeval_into_traditional(deepeval_file: str, traditional_file: str, output_file: str = None):
    """
    Merge DeepEval metrics into traditional metrics results to match the expected structure
    
    Args:
        deepeval_file: Path to deepeval results JSON file
        traditional_file: Path to traditional metrics results JSON  
        output_file: Output file path (optional, defaults to traditional_file)
    """
    
    # Load both files
    with open(deepeval_file, 'r', encoding='utf-8') as f:
        deepeval_data = json.load(f)
    
    with open(traditional_file, 'r', encoding='utf-8') as f:
        traditional_data = json.load(f)
    
    # Start with traditional data structure
    merged_data = traditional_data.copy()
    
    # Merge DeepEval metrics into detailed_results
    for filename, deepeval_questions in deepeval_data.items():
        if filename in merged_data['detailed_results']:
            traditional_questions = merged_data['detailed_results'][filename]
            
            # Match questions by index (assuming same order)
            for i, deepeval_q in enumerate(deepeval_questions):
                if i < len(traditional_questions):
                    # Get existing metrics (should be empty dict)
                    existing_metrics = traditional_questions[i].get('existing_metrics', {})
                    
                    # Add DeepEval metrics to existing_metrics
                    deepeval_metrics = deepeval_q.get('metrics', {})
                    
                    # Merge deepeval metrics into existing_metrics
                    for metric_name, metric_data in deepeval_metrics.items():
                        if isinstance(metric_data, dict):
                            existing_metrics[metric_name] = {
                                "score": metric_data.get('score', 0.0),
                                "success": metric_data.get('success', False),
                                "reason": metric_data.get('reason', 'N/A')
                            }
                    
                    # Update the existing_metrics field
                    traditional_questions[i]['existing_metrics'] = existing_metrics
    
    # Update aggregated metrics to include DeepEval
    _update_aggregated_metrics(merged_data)
    
    # Save merged results
    if output_file is None:
        output_file = traditional_file
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Merged results saved to {output_file}")
    return merged_data


def _update_aggregated_metrics(merged_data):
    """Update aggregated metrics to include DeepEval metrics"""
    
    all_metrics = {}
    
    # Collect all metrics from detailed results
    for filename, questions in merged_data['detailed_results'].items():
        file_metrics = {}
        
        for qa_result in questions:
            # Collect traditional metrics (direct fields)
            for key, value in qa_result.items():
                if (key not in ['question', 'expected', 'predicted', 'existing_metrics'] and 
                    key.endswith(('_score', '_f1', '_precision', '_recall', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge1', 'rouge2', 'rougeL')) and 
                    isinstance(value, (int, float))):
                    
                    if key not in file_metrics:
                        file_metrics[key] = []
                    file_metrics[key].append(value)
                    
                    # Also collect for overall stats
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
            
            # Collect DeepEval metrics from existing_metrics
            existing_metrics = qa_result.get('existing_metrics', {})
            for metric_name, metric_data in existing_metrics.items():
                if isinstance(metric_data, dict) and 'score' in metric_data:
                    score_key = f"deepeval_{metric_name}_score"
                    score_value = metric_data['score']
                    
                    if isinstance(score_value, (int, float)):
                        if score_key not in file_metrics:
                            file_metrics[score_key] = []
                        file_metrics[score_key].append(score_value)
                        
                        # Also collect for overall stats
                        if score_key not in all_metrics:
                            all_metrics[score_key] = []
                        all_metrics[score_key].append(score_value)
        
        # Calculate file-level averages for all metrics
        file_averages = merged_data['aggregated_metrics']['per_file'].get(filename, {})
        for metric, values in file_metrics.items():
            if values:
                # For DeepEval metrics, keep the original metric name structure
                if metric.startswith('deepeval_') and metric.endswith('_score'):
                    base_name = metric.replace('deepeval_', '').replace('_score', '')
                    file_averages[f'deepeval_{base_name}_mean'] = np.mean(values)
                    file_averages[f'deepeval_{base_name}_std'] = np.std(values)
                    file_averages[f'deepeval_{base_name}_count'] = len(values)
                else:
                    # For traditional metrics
                    file_averages[f'{metric}_mean'] = np.mean(values)
                    file_averages[f'{metric}_std'] = np.std(values)
                    file_averages[f'{metric}_count'] = len(values)
        
        merged_data['aggregated_metrics']['per_file'][filename] = file_averages
    
    # Calculate overall averages for all metrics
    overall_averages = merged_data['aggregated_metrics']['overall']
    for metric, values in all_metrics.items():
        if values:
            # For DeepEval metrics, keep the original metric name structure
            if metric.startswith('deepeval_') and metric.endswith('_score'):
                base_name = metric.replace('deepeval_', '').replace('_score', '')
                overall_averages[f'deepeval_{base_name}_mean'] = np.mean(values)
                overall_averages[f'deepeval_{base_name}_std'] = np.std(values)
                overall_averages[f'deepeval_{base_name}_count'] = len(values)
            else:
                # For traditional metrics
                overall_averages[f'{metric}_mean'] = np.mean(values)
                overall_averages[f'{metric}_std'] = np.std(values)
                overall_averages[f'{metric}_count'] = len(values)


def main():
    parser = argparse.ArgumentParser(description="Merge DeepEval results into traditional metrics")
    parser.add_argument('--deepeval-file', required=True, 
                        help='Path to DeepEval results JSON file')
    parser.add_argument('--traditional-file', required=True,
                        help='Path to traditional metrics results JSON file')
    parser.add_argument('--output-file', 
                        help='Output file path (optional, defaults to traditional-file)')
    
    args = parser.parse_args()
    
    merge_deepeval_into_traditional(
        args.deepeval_file, 
        args.traditional_file, 
        args.output_file
    )


if __name__ == '__main__':
    main()