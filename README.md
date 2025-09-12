# LlamaLora

A comprehensive machine learning project for training, evaluating, and deploying LoRA (Low-Rank Adaptation) models on educational content, specifically focused on German educational texts about counseling and guidance.

## Project Overview

This project implements a complete pipeline for:
- Training LoRA models using the Llama 3.1 8B architecture
- Retrieval-Augmented Generation (RAG) evaluation
- Question-answer generation for educational content
- Model evaluation using various metrics and benchmarks

## Abstract

This study investigates the performance and efficiency of Low-Rank Adaptation (LoRA) as a parameter-efficient fine-tuning method for injecting knowledge into large language models (LLMs) in the context of academic and career counseling. A foundation model was fine-tuned with LoRA on an AI-generated domain-specific dataset and compared to both the baseline and a Retrieval-Augmented Generation (RAG) pipeline. Evaluation combined LLM-as-a-judge metrics, general capabilities benchmarks, and efficiency indicators such as training time, inference latency, and VRAM usage. Results show that LoRA-enhanced models generally outperform the baseline in answer relevancy, hallucination reduction, and factual correctness, though performance varies across configurations and sometimes even declines in factual correctness and general capabilities benchmarks. Training is highly efficient, requiring only $\approx$ 30 minutes on a single GPU, but dataset creation introduces significant overhead. RAG achieved stronger factual accuracy, faithfulness, and output stability than LoRA-models, albeit with higher inference latency and memory demands. LoRA proved particularly effective for domain-specific contextual alignment, shown by the answer relevancy metric, whereas RAG excelled in factual grounding and robustness. These findings highlight the trade-offs and complementary advantages of both methods, suggesting that the choice between LoRA and RAG should depend on application-specific priorities concerning accuracy, efficiency, and resource constraints.

## Project Structure

```
├── data/                          # Data directories
│   ├── raw_splitted_pdfs/        # Original PDF text extracts
│   ├── raw_splitted_txt/         # Original text files
│   ├── splitted_pdfs/            # Processed PDF data
│   └── splitted_txts/            # Processed text data
├── src/                          # Source code
│   ├── eval/                     # Evaluation modules
│   ├── rag/                      # RAG pipeline implementation
│   ├── train/                    # Training modules
│   └── util/                     # Utility functions
└── results/                      # Experiment results
```

## Key Features

### Training Pipeline
- **LoRA Training**: Efficient fine-tuning using LoRA adapters
- **Model**: Meta-Llama-3.1-8B-Instruct with 4-bit quantization
- **Configurable Parameters**: Rank (r) and Alpha (α) values via environment variables

### Evaluation Framework
- **RAG Pipeline**: [rag_pipeline.py](src/rag/rag_pipeline.py) for retrieval-augmented generation
- **Question Generation**: [generate_goldens_openapi.py](src/eval/generate_goldens_openapi.py) using OpenAI GPT-4.1-mini
- **Comprehensive Evaluation**: Multiple benchmarks including MMLU, SuperGLUE, and TruthfulQA
- **Memory Monitoring**: Built-in memory usage tracking and reporting

### Data Processing
The project works with German open-access educational texts covering topics such as:
- Educational counseling ("Pädagogische Beratung")
- Career guidance ("Berufsberatung")
- Learning consultation ("Lernberatung")
- Nursing education and counseling

## Installation

1. Clone the repository
2. Install dependencies (Python3.11 with CUDA 11.8 is standard. Switching versions requires changed requirements.):
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp example.env .env
# Edit .env with your PYTHONPATH
```

## Configuration

Key configuration parameters in [src/util/args.py](src/util/args.py):

```python
RANK = int(os.getenv("RANK", "16"))          # LoRA rank
ALPHA = int(os.getenv("ALPHA", "16"))        # LoRA alpha
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 4096
```

## Usage

### Training

```bash
cd src/train
python train.py
```

For grid search training (!Recommended!):
```bash
python grid_train.py
```

### Golden Generation

Generate evaluation questions from text files using OpenAI API:

```bash
cd src/eval
python generate_goldens_openapi.py -d /path/to/texts -o questions.json --api-key YOUR_OPENAI_KEY
```

### RAG Evaluation

Run RAG pipeline evaluation:

```bash
cd src/rag
python rag_pipeline.py
```

### Comprehensive Evaluation

Run full evaluation suite(!Recommended!):

```bash
cd src/eval
python grid_eval.py
```

Run full rag evaluation suite:

```bash
cd src/eval
python rag_eval.py
```

## Environment Variables

- `OPENAI_API_KEY`: Required for question generation
- `RANK`: LoRA rank parameter (default: 16) set dynamically if running [grid_train.py](src/train/grid_train.py)
- `ALPHA`: LoRA alpha parameter (default: 16) set dynamically if running [grid_train.py](src/train/grid_train.py)

## Results and Monitoring

- **Memory Usage**: Automatic memory monitoring with reports in [results/](results/)
- **Model Outputs**: Generated content and evaluation metrics
- **Visualization**: Built-in plotting tools for result analysis

## Data Sources

The project uses open-access educational texts covering:
- Counseling methodologies
- Educational guidance frameworks
- Professional development in educational settings
- Systematic approaches to learning consultation