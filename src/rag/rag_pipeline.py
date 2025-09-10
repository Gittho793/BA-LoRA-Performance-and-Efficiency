"""
RAG Pipeline with GPU Memory Monitoring
"""
import unsloth
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time
# Core Haystack imports
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.rankers import SentenceTransformersSimilarityRanker


# Additional imports for local model handling
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch

from dotenv import load_dotenv

load_dotenv("../../.env")

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.eval.expanded_eval import load_question_json

import contextlib
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StepStat:
    """
    Statistics for a single monitored step.
    """
    label: str
    start_time_s: float
    end_time_s: float
    runtime_s: float
    baseline_reserved_gb: float
    step_peak_reserved_gb: float
    step_increment_gb: float
    pct_of_max_total: float
    pct_increment_of_max: float


class GPUMemoryMonitor:
    """
    Per-step GPU memory monitor. Use with MonitoredComponent to instrument Haystack components.
    Produces both a .txt and a .json report.
    """

    def __init__(self, device_index: Optional[int] = None):
        self.cuda = torch.cuda.is_available()
        self.device_index = (torch.cuda.current_device(
        ) if self.cuda else None) if device_index is None else device_index
        self.device_name = ""
        self.total_gb = 0.0
        self.sections: List[StepStat] = []
        self._overall_peak_reserved_b = 0
        self.session_started = False
        if self.cuda:
            props = torch.cuda.get_device_properties(self.device_index)
            self.device_name = props.name
            self.total_gb = round(props.total_memory / 1024 / 1024 / 1024, 3)

    def _bytes_to_gb(self, b: int) -> float:
        return round(b / 1024 / 1024 / 1024, 3)

    @contextlib.contextmanager
    def section(self, label: str):
        """
        Context manager to measure a single step (e.g., 'rag:llm', 'index:document_embedder').
        Resets CUDA peak stats inside the section to isolate the step.
        """
        start = time.time()
        if self.cuda:
            torch.cuda.synchronize(self.device_index)
            baseline_reserved_b = torch.cuda.memory_reserved(self.device_index)
            # isolate this step's peak
            torch.cuda.reset_peak_memory_stats(self.device_index)
        else:
            baseline_reserved_b = 0

        try:
            yield {}
        finally:
            end = time.time()
            if self.cuda:
                torch.cuda.synchronize(self.device_index)
                step_peak_b = torch.cuda.max_memory_reserved(self.device_index)
                step_increment_b = max(0, step_peak_b - baseline_reserved_b)
                # track overall absolute peak across all steps
                self._overall_peak_reserved_b = max(
                    self._overall_peak_reserved_b, step_peak_b)
            else:
                step_peak_b = 0
                step_increment_b = 0

            step = StepStat(
                label=label,
                start_time_s=start,
                end_time_s=end,
                runtime_s=round(end - start, 4),
                baseline_reserved_gb=self._bytes_to_gb(baseline_reserved_b),
                step_peak_reserved_gb=self._bytes_to_gb(step_peak_b),
                step_increment_gb=self._bytes_to_gb(step_increment_b),
                pct_of_max_total=(self._bytes_to_gb(
                    step_peak_b) / self.total_gb * 100.0) if self.cuda and self.total_gb > 0 else 0.0,
                pct_increment_of_max=(self._bytes_to_gb(
                    step_increment_b) / self.total_gb * 100.0) if self.cuda and self.total_gb > 0 else 0.0,
            )
            self.sections.append(step)

            # Console summary for the step (matches your style)
            print(f"\n[GPU] STEP '{label}' complete in {step.runtime_s:.2f}s")
            if self.cuda:
                print(f"      Peak reserved = {step.step_peak_reserved_gb} GB "
                      f"({step.pct_of_max_total:.2f}% of {self.total_gb} GB). "
                      f"Increment = {step.step_increment_gb} GB "
                      f"({step.pct_increment_of_max:.2f}%).")
            else:
                print("      CUDA not available - no GPU stats for this step.")

    def print_session_header(self, title: str = "GPU MEMORY MONITORING"):
        """
        Print the header for the GPU memory monitoring session.
        """
        print("\n" + "=" * 60)
        print(f"{title} - START")
        print("=" * 60)
        if self.cuda:
            print(
                f"GPU = {self.device_name}. Max memory = {self.total_gb} GB.")
        else:
            print("CUDA not available - GPU monitoring disabled")
        self.session_started = True

    def print_session_footer(self):
        """
        Print the footer for the GPU memory monitoring session.
        """
        print("\n" + "=" * 60)
        print("GPU MEMORY MONITORING - COMPLETE")
        print("=" * 60)
        if self.cuda:
            print(f"Overall peak reserved (across steps) = {self._bytes_to_gb(self._overall_peak_reserved_b)} GB "
                  f"({(self._bytes_to_gb(self._overall_peak_reserved_b) / self.total_gb * 100.0):.2f}% of total).")
        else:
            print("CUDA not available - no overall GPU stats.")

    def write_reports(self, output_dir: Path, filename_prefix: str = "gpu_memory_report",
                      extra_meta: Optional[Dict[str, Any]] = None):
        """
        Write GPU memory monitoring reports to the specified output directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        txt_path = output_dir / f"{filename_prefix}.txt"
        json_path = output_dir / f"{filename_prefix}.json"

        # TXT (human-readable)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"GPU Memory Report\n")
            f.write(f"=================\n")
            if self.cuda:
                f.write(f"GPU Device: {self.device_name}\n")
                f.write(f"Total GPU Memory: {self.total_gb} GB\n")
                f.write(
                    f"Overall Peak Reserved: {self._bytes_to_gb(self._overall_peak_reserved_b)} GB\n\n")
            else:
                f.write("CUDA not available - no GPU stats\n\n")

            if extra_meta:
                f.write("Additional Metadata:\n")
                for k, v in extra_meta.items():
                    f.write(f"- {k}: {v}\n")
                f.write("\n")

            f.write("Per-step Statistics:\n")
            for s in self.sections:
                f.write(
                    f"- {s.label}\n"
                    f"  runtime: {s.runtime_s:.2f}s\n"
                    f"  baseline_reserved: {s.baseline_reserved_gb} GB\n"
                    f"  step_peak_reserved: {s.step_peak_reserved_gb} GB "
                    f"({s.pct_of_max_total:.2f}% of total)\n"
                    f"  step_increment: {s.step_increment_gb} GB "
                    f"({s.pct_increment_of_max:.2f}% of total)\n"
                )

        # JSON (machine-readable)
        payload = {
            "cuda_available": self.cuda,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "total_memory_gb": self.total_gb,
            "overall_peak_reserved_gb": self._bytes_to_gb(self._overall_peak_reserved_b),
            "steps": [asdict(s) for s in self.sections],
            "meta": extra_meta or {},
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)

        print(f"\nGPU memory reports saved:\n- {txt_path}\n- {json_path}")


class MonitoredComponent:
    """
    Thin proxy for any Haystack component that intercepts .run() and measures GPU memory.
    """

    def __init__(self, inner, label: str, monitor: GPUMemoryMonitor):
        self._inner = inner
        self._label = label
        self._monitor = monitor

    def run(self, *args, **kwargs):
        with self._monitor.section(self._label):
            return self._inner.run(*args, **kwargs)

    def __getattr__(self, name):
        # Delegate all attribute access to the wrapped component
        return getattr(self._inner, name)


class RAGPipeline:
    """
    RAG Pipeline with document indexing and querying capabilities.
    Integrates GPU memory monitoring for key components.
    """
    def __init__(self, model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.document_store = InMemoryDocumentStore(
            embedding_similarity_function="cosine")
        self.indexing_pipeline = None
        self.rag_pipeline = None

        # NEW: GPU monitor
        self.gpu_monitor = GPUMemoryMonitor()  # uses current CUDA device if available

        # Initialize components
        self._setup_components()
        self._build_indexing_pipeline()
        self._build_rag_pipeline()

    def _setup_components(self):
        """Initialize all pipeline components."""

        # Document processing components
        self.file_type_router = FileTypeRouter(
            mime_types=["text/plain", "application/pdf"])
        self.text_file_converter = TextFileToDocument()
        self.pdf_converter = PyPDFToDocument()
        self.document_joiner = DocumentJoiner()
        self.document_cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=False
        )
        self.document_splitter = DocumentSplitter(
            split_by="word",
            split_length=200,
            split_overlap=50
        )

        # Embedding components
        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Storage and retrieval components
        self.document_writer = DocumentWriter(
            document_store=self.document_store)
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store)

        self.reranker = SentenceTransformersSimilarityRanker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=5)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # Local LLM generator using unsloth model
        self.llm_generator = HuggingFaceLocalGenerator(
            model=self.model_name,
            task="text-generation",
            huggingface_pipeline_kwargs={
                "device_map": "auto",
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported()
                else (torch.float16 if torch.cuda.is_available() else torch.float32),
                "model_kwargs": {
                    "load_in_4bit": False,
                }
            },
            generation_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "pad_token_id": pad_id
            }
        )

        # Prompt template for RAG
        self.prompt_template = '''
        Answer the following question based on the provided context. If the answer cannot be found in the context, say "I cannot find the answer in the provided context."

        Context:
        {% for document in documents %}
        {{ document.content }}
        {% endfor %}

        Question: {{ question }}

        Answer:
        '''

        self.prompt_builder = PromptBuilder(template=self.prompt_template)

    def _build_indexing_pipeline(self):
        self.indexing_pipeline = Pipeline()

        def wrap(comp, label):
            return MonitoredComponent(comp, label, self.gpu_monitor)

        # Add components (wrapped)
        self.indexing_pipeline.add_component("file_type_router", wrap(
            self.file_type_router, "index:file_type_router"))
        self.indexing_pipeline.add_component("text_file_converter", wrap(
            self.text_file_converter, "index:text_file_converter"))
        self.indexing_pipeline.add_component(
            "pdf_converter", wrap(self.pdf_converter, "index:pdf_converter"))
        self.indexing_pipeline.add_component("document_joiner", wrap(
            self.document_joiner, "index:document_joiner"))
        self.indexing_pipeline.add_component("document_cleaner", wrap(
            self.document_cleaner, "index:document_cleaner"))
        self.indexing_pipeline.add_component("document_splitter", wrap(
            self.document_splitter, "index:document_splitter"))
        self.indexing_pipeline.add_component("document_embedder", wrap(
            self.document_embedder, "index:document_embedder"))
        self.indexing_pipeline.add_component("document_writer", wrap(
            self.document_writer, "index:document_writer"))

        # Connect unchanged
        self.indexing_pipeline.connect(
            "file_type_router.text/plain", "text_file_converter.sources")
        self.indexing_pipeline.connect(
            "file_type_router.application/pdf", "pdf_converter.sources")
        self.indexing_pipeline.connect(
            "text_file_converter", "document_joiner.documents")
        self.indexing_pipeline.connect(
            "pdf_converter", "document_joiner.documents")
        self.indexing_pipeline.connect("document_joiner", "document_cleaner")
        self.indexing_pipeline.connect("document_cleaner", "document_splitter")
        self.indexing_pipeline.connect(
            "document_splitter", "document_embedder")
        self.indexing_pipeline.connect("document_embedder", "document_writer")

    def _build_rag_pipeline(self):
        self.rag_pipeline = Pipeline()

        def wrap(comp, label):
            return MonitoredComponent(comp, label, self.gpu_monitor)

        # Add components (wrapped)
        self.rag_pipeline.add_component("text_embedder", wrap(
            self.text_embedder, "rag:text_embedder"))
        self.rag_pipeline.add_component(
            "retriever", wrap(self.retriever, "rag:retriever"))
        self.rag_pipeline.add_component(
            "reranker", wrap(self.reranker, "rag:reranker"))
        self.rag_pipeline.add_component("prompt_builder", wrap(
            self.prompt_builder, "rag:prompt_builder"))
        self.rag_pipeline.add_component(
            "llm", wrap(self.llm_generator, "rag:llm"))

        # Connect unchanged
        self.rag_pipeline.connect(
            "text_embedder.embedding", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever", "reranker.documents")
        self.rag_pipeline.connect("reranker", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder", "llm")

    def index_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Index documents from the given file paths.

        Args:
            file_paths: List of paths to PDF or TXT files

        Returns:
            Result of the indexing pipeline
        """
        logger.info(f"Indexing {len(file_paths)} documents...")

        # Validate file paths
        valid_paths = []
        for path in file_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                logger.warning(f"File not found: {path}")

        if not valid_paths:
            raise ValueError("No valid file paths provided")

        # Run indexing pipeline
        result = self.indexing_pipeline.run({
            "file_type_router": {"sources": valid_paths}
        })

        dw = result.get("document_writer", {})
        written = dw.get("documents_written")
        if written is None:
            docs_list = dw.get("documents")
            if docs_list is not None:
                written = len(docs_list)
        if written is None:
            written = 0

        logger.info(f"Successfully indexed {written} document chunks")

        return result

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question.
        """
        logger.info(f"Processing query: {question}")
        retrieve_k = max(top_k * 3, top_k)

        result = self.rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "retriever": {"top_k": retrieve_k},
                "reranker": {"query": question, "top_k": top_k},
                "prompt_builder": {"question": question}
            },
            include_outputs_from={"retriever", "reranker"}  # <- add this
        )

        return result

    def get_answer(self, question: str, top_k: int = 5) -> str:
        """
        Get a simple answer string from the RAG pipeline.

        Args:
            question: The question to ask
            top_k: Number of documents to retrieve

        Returns:
            Generated answer as string
        """
        result = self.query(question, top_k)

        # Extract the answer from the result
        replies = result.get("llm", {}).get("replies", [])
        if replies:
            return replies[0].strip()
        else:
            return "Sorry, I couldn't generate an answer."

    def get_answer_and_retrieval(self, question: str, top_k: int = 5, keep_content_chars: Optional[int] = None):
        result = self.query(question, top_k)

        replies = result.get("llm", {}).get("replies", [])
        answer = replies[0].strip(
        ) if replies else "Sorry, I couldn't generate an answer."

        reranked_docs = (result.get("reranker") or {}).get("documents") or []
        if reranked_docs and isinstance(reranked_docs[0], list):
            reranked_docs = reranked_docs[0]
        if not reranked_docs:
            reranked_docs = (result.get("retriever") or {}
                             ).get("documents") or []

        def _to_serializable(val):
            try:
                import numpy as np  # local import
                if isinstance(val, (np.floating, np.integer)):
                    return val.item()
            except Exception:
                pass
            if hasattr(val, "item") and callable(getattr(val, "item")):
                # torch / numpy scalar
                try:
                    return val.item()
                except Exception:
                    pass
            if isinstance(val, (list, tuple)):
                return [_to_serializable(v) for v in val]
            if isinstance(val, dict):
                return {k: _to_serializable(v) for k, v in val.items()}
            if isinstance(val, (float, int, str, bool)) or val is None:
                return val
            # Fallback
            return str(val)

        retrieval = []
        for rank, d in enumerate(reranked_docs, start=1):
            score = getattr(d, "score", None)
            try:
                if score is not None:
                    score = float(score)
            except Exception:
                score = _to_serializable(score)

            content = getattr(d, "content", None)
            if keep_content_chars is not None and isinstance(content, str):
                content = content[:keep_content_chars]

            # meta may include non-serializable values
            raw_meta = getattr(d, "meta", {}) or getattr(
                d, "metadata", {}) or {}
            meta = _to_serializable(raw_meta)

            retrieval.append({
                "rank": rank,
                "id": getattr(d, "id", None),
                "score": score,
                "meta": meta,
                "content": content,
            })

        return answer, retrieval


def _bytes_to_gb(b: int) -> float:
    try:
        return float(b) / (1024 ** 3)
    except Exception:
        return 0.0


def _write_memory_report(output_dir: Path,
                         overall_peak_b: int,
                         total_mem_b: int,
                         generation_peak_b: int,
                         generation_minutes: float):
    total_mem_b = max(total_mem_b, 1)  # avoid div by zero
    overall_pct = (overall_peak_b / total_mem_b) * 100.0
    generation_pct = (generation_peak_b / total_mem_b) * 100.0

    lines = [
        f"Peak reserved memory = {_bytes_to_gb(overall_peak_b):.3f} GB.",
        f"Peak reserved memory % of max memory = {overall_pct:.3f} %.",
        f"Peak reserved memory for generation % of max memory = {generation_pct:.1f} %.",
        f"{generation_minutes:.1f} minutes used for generation."
    ]
    (output_dir / "memory_usage.txt").write_text("\n".join(lines), encoding="utf-8")


def main():

    # Initialize RAG pipeline + monitor
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline()
    monitor = rag.gpu_monitor
    monitor.print_session_header("GPU MEMORY MONITORING (RAG)")

    cuda_available = torch.cuda.is_available()
    device_index = torch.cuda.current_device() if cuda_available else None
    total_mem_b = torch.cuda.get_device_properties(
        device_index).total_memory if cuda_available else 0

    # Track overall (compatible with your existing summary file)
    if cuda_available:
        torch.cuda.reset_peak_memory_stats(device_index)
    overall_peak_before_gen_b = 0

    # Index documents (per-step stats will be recorded automatically)
    print("Indexing documents...")
    rag.index_documents([str(p) for p in Path(
        "../../data/raw_splitted_pdfs").rglob("*") if p.is_file()])

    if cuda_available:
        overall_peak_before_gen_b = torch.cuda.max_memory_reserved(
            device_index)
        torch.cuda.reset_peak_memory_stats(device_index)

    # Query/eval loop (per-step includes retriever/reranker/prompt/llm)
    questions_map, answer_map = load_question_json(
        "../eval/output/pdf_questions.json")

    output_dir = Path("../../results/rag/pdf")
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    results_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for source_file, qs in questions_map.items():
        for i, q in enumerate(qs):
            print(f"\nQuestion ({source_file}):\n{q}")
            answer, retrieval = rag.get_answer_and_retrieval(
                q, top_k=5, keep_content_chars=1000)
            best_context = retrieval[0]["content"] if retrieval else None

            print(f"\n\n\n\nAnswer: {answer}")
            print(f"\n\n\n\nExpected: {answer_map[source_file][i]}")
            print("-" * 30)

            results_by_file.setdefault(source_file, []).append({
                "question": q,
                "predicted_answer": answer,
                "best_context": best_context
            })

    minutes = (time.time() - start) / 60.0

    # Save results
    out_name = output_dir / "pdf_rag_results.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(results_by_file, f, ensure_ascii=False, indent=2)
    print(f"Saved all results to {out_name}")

    # Your original overall summary (kept intact)
    if cuda_available:
        generation_peak_b = torch.cuda.max_memory_reserved(device_index)
        overall_peak_b = max(overall_peak_before_gen_b, generation_peak_b)
    else:
        generation_peak_b = 0
        overall_peak_b = 0
    _write_memory_report(
        output_dir=output_dir,
        overall_peak_b=overall_peak_b,
        total_mem_b=total_mem_b,
        generation_peak_b=generation_peak_b,
        generation_minutes=minutes,
    )

    # NEW: write detailed per-step + overall GPU memory reports
    monitor.print_session_footer()
    meta = {
        "model_name": rag.model_name,
        "total_minutes": round(minutes, 2),
    }
    monitor.write_reports(output_dir=output_dir,
                          filename_prefix="gpu_memory_report", extra_meta=meta)


if __name__ == "__main__":
    main()
