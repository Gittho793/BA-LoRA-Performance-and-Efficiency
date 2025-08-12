import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import unsloth
import json
import time
# Core Haystack imports
from haystack import Pipeline, Document
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
from transformers import AutoTokenizer
import torch
from src.eval.expanded_eval import load_question_json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    A comprehensive RAG pipeline using Haystack with local unsloth/Meta-Llama-3.1-8B-Instruct model.
    Supports PDF and TXT file processing.
    """

    def __init__(self, model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.document_store = InMemoryDocumentStore(
            embedding_similarity_function="cosine")
        self.indexing_pipeline = None
        self.rag_pipeline = None

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
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() \
                    else (torch.float16 if torch.cuda.is_available() else torch.float32)
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
        """Build the document indexing pipeline."""

        self.indexing_pipeline = Pipeline()

        # Add components to pipeline
        self.indexing_pipeline.add_component(
            "file_type_router", self.file_type_router)
        self.indexing_pipeline.add_component(
            "text_file_converter", self.text_file_converter)
        self.indexing_pipeline.add_component(
            "pdf_converter", self.pdf_converter)
        self.indexing_pipeline.add_component(
            "document_joiner", self.document_joiner)
        self.indexing_pipeline.add_component(
            "document_cleaner", self.document_cleaner)
        self.indexing_pipeline.add_component(
            "document_splitter", self.document_splitter)
        self.indexing_pipeline.add_component(
            "document_embedder", self.document_embedder)
        self.indexing_pipeline.add_component(
            "document_writer", self.document_writer)

        # Connect components
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
        """Build the RAG query pipeline."""

        self.rag_pipeline = Pipeline()

        # Add components
        self.rag_pipeline.add_component("text_embedder", self.text_embedder)
        self.rag_pipeline.add_component("retriever", self.retriever)
        self.rag_pipeline.add_component("reranker", self.reranker)
        self.rag_pipeline.add_component("prompt_builder", self.prompt_builder)
        self.rag_pipeline.add_component("llm", self.llm_generator)

        # Connect components
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
    """Main function demonstrating RAG pipeline usage."""

    start = time.time()

    # Initialize RAG pipeline
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline()

    cuda_available = torch.cuda.is_available()
    device_index = torch.cuda.current_device() if cuda_available else None
    total_mem_b = torch.cuda.get_device_properties(
        device_index).total_memory if cuda_available else 0
    # Track overall peak (indexing + generation). Reset at very start.
    if cuda_available:
        torch.cuda.reset_peak_memory_stats(device_index)
    overall_peak_before_gen_b = 0

    # Index documents
    print("Indexing documents...")
    rag.index_documents([str(p) for p in Path(
        "../../data/raw_splitted_txt").rglob("*") if p.is_file()])

    if cuda_available:
        overall_peak_before_gen_b = torch.cuda.max_memory_reserved(
            device_index)
        # Reset before generation to isolate generation peak
        torch.cuda.reset_peak_memory_stats(device_index)

    # Query the system
    questions_map, answer_map = load_question_json(
        "../eval/output/txt_questions.json")

    """for source_file, qs in questions_map.items():
        for i, q in enumerate(qs):
            print(f"\nQuestion ({source_file}):\n{q}")
            answer = rag.get_answer(q)
            print(f"\n\n\n\nAnswer: {answer}")
            print(f"\n\n\n\nExpected: {answer_map[source_file][i]}")
            print("-" * 30)"""

    output_dir = Path("../../results/rag/txt")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_file: Dict[str, List[Dict[str, Any]]] = {}

    for source_file, qs in questions_map.items():
        for i, q in enumerate(qs):
            print(f"\nQuestion ({source_file}):\n{q}")
            answer, retrieval = rag.get_answer_and_retrieval(
                q, top_k=5, keep_content_chars=1000
            )

            # best retrieval context = top-1 reranked doc (rank 1 is first in list)
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

    # Write to a single file with the new structure
    out_name = output_dir / "txt_rag_results.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(results_by_file, f, ensure_ascii=False, indent=2)

    print(f"Saved all results to {out_name}")

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


if __name__ == "__main__":
    main()
