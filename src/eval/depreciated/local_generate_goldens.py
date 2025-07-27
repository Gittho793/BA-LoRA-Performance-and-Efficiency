"""
Doesn't work. Don't use
"""
import os
import ollama
from typing import List
from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from deepeval.synthesizer.config import ContextConstructionConfig, FiltrationConfig
from deepeval.models.base_model import DeepEvalBaseLLM
from sentence_transformers import SentenceTransformer

PDF_FILES = "/cluster/user/thoadelt/LoraData/raws/data/pdf"


class OllamaCustomLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama3.1:8b-instruct-q8_0"):
        super().__init__(model_name)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']

    def load_model(self):
        """REQUIRED: Abstract method that returns the model object"""
        return self.model_name

    # REQUIRED: Add async method
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    # REQUIRED: Add model name method
    def get_model_name(self) -> str:
        return self.model_name


class SentenceTransformersEmbedding(DeepEvalBaseEmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.model = SentenceTransformer(model_name)

    def load_model(self):
        """REQUIRED: Abstract method that returns the model object"""
        return self.model_name

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

    # REQUIRED: Add batch embedding method
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    # REQUIRED: Add async methods
    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    def get_model_name(self) -> str:
        return "SentenceTransformers Custom Embedding"


# Instantiate your custom models
custom_llm = OllamaCustomLLM("llama3.1:8b-instruct-q8_0")
custom_embedding = SentenceTransformersEmbedding()

# Configure filtration with custom critic model
filtration_config = FiltrationConfig(
    critic_model=custom_llm,  # Use YOUR custom LLM for filtering
    synthetic_input_quality_threshold=0.5
)

# Configure context construction with custom embedding
context_construction_config = ContextConstructionConfig(
    embedder=custom_embedding,  # Use YOUR custom embedding
    critic_model=custom_llm     # Use YOUR custom LLM for context evaluation
)

# Initialize synthesizer with ALL custom components
synthesizer = Synthesizer(
    # Main generation model (INSTANCE not string)
    model=custom_llm,
    filtration_config=filtration_config,  # Custom filtering
)

# Generate goldens - now completely offline!
synthesizer.generate_goldens_from_docs(
    document_paths=[os.path.join(PDF_FILES, f)
                    for f in os.listdir(PDF_FILES)
                    if f.lower().endswith('.pdf')],
    context_construction_config=context_construction_config
)
