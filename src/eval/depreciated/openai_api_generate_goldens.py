import os
from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig, EvolutionConfig



PDF_FILES = "/cluster/user/thoadelt/LoraData/raws/data/output"
MODEL = "gpt-4.1-mini"


synthesizer = Synthesizer(model=MODEL,
                          async_mode=True, 
                          max_concurrent=3,
                          evolution_config=EvolutionConfig(num_evolutions=0))
synthesizer.generate_goldens_from_docs(
    document_paths=[os.path.join(PDF_FILES, f)
                    for f in os.listdir(PDF_FILES)
                    if f.lower().endswith('.txt')],
    max_goldens_per_context=5,
    context_construction_config=ContextConstructionConfig(max_contexts_per_document=5,
                                                          embedder="text-embedding-3-small",
                                                          critic_model=MODEL,
                                                          max_retries=3)
)
