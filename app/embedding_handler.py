import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import os


class EmbeddingHandler:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_model(self):
        try:
            print(f"Loading embedding model {self.model_name}...")
            self.model = SentenceTransformer(
                self.model_name, 
                cache_folder=self.models_dir
            )
            print(f"Embedding model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            return False
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            embeddings = self.model.encode(
                text, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            return None
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        print(f"Encoding {len(documents)} documents...")
        embeddings = self.encode_text(documents)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_text(query)
    
    def get_embedding_dimension(self) -> int:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model.get_sentence_embedding_dimension()
    
    def is_loaded(self) -> bool:
        return self.model is not None