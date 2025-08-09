import faiss
import numpy as np
import json
import os
from typing import List, Tuple, Optional


class VectorStore:
    def __init__(self, embedding_dim: int, store_path: str = "data/vector_store"):
        self.embedding_dim = embedding_dim
        self.store_path = store_path
        self.index = None
        self.documents = []
        self.document_metadata = []
        
        os.makedirs(self.store_path, exist_ok=True)
        self.index_path = os.path.join(self.store_path, "faiss_index")
        self.metadata_path = os.path.join(self.store_path, "metadata.json")
    
    def create_index(self):
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        print(f"Created FAISS index with dimension {self.embedding_dim}")
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: Optional[List[dict]] = None):
        if self.index is None:
            self.create_index()
        
        if embeddings.shape[0] != len(documents):
            raise ValueError("Number of documents and embeddings must match")
        
        self.documents.extend(documents)
        
        if metadata:
            self.document_metadata.extend(metadata)
        else:
            self.document_metadata.extend([{"doc_id": i} for i in range(len(documents))])
        
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        
        print(f"Added {len(documents)} documents to vector store")
        print(f"Total documents in store: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[dict]]:
        if self.index is None or self.index.ntotal == 0:
            return [], [], []
        
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        scores = []
        metadata = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append(self.documents[idx])
                scores.append(float(distances[0][i]))
                metadata.append(self.document_metadata[idx])
        
        return results, scores, metadata
    
    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'documents': self.documents,
                    'metadata': self.document_metadata,
                    'embedding_dim': self.embedding_dim
                }, f, ensure_ascii=False, indent=2)
            
            print(f"Saved vector store to {self.store_path}")
    
    def load_index(self) -> bool:
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.index_path)
                
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data['documents']
                    self.document_metadata = data['metadata']
                    self.embedding_dim = data['embedding_dim']
                
                print(f"Loaded vector store with {self.index.ntotal} documents")
                return True
            else:
                print("No existing vector store found")
                return False
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        return self.index.ntotal if self.index else 0
    
    def clear_store(self):
        self.index = None
        self.documents = []
        self.document_metadata = []
        
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)