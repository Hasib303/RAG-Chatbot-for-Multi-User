import os
from typing import List, Tuple, Optional
from .embedding_handler import EmbeddingHandler
from .vector_store import VectorStore


class RAGRetriever:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_handler = EmbeddingHandler(embedding_model)
        self.vector_store = None
        self.knowledge_base_path = "data/knowledge_base"
        
    def initialize(self):
        success = self.embedding_handler.load_model()
        if not success:
            return False
        
        embedding_dim = self.embedding_handler.get_embedding_dimension()
        self.vector_store = VectorStore(embedding_dim)
        
        return self.vector_store.load_index()
    
    def load_documents_from_directory(self, directory_path: str = None) -> bool:
        if directory_path is None:
            directory_path = self.knowledge_base_path
        
        if not os.path.exists(directory_path):
            print(f"Knowledge base directory not found: {directory_path}")
            return False
        
        documents = []
        metadata = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            documents.append(content)
                            metadata.append({
                                "filename": filename,
                                "file_path": file_path,
                                "doc_id": len(documents)
                            })
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
        
        if not documents:
            print("No documents found in knowledge base")
            return False
        
        print(f"Found {len(documents)} documents in knowledge base")
        return self.add_documents(documents, metadata)
    
    def add_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> bool:
        try:
            embeddings = self.embedding_handler.encode_documents(documents)
            if embeddings is None:
                return False
            
            if self.vector_store is None:
                embedding_dim = self.embedding_handler.get_embedding_dimension()
                self.vector_store = VectorStore(embedding_dim)
            
            self.vector_store.add_documents(documents, embeddings, metadata)
            self.vector_store.save_index()
            
            return True
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> Tuple[List[str], List[float]]:
        if not self.vector_store or self.vector_store.get_document_count() == 0:
            return [], []
        
        try:
            query_embedding = self.embedding_handler.encode_query(query)
            if query_embedding is None:
                return [], []
            
            documents, scores, metadata = self.vector_store.search(query_embedding, top_k)
            return documents, scores
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return [], []
    
    def get_context_for_query(self, query: str, top_k: int = 3, max_context_length: int = 2000) -> str:
        documents, scores = self.retrieve_relevant_documents(query, top_k)
        
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        
        for doc in documents:
            if current_length + len(doc) <= max_context_length:
                context_parts.append(doc)
                current_length += len(doc)
            else:
                remaining_space = max_context_length - current_length
                if remaining_space > 100:
                    context_parts.append(doc[:remaining_space] + "...")
                break
        
        return "\n\n".join(context_parts)
    
    def get_document_count(self) -> int:
        return self.vector_store.get_document_count() if self.vector_store else 0