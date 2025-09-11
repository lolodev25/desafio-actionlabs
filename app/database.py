import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid
import os

class ChromaDBManager:
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        # Criar diretório se não existir
        os.makedirs(persist_directory, exist_ok=True)
        
        # Inicializar cliente ChromaDB com persistência
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Criar ou obter coleção padrão
        self.collection_name = "documents"
        try:

            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:

            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document collection for RAG system"}
            )
    
    def add_document(self, text: str, embedding: List[float], metadata: Dict = {}) -> str:
        doc_id = str(uuid.uuid4())
        
        # Adicionar documento à coleção
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        # Buscar documentos similares
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else []
        }
    
    def get_all_documents(self, limit: int = 1000) -> Dict[str, Any]:
        """Busca todos os documentos da coleção para listagem."""
        try:
            # Buscar todos os documentos usando uma busca vazia
            results = self.collection.query(
                query_embeddings=[[0.0] * 384], 
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else []
            }
        except Exception as e:
            return {"documents": [], "distances": [], "metadatas": []}
    
    def get_collection_count(self) -> int:
        """Retorna o número total de documentos na coleção."""
        try:
            return self.collection.count()
        except Exception:
            return 0