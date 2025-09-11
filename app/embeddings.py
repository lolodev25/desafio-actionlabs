from sentence_transformers import SentenceTransformer
from typing import List
import logging
import os


logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class EmbeddingGenerator:
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Carregar modelo sentence transformer
        try:

            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            
            self.model = SentenceTransformer(
                model_name,
                cache_folder='./model_cache',  
                use_auth_token=False,
                device='cpu'
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logging.info(f"EmbeddingGenerator inicializado com {model_name}")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo {model_name}: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Gera embeddings para uma lista de textos.
        
        Args:
            texts: Lista de textos para gerar embeddings
            
        Returns:
            Lista de embeddings (cada embedding é uma lista de floats)
        """
        if not texts:
            return []
        
        try:
            # Gerar embeddings usando o modelo
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,  # Retornar como numpy array
                normalize_embeddings=True,  # Normalizar para melhor performance de busca
                show_progress_bar=False  # Desabilitar barra de progresso
            )
            
            # Converter para lista de listas de floats
            return [embedding.tolist() for embedding in embeddings]
            
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar embeddings: {str(e)}")
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Gera embedding para um único texto.
        
        Args:
            text: Texto para gerar embedding
            
        Returns:
            Embedding como lista de floats
        """
        return self.generate_embeddings([text])[0]