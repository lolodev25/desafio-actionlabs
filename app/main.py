from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import io

from app.models import AddDocumentRequest, SearchResult, ChatRequest, ChatResponse
from app.database import ChromaDBManager
from app.embeddings import EmbeddingGenerator
from app.rag import RAGPipeline
from app.config import GOOGLE_API_KEY, CHROMADB_PATH, EMBEDDING_MODEL, MODEL_NAME

import uvicorn
app = FastAPI(title="RAG System API")

# Enable CORS for browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db_manager = ChromaDBManager(CHROMADB_PATH)
embedding_generator = EmbeddingGenerator(EMBEDDING_MODEL)
rag_pipeline = RAGPipeline(db_manager, embedding_generator, GOOGLE_API_KEY, MODEL_NAME)


@app.get("/")
def root():
    return {"name": "RAG System", "version": "1.0"}




@app.get("/database_stats")
async def database_stats():
    """
    Mostra estatísticas da base de dados para monitoramento.
    """
    try:
        # Buscar todos os documentos para estatísticas
        all_docs = db_manager.get_all_documents(limit=1000)
        
        # Contar documentos por arquivo
        file_stats = {}
        total_chunks = 0
        
        for metadata in all_docs.get("metadatas", []):
            if metadata:
                filename = metadata.get("filename", "Unknown")
                if filename not in file_stats:
                    file_stats[filename] = {
                        "chunks": 0,
                        "file_type": metadata.get("file_type", "unknown"),
                        "file_size": metadata.get("file_size", 0)
                    }
                file_stats[filename]["chunks"] += 1
                total_chunks += 1
        
        return {
            "total_documents": total_chunks,
            "unique_files": len(file_stats),
            "files": file_stats,
            "message": f"Base de dados contém {total_chunks} chunks de {len(file_stats)} arquivos únicos"
        }
    except Exception as e:
        return {
            "error": str(e),
            "total_documents": 0,
            "unique_files": 0
        }


@app.get("/search_chunks")
async def search_chunks(
    query: str = Query(...),
    filename: str = Query(None),
    limit: int = Query(default=10)
):
    """
    Busca chunks específicos para debug e análise.
    Útil para ver como os chunks estão sendo encontrados.
    """
    try:
        # Gerar embedding da consulta
        query_embedding = embedding_generator.generate_single_embedding(query)
        
        # Buscar documentos similares
        results = db_manager.search(query_embedding, n_results=limit)
        
        # Filtrar por arquivo se especificado
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        
        if filename:
            filtered_results = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                if meta.get("filename") == filename:
                    filtered_results.append({
                        "content": doc,
                        "similarity_score": round(1 - dist, 3),
                        "chunk_index": meta.get("chunk_index", 0),
                        "total_chunks": meta.get("total_chunks", 1),
                        "metadata": meta
                    })
            documents = filtered_results
        else:
            documents = [{
                "content": doc,
                "similarity_score": round(1 - dist, 3),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1),
                "filename": meta.get("filename", "Unknown"),
                "metadata": meta
            } for doc, meta, dist in zip(documents, metadatas, distances)]
        
        return {
            "query": query,
            "filename_filter": filename,
            "total_results": len(documents),
            "results": documents
        }
        
    except Exception as e:
        return {"error": str(e), "results": []}


@app.post("/add_document")
async def add_document(request: AddDocumentRequest):
    try:
        # Gerar embedding do texto
        embedding = embedding_generator.generate_single_embedding(request.text)
        
        # Armazenar no ChromaDB
        doc_id = db_manager.add_document(
            text=request.text,
            embedding=embedding,
            metadata=request.metadata
        )
        
        return {"success": True, "id": doc_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/upload_document")
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = ""
):
    """
    Upload de documento - suporta texto, PDF e Word.
    Extrai texto automaticamente e processa em chunks.
    """
    try:
        # Verificar se é um arquivo suportado
        allowed_extensions = {'.txt', '.md', '.py', '.json', '.csv', '.xml', '.html', '.htm', '.pdf', '.docx', '.doc'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Tipo de arquivo não suportado. Use: {', '.join(allowed_extensions)}"
            )
        
        # Ler conteúdo do arquivo
        content = await file.read()
        
        # Extrair texto baseado no tipo de arquivo
        if file_extension == '.pdf':
            text_content = _extract_text_from_pdf(content)
        elif file_extension in ['.docx', '.doc']:
            text_content = _extract_text_from_word(content)
        else:
            text_content = content.decode('utf-8')
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Arquivo vazio ou inválido")
        
        # Processar texto em chunks
        chunks = _split_text_into_chunks(text_content, max_chunk_size=2000)
        
        # Gerar embeddings para cada chunk
        embeddings = embedding_generator.generate_embeddings(chunks)
        
        # Armazenar cada chunk no ChromaDB
        document_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_metadata = {
                "filename": file.filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_size": len(content),
                "file_type": file_extension,
                "custom_metadata": metadata
            }
            
            doc_id = db_manager.add_document(
                text=chunk,
                embedding=embedding,
                metadata=chunk_metadata
            )
            document_ids.append(doc_id)
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks_processed": len(chunks),
            "message": f"Arquivo '{file.filename}' processado com sucesso! {len(chunks)} chunks criados."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


def _split_text_into_chunks(text: str, max_chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """
    Divide o texto em chunks menores para processamento.
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    text = text.strip()
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(paragraph) > max_chunk_size:
            sentences = paragraph.split('. ')
            for sentence in sentences:
                if len(current_chunk + sentence) > max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            if len(current_chunk + "\n\n" + paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                current_chunk += ("\n\n" + paragraph) if current_chunk else paragraph
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


@app.get("/search", response_model=List[SearchResult])
async def search(
    query: str = Query(...),
    limit: int = Query(default=5)
):
    try:
        # Gerar embedding da consulta
        query_embedding = embedding_generator.generate_single_embedding(query)
        
        # Buscar documentos similares
        results = db_manager.search(query_embedding, n_results=limit)
        
        # Converter para formato de resposta
        search_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results.get("documents", []),
            results.get("metadatas", []),
            results.get("distances", [])
        )):
            # Converter distância para score de similaridade
            similarity_score = 1 - distance
            
            search_results.append(SearchResult(
                content=doc,
                score=similarity_score,
                metadata=metadata
            ))
        
        return search_results
    except Exception as e:
        return []


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        max_results = min(request.max_results, 30) 
        
        # Implementar pipeline RAG
        result = rag_pipeline.generate_answer(
            request.question,
            max_results
        )
        
        chat_response = ChatResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            model_used=result.get("model_used", rag_pipeline.model_name),
            tokens_used=result.get("tokens_used", 0)
        )
        
        return chat_response
    except Exception as e:
        return ChatResponse(
            answer=f"Erro ao processar pergunta: {str(e)}",
            sources=[],
            model_used=rag_pipeline.model_name,
            tokens_used=0
        )


def _extract_text_from_pdf(content: bytes) -> str:
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        raise HTTPException(
            status_code=400, 
            detail="PyPDF2 não instalado. Execute: pip install PyPDF2"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Erro ao processar PDF: {str(e)}"
        )


def _extract_text_from_word(content: bytes) -> str:
    try:
        import docx
        from docx import Document
        
        doc = Document(io.BytesIO(content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except ImportError:
        raise HTTPException(
            status_code=400, 
            detail="python-docx não instalado. Execute: pip install python-docx"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Erro ao processar Word: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)