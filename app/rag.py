from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List
import logging

class RAGPipeline:
    
    def __init__(self, db_manager, embedding_generator, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.db_manager = db_manager
        self.embedding_generator = embedding_generator
        self.model_name = model_name
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY é obrigatória")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=1000
        )
    
    def generate_answer(self, question: str, max_results: int = 20) -> Dict[str, Any]:
        try:
            question_embedding = self.embedding_generator.generate_single_embedding(question)
            search_results = self.db_manager.search(question_embedding, n_results=max_results)
            
            context_documents = search_results.get("documents", [])
            context_metadatas = search_results.get("metadatas", [])
            context_distances = search_results.get("distances", [])
            
            file_groups = {}
            for doc, meta, dist in zip(context_documents, context_metadatas, context_distances):
                filename = meta.get("filename", "unknown")
                if filename not in file_groups:
                    file_groups[filename] = []
                file_groups[filename].append({
                    "doc": doc,
                    "meta": meta,
                    "dist": dist,
                    "similarity": 1 - dist
                })
            
            filtered_docs = []
            filtered_metas = []
            filtered_dists = []
            
            for filename, chunks in file_groups.items():
                chunks.sort(key=lambda x: x["similarity"], reverse=True)
                relevant_chunks = [c for c in chunks if c["similarity"] > 0.10]
                
                if not relevant_chunks:
                    relevant_chunks = chunks[:3]
                
                added_chunks = set()
                
                for chunk in relevant_chunks:
                    chunk_index = chunk["meta"].get("chunk_index", 0)
                    
                    if chunk_index not in added_chunks:
                        filtered_docs.append(chunk["doc"])
                        filtered_metas.append(chunk["meta"])
                        filtered_dists.append(chunk["dist"])
                        added_chunks.add(chunk_index)
                    
                    for other_chunk in chunks:
                        other_index = other_chunk["meta"].get("chunk_index", 0)
                        
                        if (abs(other_index - chunk_index) <= 1 and 
                            other_index not in added_chunks and
                            len(added_chunks) < 5):
                            
                            filtered_docs.append(other_chunk["doc"])
                            filtered_metas.append(other_chunk["meta"])
                            filtered_dists.append(other_chunk["dist"])
                            added_chunks.add(other_index)
                
                file_chunks = [(doc, meta, dist) for doc, meta, dist in zip(filtered_docs, filtered_metas, filtered_dists) 
                              if meta.get("filename") == filename]
                file_chunks.sort(key=lambda x: x[1].get("chunk_index", 0))
                
                for i, (doc, meta, dist) in enumerate(file_chunks):
                    if i < len(filtered_docs):
                        filtered_docs[i] = doc
                        filtered_metas[i] = meta
                        filtered_dists[i] = dist
            
            if not filtered_docs:
                keywords = self._extract_keywords(question)
                keyword_results = self._search_by_keywords(keywords, file_groups)
                
                if keyword_results:
                    filtered_docs = keyword_results["docs"]
                    filtered_metas = keyword_results["metas"]
                    filtered_dists = keyword_results["dists"]
                else:
                    filtered_docs = context_documents[:5]
                    filtered_metas = context_metadatas[:5]
                    filtered_dists = context_distances[:5]
            
            if len(filtered_docs) > 12:
                chunk_scores = [(i, 1 - dist) for i, dist in enumerate(filtered_dists)]
                chunk_scores.sort(key=lambda x: x[1], reverse=True)
                
                top_indices = [i for i, _ in chunk_scores[:12]]
                filtered_docs = [filtered_docs[i] for i in top_indices]
                filtered_metas = [filtered_metas[i] for i in top_indices]
                filtered_dists = [filtered_dists[i] for i in top_indices]
            
            context_text = self._build_context(filtered_docs, filtered_metas, filtered_dists)
            response = self._call_llm(question, context_text)
            sources = self._prepare_sources(filtered_docs, filtered_metas, filtered_dists)
            
            return {
                "answer": response["answer"],
                "sources": sources,
                "model_used": self.model_name,
                "tokens_used": response.get("tokens_used", 0)
            }
            
        except Exception as e:
            logging.error(f"Erro no pipeline RAG: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return {
                "answer": f"Erro ao processar pergunta: {str(e)}",
                "sources": [],
                "model_used": self.model_name,
                "tokens_used": 0
            }
    
    def _build_context(self, documents: List[str], metadatas: List[Dict], distances: List[float]) -> str:
        """Constrói o contexto a partir dos documentos encontrados, agrupando por arquivo."""
        if not documents:
            return "Nenhum documento relevante encontrado na base de conhecimento."
        
        # Agrupar por arquivo para melhor organização
        file_groups = {}
        for doc, metadata, distance in zip(documents, metadatas, distances):
            filename = metadata.get("filename", "Documento desconhecido")
            chunk_index = metadata.get("chunk_index", 0)
            total_chunks = metadata.get("total_chunks", 1)
            similarity_score = 1 - distance
            
            if filename not in file_groups:
                file_groups[filename] = []
            
            file_groups[filename].append({
                "content": doc,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "similarity": similarity_score
            })
        
        # Ordenar chunks dentro de cada arquivo por índice
        for filename in file_groups:
            file_groups[filename].sort(key=lambda x: x["chunk_index"])
        
        context_parts = []
        doc_counter = 1
        
        for filename, chunks in file_groups.items():
            context_parts.append(f"=== ARQUIVO: {filename} ===")
            context_parts.append(f"Total de chunks: {chunks[0]['total_chunks']}")
            context_parts.append(f"Chunks relevantes encontrados: {len(chunks)}")
            context_parts.append("")
            
            for chunk in chunks:
                chunk_num = chunk["chunk_index"] + 1
                similarity = chunk["similarity"]
                content = chunk["content"]
                
                context_parts.append(f"--- CHUNK {chunk_num}/{chunk['total_chunks']} (Relevância: {similarity:.3f}) ---")
                context_parts.append(content)
                context_parts.append("")
            
            context_parts.append("=" * 50)
            context_parts.append("")
            doc_counter += 1
        
        return "\n".join(context_parts)
    
    def _call_llm(self, question: str, context: str) -> Dict[str, Any]:
        if not context or context.strip() == "Nenhum documento relevante encontrado na base de conhecimento.":
            return {
                "answer": "Desculpe, não encontrei informações relevantes na base de conhecimento para responder sua pergunta.",
                "tokens_used": 0
            }
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente que responde perguntas baseado em documentos. 
IMPORTANTE: Use APENAS as informações do contexto fornecido. NÃO invente ou adicione informações externas.

Instruções:
1. Leia cuidadosamente o contexto fornecido
2. Identifique as informações relevantes para a pergunta
3. Responda de forma clara e organizada
4. Se não encontrar a informação no contexto, diga "Não encontrei essa informação nos documentos fornecidos"
5. Seja amigável e profissional e use emojis nas suas respostas.

Contexto:
{context}"""),
            ("human", "Pergunta: {question}")
        ])
        
        chain = prompt_template | self.llm
        
        try:
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            answer = response.content
            
            if not answer or answer.strip() == '':
                simple_prompt = f"""Baseado no contexto abaixo, responda à pergunta de forma clara e concisa.

Contexto:
{context[:2000]}

Pergunta: {question}

Resposta:"""
                
                try:
                    direct_response = self.llm.invoke(simple_prompt)
                    answer = direct_response.content if hasattr(direct_response, 'content') else str(direct_response)
                except Exception as e:
                    answer = "Desculpe, não consegui processar sua pergunta no momento. Tente reformular a pergunta."
            
            tokens_used = len(answer.split()) * 1.3
            
            return {
                "answer": answer,
                "tokens_used": int(tokens_used)
            }
            
        except Exception as e:
            return {
                "answer": f"Erro ao processar pergunta com o modelo de linguagem: {str(e)}",
                "tokens_used": 0
            }
    
    def _prepare_sources(self, documents: List[str], metadatas: List[Dict], distances: List[float]) -> List[Dict[str, Any]]:
        sources = []
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances), 1):
            similarity_score = 1 - distance
            sources.append({
                "document_id": i,
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                "similarity_score": round(similarity_score, 3),
                "metadata": metadata
            })
        
        return sources
    
    def _extract_keywords(self, question: str) -> List[str]:
        import re
        
        text = re.sub(r'[^\w\s]', ' ', question.lower())
        
        stop_words = {
            'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'da', 'do', 'das', 'dos',
            'em', 'na', 'no', 'nas', 'nos', 'para', 'por', 'com', 'sem', 'sobre', 'entre',
            'que', 'quem', 'qual', 'quais', 'onde', 'quando', 'como', 'porque', 'porquê',
            'é', 'são', 'foi', 'foram', 'será', 'serão', 'tem', 'têm', 'tinha', 'tinham',
            'pode', 'podem', 'deve', 'devem', 'quer', 'querem', 'gosta', 'gostam',
            'fale', 'falar', 'diga', 'dizer', 'conte', 'contar', 'explique', 'explicar',
            'quais', 'quais', 'o', 'que', 'são', 'foi', 'tem', 'pode', 'deve'
        }
        
        words = text.split()
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:10]
    
    def _search_by_keywords(self, keywords: List[str], file_groups: Dict) -> Dict:
        if not keywords:
            return None
        
        results = {
            "docs": [],
            "metas": [],
            "dists": []
        }
        
        for filename, chunks in file_groups.items():
            for chunk in chunks:
                content = chunk["doc"].lower()
                matches = sum(1 for keyword in keywords if keyword in content)
                
                if matches > 0:
                    score = matches / len(keywords)
                    results["docs"].append(chunk["doc"])
                    results["metas"].append(chunk["meta"])
                    results["dists"].append(1 - score)
        
        if results["docs"]:
            sorted_indices = sorted(range(len(results["dists"])), 
                                  key=lambda i: results["dists"][i])
            
            results["docs"] = [results["docs"][i] for i in sorted_indices]
            results["metas"] = [results["metas"][i] for i in sorted_indices]
            results["dists"] = [results["dists"][i] for i in sorted_indices]
        
        return results if results["docs"] else None