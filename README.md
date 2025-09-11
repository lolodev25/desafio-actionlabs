# RAG System API

Sistema de Retrieval-Augmented Generation (RAG) construído com FastAPI, ChromaDB e Google Gemini para busca semântica e geração de respostas baseadas em documentos.

## Tecnologias

- **FastAPI** - Framework web para API
- **ChromaDB** - Banco de dados vetorial
- **all-MiniLM-L6-v2** - Modelo de embeddings
- **Google Gemini** - Modelo de linguagem
- **LangChain** - Framework para LLM


## Instalação

1. Clone o repositório
2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate  # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure as variáveis de ambiente:
   - Crie um arquivo `.env` na raiz do projeto
   - Adicione sua chave da Google AI:
   ```
   GOOGLE_API_KEY=sua_chave_da_google_ai_aqui
   CHROMADB_PATH=./chroma_db
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   MODEL_NAME=gemini-2.5-flash
   ```

## Execução

```bash
python run_server.py
```

A API estará disponível em `http://localhost:8000`

## Endpoints

### 1. Adicionar Documento
**POST** `/add_document`

Adiciona um documento à base de conhecimento.

**Body:**
```json
{
  "text": "Conteúdo do documento",
  "filename": "documento.txt",
  "metadata": {}
}
```

**Suporte a arquivos:**
- Texto simples (.txt)
- PDF (.pdf)
- Word (.docx, .doc)

**Exemplo com arquivo:**
```bash
curl -X POST "http://localhost:8000/add_document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@documento.pdf"
```

### 2. Buscar Documentos
**GET** `/search`

Busca documentos similares por consulta semântica.

**Parâmetros:**
- `query` (string): Consulta de busca
- `n_results` (int, opcional): Número de resultados (padrão: 10)

**Exemplo:**
```bash
curl "http://localhost:8000/search?query=aprendizado%20máquina&n_results=5"
```

### 3. Chat RAG
**POST** `/chat`

Gera respostas baseadas nos documentos armazenados.

**Body:**
```json
{
  "question": "Qual é o tema principal do documento?",
  "max_results": 10
}
```

**Resposta:**
```json
{
  "answer": "Resposta gerada pelo modelo...",
  "sources": [
    {
      "document_id": 1,
      "content": "Trecho relevante...",
      "similarity_score": 0.85,
      "metadata": {
        "filename": "documento.pdf",
        "chunk_index": 0
      }
    }
  ],
  "model_used": "gemini-2.5",
  "tokens_used": 150
}
```

### 4. Estatísticas do Banco
**GET** `/database_stats`

Retorna estatísticas sobre os documentos armazenados.

### 5. Buscar Chunks
**GET** `/search_chunks`

Busca chunks específicos por consulta (endpoint de debug).

## Funcionalidades

### Busca Semântica
- Utiliza embeddings para encontrar documentos semanticamente similares
- Busca por palavras-chave como fallback
- Filtragem automática por relevância

### Processamento de Documentos
- Divisão automática em chunks para melhor processamento
- Suporte a múltiplos formatos de arquivo
- Metadados preservados para rastreabilidade

### Geração de Respostas
- Contexto inteligente com chunks adjacentes
- Respostas baseadas exclusivamente nos documentos
- Fallback robusto em caso de falhas

## Estrutura do Projeto

```
llm-rag-test/
├── app/
│   ├── __init__.py
│   ├── config.py          # Configurações
│   ├── database.py        # Gerenciamento ChromaDB
│   ├── embeddings.py      # Geração de embeddings
│   ├── main.py           # API FastAPI
│   ├── models.py         # Modelos Pydantic
│   └── rag.py            # Pipeline RAG
├── requirements.txt       # Dependências
├── run_server.py         # Script de execução
└── README.md
```

## Configuração Avançada

### Modelos de Embedding
Altere o modelo de embedding no arquivo `.env`:
```
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Modelo de Linguagem
Configure o modelo Gemini:
```
MODEL_NAME=gemini-2.5-flash
```

### Caminho do Banco
Defina onde armazenar o ChromaDB:
```
CHROMADB_PATH=./chroma_db
```

## Limitações

- Máximo de 30 resultados por busca para performance
- Contexto limitado a 4000 tokens por resposta
- Requer chave da Google AI para funcionamento

## Troubleshooting

### Erro de API Key
Certifique-se de que a `GOOGLE_API_KEY` está configurada corretamente no arquivo `.env`.

### Erro de Dependências
Execute `pip install -r requirements.txt` para instalar todas as dependências.

### Problemas com PDFs
Verifique se o arquivo PDF não está corrompido e contém texto extraível.

## Desenvolvimento

Para executar em modo de desenvolvimento:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

A documentação interativa da API estará disponível em `http://localhost:8000/docs`.