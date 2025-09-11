import sys
import os

import uvicorn
from app.main import app


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":

    
    print("Iniciando servidor RAG...")
    print("Acesse: http://localhost:8000")
    print("Documentação: http://localhost:8000/docs")
    print("Pressione Ctrl+C para parar")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
