"""
ETAPA 1 — Fundamentos Essenciais
Módulo para ensinar fundamentos de desenvolvimento para IA
"""
import streamlit as st
import sys
from io import StringIO


def render_etapa1():
    """Renderiza o conteúdo da Etapa 1"""
    
    st.title("🧠 ETAPA 1 — Fundamentos Essenciais")
    st.markdown("**Duração:** 7 dias")
    
    # Introdução
    st.markdown("""
    Esta etapa estabelece as bases sólidas para sua jornada como desenvolvedor de IA.
    Você dominará as ferramentas e conceitos essenciais necessários para criar aplicações de IA robustas.
    """)
    
    # Tópicos
    st.header("📚 Tópicos que você vai dominar:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - ✅ **Python Avançado**
        - ✅ **Estruturas de Dados**
        - ✅ **APIs FastAPI**
        - ✅ **Versionamento com Git**
        """)
    
    with col2:
        st.markdown("""
        - ✅ **Testes Automatizados**
        - ✅ **Docker para IA**
        - ✅ **Boas Práticas**
        - ✅ **Debugging Avançado**
        """)
    
    st.success("🎯 **Resultado:** Você vira Dev de IA básico capaz de criar APIs que usam modelos.")
    
    # Seções interativas
    st.markdown("---")
    
    tabs = st.tabs(["Python Avançado", "FastAPI", "Docker", "Git & Testes", "Exercícios"])
    
    with tabs[0]:
        render_python_avancado()
    
    with tabs[1]:
        render_fastapi()
    
    with tabs[2]:
        render_docker()
    
    with tabs[3]:
        render_git_testes()
    
    with tabs[4]:
        render_exercicios_etapa1()


def render_python_avancado():
    """Seção de Python Avançado"""
    st.subheader("🐍 Python Avançado")
    
    st.markdown("""
    ### Estruturas de Dados Essenciais
    
    Python oferece estruturas de dados poderosas que são fundamentais para IA:
    """)
    
    # Exemplo interativo de list comprehensions
    st.markdown("#### 1. List Comprehensions e Generators")
    
    code = """
# List Comprehension - carrega tudo na memória
squares = [x**2 for x in range(10)]
print(f"Quadrados: {squares}")

# Generator - lazy evaluation (melhor para grandes datasets)
squares_gen = (x**2 for x in range(10))
print(f"Generator: {squares_gen}")
print(f"Primeiro valor: {next(squares_gen)}")

# Filtragem com comprehension
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"Quadrados pares: {even_squares}")
"""
    
    st.code(code, language="python")
    
    if st.button("🚀 Executar Exemplo - List Comprehensions", key="py_comp"):
        output = StringIO()
        sys.stdout = output
        try:
            squares = [x**2 for x in range(10)]
            print(f"Quadrados: {squares}")
            
            squares_gen = (x**2 for x in range(10))
            print(f"Generator: {squares_gen}")
            print(f"Primeiro valor: {next(squares_gen)}")
            
            even_squares = [x**2 for x in range(10) if x % 2 == 0]
            print(f"Quadrados pares: {even_squares}")
        finally:
            sys.stdout = sys.__stdout__
        
        st.code(output.getvalue())
    
    st.markdown("---")
    
    # Decorators
    st.markdown("#### 2. Decorators - Essenciais para frameworks de IA")
    
    code_decorator = """
import time
from functools import wraps

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executou em {end-start:.4f}s")
        return result
    return wrapper

@timer_decorator
def processar_dados(n):
    return sum([i**2 for i in range(n)])

resultado = processar_dados(100000)
print(f"Resultado: {resultado}")
"""
    
    st.code(code_decorator, language="python")
    
    if st.button("🚀 Executar Exemplo - Decorator", key="py_dec"):
        output = StringIO()
        sys.stdout = output
        try:
            import time
            from functools import wraps
            
            def timer_decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    start = time.time()
                    result = func(*args, **kwargs)
                    end = time.time()
                    print(f"{func.__name__} executou em {end-start:.4f}s")
                    return result
                return wrapper
            
            @timer_decorator
            def processar_dados(n):
                return sum([i**2 for i in range(n)])
            
            resultado = processar_dados(100000)
            print(f"Resultado: {resultado}")
        finally:
            sys.stdout = sys.__stdout__
        
        st.code(output.getvalue())
    
    # Context Managers
    st.markdown("#### 3. Context Managers - Gerenciamento de Recursos")
    
    code_context = """
from contextlib import contextmanager

@contextmanager
def gerenciar_conexao(nome):
    print(f"Abrindo conexão: {nome}")
    conexao = {"nome": nome, "ativa": True}
    try:
        yield conexao
    finally:
        print(f"Fechando conexão: {nome}")
        conexao["ativa"] = False

# Uso
with gerenciar_conexao("Database") as conn:
    print(f"Trabalhando com: {conn}")
"""
    
    st.code(code_context, language="python")
    
    st.info("💡 **Dica:** Context managers são essenciais para gerenciar conexões com bancos vetoriais e modelos de IA!")


def render_fastapi():
    """Seção de FastAPI"""
    st.subheader("⚡ FastAPI - APIs para IA")
    
    st.markdown("""
    FastAPI é o framework mais popular para criar APIs de IA. Veja um exemplo completo:
    """)
    
    code = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="API de IA", version="1.0")

# Modelo de dados
class PredictionRequest(BaseModel):
    texto: str
    temperatura: Optional[float] = 0.7

class PredictionResponse(BaseModel):
    resultado: str
    confianca: float

# Endpoint de predição
@app.post("/predict", response_model=PredictionResponse)
async def fazer_predicao(request: PredictionRequest):
    \"\"\"
    Endpoint para fazer predições usando modelo de IA
    \"\"\"
    if not request.texto:
        raise HTTPException(status_code=400, detail="Texto não pode estar vazio")
    
    # Simulação de predição
    resultado = f"Processado: {request.texto[:50]}..."
    confianca = 0.95
    
    return PredictionResponse(
        resultado=resultado,
        confianca=confianca
    )

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "IA API"}

# Para executar: uvicorn main:app --reload
"""
    
    st.code(code, language="python")
    
    st.markdown("### Recursos Importantes do FastAPI:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Vantagens:**
        - ⚡ Alta performance (async)
        - 📝 Documentação automática
        - ✅ Validação automática
        - 🔒 Type hints nativos
        """)
    
    with col2:
        st.markdown("""
        **Uso em IA:**
        - Deploy de modelos
        - APIs de inferência
        - Streaming de respostas
        - Batch processing
        """)
    
    st.info("💡 **Dica:** Use FastAPI com `uvicorn` para criar APIs de IA prontas para produção!")


def render_docker():
    """Seção de Docker"""
    st.subheader("🐳 Docker para Ambientes de IA")
    
    st.markdown("""
    Docker é essencial para garantir que seus modelos funcionem em qualquer ambiente.
    """)
    
    st.markdown("### Dockerfile para Aplicação de IA:")
    
    dockerfile = """
FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models

# Porta da API
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    st.code(dockerfile, language="dockerfile")
    
    st.markdown("### Docker Compose para Stack Completa:")
    
    docker_compose = """
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/aidb
    volumes:
      - ./models:/app/models
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: pass
      POSTGRES_USER: user
      POSTGRES_DB: aidb
    volumes:
      - pgdata:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
"""
    
    st.code(docker_compose, language="yaml")
    
    st.success("🎯 Com Docker você garante que seu ambiente de IA seja reproduzível!")


def render_git_testes():
    """Seção de Git e Testes"""
    st.subheader("🔧 Git & Testes Automatizados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 Git para Projetos de IA")
        
        git_commands = """
# Inicializar repositório
git init
git add .
git commit -m "Initial commit"

# Branches para experimentos
git checkout -b experimento-novo-modelo

# Git LFS para modelos grandes
git lfs install
git lfs track "*.pkl"
git lfs track "*.h5"
git lfs track "*.bin"

# .gitignore para IA
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".env" >> .gitignore
echo "models/*.bin" >> .gitignore
echo "data/*.csv" >> .gitignore
"""
        
        st.code(git_commands, language="bash")
    
    with col2:
        st.markdown("### ✅ Testes para IA")
        
        test_code = """
import pytest
from api import fazer_predicao

def test_predicao_valida():
    request = {
        "texto": "Teste de IA",
        "temperatura": 0.7
    }
    resultado = fazer_predicao(request)
    
    assert resultado["confianca"] > 0
    assert resultado["resultado"] != ""

def test_predicao_texto_vazio():
    request = {"texto": ""}
    
    with pytest.raises(ValueError):
        fazer_predicao(request)

def test_temperatura_invalida():
    request = {
        "texto": "Teste",
        "temperatura": 2.0  # > 1.0
    }
    
    with pytest.raises(ValueError):
        fazer_predicao(request)
"""
        
        st.code(test_code, language="python")
    
    st.info("💡 **Dica:** Use pytest + pytest-cov para garantir qualidade nos seus projetos de IA!")


def render_exercicios_etapa1():
    """Exercícios práticos da Etapa 1"""
    st.subheader("💪 Exercícios Práticos")
    
    st.markdown("""
    ### Exercícios para Consolidar o Conhecimento:
    """)
    
    exercicios = [
        {
            "titulo": "1. API de Classificação de Texto",
            "descricao": "Crie uma API FastAPI que recebe texto e retorna uma classificação (positivo/negativo/neutro).",
            "dificuldade": "Médio",
            "requisitos": [
                "Endpoint POST /classify",
                "Validação com Pydantic",
                "Testes com pytest",
                "Dockerfile funcional"
            ]
        },
        {
            "titulo": "2. Sistema de Cache com Decorators",
            "descricao": "Implemente um decorator que faz cache de resultados de funções custosas.",
            "dificuldade": "Médio",
            "requisitos": [
                "Decorator funcional",
                "TTL configurável",
                "Uso de Redis (opcional)",
                "Testes unitários"
            ]
        },
        {
            "titulo": "3. Pipeline de Dados",
            "descricao": "Crie um pipeline usando context managers para processar arquivos CSV.",
            "dificuldade": "Fácil",
            "requisitos": [
                "Context manager customizado",
                "Tratamento de erros",
                "Logging adequado",
                "Git com commits organizados"
            ]
        }
    ]
    
    for ex in exercicios:
        with st.expander(f"{ex['titulo']} - {ex['dificuldade']}"):
            st.markdown(f"**Descrição:** {ex['descricao']}")
            st.markdown("**Requisitos:**")
            for req in ex['requisitos']:
                st.markdown(f"- {req}")
    
    # Checklist de progresso
    st.markdown("---")
    st.markdown("### 📋 Checklist de Progresso:")
    
    checklist_items = [
        "Domino list comprehensions e generators",
        "Sei criar decorators customizados",
        "Consigo criar APIs com FastAPI",
        "Entendo async/await em Python",
        "Sei usar Git para versionamento",
        "Escrevo testes automatizados",
        "Consigo criar containers Docker",
        "Entendo pydantic para validação"
    ]
    
    for i, item in enumerate(checklist_items):
        st.checkbox(item, key=f"check_etapa1_{i}")
    
    st.success("✅ Complete todos os itens antes de avançar para a Etapa 2!")

