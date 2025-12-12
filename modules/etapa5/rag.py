"""
ETAPA 5 — RAG (Retrieval Augmented Generation)
Módulo para ensinar sistemas RAG e bancos vetoriais
"""
import streamlit as st


def render_etapa5():
    """Renderiza o conteúdo da Etapa 5"""
    
    st.title("📚 ETAPA 5 — RAG (Retrieval Augmented Generation)")
    st.markdown("**Duração:** Flexível")
    
    st.markdown("""
    RAG é a técnica mais importante para criar sistemas de IA empresariais com conhecimento específico.
    """)
    
    # Tópicos
    st.header("📚 O que você vai aprender:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 📊 **Vetorização de Dados**
        - 🗄️ **Chroma, Pinecone, Milvus**
        - 🔄 **Pipelines RAG**
        - 🔍 **Query Transformation**
        """)
    
    with col2:
        st.markdown("""
        - 📈 **Re-ranking**
        - 🎯 **Otimização de Contexto**
        - 💾 **Chunking Strategies**
        - ⚡ **Hybrid Search**
        """)
    
    st.success("🎯 **Resultado:** Capaz de criar sistemas empresariais com memória e conhecimento específico.")
    
    st.markdown("---")
    
    tabs = st.tabs([
        "Conceitos RAG",
        "Vector Databases",
        "Chunking",
        "Advanced RAG",
        "Exercícios"
    ])
    
    with tabs[0]:
        render_conceitos_rag()
    
    with tabs[1]:
        render_vector_dbs()
    
    with tabs[2]:
        render_chunking()
    
    with tabs[3]:
        render_advanced_rag()
    
    with tabs[4]:
        render_exercicios_etapa5()


def render_conceitos_rag():
    """Conceitos básicos de RAG"""
    st.subheader("📚 O que é RAG?")
    
    st.markdown("""
    ### Retrieval Augmented Generation
    
    RAG combina busca de informações com geração de texto para criar respostas baseadas em conhecimento específico.
    
    **Fluxo básico:**
    1. 📄 Usuário faz uma pergunta
    2. 🔍 Sistema busca documentos relevantes
    3. 📝 Contexto é adicionado ao prompt
    4. 🤖 LLM gera resposta baseada no contexto
    5. ✅ Resposta é retornada com fontes
    """)
    
    code = """
from openai import OpenAI
from chromadb import Client
import chromadb

# Inicializar
client_openai = OpenAI()
chroma_client = chromadb.Client()

# Criar coleção
collection = chroma_client.create_collection(name="documentos")

# 1. Adicionar documentos
documentos = [
    "Python é uma linguagem de programação de alto nível.",
    "Machine Learning é um subcampo da Inteligência Artificial.",
    "RAG combina busca e geração de texto.",
]

collection.add(
    documents=documentos,
    ids=[f"doc{i}" for i in range(len(documentos))]
)

# 2. Fazer pergunta
pergunta = "O que é RAG?"

# 3. Buscar documentos relevantes
resultados = collection.query(
    query_texts=[pergunta],
    n_results=2
)

documentos_relevantes = resultados['documents'][0]

# 4. Criar prompt com contexto
contexto = "\\n".join(documentos_relevantes)
prompt = f\"\"\"
Contexto:
{contexto}

Pergunta: {pergunta}

Responda baseado no contexto fornecido:
\"\"\"

# 5. Gerar resposta
response = client_openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
"""
    
    st.code(code, language="python")
    
    st.info("""
    💡 **Vantagens do RAG:**
    - ✅ Conhecimento atualizado
    - ✅ Reduz alucinações
    - ✅ Cita fontes
    - ✅ Domínio específico
    """)


def render_vector_dbs():
    """Bancos de dados vetoriais"""
    st.subheader("🗄️ Vector Databases")
    
    st.markdown("### Chroma - Simples e Local")
    
    code_chroma = """
import chromadb
from chromadb.utils import embedding_functions

# Cliente persistente
client = chromadb.PersistentClient(path="./chroma_db")

# Função de embedding
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-3-small"
)

# Criar coleção
collection = client.create_collection(
    name="meus_docs",
    embedding_function=openai_ef,
    metadata={"description": "Documentação técnica"}
)

# Adicionar documentos
collection.add(
    documents=[
        "FastAPI é um framework web moderno para Python.",
        "Streamlit permite criar apps de dados rapidamente.",
        "Docker containeriza aplicações para deploy."
    ],
    metadatas=[
        {"tipo": "framework", "linguagem": "python"},
        {"tipo": "ui", "linguagem": "python"},
        {"tipo": "devops", "linguagem": "agnostic"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Buscar
resultados = collection.query(
    query_texts=["Como criar interfaces web?"],
    n_results=2,
    where={"linguagem": "python"}  # filtro
)

print(resultados['documents'])
print(resultados['distances'])
"""
    
    st.code(code_chroma, language="python")
    
    st.markdown("---")
    
    st.markdown("### Pinecone - Produção e Escala")
    
    code_pinecone = """
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Inicializar
pc = Pinecone(api_key="your-pinecone-key")
openai_client = OpenAI()

# Criar índice
index_name = "meu-rag-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimensão do embedding
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# Função para embedar
def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Upsert documentos
documentos = [
    {"id": "doc1", "text": "Python é ótimo para IA"},
    {"id": "doc2", "text": "JavaScript domina a web"},
]

vectors = []
for doc in documentos:
    vector = get_embedding(doc["text"])
    vectors.append({
        "id": doc["id"],
        "values": vector,
        "metadata": {"text": doc["text"]}
    })

index.upsert(vectors=vectors)

# Query
query = "linguagens para inteligência artificial"
query_vector = get_embedding(query)

results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)

for match in results['matches']:
    print(f"Score: {match['score']:.3f}")
    print(f"Text: {match['metadata']['text']}")
"""
    
    st.code(code_pinecone, language="python")
    
    st.markdown("### Comparação:")
    
    import pandas as pd
    comparison = {
        "Database": ["Chroma", "Pinecone", "Milvus", "Weaviate", "Qdrant"],
        "Tipo": ["Local/Cloud", "Cloud", "Self-hosted", "Cloud/Self", "Cloud/Self"],
        "Uso": ["Dev/Pequeno", "Produção", "Grande escala", "Semântico", "Performance"],
        "Facilidade": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐"]
    }
    
    df = pd.DataFrame(comparison)
    st.table(df)


def render_chunking():
    """Estratégias de chunking"""
    st.subheader("💾 Chunking Strategies")
    
    st.markdown("""
    ### Por que Chunking é Importante?
    
    Dividir documentos em pedaços menores melhora a relevância da busca.
    """)
    
    code = """
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# 1. Character Text Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\\n\\n", "\\n", " ", ""]
)

texto = \"\"\"
[seu texto longo aqui]
\"\"\"

chunks = splitter.split_text(texto)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} caracteres")

# 2. Token-based Splitter
from langchain.text_splitter import TokenTextSplitter

token_splitter = TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks_tokens = token_splitter.split_text(texto)

# 3. Semantic Chunking (por similaridade)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"
)

semantic_chunks = semantic_splitter.split_text(texto)

# 4. Markdown/Code-aware Splitter
from langchain.text_splitter import MarkdownTextSplitter

md_splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# Mantém estrutura do markdown
md_chunks = md_splitter.split_text(markdown_text)
"""
    
    st.code(code, language="python")
    
    st.markdown("### Melhores Práticas:")
    
    st.markdown("""
    - 📏 **Tamanho:** 500-1000 tokens geralmente funciona bem
    - 🔄 **Overlap:** 10-20% para manter contexto
    - 📝 **Metadados:** Adicione source, page, section
    - 🎯 **Semântico:** Considere quebrar por tópicos
    - 🔍 **Teste:** Diferentes estratégias para seu domínio
    """)


def render_advanced_rag():
    """RAG avançado"""
    st.subheader("🚀 Advanced RAG Techniques")
    
    st.markdown("### 1. Query Transformation")
    
    code_query = """
from openai import OpenAI

client = OpenAI()

def expand_query(query):
    \"\"\"Expande query para melhorar busca\"\"\"
    prompt = f\"\"\"
    Dada a pergunta do usuário, gere 3 versões alternativas 
    que ajudariam a encontrar informações relevantes:
    
    Pergunta original: {query}
    
    Versões alternativas (uma por linha):
    \"\"\"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    alternativas = response.choices[0].message.content.strip().split('\\n')
    return [query] + alternativas

# Uso
query_original = "Como treinar modelos de ML?"
queries = expand_query(query_original)

# Buscar com todas as queries
all_results = []
for q in queries:
    results = collection.query(query_texts=[q], n_results=3)
    all_results.extend(results['documents'][0])

# Remover duplicatas e pegar top results
unique_results = list(set(all_results))[:5]
"""
    
    st.code(code_query, language="python")
    
    st.markdown("---")
    
    st.markdown("### 2. Re-ranking")
    
    code_rerank = """
from sentence_transformers import CrossEncoder

# Modelo de re-ranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query, documents, top_k=3):
    \"\"\"Re-rankeia documentos por relevância\"\"\"
    
    # Criar pares (query, doc)
    pairs = [[query, doc] for doc in documents]
    
    # Calcular scores
    scores = reranker.predict(pairs)
    
    # Ordenar por score
    ranked = sorted(zip(documents, scores), 
                   key=lambda x: x[1], 
                   reverse=True)
    
    return [doc for doc, score in ranked[:top_k]]

# Uso
query = "Como funciona RAG?"
initial_results = collection.query(query_texts=[query], n_results=10)
docs = initial_results['documents'][0]

# Re-rankear
final_docs = rerank_results(query, docs, top_k=3)
"""
    
    st.code(code_rerank, language="python")
    
    st.markdown("---")
    
    st.markdown("### 3. Hybrid Search (Keyword + Semantic)")
    
    code_hybrid = """
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearch:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = np.array(embeddings)
        
        # Preparar BM25
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query, query_embedding, top_k=5, alpha=0.5):
        \"\"\"
        alpha: peso entre keyword (0) e semantic (1)
        \"\"\"
        
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        
        # Semantic scores
        query_emb = np.array(query_embedding)
        semantic_scores = np.dot(self.embeddings, query_emb)
        semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
        
        # Combinar
        hybrid_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
        
        # Top K
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        return [(self.documents[i], hybrid_scores[i]) for i in top_indices]

# Uso
searcher = HybridSearch(documents, embeddings)
results = searcher.search(query, query_embedding, alpha=0.7)
"""
    
    st.code(code_hybrid, language="python")
    
    st.success("""
    ✅ **Técnicas Avançadas:**
    - Multi-query retrieval
    - Parent document retrieval
    - Self-query
    - Contextual compression
    - Ensemble retrieval
    """)


def render_exercicios_etapa5():
    """Exercícios da Etapa 5"""
    st.subheader("💪 Exercícios Práticos")
    
    exercicios = [
        {
            "titulo": "1. Sistema Q&A de Documentação",
            "descricao": "Crie um sistema RAG para responder perguntas sobre documentação técnica.",
            "requisitos": [
                "Carregar múltiplos PDFs/docs",
                "Chunking inteligente",
                "Chroma ou Pinecone",
                "Busca semântica",
                "Citar fontes com páginas",
                "Interface Streamlit"
            ]
        },
        {
            "titulo": "2. RAG com Re-ranking",
            "descricao": "Implemente sistema RAG com re-ranking de resultados.",
            "requisitos": [
                "Initial retrieval com embedding",
                "Re-rank com CrossEncoder",
                "Comparar com/sem rerank",
                "Métricas de relevância",
                "Query expansion",
                "Logging de queries"
            ]
        },
        {
            "titulo": "3. Hybrid Search System",
            "descricao": "Combine busca keyword e semântica.",
            "requisitos": [
                "BM25 + embeddings",
                "Tuning do alpha",
                "Comparar abordagens",
                "Dataset de avaliação",
                "Métricas: MRR, NDCG",
                "API de busca"
            ]
        },
        {
            "titulo": "4. RAG Empresarial",
            "descricao": "Sistema RAG completo para empresa.",
            "requisitos": [
                "Múltiplas fontes de dados",
                "Filtros de metadados",
                "Permissões de acesso",
                "Auditoria de queries",
                "Feedback loop",
                "Deploy em produção"
            ]
        }
    ]
    
    for ex in exercicios:
        with st.expander(f"{ex['titulo']}"):
            st.markdown(f"**Descrição:** {ex['descricao']}")
            st.markdown("**Requisitos:**")
            for req in ex['requisitos']:
                st.markdown(f"- {req}")
    
    st.markdown("---")
    st.markdown("### 📋 Checklist de Domínio:")
    
    checklist = [
        "Entendo o conceito e arquitetura RAG",
        "Sei usar vector databases (Chroma/Pinecone)",
        "Domino estratégias de chunking",
        "Consigo implementar busca semântica",
        "Sei fazer query transformation",
        "Entendo re-ranking",
        "Implementei hybrid search",
        "Sei avaliar qualidade do RAG"
    ]
    
    for i, item in enumerate(checklist):
        st.checkbox(item, key=f"check_etapa5_{i}")

