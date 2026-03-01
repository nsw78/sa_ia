"""
ETAPA 4 — LLMs + Engenharia de Prompt
Módulo para ensinar Large Language Models e técnicas de prompting
"""
import streamlit as st


def render_etapa4():
    """Renderiza o conteúdo da Etapa 4"""
    
    st.title("🤖 ETAPA 4 — LLMs + Engenharia de Prompt")
    st.markdown("**Duração:** 10 dias")
    
    st.markdown("""
    Domine os Large Language Models e aprenda a extrair o máximo deles através 
    de engenharia de prompt avançada.
    """)
    
    # Tópicos
    st.header("📚 O que você vai dominar:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 🌐 **OpenAI, Gemini, Claude, Llama**
        - 💻 **Modelos Locais com Ollama**
        - 🔤 **Tokenização**
        - 📊 **Embeddings**
        """)
    
    with col2:
        st.markdown("""
        - 💡 **Técnicas de Prompting**
        - 🎯 **Zero-shot, Few-shot, CoT**
        - 📝 **Instruções Personalizadas**
        - 🤖 **Agentes de IA**
        """)
    
    st.success("🎯 **Resultado:** Capaz de construir chatbots e aplicações avançadas com LLMs.")
    
    st.markdown("---")
    
    tabs = st.tabs([
        "APIs de LLMs",
        "Ollama Local",
        "Tokenização",
        "Prompting Avançado",
        "Agentes",
        "Exercícios"
    ])
    
    with tabs[0]:
        render_apis_llms()
    
    with tabs[1]:
        render_ollama()
    
    with tabs[2]:
        render_tokenizacao()
    
    with tabs[3]:
        render_prompting()
    
    with tabs[4]:
        render_agentes()
    
    with tabs[5]:
        render_exercicios_etapa4()


def render_apis_llms():
    """Seção de APIs de LLMs"""
    st.subheader("🌐 APIs de LLMs Principais")
    
    st.markdown("### OpenAI API")
    
    code_openai = """
from openai import OpenAI
import os

# Inicializar cliente
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chat completion
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "Você é um assistente especializado em IA."},
        {"role": "user", "content": "Explique o que são transformers em 3 linhas."}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Conte uma história curta"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Embeddings
embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Texto para embedar"
)

embedding = embedding_response.data[0].embedding
print(f"\\nDimensão do embedding: {len(embedding)}")
"""
    
    st.code(code_openai, language="python")
    
    st.markdown("---")
    
    st.markdown("### Google Gemini API")
    
    code_gemini = """
import google.generativeai as genai
import os

# Configurar
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Criar modelo
model = genai.GenerativeModel('gemini-pro')

# Gerar texto
response = model.generate_content("Explique computação quântica")
print(response.text)

# Chat
chat = model.start_chat(history=[])

response = chat.send_message("Olá! Como você funciona?")
print(response.text)

response = chat.send_message("Me dê um exemplo de uso")
print(response.text)

# Ver histórico
for message in chat.history:
    print(f"{message.role}: {message.parts[0].text}")
"""
    
    st.code(code_gemini, language="python")
    
    st.markdown("---")
    
    st.markdown("### Anthropic Claude API")
    
    code_claude = """
import anthropic
import os

# Inicializar
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Chat
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explique redes neurais de forma simples"}
    ]
)

print(message.content[0].text)

# Com system prompt
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="Você é um professor de IA que explica conceitos de forma clara.",
    messages=[
        {"role": "user", "content": "O que é overfitting?"}
    ]
)

print(message.content[0].text)

# Streaming
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Conte uma piada sobre IA"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
"""
    
    st.code(code_claude, language="python")
    
    st.markdown("### Comparação de Modelos:")
    
    comparison = {
        "Modelo": ["GPT-5", "GPT-4.1", "Claude Opus 4", "Claude Sonnet 4", "Gemini 2.5 Pro", "Llama 4"],
        "Contexto": ["256K tokens", "1M tokens", "200K tokens", "200K tokens", "2M tokens", "128K tokens"],
        "Uso": ["Frontier, reasoning", "Rapido, longo contexto", "Deep analysis, code", "Fast, balanced", "Contexto longo", "Open source"]
    }
    
    import pandas as pd
    df = pd.DataFrame(comparison)
    st.table(df)


def render_ollama():
    """Seção de Ollama"""
    st.subheader("💻 Ollama - LLMs Locais")
    
    st.markdown("""
    ### Rodar LLMs Localmente
    
    Ollama permite rodar modelos como Llama, Mistral, etc localmente.
    """)
    
    installation = """
# Instalação
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows: baixar de https://ollama.com/download

# Verificar instalação
ollama --version

# Baixar modelos
ollama pull llama4
ollama pull mistral
ollama pull codellama

# Listar modelos
ollama list

# Rodar modelo
ollama run llama4
"""
    
    st.code(installation, language="bash")
    
    st.markdown("### Usar Ollama com Python:")
    
    code = """
import requests
import json

# API endpoint
url = "http://localhost:11434/api/generate"

# Fazer requisição
data = {
    "model": "llama4",
    "prompt": "Explique machine learning em 3 linhas",
    "stream": False
}

response = requests.post(url, json=data)
result = response.json()
print(result['response'])

# Streaming
data["stream"] = True
response = requests.post(url, json=data, stream=True)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line)
        print(chunk.get('response', ''), end='', flush=True)

# Usando biblioteca ollama
import ollama

# Gerar texto
response = ollama.generate(
    model='llama4',
    prompt='Por que o céu é azul?'
)
print(response['response'])

# Chat
messages = [
    {'role': 'user', 'content': 'Por que Python é popular em IA?'}
]

response = ollama.chat(model='llama4', messages=messages)
print(response['message']['content'])
"""
    
    st.code(code, language="python")
    
    st.success("""
    ✅ **Vantagens do Ollama:**
    - 100% privado e offline
    - Sem custos de API
    - Controle total
    - Rápido para testes
    """)


def render_tokenizacao():
    """Seção de Tokenização"""
    st.subheader("🔤 Tokenização e Embeddings")
    
    st.markdown("""
    ### O que é Tokenização?
    
    LLMs não entendem texto diretamente - eles processam tokens.
    """)
    
    code = """
import tiktoken

# Encoder para GPT-4.1
encoding = tiktoken.encoding_for_model("gpt-4.1")

# Tokenizar texto
text = "Inteligência Artificial está revolucionando o mundo!"
tokens = encoding.encode(text)

print(f"Texto: {text}")
print(f"Tokens: {tokens}")
print(f"Número de tokens: {len(tokens)}")

# Decodificar
decoded = encoding.decode(tokens)
print(f"Decodificado: {decoded}")

# Ver cada token
for token in tokens:
    print(f"{token} -> {encoding.decode([token])!r}")

# Calcular custo
def calcular_custo(num_tokens, preco_por_1k=0.03):
    return (num_tokens / 1000) * preco_por_1k

texto_longo = "..." * 1000  # seu texto
num_tokens = len(encoding.encode(texto_longo))
custo = calcular_custo(num_tokens)
print(f"\\nCusto estimado: ${custo:.4f}")
"""
    
    st.code(code, language="python")
    
    st.markdown("---")
    
    st.markdown("""
    ### Embeddings - Representações Vetoriais
    
    Embeddings capturam significado semântico de textos.
    """)
    
    code_embeddings = """
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Criar embeddings
textos = [
    "Cachorro é um animal de estimação",
    "Gato é um pet doméstico",
    "Python é uma linguagem de programação"
]

embeddings = [get_embedding(t) for t in textos]

# Calcular similaridade coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Comparar similaridades
for i, texto1 in enumerate(textos):
    for j, texto2 in enumerate(textos):
        if i < j:
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"Similaridade entre '{texto1}' e '{texto2}': {sim:.3f}")

# Busca semântica
query = "animais domésticos"
query_embedding = get_embedding(query)

for texto, emb in zip(textos, embeddings):
    sim = cosine_similarity(query_embedding, emb)
    print(f"'{texto}': {sim:.3f}")
"""
    
    st.code(code_embeddings, language="python")
    
    st.info("💡 **Embeddings são a base de:** RAG, busca semântica, recomendações!")


def render_prompting():
    """Seção de Prompting Avançado"""
    st.subheader("💡 Técnicas de Prompting Avançado")
    
    st.markdown("### 1. Zero-Shot Prompting")
    
    zero_shot = """
prompt = \"\"\"
Classifique o sentimento do seguinte texto como Positivo, Negativo ou Neutro:

Texto: "Este produto superou minhas expectativas! Qualidade excelente."

Sentimento:
\"\"\"
"""
    
    st.code(zero_shot, language="python")
    
    st.markdown("### 2. Few-Shot Prompting")
    
    few_shot = """
prompt = \"\"\"
Classifique o sentimento dos textos:

Texto: "Adorei este filme, muito emocionante!"
Sentimento: Positivo

Texto: "Péssimo atendimento, não recomendo."
Sentimento: Negativo

Texto: "O produto é ok, nada excepcional."
Sentimento: Neutro

Texto: "Melhor compra que já fiz, super recomendo!"
Sentimento:
\"\"\"
"""
    
    st.code(few_shot, language="python")
    
    st.markdown("### 3. Chain-of-Thought (CoT)")
    
    cot = """
prompt = \"\"\"
Resolva este problema passo a passo:

Problema: João tinha 15 maçãs. Ele deu 3 para Maria e comprou 8 mais. 
Depois comeu 2. Quantas maçãs João tem agora?

Vamos pensar passo a passo:
1. João começou com: 15 maçãs
2. Deu 3 para Maria: 15 - 3 = 12 maçãs
3. Comprou 8 mais: 12 + 8 = 20 maçãs
4. Comeu 2: 20 - 2 = 18 maçãs

Resposta: João tem 18 maçãs.

---

Agora resolva:
Pedro tinha 50 reais. Gastou 12 em almoço, ganhou 20 do pai, 
e gastou 15 em um livro. Quanto Pedro tem agora?

Vamos pensar passo a passo:
\"\"\"
"""
    
    st.code(cot, language="python")
    
    st.markdown("### 4. ReAct (Reasoning + Acting)")
    
    react = """
prompt = \"\"\"
Você é um assistente que pensa e age passo a passo.

Tarefa: Encontre a capital da França e sua população.

Pensamento: Preciso encontrar a capital da França.
Ação: Buscar[Capital da França]
Observação: A capital da França é Paris.

Pensamento: Agora preciso encontrar a população de Paris.
Ação: Buscar[População de Paris]
Observação: Paris tem aproximadamente 2.2 milhões de habitantes.

Pensamento: Tenho todas as informações necessárias.
Resposta: A capital da França é Paris, com população de cerca de 2.2 milhões.
\"\"\"
"""
    
    st.code(react, language="python")
    
    st.markdown("### 5. Self-Consistency")
    
    self_consistency = """
# Gerar múltiplas respostas e escolher a mais comum
prompts = [
    "Resolva: 25 * 4 + 12 / 3 = ?",
    "Calcule passo a passo: 25 * 4 + 12 / 3",
    "Qual o resultado de 25 * 4 + 12 / 3?"
]

respostas = []
for prompt in prompts:
    # Gerar 3 respostas para cada
    for _ in range(3):
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        respostas.append(response.choices[0].message.content)

# Encontrar resposta mais comum (maioria)
from collections import Counter
resposta_final = Counter(respostas).most_common(1)[0][0]
"""
    
    st.code(self_consistency, language="python")
    
    st.success("""
    ✅ **Técnicas Avançadas:**
    - Tree of Thoughts
    - Self-Refine
    - ReWOO
    - Skeleton-of-Thought
    """)


def render_agentes():
    """Seção de Agentes"""
    st.subheader("🤖 Agentes de IA")
    
    st.markdown("""
    ### O que são Agentes?
    
    Agentes podem usar ferramentas e tomar decisões para completar tarefas.
    """)
    
    code = """
from openai import OpenAI
import json

client = OpenAI()

# Definir ferramentas
tools = [
    {
        "type": "function",
        "function": {
            "name": "buscar_clima",
            "description": "Busca informações do clima atual de uma cidade",
            "parameters": {
                "type": "object",
                "properties": {
                    "cidade": {
                        "type": "string",
                        "description": "Nome da cidade"
                    },
                    "unidade": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["cidade"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calcular",
            "description": "Calcula expressões matemáticas",
            "parameters": {
                "type": "object",
                "properties": {
                    "expressao": {
                        "type": "string",
                        "description": "Expressão matemática"
                    }
                },
                "required": ["expressao"]
            }
        }
    }
]

# Implementar funções
def buscar_clima(cidade, unidade="celsius"):
    # Simulação - em produção usar API real
    return {
        "cidade": cidade,
        "temperatura": 25,
        "condicao": "Ensolarado",
        "unidade": unidade
    }

def calcular(expressao):
    try:
        resultado = eval(expressao)
        return {"resultado": resultado}
    except:
        return {"erro": "Expressão inválida"}

# Mapa de funções
available_functions = {
    "buscar_clima": buscar_clima,
    "calcular": calcular
}

# Agente em ação
messages = [
    {"role": "user", "content": "Qual o clima em São Paulo e quanto é 25 * 4?"}
]

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# Processar tool calls
response_message = response.choices[0].message
tool_calls = response_message.tool_calls

if tool_calls:
    messages.append(response_message)
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # Executar função
        function_response = available_functions[function_name](**function_args)
        
        # Adicionar resultado
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": json.dumps(function_response)
        })
    
    # Gerar resposta final
    final_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages
    )
    
    print(final_response.choices[0].message.content)
"""
    
    st.code(code, language="python")
    
    st.markdown("### Tipos de Agentes:")
    
    st.markdown("""
    - 🔧 **ReAct Agent**: Reasoning + Acting
    - 🎯 **Plan-and-Execute**: Planeja antes de agir
    - 🔄 **Multi-Agent**: Múltiplos agentes colaborando
    - 🌳 **Tree-of-Thoughts**: Explora múltiplos caminhos
    """)


def render_exercicios_etapa4():
    """Exercícios da Etapa 4"""
    st.subheader("💪 Exercícios Práticos")
    
    exercicios = [
        {
            "titulo": "1. Chatbot Avançado",
            "descricao": "Crie um chatbot com memória e personalidade.",
            "requisitos": [
                "Usar OpenAI ou Claude",
                "Implementar memória de conversação",
                "System prompt bem definido",
                "Streaming de respostas",
                "Interface Streamlit",
                "Salvar histórico"
            ]
        },
        {
            "titulo": "2. Sistema de Classificação",
            "descricao": "Classifique tickets de suporte automaticamente.",
            "requisitos": [
                "Few-shot prompting",
                "Classificar em categorias",
                "Extrair informações chave",
                "Sugerir prioridade",
                "API FastAPI",
                "Logging com MLflow"
            ]
        },
        {
            "titulo": "3. Agente com Ferramentas",
            "descricao": "Crie um agente que usa múltiplas ferramentas.",
            "requisitos": [
                "Mínimo 3 ferramentas",
                "Busca na web",
                "Cálculos matemáticos",
                "Acesso a banco de dados",
                "ReAct pattern",
                "Tratamento de erros"
            ]
        },
        {
            "titulo": "4. Sistema RAG Básico",
            "descricao": "Crie um sistema Q&A sobre documentos.",
            "requisitos": [
                "Carregar PDFs/textos",
                "Gerar embeddings",
                "Busca semântica",
                "Gerar resposta com contexto",
                "Interface web",
                "Citar fontes"
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
        "Sei usar APIs de LLMs principais",
        "Consigo rodar modelos locais com Ollama",
        "Entendo tokenização e seus impactos",
        "Domino técnicas de prompting",
        "Sei criar few-shot prompts efetivos",
        "Entendo Chain-of-Thought",
        "Consigo criar agentes com ferramentas",
        "Sei quando usar cada modelo/técnica"
    ]
    
    for i, item in enumerate(checklist):
        st.checkbox(item, key=f"check_etapa4_{i}")

