"""
ETAPA 6 — Agentes de IA (2026)
Módulo para ensinar criação de agentes autônomos
"""
import streamlit as st


def render_etapa6():
    """Renderiza o conteúdo da Etapa 6"""
    
    st.title("🧱 ETAPA 6 — Agentes de IA (2026)")
    st.markdown("**Duração:** Flexível")
    
    st.markdown("""
    Agentes de IA são o futuro - sistemas capazes de planejamento, uso de ferramentas 
    e tomada de decisões autônomas para completar tarefas complexas.
    """)
    
    # Tópicos
    st.header("📚 O que você vai dominar:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 🔗 **LangChain**
        - 📊 **LangGraph**
        - 📚 **LlamaIndex**
        - 🔧 **Agentes com Ferramentas**
        """)
    
    with col2:
        st.markdown("""
        - 🎯 **Planejamento de Longo Prazo**
        - 🔄 **Autoavaliação de Respostas**
        - 🤖 **Multi-Agent Systems**
        - ⚡ **Agentes Autônomos**
        """)
    
    st.success("🎯 **Resultado:** Capaz de criar agentes empresariais, automações complexas e sistemas autônomos.")
    
    st.markdown("---")
    
    tabs = st.tabs([
        "LangChain",
        "LangGraph",
        "LlamaIndex",
        "Multi-Agent",
        "Exercícios"
    ])
    
    with tabs[0]:
        render_langchain()
    
    with tabs[1]:
        render_langgraph()
    
    with tabs[2]:
        render_llamaindex()
    
    with tabs[3]:
        render_multiagent()
    
    with tabs[4]:
        render_exercicios_etapa6()


def render_langchain():
    """LangChain básico"""
    st.subheader("🔗 LangChain - Framework de Agentes")
    
    st.markdown("""
    ### Chains Básicos
    """)
    
    code = """
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Modelo
llm = ChatOpenAI(model="gpt-4.1", temperature=0.7)

# Template de prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente especializado em {especialidade}."),
    ("user", "{pergunta}")
])

# Chain
chain = prompt | llm

# Executar
response = chain.invoke({
    "especialidade": "Python",
    "pergunta": "Como funcionam decorators?"
})

print(response.content)

# Chain com parser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Resposta(BaseModel):
    resumo: str = Field(description="Resumo da resposta")
    detalhes: list[str] = Field(description="Lista de pontos detalhados")
    exemplo: str = Field(description="Exemplo de código")

parser = PydanticOutputParser(pydantic_object=Resposta)

prompt_estruturado = ChatPromptTemplate.from_template(
    \"\"\"
    Responda a pergunta de forma estruturada.
    
    {format_instructions}
    
    Pergunta: {pergunta}
    \"\"\"
)

chain_estruturado = prompt_estruturado | llm | parser

resultado = chain_estruturado.invoke({
    "pergunta": "O que são list comprehensions?",
    "format_instructions": parser.get_format_instructions()
})

print(resultado.resumo)
print(resultado.detalhes)
"""
    
    st.code(code, language="python")
    
    st.markdown("---")
    
    st.markdown("""
    ### Agentes com Ferramentas
    """)
    
    code_agents = """
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests

# Definir ferramentas
def buscar_clima(cidade: str) -> str:
    \"\"\"Busca informações do clima de uma cidade\"\"\"
    # Simulação - use API real em produção
    return f"Temperatura em {cidade}: 25°C, Ensolarado"

def calcular(expressao: str) -> str:
    \"\"\"Calcula expressões matemáticas\"\"\"
    try:
        resultado = eval(expressao)
        return f"Resultado: {resultado}"
    except:
        return "Erro ao calcular"

def buscar_web(query: str) -> str:
    \"\"\"Busca informações na web\"\"\"
    # Usar DuckDuckGo ou outra API
    return f"Resultados para '{query}': [informações simuladas]"

tools = [
    Tool(
        name="BuscarClima",
        func=buscar_clima,
        description="Útil para buscar informações do clima de uma cidade"
    ),
    Tool(
        name="Calcular",
        func=calcular,
        description="Útil para calcular expressões matemáticas"
    ),
    Tool(
        name="BuscarWeb",
        func=buscar_web,
        description="Útil para buscar informações na internet"
    )
]

# Modelo
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil que usa ferramentas para responder perguntas."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Criar agente
agent = create_openai_functions_agent(llm, tools, prompt)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# Executar
resultado = agent_executor.invoke({
    "input": "Qual o clima em São Paulo e quanto é 25 * 47?"
})

print(resultado["output"])
"""
    
    st.code(code_agents, language="python")
    
    st.info("💡 **LangChain simplifica:** Chains, Agents, Memory, RAG e muito mais!")


def render_langgraph():
    """LangGraph"""
    st.subheader("📊 LangGraph - Agentes com Estado")
    
    st.markdown("""
    ### Fluxos Complexos com Grafo
    
    LangGraph permite criar agentes com múltiplos estados e decisões.
    """)
    
    code = """
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Definir estado
class AgentState(TypedDict):
    mensagens: Annotated[list, operator.add]
    proximo: str
    resultado: str

# Funções dos nós
def analisar_pergunta(state: AgentState):
    pergunta = state["mensagens"][-1]
    
    # Decidir próxima ação
    if "clima" in pergunta.lower():
        proximo = "buscar_clima"
    elif any(op in pergunta for op in ["+", "-", "*", "/"]):
        proximo = "calcular"
    else:
        proximo = "responder_geral"
    
    return {"proximo": proximo}

def buscar_clima(state: AgentState):
    resultado = "Clima: 25°C, Ensolarado"
    return {"resultado": resultado, "proximo": "finalizar"}

def calcular(state: AgentState):
    # Extrair e calcular
    resultado = "Resultado: 42"
    return {"resultado": resultado, "proximo": "finalizar"}

def responder_geral(state: AgentState):
    resultado = "Resposta geral usando LLM"
    return {"resultado": resultado, "proximo": "finalizar"}

def finalizar(state: AgentState):
    return {"mensagens": [state["resultado"]]}

# Criar grafo
workflow = StateGraph(AgentState)

# Adicionar nós
workflow.add_node("analisar", analisar_pergunta)
workflow.add_node("buscar_clima", buscar_clima)
workflow.add_node("calcular", calcular)
workflow.add_node("responder_geral", responder_geral)
workflow.add_node("finalizar", finalizar)

# Definir edges
workflow.set_entry_point("analisar")

workflow.add_conditional_edges(
    "analisar",
    lambda x: x["proximo"],
    {
        "buscar_clima": "buscar_clima",
        "calcular": "calcular",
        "responder_geral": "responder_geral"
    }
)

workflow.add_edge("buscar_clima", "finalizar")
workflow.add_edge("calcular", "finalizar")
workflow.add_edge("responder_geral", "finalizar")
workflow.add_edge("finalizar", END)

# Compilar
app = workflow.compile()

# Executar
resultado = app.invoke({
    "mensagens": ["Qual o clima hoje?"],
    "proximo": "",
    "resultado": ""
})

print(resultado)
"""
    
    st.code(code, language="python")
    
    st.markdown("### Agente ReAct com LangGraph")
    
    code_react = """
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

class ReactState(TypedDict):
    pensamento: str
    acao: str
    observacao: str
    resposta_final: str
    iteracao: int

llm = ChatOpenAI(model="gpt-4.1")

def pensar(state: ReactState):
    prompt = f\"\"\"
    Baseado no histórico, qual o próximo passo?
    
    Pensamento anterior: {state.get('pensamento', 'Início')}
    Observação anterior: {state.get('observacao', 'Nenhuma')}
    
    Pense no próximo passo:
    \"\"\"
    
    pensamento = llm.invoke(prompt).content
    return {"pensamento": pensamento, "iteracao": state["iteracao"] + 1}

def agir(state: ReactState):
    # Extrair ação do pensamento
    acao = "buscar informação"  # Simplificado
    return {"acao": acao}

def observar(state: ReactState):
    # Executar ação e observar resultado
    observacao = f"Resultado da ação: {state['acao']}"
    return {"observacao": observacao}

def decidir_proximo(state: ReactState) -> str:
    if state["iteracao"] >= 3 or "resposta encontrada" in state.get("pensamento", ""):
        return "finalizar"
    return "pensar"

# Criar grafo ReAct
react_workflow = StateGraph(ReactState)

react_workflow.add_node("pensar", pensar)
react_workflow.add_node("agir", agir)
react_workflow.add_node("observar", observar)

react_workflow.set_entry_point("pensar")
react_workflow.add_edge("pensar", "agir")
react_workflow.add_edge("agir", "observar")

react_workflow.add_conditional_edges(
    "observar",
    decidir_proximo,
    {
        "pensar": "pensar",
        "finalizar": END
    }
)

react_app = react_workflow.compile()
"""
    
    st.code(code_react, language="python")


def render_llamaindex():
    """LlamaIndex"""
    st.subheader("📚 LlamaIndex - RAG e Agentes")
    
    st.markdown("""
    ### LlamaIndex para RAG Avançado
    """)
    
    code = """
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configurar
Settings.llm = OpenAI(model="gpt-4.1", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Carregar documentos
documents = SimpleDirectoryReader("./data").load_data()

# Criar índice
index = VectorStoreIndex.from_documents(documents)

# Query engine
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="tree_summarize"
)

# Fazer pergunta
response = query_engine.query("O que é machine learning?")
print(response)

# Ver fontes
for node in response.source_nodes:
    print(f"Score: {node.score:.3f}")
    print(f"Texto: {node.text[:200]}...")
    print(f"Metadata: {node.metadata}")
"""
    
    st.code(code, language="python")
    
    st.markdown("---")
    
    st.markdown("### Agentes com LlamaIndex")
    
    code_agents = """
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# Definir ferramentas
def buscar_documento(query: str) -> str:
    \"\"\"Busca informações nos documentos\"\"\"
    response = query_engine.query(query)
    return str(response)

def calcular_metricas(modelo: str) -> dict:
    \"\"\"Calcula métricas de um modelo\"\"\"
    return {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.92
    }

# Criar tools
buscar_tool = FunctionTool.from_defaults(fn=buscar_documento)
metricas_tool = FunctionTool.from_defaults(fn=calcular_metricas)

# Criar agente
agent = ReActAgent.from_tools(
    [buscar_tool, metricas_tool],
    verbose=True
)

# Executar
response = agent.chat("Me explique sobre ML e me dê as métricas do modelo XGBoost")
print(response)
"""
    
    st.code(code_agents, language="python")


def render_multiagent():
    """Sistemas Multi-Agent"""
    st.subheader("🤖 Multi-Agent Systems")
    
    st.markdown("""
    ### Múltiplos Agentes Colaborando
    
    Agentes especializados trabalhando juntos para resolver tarefas complexas.
    """)
    
    code = """
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

class MultiAgentSystem:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0)
        
        # Agente Pesquisador
        self.pesquisador = self.criar_agente(
            "pesquisador",
            "Você é um pesquisador especializado em coletar informações."
        )
        
        # Agente Analista
        self.analista = self.criar_agente(
            "analista",
            "Você é um analista que processa e interpreta informações."
        )
        
        # Agente Escritor
        self.escritor = self.criar_agente(
            "escritor",
            "Você é um escritor que cria conteúdo claro e bem estruturado."
        )
    
    def criar_agente(self, nome, descricao):
        prompt = ChatPromptTemplate.from_messages([
            ("system", descricao),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(self.llm, [], prompt)
        return AgentExecutor(agent=agent, tools=[], verbose=True)
    
    def executar_tarefa(self, tarefa: str):
        # 1. Pesquisar
        print("\\n=== PESQUISADOR ===")
        pesquisa = self.pesquisador.invoke({
            "input": f"Pesquise sobre: {tarefa}"
        })
        
        # 2. Analisar
        print("\\n=== ANALISTA ===")
        analise = self.analista.invoke({
            "input": f"Analise estas informações: {pesquisa['output']}"
        })
        
        # 3. Escrever
        print("\\n=== ESCRITOR ===")
        resultado = self.escritor.invoke({
            "input": f"Escreva um artigo baseado nesta análise: {analise['output']}"
        })
        
        return resultado['output']

# Usar sistema
sistema = MultiAgentSystem()
resultado = sistema.executar_tarefa("Impacto da IA na medicina")
print("\\n=== RESULTADO FINAL ===")
print(resultado)
"""
    
    st.code(code, language="python")
    
    st.markdown("---")
    
    st.markdown("""
    ### AutoGen - Framework Microsoft
    """)
    
    code_autogen = """
import autogen

# Configurar LLM
config_list = [
    {
        "model": "gpt-4.1",
        "api_key": "your-key"
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0
}

# Criar agentes
assistente = autogen.AssistantAgent(
    name="Assistente",
    llm_config=llm_config,
    system_message="Você é um assistente útil que ajuda a resolver problemas."
)

usuario = autogen.UserProxyAgent(
    name="Usuario",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

# Iniciar conversa
usuario.initiate_chat(
    assistente,
    message="Crie um script Python que baixa dados da API do GitHub e cria um gráfico."
)

# Multi-agent com especialização
critico = autogen.AssistantAgent(
    name="Critico",
    llm_config=llm_config,
    system_message="Você revisa código e sugere melhorias."
)

# Group chat
groupchat = autogen.GroupChat(
    agents=[usuario, assistente, critico],
    messages=[],
    max_round=10
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

usuario.initiate_chat(
    manager,
    message="Vamos criar um sistema de recomendação."
)
"""
    
    st.code(code_autogen, language="python")
    
    st.success("""
    ✅ **Padrões Multi-Agent:**
    - Hierárquico: Manager + Workers
    - Colaborativo: Todos iguais
    - Competitivo: Melhor resposta vence
    - Sequential: Pipeline de agentes
    """)


def render_exercicios_etapa6():
    """Exercícios da Etapa 6"""
    st.subheader("💪 Exercícios Práticos")
    
    exercicios = [
        {
            "titulo": "1. Agente Pessoal Autônomo",
            "descricao": "Crie um agente capaz de gerenciar tarefas e responder perguntas.",
            "requisitos": [
                "Múltiplas ferramentas (busca, cálculo, etc)",
                "Planejamento de tarefas",
                "Memória de conversação",
                "Interface de chat",
                "Logging de ações",
                "Error handling robusto"
            ]
        },
        {
            "titulo": "2. Sistema RAG com Agente",
            "descricao": "Combine RAG com agente que decide quando buscar informações.",
            "requisitos": [
                "LlamaIndex ou LangChain",
                "Decisão inteligente de busca",
                "Múltiplas fontes de dados",
                "Query refinement",
                "Citação de fontes",
                "Métricas de qualidade"
            ]
        },
        {
            "titulo": "3. Multi-Agent Research System",
            "descricao": "Sistema com agentes especializados pesquisando um tópico.",
            "requisitos": [
                "Agente Pesquisador",
                "Agente Analista",
                "Agente Escritor",
                "Comunicação entre agentes",
                "Produzir relatório final",
                "Visualizar fluxo"
            ]
        },
        {
            "titulo": "4. Agente de Automação",
            "descricao": "Agente que automatiza workflows empresariais.",
            "requisitos": [
                "Integração com APIs",
                "Processar emails/documentos",
                "Tomar decisões",
                "Notificações",
                "Auditoria completa",
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
        "Domino LangChain (chains, agents)",
        "Sei usar LangGraph para fluxos complexos",
        "Conheço LlamaIndex para RAG",
        "Consigo criar agentes com ferramentas",
        "Entendo padrões ReAct e CoT",
        "Implementei sistema multi-agent",
        "Sei planejar arquitetura de agentes",
        "Domino debugging de agentes"
    ]
    
    for i, item in enumerate(checklist):
        st.checkbox(item, key=f"check_etapa6_{i}")

