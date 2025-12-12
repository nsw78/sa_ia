"""
Sistema de Aprendizado de IA
Roadmap completo de 8 etapas para se tornar Engenheiro de IA
"""
import streamlit as st
from modules.etapa1 import render_etapa1
from modules.etapa2 import render_etapa2
from modules.etapa3 import render_etapa3
from modules.etapa4 import render_etapa4
from modules.etapa5 import render_etapa5
from modules.etapa6 import render_etapa6
from modules.etapa7 import render_etapa7
from modules.etapa8 import render_etapa8

# Configuração da página
st.set_page_config(
    page_title="Sistema de Aprendizado de IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .etapa-card {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 0.5rem 0;
        transition: all 0.3s;
    }
    
    .etapa-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    .stats-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def render_home():
    """Renderiza a página inicial"""
    
    st.markdown('<h1 class="main-header">🤖 Sistema de Aprendizado de IA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Roadmap completo para se tornar Engenheiro de IA em 8 etapas</p>', unsafe_allow_html=True)
    
    # Introdução
    st.markdown("""
    ## 👋 Bem-vindo!
    
    Este é um sistema completo e estruturado para você dominar **Inteligência Artificial** 
    do zero até o nível avançado. Cada etapa foi cuidadosamente planejada para construir 
    suas habilidades de forma progressiva.
    
    ### 🎯 O que você vai alcançar:
    
    - ✅ **Fundamentos sólidos** de programação e ML
    - ✅ **Domínio completo** de Deep Learning
    - ✅ **Expertise** em LLMs e Agentes
    - ✅ **Capacidade** de deploy em produção
    - ✅ **Especialização** em AI Security
    
    ### 📊 Estatísticas do Programa:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-box">
            <h2>8</h2>
            <p>Etapas Completas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-box">
            <h2>50+</h2>
            <p>Dias de Conteúdo</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-box">
            <h2>100+</h2>
            <p>Exemplos de Código</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-box">
            <h2>30+</h2>
            <p>Exercícios Práticos</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Roadmap visual
    st.markdown("## 🗺️ Roadmap Completo")
    
    etapas = [
        {
            "numero": "1",
            "nome": "Fundamentos Essenciais",
            "duracao": "7 dias",
            "emoji": "🧠",
            "descricao": "Python avançado, APIs FastAPI, Docker, Git",
            "nivel": "Iniciante"
        },
        {
            "numero": "2",
            "nome": "Machine Learning Clássico",
            "duracao": "7 dias",
            "emoji": "🔢",
            "descricao": "Regressão, Random Forest, XGBoost, MLflow",
            "nivel": "Iniciante/Intermediário"
        },
        {
            "numero": "3",
            "nome": "Deep Learning + PyTorch",
            "duracao": "10 dias",
            "emoji": "🧠",
            "descricao": "Redes neurais, CNN, RNN/LSTM, Transformers",
            "nivel": "Intermediário"
        },
        {
            "numero": "4",
            "nome": "LLMs + Engenharia de Prompt",
            "duracao": "10 dias",
            "emoji": "🤖",
            "descricao": "OpenAI, Claude, Gemini, Ollama, Prompting",
            "nivel": "Intermediário"
        },
        {
            "numero": "5",
            "nome": "RAG",
            "duracao": "Flexível",
            "emoji": "📚",
            "descricao": "Vector databases, Chunking, Hybrid search",
            "nivel": "Intermediário/Avançado"
        },
        {
            "numero": "6",
            "nome": "Agentes de IA",
            "duracao": "Flexível",
            "emoji": "🧱",
            "descricao": "LangChain, LangGraph, LlamaIndex, Multi-agent",
            "nivel": "Avançado"
        },
        {
            "numero": "7",
            "nome": "Deploy e Infraestrutura",
            "duracao": "Flexível",
            "emoji": "🏗️",
            "descricao": "Cloud, Kubernetes, MLOps, Monitoring",
            "nivel": "Avançado"
        },
        {
            "numero": "8",
            "nome": "AI Security",
            "duracao": "Avançado",
            "emoji": "🔒",
            "descricao": "Prompt injection, Jailbreak, Red teaming",
            "nivel": "Expert"
        }
    ]
    
    for i, etapa in enumerate(etapas):
        col1, col2, col3 = st.columns([1, 5, 2])
        
        with col1:
            st.markdown(f"### {etapa['emoji']}")
        
        with col2:
            st.markdown(f"""
            **Etapa {etapa['numero']}: {etapa['nome']}**  
            {etapa['descricao']}
            """)
        
        with col3:
            st.markdown(f"""
            ⏱️ {etapa['duracao']}  
            📊 {etapa['nivel']}
            """)
        
        if i < len(etapas) - 1:
            st.markdown("↓")
    
    st.markdown("---")
    
    # Como usar
    st.markdown("""
    ## 📖 Como Usar Este Sistema
    
    1. **Navegue** pelo menu lateral para escolher uma etapa
    2. **Estude** o conteúdo teórico e exemplos de código
    3. **Execute** os exemplos interativos
    4. **Pratique** com os exercícios propostos
    5. **Complete** as checklists antes de avançar
    6. **Documente** seu progresso e projetos
    
    ### 💡 Dicas para Maximizar seu Aprendizado:
    
    - ✍️ **Pratique ativamente**: Digite o código, não apenas copie
    - 🔄 **Revise regularmente**: Volte aos conceitos anteriores
    - 🚀 **Construa projetos**: Aplique o conhecimento em projetos reais
    - 👥 **Compartilhe**: Ensine outros para solidificar seu conhecimento
    - 📚 **Aprofunde**: Use as referências para estudar mais
    
    ### 🎓 Certificação e Portfólio:
    
    Ao completar cada etapa:
    - ✅ Complete todos os exercícios
    - ✅ Construa um projeto demonstrativo
    - ✅ Documente no GitHub
    - ✅ Adicione ao seu portfólio
    
    ---
    
    ## 🚀 Comece Agora!
    
    Escolha a **Etapa 1** no menu lateral para começar sua jornada!
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>💻 Desenvolvido com Streamlit | 🤖 Sistema de Aprendizado de IA</p>
        <p>⭐ Se este conteúdo te ajudou, compartilhe com outros!</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Função principal do aplicativo"""
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 Menu de Navegação")
        
        st.markdown("---")
        
        # Seleção de etapa
        etapa_selecionada = st.radio(
            "Escolha uma etapa:",
            [
                "🏠 Início",
                "🧠 Etapa 1: Fundamentos",
                "🔢 Etapa 2: ML Clássico",
                "🧠 Etapa 3: Deep Learning",
                "🤖 Etapa 4: LLMs",
                "📚 Etapa 5: RAG",
                "🧱 Etapa 6: Agentes",
                "🏗️ Etapa 7: Deploy",
                "🔒 Etapa 8: Security"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Progresso
        st.markdown("### 📊 Seu Progresso")
        
        # Simulação de progresso (pode ser conectado a um banco de dados)
        progresso_total = 0
        st.progress(progresso_total / 100)
        st.caption(f"{progresso_total}% completo")
        
        st.markdown("---")
        
        # Recursos adicionais
        st.markdown("### 📚 Recursos")
        st.markdown("""
        - [📖 Documentação](https://docs.python.org)
        - [💻 GitHub](https://github.com)
        - [🎓 Kaggle](https://kaggle.com)
        - [📝 Papers](https://arxiv.org)
        """)
        
        st.markdown("---")
        
        # Info
        st.info("""
        💡 **Dica:**  
        Complete cada etapa antes de avançar para garantir uma base sólida!
        """)
    
    # Conteúdo principal
    if etapa_selecionada == "🏠 Início":
        render_home()
    elif "Etapa 1" in etapa_selecionada:
        render_etapa1()
    elif "Etapa 2" in etapa_selecionada:
        render_etapa2()
    elif "Etapa 3" in etapa_selecionada:
        render_etapa3()
    elif "Etapa 4" in etapa_selecionada:
        render_etapa4()
    elif "Etapa 5" in etapa_selecionada:
        render_etapa5()
    elif "Etapa 6" in etapa_selecionada:
        render_etapa6()
    elif "Etapa 7" in etapa_selecionada:
        render_etapa7()
    elif "Etapa 8" in etapa_selecionada:
        render_etapa8()


if __name__ == "__main__":
    main()

