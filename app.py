"""
Sistema de Aprendizado de IA
Roadmap completo de 8 etapas para se tornar Engenheiro de IA
"""
import pathlib
import streamlit as st
from modules.etapa1 import render_etapa1
from modules.etapa2 import render_etapa2
from modules.etapa3 import render_etapa3
from modules.etapa4 import render_etapa4
from modules.etapa5 import render_etapa5
from modules.etapa6 import render_etapa6
from modules.etapa7 import render_etapa7
from modules.etapa8 import render_etapa8
from landing import render_landing

# Favicon
_FAVICON = pathlib.Path(__file__).parent / "assets" / "favicon.svg"
_ICON = str(_FAVICON) if _FAVICON.exists() else "🤖"

# Gate: se o usuario ja se inscreveu, mostra a plataforma; senao, landing page
_enrolled = st.session_state.get("enrolled", False)

# Configuração da página
st.set_page_config(
    page_title="SA-IA Academy | Formacao em Engenharia de IA",
    page_icon=_ICON,
    layout="wide",
    initial_sidebar_state="expanded" if _enrolled else "collapsed",
)

# =====================================================================
# CSS da plataforma (carregado apenas quando enrolled)
# =====================================================================
PLATFORM_CSS = """
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
        color: #a0a0b0;
        margin-bottom: 2rem;
    }

    .stat-card {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 12px;
        padding: 1.5rem 1rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
    }

    .stat-number {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }

    .stat-label {
        font-size: 0.9rem;
        color: #a0a0b0;
        margin: 0.3rem 0 0 0;
        font-weight: 500;
    }

    .roadmap-container {
        display: flex;
        flex-direction: column;
        gap: 0;
        margin: 0.5rem 0 1.5rem 0;
    }

    .roadmap-item {
        display: flex;
        align-items: stretch;
        min-height: 56px;
    }

    .roadmap-left {
        width: 48px;
        display: flex;
        flex-direction: column;
        align-items: center;
        flex-shrink: 0;
    }

    .roadmap-dot {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
    }

    .roadmap-line {
        width: 2px;
        flex: 1;
        background: linear-gradient(180deg, #667eea88, #764ba288);
    }

    .roadmap-body {
        flex: 1;
        background: linear-gradient(135deg, #667eea10, #764ba210);
        border: 1px solid #667eea30;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin: 0 0 6px 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }

    .roadmap-body:hover {
        border-color: #667eea88;
        box-shadow: 0 2px 12px rgba(102, 126, 234, 0.15);
    }

    .roadmap-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #e0e0f0;
        margin: 0;
    }

    .roadmap-desc {
        font-size: 0.8rem;
        color: #9090a0;
        margin: 2px 0 0 0;
    }

    .roadmap-meta {
        text-align: right;
        flex-shrink: 0;
        white-space: nowrap;
    }

    .roadmap-dur {
        font-size: 0.78rem;
        color: #a0a0b0;
        margin: 0;
    }

    .roadmap-nivel {
        font-size: 0.72rem;
        color: #667eea;
        font-weight: 600;
        margin: 2px 0 0 0;
    }
</style>
"""


def render_home():
    """Renderiza a página inicial (pos-inscricao)."""

    user_name = st.session_state.get("user_name", "")
    greeting = f", {user_name.split()[0]}" if user_name else ""

    st.markdown(
        f'<h1 class="main-header">SA-IA Academy</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="subtitle">Bem-vindo(a){greeting}! Sua jornada para se tornar Engenheiro(a) de IA comeca aqui.</p>',
        unsafe_allow_html=True,
    )

    # Estatisticas do programa
    st.markdown("### Estatisticas do Programa")

    stats = [
        ("8", "Etapas Completas"),
        ("50+", "Dias de Conteudo"),
        ("100+", "Exemplos de Codigo"),
        ("30+", "Exercicios Praticos"),
    ]

    cols = st.columns(4)
    for col, (number, label) in zip(cols, stats):
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<p class="stat-number">{number}</p>'
                f'<p class="stat-label">{label}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Roadmap visual
    st.markdown("## Roadmap Completo")

    etapas = [
        {"numero": "1", "nome": "Fundamentos Essenciais", "duracao": "7 dias",
         "emoji": "🧠", "descricao": "Python avancado, APIs FastAPI, Docker, Git", "nivel": "Iniciante"},
        {"numero": "2", "nome": "Machine Learning Classico", "duracao": "7 dias",
         "emoji": "🔢", "descricao": "Regressao, Random Forest, XGBoost, MLflow", "nivel": "Iniciante/Intermediario"},
        {"numero": "3", "nome": "Deep Learning + PyTorch", "duracao": "10 dias",
         "emoji": "🧠", "descricao": "Redes neurais, CNN, RNN/LSTM, Transformers", "nivel": "Intermediario"},
        {"numero": "4", "nome": "LLMs + Engenharia de Prompt", "duracao": "10 dias",
         "emoji": "🤖", "descricao": "OpenAI, Claude, Gemini, Ollama, Prompting", "nivel": "Intermediario"},
        {"numero": "5", "nome": "RAG", "duracao": "Flexivel",
         "emoji": "📚", "descricao": "Vector databases, Chunking, Hybrid search", "nivel": "Intermediario/Avancado"},
        {"numero": "6", "nome": "Agentes de IA", "duracao": "Flexivel",
         "emoji": "🧱", "descricao": "LangChain, LangGraph, LlamaIndex, Multi-agent", "nivel": "Avancado"},
        {"numero": "7", "nome": "Deploy e Infraestrutura", "duracao": "Flexivel",
         "emoji": "🏗️", "descricao": "Cloud, Kubernetes, MLOps, Monitoring", "nivel": "Avancado"},
        {"numero": "8", "nome": "AI Security", "duracao": "Avancado",
         "emoji": "🔒", "descricao": "Prompt injection, Jailbreak, Red teaming", "nivel": "Expert"},
    ]

    items_html = ""
    for i, etapa in enumerate(etapas):
        show_line = "block" if i < len(etapas) - 1 else "none"
        items_html += (
            f'<div class="roadmap-item">'
            f'  <div class="roadmap-left">'
            f'    <div class="roadmap-dot">{etapa["emoji"]}</div>'
            f'    <div class="roadmap-line" style="display:{show_line}"></div>'
            f'  </div>'
            f'  <div class="roadmap-body">'
            f'    <div>'
            f'      <p class="roadmap-title">Etapa {etapa["numero"]}: {etapa["nome"]}</p>'
            f'      <p class="roadmap-desc">{etapa["descricao"]}</p>'
            f'    </div>'
            f'    <div class="roadmap-meta">'
            f'      <p class="roadmap-dur">⏱️ {etapa["duracao"]}</p>'
            f'      <p class="roadmap-nivel">{etapa["nivel"]}</p>'
            f'    </div>'
            f'  </div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="roadmap-container">{items_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Como usar
    st.markdown("""
    ## Como Usar Este Sistema

    1. **Navegue** pelo menu lateral para escolher uma etapa
    2. **Estude** o conteudo teorico e exemplos de codigo
    3. **Execute** os exemplos interativos
    4. **Pratique** com os exercicios propostos
    5. **Complete** as checklists antes de avancar
    6. **Documente** seu progresso e projetos

    ### Dicas para Maximizar seu Aprendizado

    - **Pratique ativamente**: Digite o codigo, nao apenas copie
    - **Revise regularmente**: Volte aos conceitos anteriores
    - **Construa projetos**: Aplique o conhecimento em projetos reais
    - **Compartilhe**: Ensine outros para solidificar seu conhecimento
    - **Aprofunde**: Use as referencias para estudar mais

    ---

    Escolha a **Etapa 1** no menu lateral para comecar!
    """)


def main():
    """Funcao principal do aplicativo."""

    # ── Gate: Landing page se nao inscrito ────────────────────────────
    if not st.session_state.get("enrolled", False):
        render_landing()
        return

    # ── Plataforma (inscrito) ─────────────────────────────────────────
    st.markdown(PLATFORM_CSS, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        user_name = st.session_state.get("user_name", "Aluno")
        st.markdown(f"**{user_name}**")
        st.caption(st.session_state.get("user_email", ""))
        st.markdown("---")

        etapa_selecionada = st.radio(
            "Escolha uma etapa:",
            [
                "🏠 Inicio",
                "🧠 Etapa 1: Fundamentos",
                "🔢 Etapa 2: ML Classico",
                "🧠 Etapa 3: Deep Learning",
                "🤖 Etapa 4: LLMs",
                "📚 Etapa 5: RAG",
                "🧱 Etapa 6: Agentes",
                "🏗️ Etapa 7: Deploy",
                "🔒 Etapa 8: Security",
            ],
            index=0,
        )

        st.markdown("---")

        # Progresso
        st.markdown("### Seu Progresso")
        progresso_total = 0
        st.progress(progresso_total / 100)
        st.caption(f"{progresso_total}% completo")

        st.markdown("---")

        st.markdown("### Recursos")
        st.markdown("""
        - [Documentacao Python](https://docs.python.org)
        - [GitHub](https://github.com)
        - [Kaggle](https://kaggle.com)
        - [arXiv Papers](https://arxiv.org)
        """)

        st.markdown("---")

        if st.button("Sair da plataforma", use_container_width=True):
            for key in ["enrolled", "user_name", "user_email", "user_role"]:
                st.session_state.pop(key, None)
            st.rerun()

    # Conteudo principal
    if etapa_selecionada == "🏠 Inicio":
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
