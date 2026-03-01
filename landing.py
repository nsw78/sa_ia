"""
Landing Page - SA-IA Academy
Pagina institucional estilo academico (MIT-inspired).
"""
import streamlit as st


LANDING_CSS = """
<style>
/* ===== Global reset ===== */
.lw *{box-sizing:border-box;margin:0;padding:0;}
.lw{max-width:880px;margin:0 auto;padding:0 1rem;}

/* ===== Top bar ===== */
.top-bar{
    display:flex;justify-content:space-between;align-items:center;
    padding:1rem 0;border-bottom:1px solid #ffffff10;margin-bottom:2rem;
}
.top-logo{font-size:1.05rem;font-weight:800;color:#e0e0f0;letter-spacing:0.3px;}
.top-logo span{
    background:linear-gradient(90deg,#667eea,#764ba2);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.top-tag{
    font-size:0.68rem;color:#667eea;border:1px solid #667eea44;
    border-radius:12px;padding:0.2rem 0.75rem;font-weight:600;
}

/* ===== Hero ===== */
.hero-s{text-align:center;padding:2rem 0 2.2rem;}
.hero-s h1{
    font-size:2.5rem;font-weight:800;line-height:1.18;
    color:#f0f0ff;margin-bottom:0.9rem;
}
.hero-s h1 em{
    font-style:normal;
    background:linear-gradient(90deg,#667eea,#a78bfa,#764ba2);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.hero-p{
    font-size:1.02rem;color:#9898b0;line-height:1.7;
    max-width:620px;margin:0 auto;
}

/* ===== Metrics ===== */
.met-row{
    display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;
    padding:1.3rem 0;
    border-top:1px solid #ffffff08;border-bottom:1px solid #ffffff08;
    margin:1.5rem 0 2rem;
}
.met{text-align:center;}
.met-v{font-size:1.45rem;font-weight:800;color:#e0e0f0;}
.met-l{font-size:0.7rem;color:#6a6a80;text-transform:uppercase;letter-spacing:1px;margin-top:1px;}

/* ===== Section ===== */
.sec{margin-bottom:2.2rem;}
.sec-tag{
    font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;
    color:#667eea;font-weight:700;margin-bottom:0.4rem;
}
.sec-h{font-size:1.5rem;font-weight:800;color:#e0e0f0;margin-bottom:0.35rem;}
.sec-p{font-size:0.9rem;color:#8585a0;line-height:1.6;margin-bottom:1rem;}

/* ===== Standards ===== */
.stds{display:flex;gap:0.6rem;flex-wrap:wrap;}
.std{
    font-size:0.7rem;color:#9595aa;background:#ffffff06;
    border:1px solid #ffffff0c;border-radius:6px;
    padding:0.3rem 0.7rem;font-weight:500;
}

/* ===== Orgs ===== */
.orgs{display:flex;gap:0.8rem;flex-wrap:wrap;}
.org{
    font-size:0.8rem;font-weight:600;color:#6a6a80;
    background:#ffffff05;border:1px solid #ffffff0a;
    border-radius:8px;padding:0.4rem 1rem;
}

/* ===== Curriculum ===== */
.cur{display:grid;grid-template-columns:repeat(2,1fr);gap:0.7rem;}
.cc{
    background:#ffffff04;border:1px solid #ffffff0c;border-radius:10px;
    padding:0.9rem 1rem;display:flex;gap:0.7rem;align-items:flex-start;
    transition:border-color 0.2s;
}
.cc:hover{border-color:#667eea44;}
.cc-n{
    width:26px;height:26px;border-radius:50%;flex-shrink:0;
    background:linear-gradient(135deg,#667eea,#764ba2);
    display:flex;align-items:center;justify-content:center;
    font-size:0.68rem;font-weight:800;color:#fff;
}
.cc-t{font-size:0.85rem;font-weight:700;color:#d0d0e0;margin-bottom:2px;}
.cc-d{font-size:0.74rem;color:#7878908;line-height:1.4;}
.cc-m{display:flex;gap:0.5rem;margin-top:4px;}
.cc-tag{
    font-size:0.62rem;color:#667eea;background:#667eea12;
    border-radius:3px;padding:1px 5px;font-weight:600;
}

/* ===== Testimonials ===== */
.tg{display:grid;grid-template-columns:repeat(2,1fr);gap:0.7rem;}
.tc{
    background:#ffffff04;border:1px solid #ffffff0c;
    border-radius:10px;padding:0.9rem 1rem;
}
.tc-t{font-size:0.82rem;color:#a8a8c0;line-height:1.55;font-style:italic;margin-bottom:0.6rem;}
.tc-w{display:flex;align-items:center;gap:0.5rem;}
.tc-a{
    width:30px;height:30px;border-radius:50%;flex-shrink:0;
    background:linear-gradient(135deg,#667eea,#764ba2);
    display:flex;align-items:center;justify-content:center;
    font-size:0.7rem;font-weight:700;color:#fff;
}
.tc-nm{font-size:0.78rem;font-weight:600;color:#d0d0e0;}
.tc-rl{font-size:0.68rem;color:#6a6a80;}

/* ===== Pricing ===== */
.pb{
    max-width:400px;margin:0 auto;
    background:linear-gradient(135deg,#667eea08,#764ba208);
    border:1px solid #667eea30;border-radius:14px;
    padding:1.8rem;text-align:center;
}
.pb-name{font-size:1.05rem;font-weight:700;color:#e0e0f0;margin-bottom:0.5rem;}
.pb-old{font-size:0.85rem;color:#555;text-decoration:line-through;}
.pb-now{
    font-size:2.5rem;font-weight:900;line-height:1.1;margin:0.2rem 0;
    background:linear-gradient(90deg,#667eea,#764ba2);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.pb-sub{font-size:0.75rem;color:#8585a0;margin-bottom:0.9rem;}
.pb-list{text-align:left;list-style:none;padding:0;margin:0;}
.pb-list li{
    font-size:0.8rem;color:#a8a8c0;padding:0.22rem 0;
    border-bottom:1px solid #ffffff06;
}
.pb-list li::before{content:"\\2713  ";color:#667eea;font-weight:700;}

/* ===== FAQ ===== */
.fq{
    background:#ffffff04;border:1px solid #ffffff0c;border-radius:8px;
    padding:0.8rem 1rem;margin-bottom:0.45rem;
}
.fq-q{font-weight:700;font-size:0.84rem;color:#d0d0e0;margin-bottom:0.25rem;}
.fq-a{font-size:0.78rem;color:#8585a0;line-height:1.5;}

/* ===== Footer ===== */
.lf{text-align:center;padding:1.2rem 0 0.5rem;border-top:1px solid #ffffff08;margin-top:0.8rem;}
.lf p{font-size:0.7rem;color:#484860;margin:0.12rem 0;}

@media(max-width:700px){
    .met-row{grid-template-columns:repeat(2,1fr);}
    .cur,.tg{grid-template-columns:1fr;}
    .hero-s h1{font-size:1.7rem;}
}
</style>
"""


def render_landing():
    """Renderiza a landing page."""

    st.markdown(LANDING_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="lw">

    <div class="top-bar">
        <div class="top-logo"><span>SA-IA</span> Academy</div>
        <div class="top-tag">Turma 2026</div>
    </div>

    <div class="hero-s">
        <h1>Formacao completa em<br><em>Engenharia de Inteligencia Artificial</em></h1>
        <p class="hero-p">
            Programa de 8 modulos com padrao internacional, projetado para levar
            profissionais do nivel iniciante ao avancado em Engenharia de IA.
            Conteudo 2026 alinhado com praticas de Google, AWS e Microsoft.
        </p>
    </div>

    <div class="met-row">
        <div class="met"><div class="met-v">2.847</div><div class="met-l">Alunos</div></div>
        <div class="met"><div class="met-v">4.9/5</div><div class="met-l">Avaliacao</div></div>
        <div class="met"><div class="met-v">93%</div><div class="met-l">Conclusao</div></div>
        <div class="met"><div class="met-v">180+</div><div class="met-l">Empresas</div></div>
    </div>

    <div class="sec">
        <div class="sec-tag">Padroes internacionais</div>
        <div class="sec-h">Curriculo alinhado com frameworks globais</div>
        <div class="sec-p">O programa segue referenciais reconhecidos pela industria e academia global.</div>
        <div class="stds">
            <div class="std">ISO/IEC 23053 &mdash; AI Framework</div>
            <div class="std">EU AI Act Compliance</div>
            <div class="std">OWASP AI Security Top 10</div>
            <div class="std">MLOps Maturity Level 3</div>
            <div class="std">NIST AI RMF</div>
        </div>
    </div>

    <div class="sec">
        <div class="sec-tag">Reconhecimento</div>
        <div class="sec-h">Profissionais dessas organizacoes ja treinaram conosco</div>
        <div class="sec-p">Nossos alunos atuam em tech, financas, saude e startups de alto crescimento.</div>
        <div class="orgs">
            <div class="org">Google</div>
            <div class="org">AWS</div>
            <div class="org">Microsoft</div>
            <div class="org">Nubank</div>
            <div class="org">iFood</div>
            <div class="org">Itau</div>
            <div class="org">B3</div>
            <div class="org">TOTVS</div>
            <div class="org">Stone</div>
            <div class="org">C6 Bank</div>
        </div>
    </div>

    <div class="sec">
        <div class="sec-tag">Curriculo</div>
        <div class="sec-h">8 modulos progressivos, do zero ao avancado</div>
        <div class="sec-p">50+ dias de conteudo, 100+ exemplos de codigo e 30+ exercicios praticos.</div>
        <div class="cur">
            <div class="cc"><div class="cc-n">1</div><div><div class="cc-t">Fundamentos Essenciais</div><div class="cc-d">Python avancado, FastAPI, Docker, Git e testes</div><div class="cc-m"><span class="cc-tag">7 dias</span><span class="cc-tag">Iniciante</span></div></div></div>
            <div class="cc"><div class="cc-n">2</div><div><div class="cc-t">Machine Learning Classico</div><div class="cc-d">Regressao, Random Forest, XGBoost, MLflow</div><div class="cc-m"><span class="cc-tag">7 dias</span><span class="cc-tag">Intermediario</span></div></div></div>
            <div class="cc"><div class="cc-n">3</div><div><div class="cc-t">Deep Learning + PyTorch</div><div class="cc-d">CNN, RNN/LSTM, Transformers, Transfer Learning</div><div class="cc-m"><span class="cc-tag">10 dias</span><span class="cc-tag">Intermediario</span></div></div></div>
            <div class="cc"><div class="cc-n">4</div><div><div class="cc-t">LLMs + Prompt Engineering</div><div class="cc-d">GPT-5, Claude 4, Gemini 2.5, Ollama, CoT, ReAct</div><div class="cc-m"><span class="cc-tag">10 dias</span><span class="cc-tag">Intermediario</span></div></div></div>
            <div class="cc"><div class="cc-n">5</div><div><div class="cc-t">RAG Enterprise</div><div class="cc-d">Vector DBs, chunking, re-ranking, hybrid search</div><div class="cc-m"><span class="cc-tag">Flexivel</span><span class="cc-tag">Avancado</span></div></div></div>
            <div class="cc"><div class="cc-n">6</div><div><div class="cc-t">Agentes de IA</div><div class="cc-d">LangChain, LangGraph, LlamaIndex, multi-agentes</div><div class="cc-m"><span class="cc-tag">Flexivel</span><span class="cc-tag">Avancado</span></div></div></div>
            <div class="cc"><div class="cc-n">7</div><div><div class="cc-t">Deploy e MLOps</div><div class="cc-d">Cloud GPU, Kubernetes, CI/CD, monitoramento</div><div class="cc-m"><span class="cc-tag">Flexivel</span><span class="cc-tag">Avancado</span></div></div></div>
            <div class="cc"><div class="cc-n">8</div><div><div class="cc-t">AI Security</div><div class="cc-d">Prompt injection, red teaming, LlamaGuard 3</div><div class="cc-m"><span class="cc-tag">Flexivel</span><span class="cc-tag">Expert</span></div></div></div>
        </div>
    </div>

    <div class="sec">
        <div class="sec-tag">Depoimentos</div>
        <div class="sec-h">O que nossos alunos dizem</div>
        <div class="sec-p">Resultados reais de profissionais que aceleraram suas carreiras.</div>
        <div class="tg">
            <div class="tc">
                <div class="tc-t">"Em 3 meses sai de analista de dados para ML Engineer. Curriculo absurdamente pratico e enterprise-ready."</div>
                <div class="tc-w"><div class="tc-a">RM</div><div><div class="tc-nm">Rafael Mendes</div><div class="tc-rl">ML Engineer @ Google</div></div></div>
            </div>
            <div class="tc">
                <div class="tc-t">"O modulo de AI Security me diferenciou completamente. Fui promovida a Tech Lead de IA."</div>
                <div class="tc-w"><div class="tc-a">CS</div><div><div class="tc-nm">Camila Santos</div><div class="tc-rl">AI Tech Lead @ Itau</div></div></div>
            </div>
            <div class="tc">
                <div class="tc-t">"Unico curso que cobre o ciclo completo: do modelo ao deploy em Kubernetes com observabilidade."</div>
                <div class="tc-w"><div class="tc-a">LP</div><div><div class="tc-nm">Lucas Pereira</div><div class="tc-rl">AI Architect @ AWS</div></div></div>
            </div>
            <div class="tc">
                <div class="tc-t">"Implementamos RAG enterprise e agentes autonomos seguindo os padroes ensinados. Levantamos Serie A."</div>
                <div class="tc-w"><div class="tc-a">AT</div><div><div class="tc-nm">Ana Torres</div><div class="tc-rl">CTO @ NeuralOps</div></div></div>
            </div>
        </div>
    </div>

    <div class="sec">
        <div class="sec-tag">Investimento</div>
        <div class="sec-h">Acesso vitalicio por taxa simbolica</div>
        <div class="sec-p">Conteudo completo, atualizacoes por 1 ano, certificado e comunidade.</div>
        <div class="pb">
            <div class="pb-name">SA-IA Academy &mdash; Acesso Completo</div>
            <div class="pb-old">De R$ 997,00</div>
            <div class="pb-now">R$ 49,90</div>
            <div class="pb-sub">pagamento unico &bull; acesso vitalicio</div>
            <ul class="pb-list">
                <li>8 modulos completos (50+ dias)</li>
                <li>100+ exemplos de codigo executaveis</li>
                <li>30+ exercicios praticos com solucao</li>
                <li>Certificado de conclusao digital</li>
                <li>Comunidade com 2.800+ profissionais</li>
                <li>Atualizacoes gratuitas por 1 ano</li>
                <li>Modulo exclusivo de AI Security</li>
                <li>Garantia incondicional de 7 dias</li>
            </ul>
        </div>
    </div>

    </div>
    """, unsafe_allow_html=True)

    # ── Formulario (Streamlit nativo) ─────────────────────────────────
    st.markdown("---")
    st.markdown("#### Inscreva-se agora")

    col_f, _, col_i = st.columns([3, 0.3, 2])

    with col_f:
        with st.form("signup_form", clear_on_submit=False):
            nome = st.text_input("Nome completo")
            email = st.text_input("E-mail profissional")
            cargo = st.selectbox("Cargo atual", [
                "Estudante", "Dev Junior", "Dev Pleno", "Dev Senior",
                "Tech Lead", "Analista de Dados", "Data Scientist",
                "ML Engineer", "Gestor / Manager", "Outro",
            ])
            aceite = st.checkbox("Concordo com os Termos de Uso e Politica de Privacidade")
            submitted = st.form_submit_button("GARANTIR MINHA VAGA", use_container_width=True, type="primary")

            if submitted:
                if not nome or not email:
                    st.error("Preencha nome e e-mail.")
                elif "@" not in email:
                    st.error("E-mail invalido.")
                elif not aceite:
                    st.warning("Aceite os Termos de Uso.")
                else:
                    st.session_state["enrolled"] = True
                    st.session_state["user_name"] = nome
                    st.session_state["user_email"] = email
                    st.session_state["user_role"] = cargo
                    st.rerun()

    with col_i:
        st.markdown("**Por que se inscrever?**")
        st.markdown("""
        - Acesso imediato a toda a plataforma
        - Conteudo atualizado para **2026**
        - Padrao internacional (ISO/IEC 23053)
        - Modulo exclusivo de **AI Security**
        - Certificado de conclusao digital
        - Comunidade com **2.800+** profissionais
        """)
        st.markdown("---")
        st.markdown("**Garantia de 7 dias**")
        st.caption("Satisfacao garantida ou dinheiro de volta.")

    # ── FAQ ───────────────────────────────────────────────────────────
    st.markdown("---")

    faq_data = [
        ("Preciso ter experiencia previa com IA?",
         "Nao. O programa comeca do zero e vai ate topicos avancados como AI Security."),
        ("O conteudo esta atualizado para 2026?",
         "Sim. Claude 4, GPT-5, LangGraph, LlamaGuard 3 e padroes ISO/IEC 23053 e EU AI Act."),
        ("Quanto tempo preciso dedicar por dia?",
         "1 a 2 horas diarias. 50+ dias de conteudo, avance no seu ritmo."),
        ("Recebo certificado?",
         "Sim. Certificado digital verificavel ao completar os 8 modulos."),
        ("Qual a politica de reembolso?",
         "Garantia incondicional de 7 dias. 100% do valor de volta."),
    ]

    faq_html = '<div class="lw"><div class="sec"><div class="sec-tag">FAQ</div><div class="sec-h">Perguntas frequentes</div></div>'
    for q, a in faq_data:
        faq_html += f'<div class="fq"><div class="fq-q">{q}</div><div class="fq-a">{a}</div></div>'
    faq_html += '<div class="lf"><p>SA-IA Academy &mdash; Formacao Enterprise em Inteligencia Artificial</p>'
    faq_html += '<p>Termos de Uso &bull; Politica de Privacidade &bull; contato@sa-ia.academy</p>'
    faq_html += '<p>&copy; 2026 SA-IA Academy. Todos os direitos reservados.</p></div></div>'
    st.markdown(faq_html, unsafe_allow_html=True)
