<div align="center">

# SA-IA Academy

**Plataforma enterprise de formacao completa em Engenharia de Inteligencia Artificial**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker&logoColor=white)](https://docker.com)
[![License MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Health Check](https://img.shields.io/badge/Healthcheck-Enabled-22c55e)]()
[![Content](https://img.shields.io/badge/Content-2026-667eea)]()

</div>

---

## Indice

- [Visao Geral](#visao-geral)
- [Arquitetura](#arquitetura)
- [Fluxo da Aplicacao](#fluxo-da-aplicacao)
- [Roadmap de 8 Modulos](#roadmap-de-8-modulos)
- [Stack Tecnologico](#stack-tecnologico)
- [Quick Start](#quick-start)
- [Deploy com Docker](#deploy-com-docker)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Configuracao](#configuracao)
- [Observabilidade](#observabilidade)
- [Seguranca](#seguranca)
- [Contribuindo](#contribuindo)
- [Melhorias Futuras](#melhorias-futuras)
- [Licenca](#licenca)

---

## Visao Geral

O **SA-IA Academy** e uma plataforma educacional completa com landing page de vendas integrada e 8 modulos progressivos para formar Engenheiros de IA. Conteudo atualizado para 2026 com os modelos e frameworks mais recentes.

### Numeros

| Metrica | Valor |
|---------|-------|
| Modulos completos | **8** |
| Dias de conteudo | **50+** |
| Exemplos de codigo | **100+** |
| Exercicios praticos | **30+** |
| Linhas de codigo educacional | **5.000+** |
| Modelos referenciados | GPT-5, Claude 4, Gemini 2.5, Llama 4 |

---

## Arquitetura

```
+----------------------------------------------------------+
|               Browser (http://localhost:8510)              |
+----------------------------------------------------------+
                           |
+----------------------------------------------------------+
|                    Streamlit App (8501)                    |
|                                                           |
|  +--------------+     +------------------------------+   |
|  | Landing Page |---->| Plataforma (pos-inscricao)   |   |
|  | (landing.py) |     |                              |   |
|  | - Hero       |     |  +--------+ +--------+      |   |
|  | - Social     |     |  |Etapa 1 | |Etapa 2 | ...  |   |
|  |   Proof      |     |  +--------+ +--------+      |   |
|  | - Pricing    |     |  +--------+ +--------+      |   |
|  | - Signup     |     |  |  ...   | |Etapa 8 |      |   |
|  +--------------+     |  +--------+ +--------+      |   |
|         |             +------------------------------+   |
|    session_state                                          |
|    (enrolled=True)                                        |
|                                                           |
|  [ Healthcheck ] [ Config TOML ] [ Favicon SVG ]         |
+----------------------------------------------------------+
|                Python 3.11-slim (Docker)                   |
+----------------------------------------------------------+
```

### Decisoes Arquiteturais

- **Landing page como gate** -- Usuarios veem a pagina de vendas primeiro. Apos inscricao via formulario, `session_state["enrolled"]` libera acesso a plataforma completa
- **Stateless** -- Sem banco de dados. Conteudo renderizado em memoria via modulos Python. Inscricao via `session_state` (sessao do browser)
- **Multi-stage Docker build** -- Stage `builder` com gcc, stage `runtime` minimo (~1.1GB)
- **Requirements separados** -- `requirements-docker.txt` exclui PyTorch/Transformers (~3.5GB) pois exemplos sao exibidos em `st.code()`, nao executados
- **Seguranca** -- Usuario nao-root, `no-new-privileges`, `read_only` rootfs, tmpfs

---

## Fluxo da Aplicacao

```
Acesso (localhost:8510)
    |
    v
[enrolled == False?] --Sim--> Landing Page
    |                           |
    |                       Formulario de inscricao
    |                           |
    |                       session_state["enrolled"] = True
    |                           |
    +<--------------------------+
    |
    v
[enrolled == True] --> Plataforma completa
                        |
                        +-- Sidebar com menu
                        +-- 8 etapas de conteudo
                        +-- Progresso
                        +-- Botao "Sair"
```

---

## Roadmap de 8 Modulos

| # | Modulo | Duracao | Nivel | Topicos (atualizado 2026) |
|---|--------|---------|-------|---------------------------|
| 1 | **Fundamentos Essenciais** | 7 dias | Iniciante | Python avancado, FastAPI, Docker, Git, Testes |
| 2 | **Machine Learning Classico** | 7 dias | Iniciante/Inter. | Regressao, Random Forest, XGBoost, MLflow |
| 3 | **Deep Learning + PyTorch** | 10 dias | Intermediario | CNN, RNN/LSTM, Transformers, Transfer Learning |
| 4 | **LLMs + Prompt Engineering** | 10 dias | Intermediario | GPT-5, Claude 4, Gemini 2.5, Ollama, Llama 4, CoT, ReAct |
| 5 | **RAG Enterprise** | Flexivel | Inter./Avancado | Vector DBs, Chunking, Re-ranking, Hybrid Search |
| 6 | **Agentes de IA** | Flexivel | Avancado | LangChain, LangGraph, LlamaIndex, Multi-Agent |
| 7 | **Deploy e MLOps** | Flexivel | Avancado | Cloud GPU, Kubernetes, CI/CD, Monitoring |
| 8 | **AI Security** | Flexivel | Expert | Prompt Injection, Red Teaming, LlamaGuard 3 |

---

## Stack Tecnologico

| Camada | Tecnologias |
|--------|-------------|
| **Frontend/UI** | Streamlit, CSS customizado |
| **Landing Page** | HTML/CSS academico, Streamlit Forms |
| **Machine Learning** | scikit-learn, XGBoost, MLflow |
| **Deep Learning** | PyTorch, Transformers (HuggingFace) |
| **LLMs** | OpenAI (GPT-5), Anthropic (Claude 4), Google (Gemini 2.5), Ollama (Llama 4) |
| **RAG** | ChromaDB, Pinecone, LangChain, LlamaIndex |
| **Agentes** | LangGraph, AutoGen |
| **MLOps** | MLflow, DVC, Evidently |
| **Deploy** | Docker multi-stage, Docker Compose, Kubernetes |
| **Monitoring** | Prometheus |

---

## Quick Start

### Pre-requisitos

- Python 3.11+
- pip
- Docker e Docker Compose (para deploy containerizado)

### Instalacao Local

```bash
# 1. Clone o repositorio
git clone <url-do-repositorio>
cd sa_ia

# 2. Ambiente virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Dependencias
pip install -r requirements.txt

# 4. Variaveis de ambiente (opcional)
cp env_example.txt .env
# Edite .env com suas API keys

# 5. Executar
streamlit run app.py
```

Acesse `http://localhost:8501`. Voce vera a landing page primeiro.

---

## Deploy com Docker

### Build e execucao

```bash
# Subir
docker compose up -d

# Verificar status
docker ps --filter "name=sa-ia-learning"

# Logs
docker logs -f sa-ia-learning

# Parar
docker compose down
```

Acesse **`http://localhost:8510`**.

### Portas utilizadas

| Servico | Porta Host | Porta Container |
|---------|-----------|----------------|
| Streamlit App | **8510** | 8501 |

> Porta 8510 escolhida para evitar conflito com 50+ containers locais.

### Variaveis de ambiente

Crie `.env` na raiz (nunca commite):

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
PINECONE_API_KEY=...
APP_ENV=production
LOG_LEVEL=INFO
```

### Comandos uteis

```bash
# Rebuild
docker compose up -d --build

# Recursos
docker stats sa-ia-learning

# Healthcheck
curl http://localhost:8510/_stcore/health

# Shell
docker exec -it sa-ia-learning /bin/bash
```

---

## Estrutura do Projeto

```
sa_ia/
├── app.py                          # App principal (gate landing -> plataforma)
├── landing.py                      # Landing page de vendas (estilo academico)
├── healthcheck.py                  # Script de healthcheck Docker
├── requirements.txt                # Dependencias completas (dev local)
├── requirements-docker.txt         # Dependencias otimizadas (Docker)
├── Dockerfile                      # Multi-stage build
├── docker-compose.yml              # Orquestracao de containers
├── .dockerignore                   # Exclusoes do build
├── .gitignore                      # Exclusoes do Git
├── env_example.txt                 # Template de variaveis
├── assets/
│   └── favicon.svg                 # Favicon (cerebro neural com gradiente)
├── .streamlit/
│   └── config.toml                 # Configuracao Streamlit (producao)
└── modules/
    ├── __init__.py
    ├── etapa1/                     # Fundamentos (Python, FastAPI, Docker, Git)
    ├── etapa2/                     # ML Classico (sklearn, XGBoost, MLflow)
    ├── etapa3/                     # Deep Learning (PyTorch, CNN, RNN, Transformers)
    ├── etapa4/                     # LLMs (GPT-5, Claude 4, Gemini 2.5, Ollama)
    ├── etapa5/                     # RAG (Vector DBs, Chunking, Hybrid Search)
    ├── etapa6/                     # Agentes (LangChain, LangGraph, Multi-Agent)
    ├── etapa7/                     # Deploy (Cloud, K8s, MLOps, Monitoring)
    └── etapa8/                     # AI Security (Injection, Red Team, LlamaGuard 3)
```

---

## Configuracao

### Streamlit (`.streamlit/config.toml`)

| Parametro | Valor | Descricao |
|-----------|-------|-----------|
| `server.port` | 8501 | Porta interna |
| `server.headless` | true | Modo Docker |
| `theme.base` | dark | Tema escuro |
| `theme.primaryColor` | #667eea | Cor primaria |
| `runner.fastReruns` | true | Re-execucao otimizada |

### Docker Compose

| Parametro | Valor | Descricao |
|-----------|-------|-----------|
| `memory limit` | 1GB | RAM maxima |
| `cpu limit` | 1.0 | CPU maxima |
| `read_only` | true | Filesystem somente leitura |
| `no-new-privileges` | true | Sem escalacao |
| `healthcheck interval` | 30s | Verificacao de saude |

---

## Observabilidade

### Healthcheck

```
GET /_stcore/health -> 200 OK ("ok")
```

- **Intervalo:** 30s
- **Timeout:** 10s
- **Retries:** 3
- **Start period:** 20s

### Logs

```bash
docker logs -f sa-ia-learning
# Rotacao: max 10MB x 3 arquivos
```

---

## Seguranca

| Controle | Implementacao |
|----------|--------------|
| **Usuario nao-root** | Container executa como `appuser` |
| **Read-only filesystem** | `read_only: true` |
| **No-new-privileges** | `security_opt` |
| **Sem secrets hardcoded** | API keys via `.env` |
| **tmpfs** | Escritas em `/tmp` (100MB max) |
| **Resource limits** | CPU e memoria limitados |
| **Log rotation** | Previne disk exhaustion |

---

## Contribuindo

1. Fork o repositorio
2. Crie uma branch: `git checkout -b feature/nova-funcionalidade`
3. Commit: `git commit -m 'feat: descricao'`
4. Push: `git push origin feature/nova-funcionalidade`
5. Abra um Pull Request

### Convencoes

- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`)
- **Branch naming:** `feature/`, `fix/`, `docs/`
- **Codigo:** PEP 8
- **Testes:** pytest

---

## Melhorias Futuras

- [ ] **Nginx reverse proxy** -- TLS/SSL termination
- [ ] **Autenticacao real** -- OAuth2 ou Streamlit Authenticator com banco
- [ ] **Persistencia** -- PostgreSQL para inscricoes e progresso
- [ ] **Payment gateway** -- Stripe/Mercado Pago para cobranca real
- [ ] **CI/CD** -- GitHub Actions com build, test, push para registry
- [ ] **Kubernetes** -- Helm chart com HPA
- [ ] **Metricas** -- Prometheus exporter por modulo
- [ ] **CDN** -- Assets estaticos via CloudFront
- [ ] **Testes E2E** -- Playwright para fluxos criticos
- [ ] **i18n** -- Internacionalizacao EN/PT-BR
- [ ] **Email** -- Confirmacao de inscricao via SMTP/SES

---

## Licenca

Este projeto esta sob a licenca [MIT](LICENSE).

---

<div align="center">

**SA-IA Academy** -- Formacao Enterprise em Inteligencia Artificial

Conteudo 2026 | Streamlit | Docker

</div>
