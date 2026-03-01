<div align="center">

# SA-IA &mdash; Sistema de Aprendizado de IA

**Plataforma educacional interativa para formacao completa em Engenharia de IA**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker&logoColor=white)](https://docker.com)
[![License MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Health Check](https://img.shields.io/badge/Healthcheck-Enabled-22c55e)]()

</div>

---

## Indice

- [Visao Geral](#visao-geral)
- [Arquitetura](#arquitetura)
- [Roadmap de 8 Etapas](#roadmap-de-8-etapas)
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

O **SA-IA** e uma plataforma educacional completa que guia desenvolvedores atraves de 8 etapas progressivas para se tornarem Engenheiros de IA. Cobre desde fundamentos de Python ate topicos avancados como AI Security e Red Teaming.

### Numeros

| Metrica | Valor |
|---------|-------|
| Etapas completas | **8** |
| Dias de conteudo | **50+** |
| Exemplos de codigo | **100+** |
| Exercicios praticos | **30+** |
| Linhas de codigo educacional | **5.000+** |

---

## Arquitetura

```
+-----------------------------------------------------+
|                   Browser (8510)                     |
+-----------------------------------------------------+
                         |
+-----------------------------------------------------+
|              Streamlit App (8501)                     |
|  +-------+  +-------+  +-------+  +-------+         |
|  |Etapa 1|  |Etapa 2|  |  ...  |  |Etapa 8|         |
|  +-------+  +-------+  +-------+  +-------+         |
|                                                       |
|  [ Healthcheck ] [ Config TOML ] [ Favicon SVG ]     |
+-----------------------------------------------------+
|              Python 3.11-slim (Docker)                |
+-----------------------------------------------------+
```

**Decisoes arquiteturais:**

- **Stateless** &mdash; Nao necessita banco de dados. O conteudo educacional e renderizado em memoria via modulos Python
- **Multi-stage Docker build** &mdash; Stage `builder` com gcc para compilacao, stage `runtime` minimo (~1.1GB)
- **Requirements separados** &mdash; `requirements-docker.txt` exclui PyTorch/Transformers (~3.5GB) pois os exemplos sao exibidos via `st.code()`, nao executados
- **Healthcheck nativo** &mdash; Script Python verifica endpoint `/_stcore/health` do Streamlit
- **Seguranca** &mdash; Usuario nao-root, `no-new-privileges`, `read_only` rootfs

---

## Roadmap de 8 Etapas

| # | Etapa | Duracao | Nivel | Topicos Principais |
|---|-------|---------|-------|--------------------|
| 1 | **Fundamentos Essenciais** | 7 dias | Iniciante | Python avancado, FastAPI, Docker, Git, Testes |
| 2 | **Machine Learning Classico** | 7 dias | Iniciante/Inter. | Regressao, Random Forest, XGBoost, MLflow |
| 3 | **Deep Learning + PyTorch** | 10 dias | Intermediario | CNN, RNN/LSTM, Transformers, Transfer Learning |
| 4 | **LLMs + Prompt Engineering** | 10 dias | Intermediario | OpenAI, Claude, Gemini, Ollama, CoT, ReAct |
| 5 | **RAG** | Flexivel | Inter./Avancado | Vector DBs, Chunking, Re-ranking, Hybrid Search |
| 6 | **Agentes de IA** | Flexivel | Avancado | LangChain, LangGraph, LlamaIndex, Multi-Agent |
| 7 | **Deploy e Infraestrutura** | Flexivel | Avancado | Cloud GPU, Kubernetes, MLOps, Monitoring |
| 8 | **AI Security** | Flexivel | Expert | Prompt Injection, Jailbreak, LlamaGuard, Red Team |

---

## Stack Tecnologico

| Camada | Tecnologias |
|--------|-------------|
| **Frontend/UI** | Streamlit |
| **Machine Learning** | scikit-learn, XGBoost, MLflow |
| **Deep Learning** | PyTorch, Transformers (HuggingFace) |
| **LLMs** | OpenAI, Anthropic Claude, Google Gemini, Ollama |
| **RAG** | ChromaDB, Pinecone, LangChain, LlamaIndex |
| **Agentes** | LangGraph, AutoGen |
| **MLOps** | MLflow, DVC, Evidently |
| **Deploy** | Docker, Kubernetes, FastAPI |
| **Monitoring** | Prometheus |
| **Containerizacao** | Docker multi-stage, Docker Compose |

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

# 2. Crie um ambiente virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Instale as dependencias
pip install -r requirements.txt

# 4. Configure variaveis de ambiente (opcional)
cp env_example.txt .env
# Edite o .env com suas API keys

# 5. Execute a aplicacao
streamlit run app.py
```

A aplicacao estara disponivel em `http://localhost:8501`.

---

## Deploy com Docker

### Build e execucao (recomendado)

```bash
# Subir com Docker Compose
docker compose up -d

# Verificar status
docker ps --filter "name=sa-ia-learning"

# Ver logs
docker logs -f sa-ia-learning

# Parar
docker compose down
```

A aplicacao estara disponivel em **`http://localhost:8510`**.

### Portas utilizadas

| Servico | Porta Host | Porta Container |
|---------|-----------|----------------|
| Streamlit App | **8510** | 8501 |

> As portas foram escolhidas para evitar conflito com outros servicos locais.

### Variaveis de ambiente

Crie um arquivo `.env` na raiz do projeto (nunca commite este arquivo):

```env
# Obrigatorias apenas se for usar demos interativas com LLMs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
PINECONE_API_KEY=...

# Opcionais
APP_ENV=production
LOG_LEVEL=INFO
```

### Comandos uteis

```bash
# Rebuild apos alteracoes
docker compose up -d --build

# Ver consumo de recursos
docker stats sa-ia-learning

# Verificar healthcheck
curl http://localhost:8510/_stcore/health

# Acessar shell do container
docker exec -it sa-ia-learning /bin/bash
```

---

## Estrutura do Projeto

```
sa_ia/
├── app.py                          # Aplicacao principal Streamlit
├── healthcheck.py                  # Script de healthcheck Docker
├── requirements.txt                # Dependencias completas (dev local)
├── requirements-docker.txt         # Dependencias otimizadas (Docker)
├── Dockerfile                      # Multi-stage build
├── docker-compose.yml              # Orquestracao de containers
├── .dockerignore                   # Exclusoes do build Docker
├── .gitignore                      # Exclusoes do Git
├── env_example.txt                 # Template de variaveis de ambiente
├── assets/
│   └── favicon.svg                 # Favicon da aplicacao
├── .streamlit/
│   └── config.toml                 # Configuracao Streamlit (producao)
└── modules/
    ├── __init__.py
    ├── etapa1/
    │   ├── __init__.py
    │   └── fundamentos.py          # Python, FastAPI, Docker, Git
    ├── etapa2/
    │   ├── __init__.py
    │   └── ml_classico.py          # Regressao, Arvores, XGBoost, MLflow
    ├── etapa3/
    │   ├── __init__.py
    │   └── deep_learning.py        # Tensores, CNN, RNN, Transformers
    ├── etapa4/
    │   ├── __init__.py
    │   └── llms_prompts.py         # APIs LLM, Ollama, Prompting
    ├── etapa5/
    │   ├── __init__.py
    │   └── rag.py                  # Vector DBs, Chunking, Hybrid Search
    ├── etapa6/
    │   ├── __init__.py
    │   └── agentes.py              # LangChain, LangGraph, Multi-Agent
    ├── etapa7/
    │   ├── __init__.py
    │   └── deploy.py               # Cloud, Kubernetes, MLOps
    └── etapa8/
        ├── __init__.py
        └── security.py             # Prompt Injection, Red Teaming
```

---

## Configuracao

### Streamlit (`.streamlit/config.toml`)

| Parametro | Valor | Descricao |
|-----------|-------|-----------|
| `server.port` | 8501 | Porta interna do Streamlit |
| `server.headless` | true | Modo sem browser (Docker) |
| `theme.base` | dark | Tema escuro padrao |
| `theme.primaryColor` | #667eea | Cor primaria (gradiente roxo) |
| `runner.fastReruns` | true | Re-execucao otimizada |

### Docker Compose

| Parametro | Valor | Descricao |
|-----------|-------|-----------|
| `memory limit` | 1GB | Limite maximo de RAM |
| `cpu limit` | 1.0 | Limite de CPU |
| `read_only` | true | Filesystem somente leitura |
| `no-new-privileges` | true | Previne escalacao de privilegios |
| `healthcheck interval` | 30s | Frequencia de verificacao |

---

## Observabilidade

### Healthcheck

O container possui healthcheck integrado que verifica o endpoint nativo do Streamlit:

```
GET /_stcore/health → 200 OK
```

Configuracao:
- **Intervalo:** 30s
- **Timeout:** 10s
- **Retries:** 3
- **Start period:** 20s

### Logs

Logs estruturados em JSON com rotacao automatica:

```bash
# Ver logs em tempo real
docker logs -f sa-ia-learning

# Configuracao de rotacao
# max-size: 10MB por arquivo
# max-file: 3 arquivos
```

---

## Seguranca

| Controle | Implementacao |
|----------|--------------|
| **Usuario nao-root** | Container executa como `appuser` |
| **Read-only filesystem** | `read_only: true` no compose |
| **No-new-privileges** | Previne escalacao via `security_opt` |
| **Sem secrets hardcoded** | API keys via `.env` e variaveis de ambiente |
| **tmpfs** | Escritas temporarias em `/tmp` (100MB max) |
| **Resource limits** | CPU e memoria limitados |
| **Log rotation** | Previne disk exhaustion |
| **XSRF protection** | Configuravel via Streamlit |

---

## Contribuindo

1. Fork o repositorio
2. Crie uma branch: `git checkout -b feature/nova-etapa`
3. Commit suas mudancas: `git commit -m 'feat: adiciona nova etapa'`
4. Push: `git push origin feature/nova-etapa`
5. Abra um Pull Request

### Convencoes

- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`)
- **Branch naming:** `feature/`, `fix/`, `docs/`
- **Codigo:** PEP 8, type hints quando aplicavel
- **Testes:** pytest para novos modulos

---

## Melhorias Futuras

- [ ] **Nginx reverse proxy** &mdash; TLS/SSL termination e caching de assets estaticos
- [ ] **Autenticacao** &mdash; OAuth2 Proxy ou Streamlit Authenticator para controle de acesso
- [ ] **CI/CD** &mdash; GitHub Actions com build, test, push para registry (ECR/GCR)
- [ ] **Kubernetes** &mdash; Helm chart com HPA baseado em conexoes WebSocket
- [ ] **Persistencia** &mdash; PostgreSQL para salvar progresso e checklists dos alunos
- [ ] **Metricas Prometheus** &mdash; Custom exporter com metricas de uso por etapa
- [ ] **CDN** &mdash; Assets estaticos via CloudFront/Cloud CDN
- [ ] **Testes E2E** &mdash; Playwright/Selenium para fluxos criticos
- [ ] **i18n** &mdash; Internacionalizacao (EN/PT-BR)
- [ ] **PWA** &mdash; Progressive Web App para acesso offline

---

## Licenca

Este projeto esta sob a licenca [MIT](LICENSE).

---

<div align="center">

**SA-IA** &mdash; Sistema de Aprendizado de IA

Desenvolvido com Streamlit | Containerizado com Docker

</div>
