# 🤖 Sistema de Aprendizado de IA

Um roadmap completo e interativo para se tornar Engenheiro de IA, do zero ao avançado, em 8 etapas estruturadas.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📚 Sobre o Projeto

Este sistema foi desenvolvido para guiar desenvolvedores através de uma jornada completa de aprendizado em Inteligência Artificial, cobrindo desde fundamentos até tópicos avançados como AI Security e MLOps.

### ✨ Características

- 📖 **8 Etapas Completas**: Roadmap estruturado e progressivo
- 💻 **100+ Exemplos de Código**: Código executável e bem documentado
- 🎯 **30+ Exercícios Práticos**: Projetos hands-on para cada etapa
- ⚡ **Interface Interativa**: Aplicação Streamlit moderna e responsiva
- 🔄 **Conteúdo Atualizado**: Tecnologias e práticas mais recentes (2025)

## 🗺️ Roadmap Completo

### 🧠 ETAPA 1 — Fundamentos Essenciais (7 dias)
- Python avançado (decorators, context managers, generators)
- APIs com FastAPI
- Docker para ambientes de IA
- Git e testes automatizados
- **Objetivo**: Virar Dev de IA básico capaz de criar APIs

### 🔢 ETAPA 2 — Machine Learning Clássico (7 dias)
- Regressão Linear e Logística
- Árvores de Decisão e Random Forest
- XGBoost
- Pipelines e Feature Engineering
- MLflow para tracking
- **Objetivo**: Entender ML e treinar modelos próprios

### 🧠 ETAPA 3 — Deep Learning + PyTorch (10 dias)
- Tensores e Autograd
- Redes Neurais Artificiais
- CNN (Visão Computacional)
- RNN/LSTM (Sequências)
- Transformers
- **Objetivo**: Treinar modelos neurais reais

### 🤖 ETAPA 4 — LLMs + Engenharia de Prompt (10 dias)
- OpenAI, Gemini, Claude, Llama
- Modelos locais com Ollama
- Tokenização e Embeddings
- Técnicas de Prompting (Zero-shot, Few-shot, CoT)
- Agentes com ferramentas
- **Objetivo**: Construir chatbots e aplicações avançadas

### 📚 ETAPA 5 — RAG (Retrieval Augmented Generation)
- Vetorização de dados
- Vector Databases (Chroma, Pinecone, Milvus)
- Chunking strategies
- Query transformation e Re-ranking
- Hybrid Search
- **Objetivo**: Criar sistemas empresariais com memória

### 🧱 ETAPA 6 — Agentes de IA (2025)
- LangChain e LangGraph
- LlamaIndex
- Agentes com múltiplas ferramentas
- Planejamento de longo prazo
- Multi-Agent Systems
- **Objetivo**: Criar agentes autônomos complexos

### 🏗️ ETAPA 7 — Deploy e Infraestrutura de IA
- GPUs no GCP/AWS/Azure
- Kubernetes para IA
- CI/CD para modelos
- MLOps end-to-end
- Monitoring e Alertas
- **Objetivo**: Virar AI Platform Engineer

### 🔒 ETAPA 8 — AI Security (Área Premium)
- Firewall de prompts
- Detecção de ataques (Injection, Jailbreak)
- LlamaGuard
- Red Teaming de IA
- Proteção de APIs
- **Objetivo**: Especialista em AI Security

## 🚀 Como Começar

### Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes Python)
- Git (opcional, mas recomendado)

### Instalação

1. **Clone o repositório** (ou baixe os arquivos):
```bash
git clone <url-do-repositorio>
cd ml_classic
```

2. **Crie um ambiente virtual** (recomendado):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente** (opcional):
```bash
# Crie um arquivo .env na raiz do projeto
OPENAI_API_KEY=sua-chave-aqui
ANTHROPIC_API_KEY=sua-chave-aqui
GEMINI_API_KEY=sua-chave-aqui
```

### Executar a Aplicação

```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no seu navegador em `http://localhost:8501`

## 📖 Estrutura do Projeto

```
ml_classic/
├── app.py                      # Aplicação principal Streamlit
├── requirements.txt            # Dependências do projeto
├── README.md                   # Este arquivo
├── .env                        # Variáveis de ambiente (criar)
├── modules/                    # Módulos das etapas
│   ├── __init__.py
│   ├── etapa1/                 # Fundamentos Essenciais
│   │   ├── __init__.py
│   │   └── fundamentos.py
│   ├── etapa2/                 # ML Clássico
│   │   ├── __init__.py
│   │   └── ml_classico.py
│   ├── etapa3/                 # Deep Learning
│   │   ├── __init__.py
│   │   └── deep_learning.py
│   ├── etapa4/                 # LLMs
│   │   ├── __init__.py
│   │   └── llms_prompts.py
│   ├── etapa5/                 # RAG
│   │   ├── __init__.py
│   │   └── rag.py
│   ├── etapa6/                 # Agentes
│   │   ├── __init__.py
│   │   └── agentes.py
│   ├── etapa7/                 # Deploy
│   │   ├── __init__.py
│   │   └── deploy.py
│   └── etapa8/                 # Security
│       ├── __init__.py
│       └── security.py
└── assets/                     # Recursos adicionais (imagens, etc)
```

## 💡 Como Usar

1. **Navegação**: Use o menu lateral para escolher uma etapa
2. **Estudo**: Leia o conteúdo teórico e analise os exemplos
3. **Prática**: Execute os exemplos interativos
4. **Exercícios**: Complete os exercícios práticos propostos
5. **Checklist**: Marque os itens da checklist ao dominar cada tópico
6. **Próxima Etapa**: Avance quando se sentir confortável

## 🎯 Dicas de Aprendizado

- ✍️ **Pratique ativamente**: Digite o código, não apenas leia
- 🔄 **Revise regularmente**: Volte aos conceitos quando necessário
- 🚀 **Construa projetos**: Aplique em projetos pessoais
- 👥 **Compartilhe**: Ensine outros para solidificar conhecimento
- 📚 **Aprofunde**: Use as referências para estudar mais
- 💪 **Seja consistente**: Estude todos os dias, mesmo que pouco tempo

## 🛠️ Tecnologias Utilizadas

- **Frontend/UI**: Streamlit
- **Machine Learning**: scikit-learn, XGBoost
- **Deep Learning**: PyTorch, Transformers
- **LLMs**: OpenAI, Anthropic, Google, Ollama
- **RAG**: Chroma, Pinecone, LangChain, LlamaIndex
- **MLOps**: MLflow, DVC, Evidently
- **Deploy**: FastAPI, Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana

## 📦 Instalação de Componentes Opcionais

### Ollama (Modelos Locais)
```bash
# Windows: Baixar de https://ollama.com/download
# Linux/Mac:
curl -fsSL https://ollama.com/install.sh | sh

# Baixar modelos
ollama pull llama3.1
ollama pull mistral
```

### Docker (Containerização)
- Windows/Mac: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Linux: Via package manager

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/NovaEtapa`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova etapa'`)
4. Push para a branch (`git push origin feature/NovaEtapa`)
5. Abrir um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🌟 Agradecimentos

Este projeto foi desenvolvido com base nas melhores práticas e tecnologias mais recentes de IA, consolidando conhecimento de múltiplas fontes e experiências práticas.

## 📧 Contato e Suporte

- **Issues**: Para reportar bugs ou sugerir melhorias, abra uma issue no GitHub
- **Discussões**: Use a aba Discussions para perguntas e compartilhar experiências

## 🎓 Próximos Passos

Após completar este roadmap, você estará preparado para:

- 🎯 **Trabalhar** como Engenheiro de IA/ML
- 🚀 **Construir** produtos de IA do zero
- 💼 **Consultorias** e projetos freelance
- 🏢 **Liderar** times de IA
- 📚 **Continuar** aprendendo (IA nunca para de evoluir!)

---

<div align="center">

**⭐ Se este projeto te ajudou, considere dar uma estrela!**

**💻 Desenvolvido com ❤️ e Streamlit**

**🤖 Bons estudos e sucesso na sua jornada de IA!**

</div>

#   s a _ i a  
 