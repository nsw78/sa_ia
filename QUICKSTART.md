# 🚀 Guia Rápido de Início

Este guia vai te ajudar a começar rapidamente com o Sistema de Aprendizado de IA.

## ⚡ Início Rápido (5 minutos)

### 1. Instalar Dependências Básicas

```bash
# Instalar apenas o essencial para começar
pip install streamlit pandas numpy scikit-learn matplotlib
```

### 2. Executar a Aplicação

```bash
streamlit run app.py
```

Pronto! A aplicação abrirá em `http://localhost:8501`

## 📦 Instalação Completa

### Passo 1: Preparar o Ambiente

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Passo 2: Instalar Todas as Dependências

```bash
pip install -r requirements.txt
```

**Nota:** A instalação completa pode levar alguns minutos.

### Passo 3: Configurar APIs (Opcional)

Se você quiser usar LLMs (Etapas 4-8), crie um arquivo `.env`:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

## 🎯 Primeiros Passos

### 1. Comece pela Etapa 1

No menu lateral, selecione **"🧠 Etapa 1: Fundamentos"**

### 2. Explore o Conteúdo

- Leia as explicações
- Execute os exemplos de código
- Pratique com os exercícios

### 3. Complete a Checklist

Marque os itens conforme você domina cada tópico.

## 📚 Roadmap Sugerido

### Iniciante (Semanas 1-3)
1. ✅ Etapa 1: Fundamentos Essenciais (7 dias)
2. ✅ Etapa 2: ML Clássico (7 dias)
3. ✅ Comece Etapa 3: Deep Learning (7 dias)

### Intermediário (Semanas 4-8)
4. ✅ Complete Etapa 3: Deep Learning
5. ✅ Etapa 4: LLMs + Prompting (10 dias)
6. ✅ Etapa 5: RAG (flexível)

### Avançado (Semanas 9+)
7. ✅ Etapa 6: Agentes de IA
8. ✅ Etapa 7: Deploy e Infraestrutura
9. ✅ Etapa 8: AI Security

## 💻 Instalação de Ferramentas Adicionais

### Ollama (Modelos Locais)

Para rodar LLMs localmente:

```bash
# Windows: Baixar instalador de https://ollama.com
# Linux/Mac:
curl -fsSL https://ollama.com/install.sh | sh

# Baixar um modelo
ollama pull llama3.1
```

### Docker (Opcional)

Para containerização:
- Windows/Mac: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Linux: `sudo apt install docker.io` (Ubuntu/Debian)

## 🔧 Solução de Problemas

### Erro: "ModuleNotFoundError"

```bash
# Reinstalar dependências
pip install -r requirements.txt --force-reinstall
```

### Erro: "Streamlit command not found"

```bash
# Verificar se está no ambiente virtual
# Se não, ativar:
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Reinstalar Streamlit
pip install streamlit
```

### Erro com APIs (OpenAI, etc)

1. Verifique se o arquivo `.env` existe
2. Confirme que as chaves de API estão corretas
3. As Etapas 1-3 funcionam sem APIs!

### Performance Lenta

```bash
# Instalar versão GPU do PyTorch (se tiver GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Conteúdo por Etapa

### 🧠 Etapa 1: Fundamentos (GRATUITO)
- Python avançado
- FastAPI
- Docker
- Git

### 🔢 Etapa 2: ML Clássico (GRATUITO)
- Regressão
- Random Forest
- XGBoost
- MLflow

### 🧠 Etapa 3: Deep Learning (GRATUITO)
- PyTorch
- CNN
- RNN/LSTM
- Transformers básico

### 🤖 Etapa 4: LLMs (Requer API)
- OpenAI ⚠️ pago
- Claude ⚠️ pago
- Gemini ⚠️ tem free tier
- Ollama ✅ gratuito e local

### 📚 Etapa 5: RAG (Requer API)
- Chroma ✅ gratuito
- Pinecone ⚠️ tem free tier
- LangChain ✅ gratuito

### 🧱 Etapa 6: Agentes (Requer API)
- LangChain ✅ gratuito
- LangGraph ✅ gratuito
- Usa APIs de LLM ⚠️

### 🏗️ Etapa 7: Deploy (Cloud pago)
- AWS/GCP/Azure ⚠️ pago
- Pode praticar localmente ✅

### 🔒 Etapa 8: Security (Avançado)
- Conceitos ✅ gratuitos
- Ferramentas ✅ open source

## 🎓 Dicas de Estudo

### Para Iniciantes
1. **Não pule etapas** - cada uma constrói sobre a anterior
2. **Pratique muito** - digite o código, não apenas leia
3. **Faça os exercícios** - são essenciais para fixar

### Para Quem Já Sabe Programar
1. Passe rápido pela Etapa 1 (revise o que não conhece)
2. Dedique tempo na Etapa 2 e 3 (fundamentos de ML/DL)
3. Foque em Etapas 4-8 (IA moderna)

### Para Quem Já Conhece ML
1. Vá direto para Etapa 4 (LLMs)
2. Etapas 5-6 são o diferencial (RAG e Agentes)
3. Etapas 7-8 são raras no mercado (Deploy e Security)

## 🌟 Próximos Passos

Após instalar e executar:

1. ✅ Explore a interface
2. ✅ Leia a Etapa 1 completa
3. ✅ Execute os exemplos interativos
4. ✅ Faça pelo menos 1 exercício
5. ✅ Avance para próxima etapa

## 📧 Suporte

- **Bug?** Abra uma issue no GitHub
- **Dúvida?** Veja a documentação completa no README.md
- **Sugestão?** Pull requests são bem-vindos!

---

## 🎯 Checklist de Instalação

- [ ] Python 3.11+ instalado
- [ ] Ambiente virtual criado
- [ ] Dependências instaladas (`pip install -r requirements.txt`)
- [ ] Aplicação executando (`streamlit run app.py`)
- [ ] Consegue navegar entre etapas
- [ ] (Opcional) Ollama instalado
- [ ] (Opcional) APIs configuradas

**Tudo pronto?** Comece pela Etapa 1! 🚀

---

<div align="center">

**💻 Desenvolvido com Streamlit**

**⭐ Bons estudos!**

</div>

