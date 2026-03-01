# =============================================================================
# Dockerfile - Sistema de Aprendizado de IA
# Multi-stage build otimizado para produção
# Decisão arquitetural: imagem leve usando apenas dependências de runtime.
# As libs pesadas (torch, transformers ~3GB) NÃO são necessárias pois a
# plataforma exibe exemplos de código, não os executa em runtime.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder - instala dependências em ambiente isolado
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Instala dependências de compilação
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements otimizado para Docker
COPY requirements-docker.txt .

# Instala dependências em diretório isolado (--prefix)
RUN pip install --no-cache-dir --prefix=/install -r requirements-docker.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime - imagem final mínima
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Metadata (OCI standard labels)
LABEL org.opencontainers.image.title="sa-ia-learning" \
      org.opencontainers.image.description="Sistema de Aprendizado de IA - Plataforma educacional interativa" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="SA-IA" \
      org.opencontainers.image.source="https://github.com/sa-ia/sa-ia-learning"

# Segurança: usuário não-root
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copia dependências do builder
COPY --from=builder /install /usr/local

# Copia código da aplicação
COPY app.py .
COPY modules/ ./modules/
COPY assets/ ./assets/
COPY healthcheck.py .
COPY .streamlit/ ./.streamlit/

# Cria diretórios necessários com permissões corretas
RUN mkdir -p /app/logs /app/data /tmp/.streamlit && \
    chown -R appuser:appuser /app /tmp/.streamlit

# Variáveis de ambiente (sem secrets hardcoded)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=dark \
    APP_ENV=production \
    LOG_LEVEL=INFO

# Porta do Streamlit
EXPOSE 8501

# Healthcheck real funcional
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python healthcheck.py || exit 1

# Executa como usuário não-root
USER appuser

# Entrypoint com Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
