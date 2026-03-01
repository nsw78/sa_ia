"""
Healthcheck script para o container Docker.
Verifica se o Streamlit está respondendo na porta configurada.
Retorna exit code 0 (healthy) ou 1 (unhealthy).
"""
import sys
import urllib.request
import os


def check_health() -> bool:
    port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
    url = f"http://localhost:{port}/_stcore/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


if __name__ == "__main__":
    healthy = check_health()
    sys.exit(0 if healthy else 1)
