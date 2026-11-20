# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv export --locked --extra app --no-dev --no-emit-project --no-hashes \
    --output-file /tmp/constraints.txt > /dev/null

RUN python - <<'PY'
import tomllib
from pathlib import Path
packages = tomllib.loads(Path("uv.lock").read_text())["package"]
roots = {"torch", "torchvision", "numpy", "pillow", "fastapi", "uvicorn", "python-multipart", "streamlit"}
versions = {}
for package in packages:
    name, version = package["name"], package["version"]
    # The lock also holds Windows CUDA builds (e.g. 2.15.0+cu132); use the plain release.
    if name in roots and "+" not in version:
        versions.setdefault(name, set()).add(version)
missing = roots - versions.keys()
ambiguous = {name: found for name, found in versions.items() if len(found) != 1}
if missing or ambiguous:
    raise SystemExit(f"Cannot pick runtime versions: missing={missing} ambiguous={ambiguous}")
lines = [
    f"{name}=={found.pop()}{'+cpu' if name in {'torch', 'torchvision'} else ''}"
    for name, found in sorted(versions.items())
]
Path("/tmp/runtime.txt").write_text("\n".join(lines))
PY
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python --no-cache \
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    --constraint /tmp/constraints.txt --requirement /tmp/runtime.txt

FROM python:3.13-slim-bookworm AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHECKPOINT_PATH=/models/best.pt
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY app/api ./app/api
COPY app/ui ./app/ui
COPY docker/healthcheck.py ./docker/healthcheck.py
USER appuser

FROM runtime AS api
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "docker/healthcheck.py", "api"]
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS demo
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "docker/healthcheck.py", "demo"]
CMD ["python", "-m", "streamlit", "run", "app/ui/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
