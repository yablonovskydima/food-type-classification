# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv export --locked --extra api --no-dev --no-emit-project --no-hashes \
    --output-file /tmp/constraints.txt
# Install only serving dependencies. CPU wheels keep CUDA libraries out of the image;
# all package versions still come from the project's lock, including transitive constraints.
RUN python - <<'PY'
import tomllib
from pathlib import Path
packages = tomllib.loads(Path("uv.lock").read_text())["package"]
roots = {"torch", "torchvision", "numpy", "pillow", "fastapi", "uvicorn", "python-multipart"}
lines = []
for package in packages:
    name = package["name"]
    if name in roots:
        suffix = "+cpu" if name in {"torch", "torchvision"} else ""
        lines.append(f"{name}=={package['version']}{suffix}")
Path("/tmp/runtime.txt").write_text("\n".join(lines))
PY
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python --no-cache \
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    --constraint /tmp/constraints.txt --requirement /tmp/runtime.txt

FROM python:3.13-slim-bookworm AS api
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REQUIRE_CHECKPOINT=1 \
    DEVICE=cpu
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY app/api ./app/api
COPY docker/healthcheck.py ./docker/healthcheck.py
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "docker/healthcheck.py"]
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
