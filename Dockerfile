FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# 1. PyTorch, CPU-only build (the default Linux build bundles GPU libraries)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Remaining dependencies, cached until pyproject.toml changes
COPY pyproject.toml README.md ./
RUN mkdir -p src/gridcast && touch src/gridcast/__init__.py \
    && pip install ".[chronos,api]"

# 3. Base model weights, baked into the image
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('amazon/chronos-2')"

# 4. Application code (changes most often, so it comes last)
COPY src ./src
RUN pip install --no-deps .

# 5. Run as an unprivileged user, fully offline
RUN useradd --create-home appuser
USER appuser
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
CMD ["uvicorn", "gridcast.api:app", "--host", "0.0.0.0", "--port", "8000"]
