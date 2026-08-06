# syntax=docker/dockerfile:1

# -----------------------------------------------------------------------------
# Builder stage: resolve + install every declared dependency into a venv.
# `--all-extras` covers every optional-dependency group declared in
# pyproject.toml (api, offline-storage, offline-llm, memory, offline, test,
# evaluation, observability, ...) so the image needs no runtime pip installs
# for the supported feature set. Heavy, non-extra stacks (torch, transformers,
# lmdeploy, chromadb, ...) are intentionally NOT preinstalled; those legacy
# modules still self-install via pipmaster at runtime when explicitly used.
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Compile toolchain for any wheel that has no prebuilt binary for linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# Project metadata + lock first for optimal layer caching.
# pyproject.toml declares `readme = "README.md"` but this repo has no root
# README, so the API-server doc is used as the in-image build readme only.
COPY pyproject.toml setup.py uv.lock ./
COPY docs/LightRAG-API-Server.md ./README.md

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --no-dev --no-install-project

# Application source, installed non-editable into the same venv.
COPY lightrag/ ./lightrag/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --no-dev --no-editable \
    && /app/.venv/bin/python -m ensurepip --upgrade

# -----------------------------------------------------------------------------
# Runtime stage: same Python ABI (3.12 bookworm), no build toolchain.
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Runtime system packages:
#   - gosu:          drop root privileges in the entrypoint
#   - ca-certificates: TLS for outbound LLM/storage calls
#   - libgomp1:      OpenMP runtime used by numpy/scipy/faiss
#   - libcairo2:     native cairo lib for cairosvg SVG->PNG rasterization
#   - libreoffice-*: legacy .doc/.ppt/.xls conversion (ENABLE_LIBREOFFICE_CONVERSION)
#   - fontconfig/fonts: CJK + Latin fonts for rendered text
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gosu \
        ca-certificates \
        libgomp1 \
        libcairo2 \
        libreoffice-writer \
        libreoffice-calc \
        libreoffice-impress \
        fontconfig \
        fonts-liberation \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Populated venv + application source from the builder stage.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/lightrag ./lightrag

# Point the venv interpreter at this image's python (same 3.12 ABI).
RUN ln -sfn /usr/local/bin/python3.12 /app/.venv/bin/python \
    && /app/.venv/bin/python -c "import sys; assert sys.version_info[:2] == (3, 12)"

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    HOST="0.0.0.0" \
    PORT="9621" \
    WORKING_DIR="/app/rag_storage" \
    INPUT_DIR="/app/inputs" \
    LOG_DIR="/app/logs" \
    TIKTOKEN_CACHE_DIR="/app/data/tiktoken" \
    PROMPT_DIR="/app/prompts"

# Minimal .env so the server's runtime-target check accepts the container
# (real config is injected at runtime via compose env_file/environment).
RUN mkdir -p /app/rag_storage /app/inputs /app/logs /app/data/tiktoken /app/prompts \
    && printf 'LIGHTRAG_RUNTIME_TARGET=compose\n' > /app/.env

# Fixed-UID non-root user + HOME so pipmaster's runtime pip installs never
# fall back to an unwritable /root.
RUN groupadd -g 1000 lightrag \
    && useradd -u 1000 -g lightrag -m -d /home/lightrag -s /usr/sbin/nologin lightrag \
    && chown -R lightrag:lightrag /app /home/lightrag

ENV HOME=/home/lightrag \
    XDG_CACHE_HOME=/home/lightrag/.cache \
    PIP_CACHE_DIR=/home/lightrag/.cache/pip

# Entrypoint fixes bind-mount ownership then drops privileges (see file).
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 9621

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9621/health', timeout=4)"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "lightrag.api.lightrag_server"]
