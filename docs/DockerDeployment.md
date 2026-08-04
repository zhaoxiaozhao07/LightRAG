# LightRAG Docker Deployment

A lightweight Knowledge Graph Retrieval-Augmented Generation system with multiple LLM backend support.

## 🚀 Preparation

### Clone the repository:

```bash
# Linux/MacOS
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
```
```powershell
# Windows PowerShell
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
```

### Configure your environment:

```bash
# Linux/MacOS
cp env.example .env
# Edit .env with your preferred configuration
```
```powershell
# Windows PowerShell
Copy-Item env.example .env
# Edit .env with your preferred configuration
```

LightRAG can be configured using environment variables in the `.env` file:

**Server Configuration**

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 9621)

**LLM Configuration**

- `LLM_BINDING`: LLM backend to use (lollms/ollama/openai)
- `LLM_BINDING_HOST`: LLM server host URL
- `LLM_MODEL`: Model name to use

**Embedding Configuration**

- `EMBEDDING_BINDING`: Embedding backend (lollms/ollama/openai)
- `EMBEDDING_BINDING_HOST`: Embedding server host URL
- `EMBEDDING_MODEL`: Embedding model name
- `EMBEDDING_ASYMMETRIC`: Explicitly enable query/document asymmetric embeddings
- `EMBEDDING_DOCUMENT_PREFIX`: Document prefix for prefix-based asymmetric embeddings (or `NO_PREFIX`)
- `EMBEDDING_QUERY_PREFIX`: Query prefix for prefix-based asymmetric embeddings (or `NO_PREFIX`)

See [Asymmetric Embedding Configuration](./AsymmetricEmbedding.md) for prefix
validation rules and provider-specific behavior.

**RAG Configuration**

- `MAX_ASYNC`: Maximum async operations
- `MAX_TOKENS`: Maximum token size
- `EMBEDDING_DIM`: Embedding dimensions

## 🐳 Docker Deployment

Docker instructions work the same on all platforms with Docker Desktop installed.

### Build Optimization

The Dockerfile uses BuildKit cache mounts to significantly improve build performance:

- **Automatic cache management**: BuildKit is automatically enabled via `# syntax=docker/dockerfile:1` directive
- **Faster rebuilds**: Only downloads changed dependencies when `uv.lock` or `bun.lock` files are modified
- **Efficient package caching**: UV and Bun package downloads are cached across builds
- **No manual configuration needed**: Works out of the box in Docker Compose and GitHub Actions

### Start LightRAG  server:

```bash
docker compose up -d
```

If you used the interactive setup, start the generated stack with:

```bash
docker compose -f docker-compose.final.yml up -d
```

The interactive setup keeps `.env` host-usable. Container-only hostnames such as `postgres` or `host.docker.internal`, along with staged SSL paths under `/app/data/certs/`, are injected into the generated `docker-compose.final.yml` for the `lightrag` service instead of being persisted back into `.env`.
On reruns, unchanged wizard-managed service blocks in `docker-compose.final.yml` are preserved by
default. To repair or fully regenerate those managed blocks from the bundled templates, rerun the
matching setup target with `make env-base-rewrite` or `make env-storage-rewrite`.

If the generated stack includes local Milvus, compose resolves `MINIO_ACCESS_KEY_ID` and
`MINIO_SECRET_ACCESS_KEY` at startup from the repo `.env` or exported shell environment. The
generated compose file does not snapshot those values, and `docker compose` exits immediately if
either variable is missing.

Before exposing the generated stack beyond localhost, run:

```bash
make env-security-check
```

That command audits the current `.env` for missing authentication, unsafe whitelist settings, weak
JWT secrets, and other setup-level security risks without rewriting any files.

LightRAG Server uses the following paths for data storage:

```
data/
├── rag_storage/    # RAG data persistence
└── inputs/         # Input documents
```

### Optional: local vLLM embedding and reranker

To run embedding and/or reranking locally with vLLM, run `make env-base` and answer `yes` when prompted to run the embedding model and rerank service locally via Docker.
That configures the embedding service to use `BAAI/bge-m3` on port 8001 with a local vLLM server, and can also add a `vllm-rerank` service on port 8000.

Alternatively, rerun `make env-base` later and enable only the rerank Docker prompt to add the `vllm-rerank` service automatically.
vLLM provides a `v1/rerank` endpoint that works with the `cohere` binding.

Example `docker-compose.override.yml` for GPU hosts (embedding + reranker):

```yaml
services:
  vllm-embed:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    command: >
      --model BAAI/bge-m3
      --port 8001
      --dtype float16
    ports:
      - "8001:8001"
    volumes:
      - ./data/hf-cache:/root/.cache/huggingface
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  vllm-rerank:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    command: >
      --model BAAI/bge-reranker-v2-m3
      --port 8000
      --dtype float16
    ports:
      - "8000:8000"
    volumes:
      - ./data/hf-cache:/root/.cache/huggingface
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

For CPU-only hosts, use the official CPU image instead:

```yaml
services:
  vllm-embed:
    image: vllm/vllm-openai-cpu:latest
    command: >
      --model BAAI/bge-m3
      --port 8001
      --dtype float32
    ports:
      - "8001:8001"
    volumes:
      - ./data/hf-cache:/root/.cache/huggingface

  vllm-rerank:
    image: vllm/vllm-openai-cpu:latest
    command: >
      --model BAAI/bge-reranker-v2-m3
      --port 8000
      --dtype float32
    ports:
      - "8000:8000"
    volumes:
      - ./data/hf-cache:/root/.cache/huggingface
```

Add the embedding and rerank config to `.env`:

```bash
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_BINDING_HOST=http://localhost:8001/v1
EMBEDDING_BINDING_API_KEY=local-key
VLLM_EMBED_DEVICE=cpu

RERANK_BINDING=cohere
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=http://localhost:8000/rerank
RERANK_BINDING_API_KEY=local-key
VLLM_RERANK_DEVICE=cpu
```

If LightRAG runs in Docker while vLLM runs on the host, the generated compose file rewrites those endpoints to:

```bash
EMBEDDING_BINDING_HOST=http://host.docker.internal:8001/v1
RERANK_BINDING_HOST=http://host.docker.internal:8000/rerank
```

For GPU, set:

```bash
VLLM_EMBED_DEVICE=cuda
VLLM_RERANK_DEVICE=cuda
```

Ensure the NVIDIA Container Toolkit is installed and the host has CUDA drivers available.
The setup wizard uses the CPU image by default for `cpu` device and the GPU image for `cuda` device.
When rerunning `make env-base`, an existing `VLLM_EMBED_DEVICE` / `VLLM_RERANK_DEVICE` value is
preserved instead of being overwritten by a fresh GPU auto-detection result.
Those templates already pin the matching vLLM `--dtype` (`float32` on CPU, `float16` on CUDA), so no separate `VLLM_*_DTYPE` environment variables are needed.

### SSL certificates

The setup wizard stages TLS certificate files under `./data/certs/` before generating the compose file.
This keeps generated host mounts under the same `./data` root used by the default Docker deployment.

### PostgreSQL image

The interactive setup defaults PostgreSQL to `gzdaniel/postgres-for-rag:pg18-age-pgvector`. This image bundles both Apache AGE and pgvector so the generated stack works with `PGGraphStorage` and `PGVectorStorage` without extra extension setup.

The image no longer ships fixed credentials; on first start it creates the user, password, and database from the `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` environment variables. The setup wizard prompts for these values (defaulting to `rag` / `rag` / `lightrag`) and injects them into the generated `docker-compose.final.yml`, so you can choose any user, password, and database name.

**Important Note**: If PGGraphStorage is not required for vector storage, you may replace the upper docker image with the latest official pgvector image `pgvector/pgvector:pg18`. Please note that data file formats are incompatible across different PostgreSQL major versions; once this Docker image is deployed, it cannot be rolled back to a previous version.

#### Build the PostgreSQL image

The default PostgreSQL image can be rebuilt from the repository root with `Dockerfile.postgres`. The Dockerfile starts from `pgvector/pgvector:pg18-trixie`, builds Apache AGE for PostgreSQL 18, and installs an init script that creates both the `vector` and `age` extensions for new databases.

For local development or single-host deployment:

```bash
docker build -f Dockerfile.postgres \
  -t gzdaniel/postgres-for-rag:pg18-age-pgvector \
  .
```

To publish the image to your own registry, use a private tag:

```bash
docker build -f Dockerfile.postgres \
  -t registry.example.com/postgres-for-rag:pg18-age-pgvector \
  .
docker push registry.example.com/postgres-for-rag:pg18-age-pgvector
```

For a multi-architecture image:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.postgres \
  -t registry.example.com/postgres-for-rag:pg18-age-pgvector \
  --push \
  .
```

After building a custom tag, update the `postgres.image` value in the generated `docker-compose.final.yml` to point at that tag. On later setup wizard reruns, existing wizard-managed service images are preserved.

### Object-storage deployment (S3 / MinIO)

LightRAG supports an object-authoritative artifact storage mode in which S3/MinIO holds source bytes and generated artifacts, while the local filesystem is only operation-scoped scratch. This is intended for horizontally-scaled / stateless deployments. Configure the object-storage backend with the `LIGHTRAG_OBJECT_STORAGE_*` variables:

| Variable | Purpose |
| -------- | ------- |
| `LIGHTRAG_OBJECT_STORAGE` | Backend selector: `local` (default, no object storage), `s3`, or `minio`. Object-authoritative mode requires `s3` or `minio`. |
| `LIGHTRAG_OBJECT_STORAGE_ENDPOINT` | S3/MinIO endpoint URL (e.g. `http://minio:9000` for a compose service, `http://host.docker.internal:9000` for a host-side MinIO). |
| `LIGHTRAG_OBJECT_STORAGE_BUCKET` | Target bucket name (validated for DNS-safe naming). |
| `LIGHTRAG_OBJECT_STORAGE_PREFIX` | Object key prefix (default `kb`); all artifact keys are namespaced under it. Empty is allowed; trailing slash is not. |
| `LIGHTRAG_OBJECT_STORAGE_ACCESS_KEY_ID` | Access key. |
| `LIGHTRAG_OBJECT_STORAGE_SECRET_ACCESS_KEY` | Secret key. |
| `LIGHTRAG_OBJECT_STORAGE_USE_SSL` | `true`/`false` — TLS to the endpoint. |
| `LIGHTRAG_OBJECT_STORAGE_REGION` | AWS region (regional S3 endpoints). |
| `LIGHTRAG_OBJECT_STORAGE_CREATE_BUCKET` | `true` lets the server create the bucket on startup if it does not exist (development convenience; verify bucket policy in production). |

When running in Docker against a host-side MinIO/S3, use `host.docker.internal` in the endpoint (the same host-rewrite the setup wizard applies to LLM endpoints). For a compose-side MinIO service, point at the service name.

**Volume / mount considerations.** In object-authoritative mode (`LIGHTRAG_ARTIFACT_STORAGE_MODE=object`), persistent local artifact storage is **not** required — artifacts live in the bucket, not under `data/inputs`. The container still needs a canonical `INPUT_DIR` (POSIX `fcntl` locking + writable scratch) for transient operation-scoped work, but it does not need to be backed up or shared across replicas. For multi-replica deployments, point every replica at the same bucket + PostgreSQL metadata backend (`LIGHTRAG_KB_METADATA_BACKEND=postgres`); do not share a local `data/inputs` across containers.

**Capability gate.** Even with all prerequisites met, production object mode is currently disabled by the code-level constant `OBJECT_AUTHORITATIVE_LIFECYCLE_IMPLEMENTED = False` (`lightrag/api/config.py`). The server refuses to start in object mode until that constant is flipped to `True` after Phase 3 Gate 3 passes. The constant is not configurable via environment variables. See [LightRAG-API-Server.md → Object-Authoritative Artifact Storage](./LightRAG-API-Server.md) for the gate, migration/orphan CLIs, and route policy, and the [Production Backup & Recovery Runbook](./生产级后端备份恢复Runbook.md) for operational detail.

**Health readiness.** When object mode is enabled, `GET /health` exposes an `artifact_lifecycle` block whose `object_store_ready` field is a cached HeadBucket probe (~30s TTL). Container orchestrators that keys readiness off `/health` should be aware that `artifact_lifecycle.object_store_ready=false` indicates the bucket is unreachable / auth failed / the probe timed out — verify endpoint, credentials, and network policy. The probe is metadata-only (a single `head_bucket`); `/health` never lists bucket contents or downloads objects, so its latency stays bounded regardless of bucket size. See the Runbook, section 12, for field-by-field interpretation.

### Updates

To update the Docker container:
```bash
docker compose pull
docker compose down
docker compose up
```

### Offline deployment

Software packages requiring `transformers`, `torch`, or `cuda` are not preinstalled in the docker images. Consequently, document extraction tools such as Docling, as well as local LLM models like Hugging Face and LMDeploy, cannot be used in an offline environment. These high-compute-resource-demanding services should not be integrated into LightRAG. Docling will be decoupled and deployed as a standalone service.

## 📦 Build Docker Images

### For local development and testing

```bash
# Build and run with Docker Compose (BuildKit automatically enabled)
docker compose up --build

# Or explicitly enable BuildKit if needed
DOCKER_BUILDKIT=1 docker compose up --build
```

**Note**: BuildKit is automatically enabled by the `# syntax=docker/dockerfile:1` directive in the Dockerfile, ensuring optimal caching performance.

### For production release

 **multi-architecture build and push**:

```bash
# Use the provided build script
./docker-build-push.sh
```

**The build script will**:

- Check Docker registry login status
- Create/use buildx builder automatically
- Build for both AMD64 and ARM64 architectures
- Push to GitHub Container Registry (ghcr.io)
- Verify the multi-architecture manifest

**Prerequisites**:

Before building multi-architecture images, ensure you have:

- Docker 20.10+ with Buildx support
- Sufficient disk space (20GB+ recommended for offline image)
- Registry access credentials (if pushing images)

### Verify official GHCR images with Cosign

Official LightRAG images published to GitHub Container Registry by GitHub Actions are signed with Sigstore Cosign using GitHub OIDC keyless signing.

Install `cosign`, then verify the image tag you want to run:

```bash
cosign verify ghcr.io/HKUDS/LightRAG:<tag> \
  --certificate-identity-regexp '^https://github.com/HKUDS/LightRAG/.github/workflows/(docker-publish|docker-build-manual|docker-build-lite)\.yml@refs/.+$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Replace `<tag>` with the version tag you want to validate, for example a release tag, `latest`, `<tag>-lite`, or `lite`.
