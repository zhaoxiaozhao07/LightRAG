# Advanced Features

## Multimodal Document Processing

LightRAG Server includes a multimodal document pipeline for text, images, tables, and equations. Document parsing is handled through external MinerU or Docling services configured by endpoint, so the server no longer needs to install or import the `raganything` package locally.

**Status:** multimodal parsing and sidecar indexing are now wired into the ingestion pipeline. MinerU/Docling/native parsers write LightRAG sidecars, the analyze stage can enrich enabled image/table/equation items with surrounding context and VLM analysis (`process_options` flags `i`/`t`/`e`), and successful multimodal analysis results are converted into dedicated chunks for graph extraction and retrieval. Deployments still need working parser endpoints and VLM role configuration for full image/table/equation understanding; if no multimodal options are enabled, ingestion falls back to normal text indexing.

**Current Capabilities:**
- End-to-end parser-to-index flow for native, MinerU, and Docling sidecars
- Universal document support through configured parser routing for PDFs, Office documents, images, and other supported suffixes
- Opt-in image/table/equation VLM analysis via per-document `process_options`
- Multimodal chunks that participate in entity/relation extraction and hybrid retrieval
- Stage observability through parse/analyze metadata such as `parse_stage_skipped`, `analyzing_stage_skipped`, and `multimodal_processed`

### Quick Start

Configure parser routing and external parser service endpoints in `.env`:

```bash
LIGHTRAG_PARSER=pdf:mineru,docx:docling,pptx:docling,xlsx:docling,*:legacy
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://localhost:8000
DOCLING_ENDPOINT=http://localhost:5001/v1/convert/file/async
```

Then upload documents through LightRAG Server. `LIGHTRAG_PARSER` rules match suffixes such as `pdf`, may be separated with commas or semicolons, and are evaluated from left to right. If a rule enables MinerU or Docling, the matching endpoint must be configured before server startup. Per-file hints such as `paper.[mineru].pdf` and `memo.[native].docx` override the default rules. Parsed multimodal sidecars are written by the pipeline and consumed by the normal indexing flow. See [File Processing Configuration](./FileProcessingConfiguration-zh.md) for detailed routing rules and examples.

---

## Token Usage Tracking

**Overview and Usage**

LightRAG provides a `TokenTracker` tool to monitor token consumption reported by supported LLM providers. This feature is useful for controlling API costs and optimizing performance.

`TokenTracker` does not automatically inject itself into LLM calls. Pass it to the provider binding directly, bind it through `llm_model_kwargs`, or capture it in your custom LLM function.

**Method 1: Track direct LLM calls**

```python
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import TokenTracker

token_tracker = TokenTracker()

with token_tracker:
    result1 = await openai_complete_if_cache(
        "gpt-4o-mini",
        "your question 1",
        token_tracker=token_tracker,
    )
    result2 = await openai_complete_if_cache(
        "gpt-4o-mini",
        "your question 2",
        token_tracker=token_tracker,
    )
```

The context manager resets the tracker when entering the block and prints usage when leaving it. The `token_tracker=token_tracker` argument is still required.

**Method 2: Track LightRAG calls**

```python
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.utils import TokenTracker

token_tracker = TokenTracker()

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=gpt_4o_mini_complete,
    llm_model_kwargs={"token_tracker": token_tracker},
    embedding_func=embedding_func,
)

await rag.initialize_storages()

token_tracker.reset()
await rag.ainsert(["document one", "document two"])
await rag.aquery("your question 1", param=QueryParam(mode="naive"))
await rag.aquery("your question 2", param=QueryParam(mode="mix"))

print("Token usage:", token_tracker.get_usage())
```

`llm_model_kwargs={"token_tracker": token_tracker}` is passed to the default role LLM wrappers used by extraction, keyword generation, querying, and VLM calls. If you configure role-specific LLM kwargs, put `token_tracker` in the relevant role kwargs as well, or use the closure pattern below.

**Robust custom wrapper pattern**

```python
from lightrag import LightRAG
from lightrag.llm.gemini import gemini_complete_if_cache
from lightrag.utils import TokenTracker


def make_llm_func(token_tracker: TokenTracker):
    async def _llm_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        **kwargs,
    ):
        return await gemini_complete_if_cache(
            "gemini-2.5-flash-lite",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            token_tracker=token_tracker,
            **kwargs,
        )

    return _llm_model_func


token_tracker = TokenTracker()

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=make_llm_func(token_tracker),
    embedding_func=embedding_func,
)

await rag.initialize_storages()

token_tracker.reset()
await rag.ainsert(["document one", "document two"])

print("Token usage:", token_tracker.get_usage())
```

**Usage Tips:**
- Use context managers for direct LLM sessions when you want automatic reset and final printing
- For segmented statistics, call `reset()` before each indexing or query phase
- LLM cache hits do not create new provider calls, so token usage does not increase for cached responses
- Regular checking of token usage helps detect abnormal consumption early

---

## Data Export Functions

LightRAG allows you to export your knowledge graph data in various formats for analysis, sharing, and backup.

**Basic Usage**

```python
# Basic CSV export (default format)
rag.export_data("knowledge_graph.csv")

# Specify any format
rag.export_data("output.xlsx", file_format="excel")
```

**Supported File Formats**

```python
rag.export_data("graph_data.csv", file_format="csv")
rag.export_data("graph_data.xlsx", file_format="excel")
rag.export_data("graph_data.md", file_format="md")
rag.export_data("graph_data.txt", file_format="txt")
```

**Additional Options**

Include vector embeddings in the export (optional):

```python
rag.export_data("complete_data.csv", include_vector_data=True)
```

All exports include entity information (names, IDs, metadata), relation data (connections between entities), and relationship information from the vector database.

---

## Cache Management

**Clear Cache**

`aclear_cache()` clears all cached entries in `llm_response_cache`. It does not support selective cleanup by mode or cache type.

```python
# Asynchronous
await rag.aclear_cache()

# Synchronous
rag.clear_cache()
```

For selective cleanup of query-related caches, use the `lightrag.tools.clean_llm_query_cache` tool and see the guide in [lightrag/tools/README_CLEAN_LLM_QUERY_CACHE.md](../lightrag/tools/README_CLEAN_LLM_QUERY_CACHE.md). It manages query caches and keywords caches for `mix`, `hybrid`, `local`, and `global` modes. It does **not** clean extraction caches such as `default:extract:*` and `default:summary:*`.

---

## Langfuse Observability Integration

Langfuse provides a drop-in replacement for the OpenAI client that automatically tracks all LLM interactions, enabling developers to monitor, debug, and optimize their RAG systems.

### Installation

```bash
pip install lightrag-hku[observability]
# Or from source:
pip install -e ".[observability]"
```

### Configuration

Add to `.env` file:

```
## Langfuse Observability (Optional)
LANGFUSE_SECRET_KEY=""
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted instance
LANGFUSE_ENABLE_TRACE=true
```

### Features

Once installed and configured, Langfuse automatically traces all OpenAI LLM calls. Dashboard features include:
- **Tracing**: View complete LLM call chains
- **Analytics**: Token usage, latency, cost metrics
- **Debugging**: Inspect prompts and responses
- **Evaluation**: Compare model outputs
- **Monitoring**: Real-time alerting

> **Note**: LightRAG currently only integrates OpenAI-compatible API calls with Langfuse. APIs such as Ollama, Azure, and AWS Bedrock are not yet supported for Langfuse observability.

---

## RAGAS-based Evaluation

**RAGAS** (Retrieval Augmented Generation Assessment) is a framework for reference-free evaluation of RAG systems using LLMs. LightRAG provides an evaluation script based on RAGAS. For detailed information, see [RAGAS-based Evaluation Framework](../lightrag/evaluation/README_EVALUASTION_RAGAS.md).
