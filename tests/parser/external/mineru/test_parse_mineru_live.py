"""Gated live integration test for the real MinerU parsing path.

Unlike ``test_parse_mineru_sidecar.py`` (which stubs ``MinerURawClient`` so no
MinerU service is contacted), this test drives the **real** MinerU client end to
end against a **live** MinerU service. It is therefore opt-in and skipped unless
a live endpoint is provided, mirroring the live-PostgreSQL contract test gating.

Enable it by pointing at a running MinerU local-API service::

    LIGHTRAG_MINERU_TEST_ENDPOINT=http://127.0.0.1:8000 \
        uv run pytest tests/parser/external/mineru/test_parse_mineru_live.py -q

Optional knobs:
    LIGHTRAG_MINERU_TEST_PDF=/abs/path/to/sample.pdf   # use a real PDF instead
                                                        # of the minimal built-in one

What it proves that the stubbed test cannot: the real HTTP client reaches the
service, uploads the source, polls/downloads the raw bundle, and the sidecar
normalizer turns the live bundle into a LightRAG-format parse result with
on-disk artifacts. When no endpoint is configured the test is skipped (never
failed), so CI stays green without a MinerU service.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.integration]

_MINERU_ENDPOINT = os.getenv("LIGHTRAG_MINERU_TEST_ENDPOINT")

# A minimal single-page PDF containing the text "Hello LightRAG MinerU".
# Used when LIGHTRAG_MINERU_TEST_PDF is not provided. Kept tiny but spec-valid.
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
    b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
    b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>endobj\n"
    b"4 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
    b"5 0 obj<< /Length 68 >>stream\n"
    b"BT /F1 24 Tf 72 700 Td (Hello LightRAG MinerU) Tj ET\n"
    b"endstream endobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000241 00000 n \n"
    b"0000000314 00000 n \n"
    b"trailer<< /Size 6 /Root 1 0 R >>\n"
    b"startxref\n433\n"
    b"%%EOF\n"
)


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 32)


async def _mock_llm(prompt: Any, **kwargs: Any) -> str:
    return '{"name":"x","summary":"s","detail_description":"d"}'


def _sample_pdf(tmp_path: Path) -> Path:
    provided = os.getenv("LIGHTRAG_MINERU_TEST_PDF")
    if provided:
        path = Path(provided)
        if not path.is_file():
            pytest.skip(f"LIGHTRAG_MINERU_TEST_PDF not found: {provided}")
        return path
    pdf_path = tmp_path / "mineru_live_sample.pdf"
    pdf_path.write_bytes(_MINIMAL_PDF)
    return pdf_path


@pytest.mark.skipif(
    not _MINERU_ENDPOINT,
    reason="live MinerU integration skipped: set LIGHTRAG_MINERU_TEST_ENDPOINT to enable",
)
async def test_parse_mineru_live_parse_build_and_query(tmp_path, monkeypatch):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc, Tokenizer

    # Point the real MinerU client at the live local-API service.
    monkeypatch.setenv("MINERU_API_MODE", "local")
    monkeypatch.setenv("MINERU_LOCAL_ENDPOINT", _MINERU_ENDPOINT)
    # Force a fresh parse so we exercise the real download path, not a cache hit.
    monkeypatch.setenv("LIGHTRAG_FORCE_REPARSE_MINERU", "true")
    # Keep the generated sidecar under the same test-scoped INPUT_DIR so the
    # production path-containment guard is exercised without escaping it.
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))

    pdf_path = _sample_pdf(tmp_path)

    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"mineru-live-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=32, max_token_size=4096, func=_mock_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        vlm_process_enable=False,
    )
    await rag.initialize_storages()
    try:
        result = await rag.parse_mineru(
            "doc-mineru-live",
            str(pdf_path),
            {
                "parse_format": "pending_parse",
                "parse_engine": "mineru",
                "process_options": "iF",
                "force_reparse": True,
                "archive_source_after_parse": False,
            },
        )

        # The live bundle must normalize into a LightRAG-format parse result.
        assert result["parse_format"] == "lightrag"
        assert result.get("parse_stage_skipped") is False
        blocks_path = result.get("blocks_path")
        assert blocks_path, f"expected blocks_path in result: {result}"
        assert Path(blocks_path).is_file()

        # The real MinerU raw bundle must have been materialized on disk.
        raw_dirs = list(Path(tmp_path).rglob("*.mineru_raw"))
        assert raw_dirs, "expected a *.mineru_raw bundle directory from live MinerU"
        # Markdown is the canonical MinerU text output.
        markdowns = list(raw_dirs[0].rglob("*.md"))
        assert markdowns, f"expected markdown in raw bundle: {list(raw_dirs[0].rglob('*'))}"

        # --- build: feed the live parse sidecar through the REAL index pipeline ---
        # Mirrors IndexBuildService.run_build's enqueue call so this exercises the
        # same parse-artifact -> chunk/extract/embed/KG-merge path a KB build uses.
        from lightrag.base import QueryParam
        from lightrag.utils_pipeline import sidecar_uri_for

        sidecar_uri = sidecar_uri_for(Path(blocks_path).parent)
        await rag.apipeline_enqueue_documents(
            input=[""],
            ids=["mineru-live-doc"],
            file_paths=["mineru_live_sample.pdf"],
            docs_format="lightrag",
            lightrag_document_paths=[sidecar_uri],
            parse_engine="mineru",
            process_options="iF",
        )
        await rag.apipeline_process_enqueue_documents()

        # After the build the document must be tracked in doc_status.
        built = await rag.doc_status.get_by_id("mineru-live-doc")
        assert built is not None, "expected a doc_status row after build"

        # --- query: the built KB must retrieve the live-parsed content ---
        data = await rag.aquery_data(
            "Hello LightRAG MinerU",
            param=QueryParam(mode="naive", top_k=10, chunk_top_k=10),
        )
        chunks = data.get("data", {}).get("chunks", [])
        assert chunks, f"expected retrievable chunks after build+query: {data}"
    finally:
        await rag.finalize_storages()
