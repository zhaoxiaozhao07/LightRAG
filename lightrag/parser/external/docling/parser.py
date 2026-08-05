"""Docling engine adapter (implements ExternalParserBase hooks)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.constants import DOCLING_RAW_DIR_SUFFIX, PARSER_ENGINE_DOCLING
from lightrag.parser.base import ParseContext, ParseResult
from lightrag.parser.external._base import ExternalParserBase
from lightrag.utils import logger

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc


class DoclingParser(ExternalParserBase):
    engine_name = PARSER_ENGINE_DOCLING
    raw_dir_suffix = DOCLING_RAW_DIR_SUFFIX
    force_reparse_env = "LIGHTRAG_FORCE_REPARSE_DOCLING"

    async def parse(self, ctx: ParseContext) -> ParseResult:
        """Enhanced parse with LibreOffice conversion support for legacy Office formats.

        Converts .doc/.ppt/.xls to .docx/.pptx/.xlsx before parsing.
        """
        from lightrag.parser.routing import should_convert_with_libreoffice

        # Check if LibreOffice conversion is needed
        rs = ctx.resolve(self.engine_name)
        source_path = rs.source_path

        if should_convert_with_libreoffice(str(source_path)):
            from lightrag.parser.external.libreoffice import (
                LibreOfficeConverter,
                raw_dir_for_parsed_dir,
            )
            from lightrag.utils_pipeline import normalize_document_file_path

            logger.info(
                f"[{self.engine_name}] Converting {source_path.name} with LibreOffice"
            )

            # Set up LibreOffice conversion cache directory
            libreoffice_raw_dir = raw_dir_for_parsed_dir(
                rs.parsed_dir, suffix=".libreoffice_raw"
            )

            # Convert using LibreOfficeConverter
            converter = LibreOfficeConverter()
            document_name = normalize_document_file_path(ctx.file_path)
            conversion_result = await converter.convert_for_docling(
                raw_dir=libreoffice_raw_dir,
                source_file_path=source_path,
                document_name=document_name,
            )

            # Temporarily replace the source path in content_data to point to converted file
            original_source = ctx.content_data.get("source_file")
            ctx.content_data["source_file"] = conversion_result.target_path.name

            try:
                # Call parent parse with converted file
                result = await super().parse(ctx)
                return result
            finally:
                # Restore original source file reference
                if original_source is not None:
                    ctx.content_data["source_file"] = original_source
                else:
                    ctx.content_data.pop("source_file", None)

        # No conversion needed, use parent implementation directly
        return await super().parse(ctx)

    def is_bundle_valid(
        self,
        raw_dir: Path,
        source_path: Path,
        *,
        engine_params: "Mapping[str, Any] | None" = None,
    ) -> bool:
        from lightrag.parser.external.docling import is_bundle_valid

        return is_bundle_valid(raw_dir, source_path, overrides=engine_params)

    async def download_into(
        self,
        raw_dir: Path,
        source_path: Path,
        *,
        upload_name: str,
        engine_params: "Mapping[str, Any] | None" = None,
    ) -> None:
        from lightrag.parser.external.docling import DoclingRawClient

        # Map the canonical ``upload_name`` onto docling-serve's multipart
        # filename so the bundle's main JSON is named ``<canonical_stem>.json``
        # (the IR builder locates it via that canonical stem).
        await DoclingRawClient(overrides=engine_params).download_into(
            raw_dir, source_path, upload_filename=upload_name
        )

    def build_ir(self, raw_dir: Path, document_name: str) -> "IRDoc":
        from lightrag.parser.external.docling import DoclingIRBuilder

        return DoclingIRBuilder().normalize_from_workdir(
            raw_dir, document_name=document_name
        )

    def validate_ir(self, ir: "IRDoc", *, file_path: str, raw_dir: Path) -> None:
        if not ir.blocks:
            raise ValueError(
                f"Docling IR builder produced zero blocks for {file_path} "
                f"(raw_dir={raw_dir})"
            )
