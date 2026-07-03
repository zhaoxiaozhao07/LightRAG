"""LibreOffice conversion adapter facade."""

from lightrag.constants import LIBREOFFICE_RAW_DIR_SUFFIX
from lightrag.parser.external.libreoffice.converter import (
    BASE_CONVERT_ARGS,
    DEFAULT_LIBREOFFICE_EXECUTABLE,
    DEFAULT_LIBREOFFICE_TIMEOUT_SECONDS,
    LEGACY_OFFICE_SUFFIX_MAP,
    MANIFEST_ENGINE,
    LibreOfficeConfig,
    LibreOfficeConversionResult,
    LibreOfficeConverter,
    compute_options_signature,
    converted_upload_filename,
    is_conversion_cache_valid,
    is_legacy_office_file,
    is_legacy_office_suffix,
    raw_dir_for_parsed_dir,
    target_suffix_for_source_suffix,
)

__all__ = [
    "BASE_CONVERT_ARGS",
    "DEFAULT_LIBREOFFICE_EXECUTABLE",
    "DEFAULT_LIBREOFFICE_TIMEOUT_SECONDS",
    "LEGACY_OFFICE_SUFFIX_MAP",
    "LIBREOFFICE_RAW_DIR_SUFFIX",
    "MANIFEST_ENGINE",
    "LibreOfficeConfig",
    "LibreOfficeConversionResult",
    "LibreOfficeConverter",
    "compute_options_signature",
    "converted_upload_filename",
    "is_conversion_cache_valid",
    "is_legacy_office_file",
    "is_legacy_office_suffix",
    "raw_dir_for_parsed_dir",
    "target_suffix_for_source_suffix",
]
