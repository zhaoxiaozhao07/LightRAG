import sys


def test_supported_document_extensions_include_legacy_office() -> None:
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        from lightrag.api.routers.document_routes import SUPPORTED_DOCUMENT_EXTENSIONS
    finally:
        sys.argv = saved_argv

    assert ".doc" in SUPPORTED_DOCUMENT_EXTENSIONS
    assert ".ppt" in SUPPORTED_DOCUMENT_EXTENSIONS
    assert ".xls" in SUPPORTED_DOCUMENT_EXTENSIONS
