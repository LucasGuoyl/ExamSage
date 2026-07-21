from pathlib import Path

from exam_predictor.workspace.models import ScanPolicy


SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".doc": "document",
    ".docx": "document",
    ".ppt": "presentation",
    ".pptx": "presentation",
    ".xls": "spreadsheet",
    ".xlsx": "spreadsheet",
    ".csv": "tabular",
    ".tsv": "tabular",
    ".json": "structured_data",
    ".yaml": "structured_data",
    ".yml": "structured_data",
    ".md": "text",
    ".txt": "text",
    ".html": "text",
    ".htm": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".zip": "archive",
}

DEFAULT_SCAN_POLICY = ScanPolicy()


def classify_format(filename: str) -> str | None:
    return SUPPORTED_EXTENSIONS.get(Path(filename).suffix.casefold())
