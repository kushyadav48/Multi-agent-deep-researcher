"""UTF-8 TXT/Markdown and page-by-page PDF loading; no OCR."""

from pathlib import Path

from pypdf import PdfReader

from rag.models import Document, DocumentPage


class DocumentLoadError(ValueError):
    """A supported document could not be read or has no extractable text."""


def load_document(path: str | Path) -> Document:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Document file not found: {source}")
    extension = source.suffix.lower()
    if extension not in {".pdf", ".txt", ".md"}:
        raise ValueError(f"Unsupported document extension: {extension}; use .pdf, .txt, or .md")

    if extension == ".pdf":
        try:
            with source.open("rb") as stream:
                reader = PdfReader(stream)
                if reader.is_encrypted and not reader.decrypt(""):
                    raise DocumentLoadError(f"PDF requires a password: {source}")
                pages = tuple(
                    DocumentPage(page.extract_text() or "", index)
                    for index, page in enumerate(reader.pages, start=1)
                )
        except DocumentLoadError:
            raise
        except Exception as error:
            raise DocumentLoadError(f"Unable to read PDF {source}: {error}") from error
    else:
        try:
            pages = (DocumentPage(source.read_text(encoding="utf-8")),)
        except (OSError, UnicodeError) as error:
            raise DocumentLoadError(f"Unable to read UTF-8 document {source}: {error}") from error

    if not any(page.text.strip() for page in pages):
        raise DocumentLoadError(f"Document is empty or has no extractable text (no OCR): {source}")
    return Document(str(source), pages)
