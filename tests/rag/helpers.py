from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject


class FakeEmbeddingProvider:
    """Orthogonal topic vectors make ranking independent of a live model."""

    def embed_text(self, text: str) -> list[float]:
        lower = text.lower()
        return [
            float(any(word in lower for word in ("protocol", "models", "tools"))),
            float(any(word in lower for word in ("plants", "photosynthesis"))),
            0.1,
        ]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


def write_pdf(path: Path) -> None:
    writer = PdfWriter()
    for text in ("Models connect to tools using a protocol.", "Plants use photosynthesis."):
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 250 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)
