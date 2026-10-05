import tempfile
import unittest
from pathlib import Path

from rag.loaders import DocumentLoadError, load_document
from tests.rag.helpers import write_pdf


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_utf8_txt_and_markdown(self):
        for extension in (".txt", ".md"):
            with self.subTest(extension=extension):
                path = self.root / ("document" + extension)
                path.write_text("MCP connects tools. \u03bb", encoding="utf-8")
                document = load_document(path)
                self.assertEqual(document.source, str(path.resolve()))
                self.assertEqual(document.pages[0].text, "MCP connects tools. \u03bb")
                self.assertIsNone(document.pages[0].page)

    def test_pdf_page_text_and_numbers(self):
        path = self.root / "paper.pdf"
        write_pdf(path)
        document = load_document(path)
        self.assertEqual([page.page for page in document.pages], [1, 2])
        self.assertIn("protocol", document.pages[0].text)
        self.assertIn("photosynthesis", document.pages[1].text)

    def test_unsupported_and_missing(self):
        path = self.root / "document.html"
        path.write_text("unsupported", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_document(path)
        with self.assertRaises(FileNotFoundError):
            load_document(self.root / "missing.txt")

    def test_empty_corrupt_and_invalid_utf8(self):
        for name, contents, message in (
            ("empty.txt", b" \n\t", "empty"),
            ("broken.pdf", b"not a PDF", "PDF"),
            ("bad.txt", b"\xff", "UTF-8"),
        ):
            with self.subTest(name=name):
                path = self.root / name
                path.write_bytes(contents)
                with self.assertRaisesRegex(DocumentLoadError, message):
                    load_document(path)

    def test_blank_pdf_has_no_extractable_text(self):
        from pypdf import PdfWriter
        path = self.root / "scan.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with path.open("wb") as output:
            writer.write(output)
        with self.assertRaisesRegex(DocumentLoadError, "empty|extractable"):
            load_document(path)
