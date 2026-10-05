import unittest

from rag.chunking import TextChunker
from rag.models import Document, DocumentPage


class ChunkingTests(unittest.TestCase):
    def test_determinism_overlap_and_coverage(self):
        document = Document("source.txt", (DocumentPage("abcdefghijklmnop"),))
        chunker = TextChunker(chunk_size=6, overlap=2)
        chunks = chunker.chunk(document)
        self.assertEqual([chunk.text for chunk in chunks], ["abcdef", "efghij", "ijklmn", "mnop"])
        self.assertEqual(chunks, chunker.chunk(document))
        self.assertEqual(len({chunk.id for chunk in chunks}), 4)
        changed = Document("source.txt", (DocumentPage("different text"),))
        self.assertNotEqual(chunks[0].id, chunker.chunk(changed)[0].id)

    def test_invalid_configuration(self):
        for size, overlap in ((0, 0), (-1, 0), (10, -1), (10, 10), (10, 11), (2.5, 0), (10, True)):
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    TextChunker(size, overlap)

    def test_metadata_pages_and_empty_windows(self):
        metadata = {"title": "Paper", "reviewed": True, "year": 2026}
        document = Document("paper.pdf", (
            DocumentPage("abcdef", page=1),
            DocumentPage("      ", page=2),
            DocumentPage("ghijkl", page=3),
        ), metadata=metadata)
        metadata["title"] = "mutated"
        chunks = TextChunker(4, 1).chunk(document)
        self.assertEqual([chunk.page for chunk in chunks], [1, 1, 3, 3])
        self.assertEqual([chunk.chunk_index for chunk in chunks], list(range(4)))
        for chunk in chunks:
            self.assertEqual(chunk.source, "paper.pdf")
            self.assertEqual(chunk.metadata["title"], "Paper")
            self.assertTrue(chunk.text.strip())
            with self.assertRaises(TypeError):
                chunk.metadata["title"] = "mutated"
        self.assertEqual(TextChunker().chunk(Document("empty.txt", (DocumentPage("  "),))), [])

    def test_no_redundant_overlap_chunk_at_end(self):
        self.assertEqual(len(TextChunker(6, 2).chunk(Document("a", (DocumentPage("abcdef"),)))), 1)
