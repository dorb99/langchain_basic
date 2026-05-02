import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retrieval.query_plan import chroma_source_filter


class TestChromaSourceFilter:
    def test_none(self):
        assert chroma_source_filter(None) is None

    def test_source_basename(self):
        assert chroma_source_filter({"source": "docs/sample.txt"}) == {"source": "sample.txt"}

    def test_ignores_other_keys(self):
        assert chroma_source_filter({"source": "a.txt", "other": "x"}) == {"source": "a.txt"}

    def test_drops_path_traversal(self):
        assert chroma_source_filter({"source": "../etc/passwd"}) is None
