import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.calculator import calculator
from tools.data_lookup import data_lookup


class TestCalculator:
    def test_basic_addition(self):
        result = calculator.invoke({"expression": "2 + 3"})
        assert "5.0" in result

    def test_multiplication(self):
        result = calculator.invoke({"expression": "6 * 7"})
        assert "42.0" in result

    def test_percentage(self):
        result = calculator.invoke({"expression": "(15/100) * 4200"})
        assert "630.0" in result

    def test_division(self):
        result = calculator.invoke({"expression": "100 / 4"})
        assert "25.0" in result

    def test_division_by_zero(self):
        result = calculator.invoke({"expression": "1 / 0"})
        assert "error" in result.lower()

    def test_invalid_expression(self):
        result = calculator.invoke({"expression": "abc"})
        assert "error" in result.lower()


class TestDataLookup:
    def test_known_topic_rag(self):
        result = data_lookup.invoke({"topic": "rag"})
        assert "rag" in result.lower()
        assert "retrieval" in result.lower()

    def test_known_topic_langsmith(self):
        result = data_lookup.invoke({"topic": "langsmith"})
        assert "tracing" in result.lower()

    def test_unknown_topic(self):
        result = data_lookup.invoke({"topic": "quantum computing"})
        assert "no matching" in result.lower()

    def test_case_insensitive(self):
        result = data_lookup.invoke({"topic": "RAG"})
        assert "rag" in result.lower()
