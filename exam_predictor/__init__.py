"""Exam Predictor — predict exam topics and generate practice questions from course materials."""

__version__ = "0.1.0"

from .pipeline import ExamPredictor
from .schema import Chunk, ExamQuestion, KnowledgePointScore, GeneratedQuestion

__all__ = [
    "ExamPredictor",
    "Chunk",
    "ExamQuestion",
    "KnowledgePointScore",
    "GeneratedQuestion",
]
