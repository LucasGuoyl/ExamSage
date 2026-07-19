"""ExamSage: a web-grounded university exam revision agent."""

__version__ = "0.4.0"

from .agent import ExamSageAgent
from .pipeline import ExamPredictor
from .schema import Chunk, ExamQuestion, GeneratedQuestion, KnowledgePointScore

__all__ = [
    "ExamSageAgent",
    "ExamPredictor",
    "Chunk",
    "ExamQuestion",
    "KnowledgePointScore",
    "GeneratedQuestion",
]
