"""Run the full pipeline on a single university course directory.

Usage:
    python examples/run_university.py --course data_samples/my_course

Expects:
    <course_dir>/
        slides/        (PDF / PPTX / MD)
        past_papers/   (JSON / PDF / MD)
        tutorials/     (optional)
        syllabus.md    (optional)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# add parent to path so we can import the package without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exam_predictor import ExamPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", required=True, help="Path to course directory")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument("--name", default=None, help="Course display name")
    args = parser.parse_args()

    predictor = ExamPredictor.from_config_file(args.config)
    report = predictor.predict(args.course, course_name=args.name)
    predictor.save_report(report, args.out)


if __name__ == "__main__":
    main()
