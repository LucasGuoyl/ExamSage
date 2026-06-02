"""Hold-out evaluation on Gaokao-style data.

Splits past papers by year: years before --test_year become "training" data
(fed into the pipeline as past papers + slides); the test year is held out
and we measure top-K coverage and MRR.

Expected directory layout:
    data_samples/gaokao_math/
        slides/                  (textbook / 教材 / 复习资料)
        past_papers/
            gaokao_2010.json
            gaokao_2011.json
            ...
            gaokao_2024.json

Usage:
    python examples/run_gaokao.py \\
        --course data_samples/gaokao_math \\
        --test_year 2024 \\
        --top_k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from rich.console import Console
from rich.table import Table

from exam_predictor import ExamPredictor
from exam_predictor.evaluator import evaluate_topk_coverage, random_baseline
from exam_predictor.ingest import discover_past_papers
from exam_predictor.chunker import questions_from_records


console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--test_year", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--coverage_threshold", type=float, default=0.55)
    parser.add_argument("--out", default="output/gaokao_eval")
    args = parser.parse_args()

    course_dir = Path(args.course)

    # 1. Split past papers into train / test by year
    all_records = discover_past_papers(course_dir)
    train_records = [r for r in all_records if r.get("year") != args.test_year]
    test_records = [r for r in all_records if r.get("year") == args.test_year]
    if not test_records:
        raise SystemExit(f"No test data for year {args.test_year}")
    console.print(f"[bold]Split:[/bold] train={len(train_records)}  test={len(test_records)}")

    # 2. Temporarily move test data out so the pipeline doesn't see it.
    #    We do this by creating a filtered staging dir.
    stage = Path(args.out) / "_staging"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "slides").symlink_to((course_dir / "slides").resolve(), target_is_directory=True) \
        if not (stage / "slides").exists() else None
    pp_dir = stage / "past_papers"
    pp_dir.mkdir(exist_ok=True)
    (pp_dir / "train.json").write_text(
        json.dumps(train_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # also pass through tutorials/syllabus if present
    for extra in ("tutorials", "syllabus.md"):
        src = course_dir / extra
        dst = stage / extra
        if src.exists() and not dst.exists():
            if src.is_dir():
                dst.symlink_to(src.resolve(), target_is_directory=True)
            else:
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # 3. Run the pipeline
    predictor = ExamPredictor.from_config_file(args.config)
    report = predictor.predict(stage, course_name=f"gaokao_eval_holdout_{args.test_year}")
    predictor.save_report(report, args.out)

    # 4. Evaluate against held-out test questions
    test_qs = questions_from_records(test_records)
    # We need access to the same chunks the pipeline used. Re-ingest here:
    from exam_predictor.ingest import ingest_directory
    from exam_predictor.chunker import chunk_units
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    chunk_cfg = cfg.get("chunking", {})
    slide_units = ingest_directory(stage, "slides")
    chunks = chunk_units(
        slide_units,
        chunk_size=chunk_cfg.get("chunk_size", 800),
        overlap=chunk_cfg.get("chunk_overlap", 100),
    )

    eval_result = evaluate_topk_coverage(
        test_qs, report.predictions, chunks, predictor.embedder,
        top_k=args.top_k,
        coverage_threshold=args.coverage_threshold,
    )
    baseline = random_baseline(
        test_qs, report.predictions, chunks, predictor.embedder,
        top_k=args.top_k,
        coverage_threshold=args.coverage_threshold,
    )

    # 5. Pretty-print
    table = Table(title=f"Hold-out evaluation (test year {args.test_year}, top_k={args.top_k})")
    table.add_column("Metric")
    table.add_column("Pipeline")
    table.add_column("Random baseline")
    table.add_row(
        "Coverage",
        f"{eval_result.coverage:.3f}",
        f"{baseline['mean_coverage']:.3f} ± {baseline['std_coverage']:.3f}",
    )
    table.add_row("MRR", f"{eval_result.mrr:.3f}", "—")
    console.print(table)

    # 6. Save detailed eval
    out_path = Path(args.out) / "eval_detail.json"
    out_path.write_text(json.dumps({
        "test_year": args.test_year,
        "top_k": args.top_k,
        "coverage": eval_result.coverage,
        "mrr": eval_result.mrr,
        "baseline": baseline,
        "per_question": eval_result.per_question,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Saved:[/green] {out_path}")


if __name__ == "__main__":
    main()
