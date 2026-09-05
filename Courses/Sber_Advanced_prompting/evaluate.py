"""Reproducible measurement of the SPIEF pipeline on the golden set.

Runs every golden question through the agent, grades each answer with an
LLM judge, and reports both quality and cost (tokens, tokens per correct
answer). Results are written to `reports/` so runs can be compared before
and after an optimization.

Usage:
    uv run evaluate.py [run-label]
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from src.config import Settings
from src.context import AgentContext
from src.judge import answer_question, judge_answer
from src.llm import get_model
from src.models import Embedder
from src.retrieval import Retrieval, load_data

ROOT = Path(__file__).parent
GOLDEN = ROOT / "data" / "golden.xlsx"
REPORTS = ROOT / "reports"
CONCURRENCY = 4


def load_golden(path: Path) -> list[dict[str, str]]:
    """Reads the golden set from Excel.

    Row 0 of the sheet is a title banner and row 1 is the real header, so the
    header is taken explicitly rather than inferred.

    Args:
        path: Path to `golden.xlsx`.

    Returns:
        One dict per question, keyed by the sheet's column names.
    """
    raw = pl.read_excel(path, has_header=False)
    header = [str(v) for v in raw.row(1)]
    records = [dict(zip(header, (str(v) for v in row))) for row in raw.slice(2).rows()]
    return [r for r in records if r.get("Вопрос")]


def summarize(results: list[dict]) -> dict:
    """Aggregates per-question results into the reported metrics.

    Args:
        results: Per-question records produced by the evaluation run.

    Returns:
        Quality metrics (accuracy) and cost metrics (token totals, tokens per
        question, and tokens per correct answer as the unit cost of a result).
    """
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    tokens = sum(r["total_tokens"] for r in results)
    return {
        "questions": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "input_tokens": sum(r["input_tokens"] for r in results),
        "output_tokens": sum(r["output_tokens"] for r in results),
        "total_tokens": tokens,
        "tokens_per_question": tokens / total if total else 0.0,
        "tokens_per_correct_answer": tokens / passed if passed else None,
    }


def write_report(label: str, summary: dict, results: list[dict]) -> Path:
    """Writes the run's metrics and per-question detail to `reports/`.

    Args:
        label: Name identifying this run (e.g. "baseline").
        summary: Aggregated metrics from `summarize`.
        results: Per-question records.

    Returns:
        Path of the written Markdown report.
    """
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{label}.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# Evaluation report — {label}",
        "",
        f"Run at {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Questions | {summary['questions']} |",
        f"| Passed | {summary['passed']} |",
        f"| Accuracy | {summary['accuracy']:.1%} |",
        f"| Input tokens | {summary['input_tokens']} |",
        f"| Output tokens | {summary['output_tokens']} |",
        f"| Total tokens | {summary['total_tokens']} |",
        f"| Tokens per question | {summary['tokens_per_question']:.0f} |",
        f"| Tokens per correct answer | "
        f"{summary['tokens_per_correct_answer']:.0f} |"
        if summary["tokens_per_correct_answer"]
        else "| Tokens per correct answer | n/a |",
        "",
        "## Per-question results",
        "",
        "| # | Verdict | Tokens | Question | Judge comment |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        question = r["question"].replace("|", "\\|")
        reason = r["reason"].replace("|", "\\|")
        mark = "PASS" if r["ok"] else "FAIL"
        lines.append(
            f"| {r['number']} | {mark} | {r['total_tokens']} | {question} | {reason} |"
        )

    path = REPORTS / f"{label}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def main():
    """Runs the golden-set evaluation and writes a report."""
    load_dotenv()
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    settings = Settings()

    data = load_data(ROOT / "data" / "SPIEF_txt" / "SPIEF_txt")
    if settings.max_documents is not None:
        data = data[: settings.max_documents]

    retrieval = Retrieval(settings.qdrant, Embedder(settings.embedder))
    await retrieval.start(data)

    model = get_model()
    context = AgentContext(retrieval=retrieval)
    golden = load_golden(GOLDEN)
    print(f"Evaluating {len(golden)} question(s) as '{label}'...\n")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def run_case(case: dict[str, str]) -> dict:
        """Answers and grades a single golden-set question."""
        question, reference = case["Вопрос"], case["Эталонный ответ"]
        async with semaphore:
            answer, usage = await answer_question(model, context, question)
            text = answer.answer if answer else ""
            verdict = await judge_answer(model, question, reference, text)
        return {
            "number": case["№"],
            "question": question,
            "reference": reference,
            "answer": text,
            "sources": [s.model_dump() for s in answer.sources] if answer else [],
            "found": answer.found if answer else False,
            "ok": verdict.ok,
            "reason": verdict.reason,
            **usage,
        }

    results = await asyncio.gather(*(run_case(case) for case in golden))
    results = sorted(results, key=lambda r: int(r["number"]))

    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"[{mark}] #{r['number']} ({r['total_tokens']} tok) {r['question']}")
        print(f"       judge: {r['reason']}")
        if not r["ok"]:
            print(f"       answer: {r['answer'][:300]}")
        print()

    summary = summarize(results)
    path = write_report(label, summary, results)

    print("=" * 60)
    print(f"Accuracy:                  {summary['passed']}/{summary['questions']} "
          f"= {summary['accuracy']:.1%}")
    print(f"Total tokens:              {summary['total_tokens']}")
    print(f"Tokens per question:       {summary['tokens_per_question']:.0f}")
    if summary["tokens_per_correct_answer"]:
        print(f"Tokens per correct answer: {summary['tokens_per_correct_answer']:.0f}")
    print(f"Report written to:         {path}")


if __name__ == "__main__":
    asyncio.run(main())
