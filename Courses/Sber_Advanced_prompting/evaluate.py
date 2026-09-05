import asyncio
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from src.config import Settings
from src.context import AgentContext
from src.judge import answer_question, judge_answer
from src.llm import get_model
from src.models import Embedder
from src.retrieval import Retrieval, load_data

GOLDEN = Path(__file__).parent / "data" / "golden.xlsx"
CONCURRENCY = 4


def load_golden(path: Path) -> list[dict[str, str]]:
    """Reads the golden set: row 0 is a title banner, row 1 the real header."""
    raw = pl.read_excel(path, has_header=False)
    header = [str(v) for v in raw.row(1)]
    rows = raw.slice(2).rows()
    records = [dict(zip(header, (str(v) for v in row))) for row in rows]
    return [r for r in records if r.get("Вопрос")]


async def main():
    load_dotenv()
    settings = Settings()

    data = load_data(Path(__file__).parent / "data" / "SPIEF_txt" / "SPIEF_txt")
    if settings.max_documents is not None:
        data = data[: settings.max_documents]

    retrieval = Retrieval(settings.qdrant, Embedder(settings.embedder))
    await retrieval.start(data)

    model = get_model()
    context = AgentContext(retrieval=retrieval)

    golden = load_golden(GOLDEN)
    print(f"Evaluating {len(golden)} question(s)...\n")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def run_case(case: dict[str, str]):
        question, reference = case["Вопрос"], case["Эталонный ответ"]
        async with semaphore:
            answer = await answer_question(model, context, question)
            verdict = await judge_answer(model, question, reference, answer)
        return case["№"], question, answer, verdict

    results = await asyncio.gather(*(run_case(case) for case in golden))

    for number, question, answer, verdict in results:
        mark = "PASS" if verdict.ok else "FAIL"
        print(f"[{mark}] #{number} {question}")
        print(f"       judge: {verdict.reason}")
        if not verdict.ok:
            print(f"       answer: {answer[:300]}")
        print()

    passed = sum(1 for *_, verdict in results if verdict.ok)
    total = len(results)
    print("=" * 50)
    print(f"Accuracy: {passed}/{total} = {passed / total:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
