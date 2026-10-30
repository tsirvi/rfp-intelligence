"""Run a fixed set of questions through the Q&A agents and write docs/qa_log.md (the sample Q&A log).

python eval/run_qa_log.py [--out docs/qa_log.md]

Uses the same path as the API and UI (Orchestrator.ask), so each entry shows how it was answered:
instantly from the validated record, written by the model, or quoted passages after the time budget.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

from agents.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parent.parent
BIDS = ["Bid1", "Bid2"]  # the assignment's sample bids; a question naming one of them narrows it further

# The brief's five example questions first, then questions of other kinds (overview, comparison table,
# single fields, free-form).
QUESTIONS = [
    "What is the submission deadline for Bid1 after all addendums?",
    "Which affidavits are required for the Dell laptop bid?",
    "Compare the warranty requirements of both bids.",
    "Is a bid bond required, and if so, how much?",
    "What changed in Addendum 2 compared to the original RFP?",
    "What are the bids about?",
    "Compare Bid1 and Bid2",
    "Who is the procurement contact for Bid2?",
    "Is the pre-proposal meeting for Bid1 mandatory?",
    "What is the contract term for Bid1?",
    "What insurance coverage is required for Bid1?",
    "Can vendors submit proposals by email for Bid1?",
]
MODE = {
    "record": "instant, from the validated record",
    "llm": "written by the model",
    "passages": "quoted passages (model did not answer in time)",
    "chat": "help text",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "docs" / "qa_log.md"))
    out = Path(parser.parse_args().out)

    orch = Orchestrator()
    orch.llm.check()
    orch.preload()
    entries = []
    for i, question in enumerate(QUESTIONS, start=1):
        started = time.perf_counter()
        answer = orch.ask(question, BIDS)
        seconds = time.perf_counter() - started
        print(f"[{i}/{len(QUESTIONS)}] {seconds:5.1f}s {answer.mode:8} {question}", flush=True)
        sources = "\n".join(f"- {c.label()}" for c in answer.citations) or "- none"
        entries.append(
            f"### {i}. {question}\n\n{answer.answer}\n\n**Sources**\n\n{sources}\n\n"
            f"*Bids: {', '.join(answer.bids) or '-'} · {MODE.get(answer.mode, answer.mode)} · "
            f"confidence {answer.confidence:.2f} · {seconds:.0f} s · trace `{answer.run_id or '-'}`*\n"
        )

    header = (
        "# Sample Q&A log\n\n"
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python eval/run_qa_log.py` with "
        f"`{orch.llm.model}` (Ollama, CPU) over the sample bids ({', '.join(BIDS)}). Each answer states how it was "
        "produced. Answers are generated from the documents and are informational only; check the cited "
        "pages before relying on them.\n\n"
        "Traces for each answer are in `outputs/traces/<trace id>.jsonl`.\n\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n".join(entries), "utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
