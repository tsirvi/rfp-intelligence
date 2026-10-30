"""RFP Intelligence: one command per mode.

python main.py --bid ./data/Bid1                     index the folder, extract the 20 fields -> outputs/Bid1.json
python main.py --ask "What is the submission deadline for Bid1 after all addendums?"
python main.py --ask "Compare the warranty requirements of both bids." --bids Bid1 Bid2
"""

import argparse
import json
import logging
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="RFP Intelligence Platform")
    parser.add_argument("--bid", type=Path, help="Bid folder to index and extract")
    parser.add_argument("--bid-id", help="Bid id (defaults to the folder name)")
    parser.add_argument("--ask", help="Question to answer over the indexed bids")
    parser.add_argument("--bids", nargs="+", help="Restrict --ask to these bid ids")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show every agent step")
    args = parser.parse_args()
    if not (args.bid or args.ask):
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    from agents.llm import LLMError
    from agents.orchestrator import Orchestrator  # imported late so --help stays fast

    orchestrator = Orchestrator()
    try:
        orchestrator.llm.check()
    except LLMError as exc:
        sys.exit(f"LLM not ready: {exc}")
    if args.bid:
        if not args.bid.is_dir():
            sys.exit(f"Not a folder: {args.bid}")
        record = orchestrator.extract_bid(args.bid_id or args.bid.name, args.bid)
        v = record.validation
        print(
            f"\n{record.bid_id}: {v.passed} passed, {v.failed} failed, {v.not_found} not found "
            f"in {record.run.get('seconds')}s ({record.run.get('llm_calls')} LLM calls)"
        )
        for name, f in record.fields.items():
            if name != "Bid Summary":
                print(f"  {name:<40} {(f.value or '—')[:70]}")
        print(f"\nSummary: {record.fields['Bid Summary'].value}")
        print(f"\nWrote outputs/{record.bid_id}.json and .md; trace: {record.run.get('trace')}")
    if args.ask:
        answer = orchestrator.ask(args.ask, args.bids)
        print(f"\nQ: {answer.question}\nA: {answer.answer}")
        for c in answer.citations:
            print(f"   - {c.label()}")
        print(
            json.dumps(
                {"mode": answer.mode, "bids": answer.bids, "confidence": answer.confidence, "run_id": answer.run_id}
            )
        )


if __name__ == "__main__":
    main()
