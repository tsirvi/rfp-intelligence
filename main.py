"""RFP Intelligence: one command per mode.

python main.py --bid ./data/Bid1                     index the folder, extract the 20 fields -> outputs/Bid1.json
python main.py --ask "What is the submission deadline for Bid1 after all addendums?"
python main.py --ask "Compare the warranty requirements of both bids." --bids Bid1 Bid2
python main.py --compare Bid1 Bid2                   side-by-side report -> outputs/comparisons/Bid1_vs_Bid2.md
python main.py --go-no-go Bid1                       bid / no-bid recommendation -> outputs/go_no_go/Bid1.md
python main.py --usage                               cost and latency of the runs so far
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
    parser.add_argument("--compare", nargs="+", metavar="BID", help="Compare these extracted bids side by side")
    parser.add_argument("--no-summary", action="store_true", help="--compare without the model-written summary")
    parser.add_argument("--go-no-go", metavar="BID", help="Bid / no-bid recommendation for an extracted bid")
    parser.add_argument("--profile", type=Path, help="Company profile for --go-no-go (default company_profile.yaml)")
    parser.add_argument("--usage", action="store_true", help="Show cost and latency per kind of run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show every agent step")
    args = parser.parse_args()
    if not (args.bid or args.ask or args.compare or args.go_no_go or args.usage):
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    from agents.llm import LLMError
    from agents.orchestrator import Orchestrator  # imported late so --help stays fast

    orchestrator = Orchestrator()
    if args.usage:
        from agents.usage import summary

        print(json.dumps(summary(orchestrator.outputs_dir), indent=2))
    if args.go_no_go:  # rule-based: no LLM needed
        from agents.go_no_go_agent import load_profile, to_markdown

        decision = orchestrator.go_no_go(args.go_no_go, load_profile(args.profile) if args.profile else None)
        print(to_markdown(decision))
    needs_llm = args.bid or args.ask or (args.compare and not args.no_summary)
    if not needs_llm:
        if args.compare:
            print_comparison(orchestrator.compare(args.compare, summary=False))
        return
    try:
        orchestrator.llm.check()
    except LLMError as exc:
        sys.exit(f"LLM not ready: {exc}")
    if args.compare:
        print_comparison(orchestrator.compare(args.compare))
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
        info = {"mode": answer.mode, "cached": answer.cached, "bids": answer.bids, "confidence": answer.confidence}
        print(json.dumps(info | {"seconds": answer.usage.get("seconds"), "run_id": answer.run_id}))


def print_comparison(result) -> None:
    from agents.comparison_agent import to_markdown

    print(to_markdown(result))
    print(f"Wrote outputs/comparisons/{'_vs_'.join(result.bids)}.md and .json")


if __name__ == "__main__":
    main()
