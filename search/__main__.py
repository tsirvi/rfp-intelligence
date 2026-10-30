"""Index CLI.

python -m search index data/Bid1 data/Bid2      # incremental: unchanged files are skipped
python -m search index data/Bid2 --force        # re-parse everything (after a parser change)
python -m search query "submission deadline" --bid Bid1          # hybrid (add --rerank for the cross-encoder)
python -m search query "new due date" --bid Bid1 --addendum 2    # metadata filters
python -m search query "210-BLYZ" --mode bm25                    # one retriever alone, for inspection
python -m search stats
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from search.hybrid import HybridSearcher
from search.store import IndexStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m search")
    sub = parser.add_subparsers(dest="command", required=True)
    idx = sub.add_parser("index", help="index one or more bid folders")
    idx.add_argument("folders", nargs="+", type=Path)
    idx.add_argument("--force", action="store_true", help="re-parse files even if unchanged")
    q = sub.add_parser("query", help="search the index")
    q.add_argument("text")
    q.add_argument("--bid")
    q.add_argument("--type", dest="doc_type", action="append", help="doc type filter, repeatable")
    q.add_argument("--addendum", dest="addendum_number", type=int, action="append", help="addendum number filter")
    q.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    q.add_argument("--rerank", action="store_true", help="cross-encoder reranking (off by default)")
    q.add_argument("--no-expand", action="store_true")
    q.add_argument("-k", type=int, default=5)
    sub.add_parser("stats", help="documents and chunks per bid")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    store = IndexStore()

    if args.command == "index":
        for folder in args.folders:
            r = store.index_folder(folder, force=args.force)
            print(
                f"{r.bid_id}: added {len(r.added)}, updated {len(r.updated)}, unchanged {len(r.unchanged)}, "
                f"removed {len(r.removed)}, failed {len(r.failures)}, {r.chunks_added} chunks, {r.seconds}s"
            )
            for f in r.failures:
                print(f"  FAILED {f.file_name}: {f.error}")
    elif args.command == "query":
        filters = {"bid_id": args.bid, "doc_type": args.doc_type, "addendum_number": args.addendum_number}
        if args.mode == "hybrid":
            result = HybridSearcher(store).search(
                args.text, filters, args.k, rerank=args.rerank, expand=not args.no_expand
            )
            print(f"variants: {result.variants}\ntimings: {result.timings}")
            ranked = [(h.chunk, h.score) for h in result.hits]
        else:
            search = store.dense_search if args.mode == "dense" else store.bm25_search
            ranked = search(args.text, args.k, filters)
        for chunk, score in ranked:
            print(f"\n[{score:.3f}] {chunk.bid_id} | {chunk.citation} | {chunk.section or '-'}")
            print("  " + chunk.text[:400].replace("\n", "\n  "))
    else:
        per_bid = Counter(c.bid_id for c in store.chunks())
        for bid in store.bids():
            docs = store.documents(bid)
            print(f"{bid}: {len(docs)} documents, {per_bid[bid]} chunks")
            for d in docs:
                print(f"  {d.doc_type.value:<9} {d.file_name}")


if __name__ == "__main__":
    main()
