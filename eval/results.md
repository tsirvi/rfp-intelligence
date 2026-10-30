# Retrieval evaluation

26 questions over 2 bids, each search filtered to its bid. Latency is per query, with models already loaded.

| config | recall@1 | recall@3 | recall@5 | mrr | avg_latency_s |
|---|---|---|---|---|---|
| bm25 | 0.500 | 0.750 | 0.769 | 0.689 | 0.006 |
| dense | 0.462 | 0.558 | 0.654 | 0.608 | 0.035 |
| hybrid | 0.615 | 0.673 | 0.750 | 0.750 | 0.041 |
| +expand | 0.615 | 0.712 | 0.750 | 0.756 | 0.154 |
| +dedupe (default) | 0.615 | 0.712 | 0.808 | 0.759 | 0.162 |
| +rerank | 0.596 | 0.750 | 0.788 | 0.747 | 6.120 |

## Misses in the best configuration (first relevant rank > 3)

- q01 (Bid1): What is the final submission deadline for proposals? — passage ranks [6, None]
- q02 (Bid1): What was the original solicitation due date before any addendum? — passage ranks [9]
- q04 (Bid1): Who is the buyer for this solicitation and what is their email address? — passage ranks [None]
- q08 (Bid1): How should vendors submit their responses? — passage ranks [5]
- q09 (Bid1): By when must vendor questions be submitted? — passage ranks [None]
- q16 (Bid2): When are proposals due for the Dell laptop solicitation? — passage ranks [4, 5]
