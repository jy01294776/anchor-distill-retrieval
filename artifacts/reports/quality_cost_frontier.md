# Quality, Performance, and Cost Frontier

| System | Signal | Human labels | Recall@1 | MRR | Params | texts/s | Peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniLM zero-shot | zero-shot | 0 | 0.6237 | 0.7251 | 22.7M | 2315.8 | 627 |
| MiniLM gold 1-shot | gold | 77 | 0.6594 | 0.7573 | 22.7M | 2315.8 | 627 |
| MiniLM gold 4-shot | gold | 308 | 0.7075 | 0.8001 | 22.7M | 2315.8 | 627 |
| MiniLM gold 16-shot | gold | 1232 | 0.8269 | 0.8885 | 22.7M | 2315.8 | 627 |
| MiniLM gold full | gold | 10003 | 0.9192 | 0.9504 | 22.7M | 2315.8 | 627 |
| MiniLM listwise hard KD | teacher hard | 0 | 0.6519 | 0.7509 | 22.7M | 2165.0 | 627 |
| MiniLM listwise soft KD | teacher soft | 0 | 0.6318 | 0.7272 | 22.7M | 2165.0 | 627 |
| MiniLM listwise hybrid | gold + teacher soft | 77 | 0.6623 | 0.7582 | 22.7M | 2165.0 | 627 |
| Sentence-T5-XL zero-shot | zero-shot | 0 | 0.6516 | 0.7497 | 1241.7M | 90.4 | 2905 |

## Student versus large encoder

The listwise-hard MiniLM point estimate differs from Sentence-T5-XL by +0.0003 Recall@1; the paired 95% interval is [-0.0146, +0.0146].
It uses 54.7x fewer parameters, measured 24.0x higher local throughput, and used 4.6x less peak RSS.

## Teacher cost

The observed teacher run cost $0.04733 for 1,920 pairs. At the same token mix, that projects to $24.65 per million pairs or $246.49 per million queries with 10 candidates.

## Interpretation limits

- Local MPS throughput is a machine-specific batch benchmark, not a service SLA.
- The test-set comparisons are exploratory because multiple candidate systems were evaluated.
- The observed confidence interval does not establish formal equivalence.
- Dollar projections use observed teacher token mix and captured API prices.
