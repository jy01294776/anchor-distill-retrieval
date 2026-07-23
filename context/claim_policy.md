# Claim-to-evidence policy

| Claim | Required evidence |
|---|---|
| Fine-tuned a compact sentence encoder | completed training manifest and model hash |
| Performed soft-target knowledge distillation | accepted token-mass audit plus KL training manifest |
| Improved retrieval quality | held-out test metric and bootstrap interval |
| Reduced latency or cost | reproducible benchmark with hardware/model metadata |
| Built a recoverable async system | fault-injection integration tests |
| Packaged the system | successful wheel and container builds |
| Deployed the system | production URL, deployment record, and monitoring evidence |

No improvement, scalability, or production claim may be inferred from code
presence alone.

Current allowed public-project claims:

- Fine-tuned a compact encoder on 77 public labels.
- Improved BANKING77 Recall@1 by 3.57 percentage points in the current run,
  with paired 95% bootstrap interval [2.76, 4.38] percentage points.
- Built and locally tested recoverable job orchestration, including injected
  worker-crash checkpoint recovery.
- Measured local MPS batch performance with the hardware, batch size, sample
  count, repeats, latency, throughput, and RSS recorded.

Current prohibited claims:

- Soft distillation improves quality or label efficiency.
- The system reduces production cost.
- The container or Compose stack has been built successfully.
- The system is deployed or productized.
