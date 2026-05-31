# Evaluation

## Evaluation Design

The system was evaluated through automatic benchmark-based evaluation. Each benchmark tests a different component of the system.

| Evaluation Lane | Dataset | Component Tested | Metrics |
|---|---|---|---|
| Judgment prediction | ILDC | Judgment prediction classifier | Accuracy, Macro F1 |
| Layperson statute identification | ILSIC-Lay | Statute/source routing | Micro F1, Macro F1, Hit@1, MRR |
| Statute retrieval | IL-PCSR | Statute retrieval lane | Macro F1@3, Recall@k, MRR, MAP |
| Precedent retrieval | IL-PCSR | Case-law retrieval lane | Recall@k, MRR, MAP |

## Final Results

| Task | Dataset | Result |
|---|---|---|
| Judgment prediction | ILDC | Accuracy 61.24%; Macro F1 61.15% |
| Layperson statute identification | ILSIC-Lay | Micro F1 21.36%; Macro F1 20.50%; Hit@1 11.48%; MRR 0.3213 |
| Statute retrieval | IL-PCSR | Macro F1@3 0.0912; Recall@10 0.1846; MRR 0.2263; MAP 0.0971 |
| Precedent retrieval | IL-PCSR | Recall@10 0.3327; MRR 0.2860; MAP 0.1797 |

## Interpretation

The judgment prediction result is moderate and useful only as a triage signal. The statute-identification and retrieval results show that retrieval remains the main bottleneck. The project is therefore best understood as a product-oriented, source-routed legal RAG prototype with benchmark evaluation, rather than a state-of-the-art benchmark system.

## Limitations

- The system is not legal advice.
- Retrieval performance is modest on strict benchmarks.
- The full artifacts and datasets are not included in GitHub due to size and licensing constraints.
- Uploaded-document Q/A is not benchmarked through a public dataset in the current version.
- Legal source currency must be verified before real-world use.
