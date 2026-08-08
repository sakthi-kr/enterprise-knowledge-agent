# Data

The project uses a local subset of [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench). Raw benchmark archives and generated processing outputs are intentionally excluded from Git because they are reproducible and can be large.

The recommended working subset uses all Confluence, Jira, and GitHub source archives from the v1.0.0 benchmark release. This keeps the corpus focused on engineering and operational knowledge while remaining practical for local development.

Expected local layout:

```text
data/
├── raw/
│   └── enterprise_rag_bench/
│       ├── archives/
│       │   └── *_slice_*.zip
│       └── questions.jsonl
└── processed/
    └── enterprise_rag_bench/
        ├── documents.jsonl
        ├── chunks.jsonl
        ├── benchmark_questions.jsonl
        └── corpus_stats.json
```

`documents.jsonl` preserves both the benchmark `doc_id` and a stable `record_id` for each physical source document. This distinction matters because the released source archives contain a small number of conflicting records that share a benchmark document ID. Those records are retained rather than silently discarded.

`chunks.jsonl` contains deterministic overlapping text windows with stable UUID chunk identifiers and source provenance. Benchmark questions are retained only when their expected evidence is present and unambiguous in the local corpus, along with information-not-found questions that remain valid for a subset corpus. Questions whose expected document IDs collide across different source records are excluded from the local evaluation set and reported in `corpus_stats.json`.
