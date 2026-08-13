# Answer evaluation artifacts

This directory stores compact, reproducible summaries from the local EnterpriseRAG-Bench answer evaluation harness.

Generated per-question rows intentionally exclude raw benchmark questions, gold answers, answer
facts, retrieved document text, and generated answer bodies. The artifacts contain question IDs,
question types, quality proxies, retrieval/citation metrics, latency, token usage, tool-routing
diagnostics, provider/model metadata, and cost estimates only.

Provider/model combinations use separate result and summary filenames. Existing partial Gemini
results therefore cannot contaminate a Groq evaluation, and a resumed run only skips successful
pairs produced by the same evaluator version, provider, and model.

Incomplete evaluations are not valid architecture comparisons. The summary records error counts
and suppresses dense-vs-candidate comparison metrics until all selected question/system pairs have
succeeded. Failed pairs can be retried by running the same evaluation command again.

The JSONL result file is append-only attempt history, so it can contain more rows than the final number of unique question/system pairs when a failed pair is retried. Summary metrics use the latest result for each pair.
