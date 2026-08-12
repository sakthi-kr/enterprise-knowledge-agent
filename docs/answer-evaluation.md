# Answer evaluation

The project evaluates three runtime paths over the same deterministic, balanced subset of the
prepared EnterpriseRAG-Bench questions:

- dense RAG: Qdrant retrieval followed by grounded answer generation;
- graph-augmented RAG: dense retrieval with bounded Neo4j evidence expansion;
- agent: LLM planning plus bounded Qdrant and optional Neo4j tools through LangGraph.

The harness is intentionally an internal engineering evaluation rather than an
EnterpriseRAG-Bench leaderboard submission. The official benchmark uses a separate judge-based
answer-evaluation protocol. This repository keeps benchmark gold answers and answer facts local
and computes deterministic or local-embedding proxies instead of sending gold references to an
external evaluator.

## Provider control

Every system in one comparison must use the same provider and model. The harness currently
supports the existing Gemini adapter and a Groq adapter using `openai/gpt-oss-20b`. Provider/model
names are stored in every result row and provider runs use separate artifact paths, preventing a
partially completed run from one model from being resumed with another model.

Groq is called through its OpenAI-compatible Chat Completions REST endpoint. Planning and grounded
answers use the same provider-neutral prompts and strict JSON schemas as the Gemini path. The
adapter records provider-reported token usage, follows `retry-after` on rate-limit responses, and
uses bounded retries for transient transport or provider failures.

For quota-bounded evaluation, a smaller shared context budget can be supplied explicitly. For
example, `--context-sources 4 --dense-context-sources 3 --graph-context-sources 1` allows the dense
system up to four dense sources while graph-enabled systems reserve at most one of four positions
for graph-derived evidence. The selected budget is written to the summary so the experiment does
not silently differ from the normal runtime configuration.

## Metrics

Answerability accuracy checks whether answerable questions receive an answer and `info_not_found`
questions produce the explicit insufficient-evidence outcome.

Citation metrics compare the document IDs actually cited by the generated answer with the
benchmark's expected document IDs. The harness reports document recall, citation precision, and
invalid extra document counts.

Two local semantic proxy metrics use the same BGE embedding model as retrieval:

- gold-answer similarity proxy: cosine similarity between the generated and reference answers;
- answer-fact coverage proxy: fraction of benchmark answer facts with a sufficiently similar
  generated-answer sentence.

These are useful for regression testing, but they are not equivalent to the benchmark's official
LLM-judge correctness and completeness scores.

Operational metrics include end-to-end latency, provider-reported prompt/output/thinking tokens,
and an estimated paid token cost for the selected model. Free-tier executions can still have zero
actual cost. Agent results additionally report planner-policy alignment, planner fallback rate,
tool-call count, graph-tool call rate, and graph-tool yield rate.

## Sampling and resumability

Sampling is deterministic and balanced by the compatible benchmark question types. Each selected
question/system attempt is appended immediately to a provider-specific JSONL results file.
Successful pairs are not repeated on later runs, while failed pairs remain eligible for retry. The
evaluator keeps only the newest attempt for each question/system pair when computing summaries.

Bulk evaluation is paced because hosted-model quotas are provider- and model-specific. Transient
provider failures are retried by the provider adapter. If a rate limit remains exhausted, the
evaluation stops immediately with progress saved instead of issuing a long sequence of doomed
requests. Provider HTTP status codes are stored without storing provider error text.

An incomplete run is marked `evaluation_complete: false`; cross-system quality comparisons are
suppressed until every selected pair has succeeded. This prevents architecture conclusions from
being calculated over different surviving subsets of questions.

The generated artifacts deliberately omit raw questions, generated answers, gold answers, answer
facts, retrieved text, and source paths. Question IDs and aggregate metrics are retained for local
debugging and reproducibility.
