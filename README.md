# ClaimBridge

ClaimBridge is a local, review-first audit for claims in an already-generated RAG answer. It compares each cited sentence to the cited source with a pretrained natural-language-inference model and points a human reviewer to the most relevant evidence span. It does **not** generate answers or certify truth.

## Who it is for

Developers and evaluators who already have an answer plus retrieved source snippets, and want a small, provider-neutral check before displaying or testing that answer. The task is: give ClaimBridge a JSON answer/source packet, inspect per-claim support signals, and review uncertain or mismatched claims.

Non-goals: retrieving documents, checking live web truth, making legal/medical decisions, or replacing a full RAG evaluation suite.

## Run

Requires Python 3.11+ and internet on the first run to download the [Apache-2.0 NLI model](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768). Inference then runs locally; no API key is needed. The model is English-trained, and long source documents should be split into concise snippets before use.

```bash
uv sync --python 3.11
uv run claimbridge examples/example.json --out report.local.json
cat report.local.json
uv run python -m unittest discover -s tests -v
```

The example has a supported adult-enrolment statement and an overclaim about children. The report contains each claim, citation ID, selected source span, NLI scores, and one of `likely_supported`, `possible_contradiction`, `mixed_evidence_review`, `needs_review`, `missing_source`, or `uncited`. Scores are model signals, **not calibrated probabilities of factual correctness**. Keep a human in the loop, especially for negation, numerical reasoning, tables, non-English text, or high-stakes content.

Input format:

```json
{
  "answer": "The trial enrolled 120 adults [study].",
  "sources": [{"id": "study", "text": "The trial enrolled 120 adults."}]
}
```

Each answer sentence should cite source IDs in square brackets. The tool never uploads the packet, but model files are fetched from Hugging Face on first use. Do not process confidential material on a shared machine without checking local cache/storage policy.

## Why this small scope

As of 2026-10-09, a [RAGFlow user report from 2025](https://github.com/infiniflow/ragflow/issues/7232) describes a correct answer attached to the wrong citation. A [LlamaIndex discussion started in May 2026](https://github.com/run-llama/llama_index/discussions/21744) separates claim-to-source support from generic faithfulness, with discussion continuing into July. These are independent problem signals across time, not proof of a large market. The cross-month discussion is the attention signal; a tiny JSON-in/JSON-out audit can be reused in CI and shared as a concrete failure report, which is a distribution hypothesis still to test.

The [citation-validator](https://github.com/va2ai/citation-validator) project already shows a richer, domain-specific five-layer validator. ClaimBridge deliberately differs by accepting a preexisting answer and source packet, requiring no generation API, and outputting a narrow, local review artifact. The NLI model adds semantic support triage; deterministic citation-ID checks alone cannot tell whether a real source supports a claim. No code or prompt text from these projects is copied.

## Limits and acceptance

- A cited source can be genuine yet stale or factually wrong; ClaimBridge cannot detect that.
- The NLI model was trained on general English sentence pairs. Domain transfer, numeric claims, and multi-hop reasoning require human checking.
- Long evidence is windowed and each model input is truncated to 512 tokens, so relevant context can be missed.
- Sentence splitting is deliberately simple; abbreviations, bullet lists, and citations spanning several sentences may need preprocessing.
- A `likely_supported` signal is not a release gate by itself. The intended MVP acceptance is: the example runs from a clean install, preserves uncited/missing-source cases, and emits reviewable source spans without network inference.

MIT-licensed application code. Model weights are not included in this repository; see the model's own Apache-2.0 license and card.
