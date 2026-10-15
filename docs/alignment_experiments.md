# Alignment experiments: LLM rerank (A) and sentence-level pairing (B)

*2026-10-15. Experiment only: the pipeline's aligner and its defaults are unchanged.*

## Question

The first evaluation (`python -m wikilens evaluate`, 54 hand-labelled en-zh chunks from *Huawei* and *Social credit system*) found that for 18 of the 54 Chinese chunks, the correct English chunk was not among the top 3 by LaBSE chunk cosine. The likely cause is granularity: a Chinese paragraph often states several facts, while English spreads them over several short paragraphs. So one English paragraph matches only part of a Chinese one, and the whole-chunk embedding dilutes that match. We tried two fixes:

- **A. LLM rerank.** For each Chinese chunk, take the top 10 English chunks by chunk cosine. The configured LLM (`SCORING_BACKEND`) labels each candidate `match`, `partial` or `none`, following the same rules as the annotation guidelines. The predicted pairs are the candidates the LLM labels match or partial (also reported: match only).
- **B. Sentence-level pairing.** Split both chunks into the fetcher's sentences and embed each sentence with LaBSE (pinned revision). A chunk pair's score is either the **max** sentence-pair cosine, or the **top-k mean**: take each Chinese sentence's best English sentence, then average the k highest of those. The score threshold is swept over 0.20–0.90.

## How to reproduce

```bash
python -m wikilens experiment                 # all methods; A calls the LLM once per labelled chunk (54 calls)
python -m wikilens experiment --cached-only --llm-model meta-llama/llama-3.3-70b-instruct   # A from cached replies
```

The output goes to `eval/experiments/alignment_en-zh_methods.md` (tables) and `.json` (everything, including the PR curves and the predicted pairs that no label covers). LLM replies are cached in `eval/experiments/llm_rerank_<model>.json`, keyed by prompt. Commit that cache so A can be re-scored without calling the LLM again. Scoring is the same as for `evaluate`: partial counts as positive, and predictions outside the labelled pairs are handled with `--unlabelled auto`.

## Results (en-zh, 54 chunks, 10 of them absent in English)

"best" means the threshold with the highest F1 on these same labels, so it is optimistic. The cross-article rows pick the threshold on one article and score it on the other.

| method | thr | P | R | F1 | F1 95% CI | held-out F1 Huawei / SCS |
|---|---|---|---|---|---|---|
| current pipeline default (chunk 0.45, routing 0.45) | 0.45 | 0.153 | 0.287 | 0.200 | 0.12–0.29 | – |
| chunk cosine, routing off (current best) | 0.70 | 0.592 | 0.362 | 0.450 | 0.32–0.56 | 0.348 / 0.474 |
| **B: sentence max** | 0.65 | 0.607 | 0.425 | **0.500** | 0.38–0.60 | **0.575** / 0.413 |
| B: sentence top-2 mean | 0.60 | 0.619 | 0.325 | 0.426 | 0.31–0.54 | 0.291 / 0.295 |
| B: sentence top-3 mean | 0.55 | 0.385 | 0.438 | 0.409 | 0.32–0.50 | 0.164 / 0.255 |
| A: LLM rerank top 10 | – | *not run yet (no LLM credentials in the session where this was written)* | | | | |

Is the correct answer among the candidates? "Chunks" is the share of the 44 non-absent chunks with a known correct target in the top k. "Outside top 3" uses the 17 matches the annotator named outside chunk cosine's top 3.

| method | chunks in top 3 | chunks in top 10 | outside-top-3 matches in top 3 | outside-top-3 matches in top 10 |
|---|---|---|---|---|
| chunk cosine | 86% | 95% | 0/17 (by construction) | 11/17 (65%) |
| B: sentence max | 80% | 86% | 5/17 (29%) | 7/17 (41%) |
| B: sentence top-2 mean | 77% | 84% | 3/17 | 8/17 |
| B: sentence top-3 mean | 77% | 82% | 3/17 | 7/17 |
| A (candidate pool = chunk cosine top 10) | – | 95% | – | 11/17 (65%) |

A's recall can never be higher than its pool: at most 95% of chunks and 11 of the 17 outside-top-3 matches. It can only reach those if the LLM accepts them and still rejects the many wrong candidates in the pool.

## Reading the numbers

- **Sentence max (B) is the best embedding-only score so far:** F1 0.500 against 0.450, with recall 0.425 against 0.362 at about the same precision. It moves 5 of the 17 matches that chunk cosine ranked 4th or lower into the top 3. That fits the granularity explanation: one strong sentence-to-sentence match is enough. The gain is within the confidence intervals, though, and it does not hold across articles. On Huawei the threshold picked on SCS gives 0.575 against 0.348. On SCS it gives 0.413 against 0.474.
- **Top-k mean is worse than max.** It asks a short English paragraph to cover k Chinese sentences, which is exactly the 1:N situation the change is meant to handle.
- **The coverage columns are biased towards chunk cosine.** Apart from the 17 named outside-top-3 matches, every known positive was found by labelling chunk cosine's top 3. When B ranks an unlabelled chunk above one of those, it loses coverage even if that chunk is also correct. The same bias appears in precision. Most of B-max's false positives (18 of 22) are pairs nobody labelled, and these count as false only because of the `auto` rule. If unlabelled pairs are ignored, precision is 0.895 for B-max against 0.690 for chunk cosine. The JSON lists these pairs (`unlabelled_predictions`). Labelling them, about 25 pairs per method, would settle how much of B's precision is real.
- **The labels are small.** 54 chunks from 2 articles, so a 95% interval on F1 is about ±0.1. The best thresholds are picked in-sample.

## Next steps

1. Run A with the configured LLM: `python -m wikilens experiment`. It needs `SCORING_BACKEND=openai` plus `OPENAI_API_KEY`, or a local Ollama. Commit `eval/experiments/llm_rerank_*.json`.
2. Label the unlabelled pairs that B-max and A predict, then re-run both scripts.
3. A natural combination to try after that: B-max as the ranking (better outside-top-3 recall), with the LLM reranking B's top 10.
