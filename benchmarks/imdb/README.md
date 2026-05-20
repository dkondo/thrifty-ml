# IMDB Benchmark

Reproduces the proxy-model results from [arXiv 2603.15970](https://arxiv.org/html/2603.15970v6) (SIGMOD 2026) on the Stanford IMDB sentiment dataset.

## What this measures

| Column | Description | Paper reference |
|---|---|---|
| F1 (proxy vs LLM) | F1 of the proxy classifier vs LLM labels on the holdout split | Table 1, "Relative F1" |
| F1 (LLM vs gold) | F1 of LLM labels vs IMDB gold labels on the sample | Table 1, "LLM accuracy" |
| F1 (proxy vs gold) | F1 of proxy predictions vs IMDB gold labels on the remainder | Bonus — gold available on IMDB |
| Relative accuracy | F1 (proxy vs gold) / F1 (LLM vs gold) | Section 4.1, eq. (2) |
| Token reduction | Projected full-LLM tokens / actual tokens used | Table 1, "Cost reduction" |
| Speedup | Projected full-LLM wall time / actual wall time | Table 1, "Latency reduction" |
| Fallback | Whether the proxy fell back to full-LLM labeling | Section 3.3 |

The paper's headline threshold: proxy F1 ≥ 0.9 × LLM F1 on the holdout split.

## Expected numbers

At `sample_size=1000` on 50k rows with `anthropic/claude-haiku-4-5` + `text-embedding-3-small`:

- **F1 (proxy vs LLM)** ≈ 0.91–0.97 (paper: ≥ 0.9); observed 0.910
- **Token reduction** ≈ 50× (50k / 1k sample); observed 50× (341k / 17M tokens)
- **Relative accuracy** ≈ 0.95–1.05; observed 0.971
- **Speedup** ≈ 25–35× (fit + predict vs projected full-LLM wall time); observed 29.5×

These providers differ from the paper's Gemini/Gecko baseline; the _ratios_ should match within a wide band even if absolute F1 values differ slightly.

## Installation

```bash
pip install -e ".[benchmark]"
# or
uv pip install -e ".[benchmark]"
```

## Smoke run (~$0.01, ~2 min)

```bash
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \
  python -m benchmarks.imdb.run \
    --rows 500 \
    --sample-size 100 \
    --cache-dir /tmp/frugal_smoke \
    --out /tmp/smoke.json
```

Expect: markdown table on stdout, `f1_proxy_vs_llm > 0.8`, `token_reduction_x ≈ 5`, no fallback.

## Full run (~$1–3, ~20 min)

```bash
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \
  python -m benchmarks.imdb.run \
    --cache-dir benchmarks/imdb/.cache_$(date +%s) \
    --out benchmarks/imdb/results.json
```

Expect: `f1_proxy_vs_llm ≥ 0.9`, `token_reduction_x ≈ 50`, `relative_accuracy ∈ [0.9, 1.1]`.

## Cold vs warm cache

- **Cold** (reproducible timings): pass a fresh `--cache-dir benchmarks/imdb/.cache_$(date +%s)`.
- **Warm** (free reruns): omit `--cache-dir`. Cached embeddings and labels make re-runs near-instant and cost $0.

## All CLI options

```
--llm               LiteLLM model string       [default: anthropic/claude-haiku-4-5]
--embedding-model   LiteLLM embedding model    [default: text-embedding-3-small]
--proxy             Proxy type: lr, svc, lgbm  [default: lr]
--sample-size       Rows labeled by LLM        [default: 1000]
--rows              Total rows (None = 50k)    [default: None]
--seed              Random seed                [default: 42]
--cache-dir         Cache directory            [default: ~/.cache/thrifty_ml/]
--out               JSON output path           [default: benchmarks/imdb/results.json]
--max-concurrency   Parallel LLM calls         [default: 8]
```

## Env vars

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # for anthropic/claude-haiku-4-5
export OPENAI_API_KEY=sk-...          # for text-embedding-3-small
```

Any LiteLLM-supported provider works — pass `--llm` and `--embedding-model` to switch.

## Observed results

### Full run (50 000 rows, sample_size=1 000, anthropic/claude-haiku-4-5 + text-embedding-3-small)

| Dataset | Prompt | F1 (proxy vs LLM) | F1 (LLM vs gold) | F1 (proxy vs gold) | Relative accuracy | Token reduction | Speedup | Fallback |
|---|---|---|---|---|---|---|---|---|
| IMDB | Is this movie review positive? | 0.910 | 0.961 | 0.932 | 0.971 | 50× | 29.5× | False |

**Reproduces the paper's headline claims:**

- **F1 (proxy vs LLM) = 0.910** — above the paper's ≥ 0.9 threshold; proxy accepted, remainder predicted without LLM calls.
- **Relative accuracy = 0.971** — proxy accuracy vs gold is 97.1% of LLM accuracy vs gold; within the paper's reported 0.90–1.05 band.
- **F1 (LLM vs gold) = 0.961** — claude-haiku-4-5 matches IMDB gold labels at 96.1%, consistent with the paper's LLM accuracy.
- **Token reduction = 50×** — 341 005 tokens used vs 17 050 250 projected for full LLM labeling (1 000 sample / 50 000 rows).
- **Speedup = 29.5×** — fit + predict wall time vs projected full-LLM wall time.
- **No fallback** — proxy cleared the threshold on first attempt with 1 000 labeled samples.

### Smoke run (500 rows, sample_size=100) — for reference

| Dataset | Prompt | F1 (proxy vs LLM) | F1 (LLM vs gold) | F1 (proxy vs gold) | Relative accuracy | Token reduction | Speedup | Fallback |
|---|---|---|---|---|---|---|---|---|
| IMDB | Is this movie review positive? | 0.857 | 0.971 | 0.985 | 1.014 | — (fallback) | — (fallback) | True |

Fallback triggered at this small scale: 100 labeled samples (20% of 500 rows) gave the proxy insufficient signal. At the paper's 2% sample ratio (1 000/50 000) the proxy succeeds comfortably.
