# thrifty-ml

Replace expensive per-row LLM calls with a lightweight ML classifier trained on your own data. Get the same answers more than 100× cheaper and faster.

## The problem

You have a DataFrame with 100 000 rows. You want to filter or classify every row using an LLM prompt. Calling the LLM once per row is slow and expensive. At $0.25 / 1M input tokens and ~100 tokens per row, that's $2.50 — for a simple yes/no filter.

## How thrifty-ml solves it

thrifty-ml implements the proxy model technique from ["100x Cost & Latency Reduction: Performance Analysis of AI Query Approximation using Lightweight Proxy Models"](https://arxiv.org/html/2603.15970v6) (Google Research, SIGMOD 2026):

1. **Sample** a small subset of rows (~1 000).
2. **Label** the sample with the LLM — the only rows that ever touch the API.
3. **Embed** all rows with an embedding model.
4. **Train** a fast classifier (logistic regression by default) on the labeled embeddings.
5. **Evaluate** the classifier on a holdout split. If it matches the LLM within a tolerance τ, use it.
6. **Predict** the remaining rows with the classifier — no LLM calls, ~0.1 ms per 1 000 rows.

On a 100 000-row dataset with `sample_size=1000`, you pay for 1 000 LLM calls instead of 100 000. **100× cost reduction**, with accuracy that matches the LLM on most tasks.

---

## Quick start

```bash
pip install thrifty-ml
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

**Filter rows** — keep only the ones that match a condition:

```python
import pandas as pd
from thrifty_ml import ml_filter

df = pd.read_csv("reviews.csv")

mask = ml_filter(
    df,
    prompt="Is this a positive movie review?",
    text_column="text",
    llm="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-large",
)

print(df[mask])
```

**Classify rows** — assign each row to a category:

```python
from thrifty_ml import ml_classify

df["topic"] = ml_classify(
    df,
    prompt="Classify this support ticket by topic.",
    text_column="body",
    llm="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-large",
    classes=["billing", "technical", "account", "other"],
)
```

**Or use the CLI** — no Python required:

```bash
thrifty-ml filter examples/reviews.csv \
  --prompt "Is this a positive movie review?" \
  --text-col text \
  --out positive.csv \
  --llm openai/gpt-4o-mini \
  --embedding-model openai/text-embedding-3-large \
  --sample-size 20
```

Both calls label a small sample with the LLM (~1 000 rows by default), train a classifier on the results, and predict the rest — without any additional LLM calls.

---

## Installation

```bash
pip install thrifty-ml

# For LightGBM support (non-linear proxy, better on hard tasks):
pip install thrifty-ml[lgbm]
```

Requires Python ≥ 3.10. LLM and embedding calls go through [LiteLLM](https://github.com/BerriAI/litellm), so any provider works — Anthropic, OpenAI, Bedrock, Vertex, local Ollama, etc.

---

## Python API

### Binary filter

Keep rows that match a natural-language condition.

```python
import pandas as pd
from thrifty_ml import ml_filter

df = pd.read_parquet("reviews.parquet")

mask = ml_filter(
    df,
    prompt="Is this a positive review?",
    text_column="review_text",
    llm="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-large",
)

positive_reviews = df[mask]
```

`ml_filter` returns a boolean numpy array of shape `(len(df),)`.

### Multi-class classification

Assign each row to one of a fixed set of labels.

```python
from thrifty_ml import ml_classify

labels = ml_classify(
    df,
    prompt="Classify this support ticket by topic.",
    text_column="body",
    llm="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-large",
    classes=["billing", "technical", "account", "other"],
)

df["topic"] = labels
```

If the LLM returns a value not in `classes`, that row is labeled `"__unknown__"`.

### All parameters

Both `ml_filter` and `ml_classify` accept:

| Parameter | Default | Description |
|---|---|---|
| `df` | required | Input DataFrame |
| `prompt` | required | Natural-language instruction for the LLM |
| `text_column` | required | Column name containing text to evaluate |
| `llm` | required | LiteLLM model string (e.g. `"openai/gpt-4o-mini"`) |
| `embedding_model` | required | LiteLLM embedding model string, or a custom `EmbeddingBackend` |
| `classes` | — | List of class labels (`ml_classify` only) |
| `proxy` | `"lr"` | Proxy model type: `"lr"`, `"svc"`, or `"lgbm"` |
| `sample_size` | `1000` | Number of rows to label with the LLM |
| `fallback_threshold` | `0.1` | τ — if proxy F1 < 1.0 − τ, fall back to full LLM |
| `max_concurrency` | `8` | Max simultaneous LLM API calls |
| `cache_dir` | `~/.cache/thrifty_ml/` | Override the embedding/label cache directory |
| `seed` | `None` | Random seed for reproducible sampling |

---

## Offline mode: fit once, predict forever

For recurring workloads, train the proxy once and reuse it without any LLM calls.

```python
from thrifty_ml import Proxy

# Train — labels a sample with the LLM, fits the proxy
proxy = Proxy(
    prompt="Is this a positive review?",
    llm="openai/gpt-4o-mini",
    embedding_model="openai/text-embedding-3-large",
    model="lgbm",         # "lr" | "svc" | "lgbm"
    sample_size=2000,
)
proxy.fit(train_df, text_column="review_text")
proxy.save("sentiment_proxy.joblib")

# Later — no LLM, no API key needed
from thrifty_ml import Proxy
proxy = Proxy.load("sentiment_proxy.joblib")
labels = proxy.predict(new_df, text_column="review_text")
```

`save` writes two files: `sentiment_proxy.joblib` (the model) and `sentiment_proxy.joblib.meta.json` (metadata). `load` reads both to restore the proxy fully.

If you used a custom `EmbeddingBackend`, pass it to `load`:

```python
proxy = Proxy.load("sentiment_proxy.joblib", embedding_model=MyBackend())
```

---

## Custom embedding backends

Pass any `EmbeddingBackend` subclass instead of a model string to use your own embeddings.

```python
import numpy as np
from thrifty_ml import EmbeddingBackend, ml_filter

class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    @property
    def model_id(self) -> str:
        return f"st:{self._model.get_sentence_embedding_dimension()}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True).astype("float32")

mask = ml_filter(
    df,
    prompt="Is this relevant?",
    text_column="text",
    llm="openai/gpt-4o-mini",
    embedding_model=SentenceTransformerBackend("all-MiniLM-L6-v2"),
)
```

`model_id` is used as the diskcache key — keep it stable across runs.

---

## Proxy model types

| Type | Key | Notes |
|---|---|---|
| Logistic Regression | `"lr"` | Default. Fast, interpretable, works well on modern embeddings. |
| Linear SVM | `"svc"` | Similar to LR; sometimes better on very high-dimensional embeddings. |
| LightGBM | `"lgbm"` | Best for non-linear tasks; requires `pip install thrifty-ml[lgbm]`. |

The fallback threshold τ (`fallback_threshold=0.1`) controls quality vs cost. If the proxy's F1 on the holdout split is below `1.0 - τ`, thrifty-ml warns you and falls back to labeling all rows with the LLM. Tighten τ (e.g. `0.05`) for higher accuracy requirements; loosen it (e.g. `0.2`) to force proxy use even when accuracy is lower.

---

## Caching

Embeddings and LLM labels are cached automatically at `~/.cache/thrifty_ml/` using [diskcache](https://grantjenks.com/docs/diskcache/). Re-running the same call with the same inputs costs nothing.

Cache keys include the model ID, prompt, and (for multiclass) the class list, so changing any of these triggers fresh calls.

```python
# Use a project-specific cache directory
mask = ml_filter(df, ..., cache_dir="./my_project_cache")
```

---

## CLI

thrifty-ml ships a CLI for use without writing Python.

### Filter rows

```bash
thrifty-ml filter reviews.parquet \
  --prompt "Is this a positive review?" \
  --text-col review_text \
  --out positive.parquet \
  --llm openai/gpt-4o-mini \
  --embedding-model openai/text-embedding-3-large
```

Writes a parquet file containing only matching rows, with an added `_thrifty_mask` column.

### Classify rows

```bash
thrifty-ml classify tickets.csv \
  --prompt "Classify this support ticket by topic." \
  --text-col body \
  --classes "billing,technical,account,other" \
  --out classified.csv \
  --llm openai/gpt-4o-mini \
  --embedding-model openai/text-embedding-3-large
```

Appends a `label` column to the output file.

### Embed a column

```bash
thrifty-ml embed reviews.parquet \
  --text-col review_text \
  --model openai/text-embedding-3-large \
  --out embeddings.npy
```

Saves embeddings as a numpy `.npy` file.

### Label a sample

```bash
thrifty-ml label reviews.parquet \
  --prompt "Is this a positive review?" \
  --text-col review_text \
  --sample 1000 \
  --out labels.csv \
  --llm openai/gpt-4o-mini
```

Labels a random sample and saves the results — useful for inspecting LLM outputs before running a full pipeline.

### Clear the cache

```bash
thrifty-ml cache clear

# Clear a specific cache directory
thrifty-ml cache clear --cache-dir ./my_project_cache
```

### Common CLI flags

All commands accept:

| Flag | Default | Description |
|---|---|---|
| `--llm` | `openai/gpt-4o-mini` | LiteLLM model string |
| `--embedding-model` | `openai/text-embedding-3-large` | LiteLLM embedding model string |
| `--proxy` | `lr` | Proxy type: `lr`, `svc`, `lgbm` |
| `--sample-size` | `1000` | Rows to label with the LLM |
| `--fallback-threshold` | `0.1` | τ quality threshold |
| `--max-concurrency` | `8` | Parallel LLM calls |
| `--cache-dir` | — | Override cache directory |
| `--seed` | — | Random seed |

Input files can be `.parquet`, `.csv`, `.json`, or `.jsonl`. Output format matches input unless you specify a different extension in `--out`.

---

## When does the proxy get used?

thrifty-ml prints a warning and falls back to full LLM labeling if:

- The labeled sample contains only one class (proxy can't learn anything).
- The proxy F1 on the holdout split is below `1.0 - fallback_threshold`.

In both cases you still get correct labels — just at full LLM cost for that run.

---

## Environment variables

Set API keys for your chosen providers:

```bash
export OPENAI_API_KEY=sk-...
```

thrifty-ml passes these through to LiteLLM, which supports all standard provider env vars. See the [LiteLLM docs](https://docs.litellm.ai/docs/providers) for the full list.

---

## Validation: parity with the SIGMOD 2026 paper

The proxy-model technique was independently validated against the benchmark in ["100x Cost & Latency Reduction: Performance Analysis of AI Query Approximation using Lightweight Proxy Models"](https://arxiv.org/html/2603.15970v6) (Google Research, SIGMOD 2026) using the Stanford IMDB sentiment dataset — the paper's primary use case (movie review classification).

**Setup:** 50 000 IMDB reviews, `sample_size=1000`, logistic regression proxy, `openai/gpt-4o-mini` + `openai/text-embedding-3-large`.

| Metric | Observed | Paper target |
|---|---|---|
| F1 (proxy vs LLM) | **0.964** | ≥ 0.9 |
| F1 (LLM vs gold labels) | **0.939** | — |
| F1 (proxy vs gold labels) | **0.953** | — |
| Relative accuracy | **1.015** | 0.90–1.05 |
| Token reduction | **50×** | ~50× at 2% sample ratio |
| Speedup | **21.8×** | — |
| Fallback triggered | **No** | — |

**What this means:**

- The proxy classifier (trained on 1 000 LLM-labeled examples) matched LLM output with F1 = 0.964, well above the paper's ≥ 0.9 threshold. The remainder of the 50 000 rows was predicted by the proxy with zero additional LLM calls.
- Relative accuracy of 1.015 means the proxy's predictions against the human gold labels are slightly better than the LLM's own predictions — within the paper's reported 0.90–1.05 band.
- Token reduction of 50× is measured directly: 331 341 tokens used vs 16 567 050 tokens projected for full LLM labeling.

The full benchmark is in [`benchmarks/imdb/`](benchmarks/imdb/) and is runnable independently. Provider differs from the paper's Gemini/Gecko baseline; the accuracy ratios and cost reduction hold regardless of provider.

---

## Appendix: thrifty-ml vs BigQuery AI.IF and AlloyDB

The proxy model technique was published in ["100x Cost & Latency Reduction: Performance Analysis of AI Query Approximation using Lightweight Proxy Models"](https://arxiv.org/html/2603.15970v6) (Google Research, SIGMOD 2026) and ships inside two Google products: `AI.IF` / `AI.LABEL` in BigQuery, and accelerated semantic functions in AlloyDB. The cost and latency wins are real — Google reports 100× or more improvement, reaching ~1 000× on 10M-row tables with pre-computed embeddings — but the implementation is locked inside Google's data warehouse SQL surface. thrifty-ml ports the same technique to Python and removes every one of those constraints.

### No infrastructure dependency

BigQuery and AlloyDB require your data to be in GCP, a billing account, and IAM setup. thrifty-ml works on a local pandas DataFrame, a parquet file, or a CSV — on your laptop, in a notebook, in a CI job, or in any Python environment. No cloud account required.

### Any LLM and any embedding provider

Google's products are wired to Vertex AI and Gemini models. thrifty-ml uses [LiteLLM](https://docs.litellm.ai/docs/providers) as an adapter, so the same API works with Anthropic, OpenAI, AWS Bedrock, Google Vertex, Azure OpenAI, a local Ollama instance, or any other provider. You can also bring your own embeddings via the `EmbeddingBackend` ABC — pre-computed vectors, a fine-tuned sentence-transformer, a proprietary model — without touching the rest of the pipeline.

### Deploy-once offline mode

In BigQuery and AlloyDB, the sample-label-train pipeline reruns on every query. thrifty-ml's `Proxy` class separates training from inference: fit once, serialize to disk, deploy the classifier wherever you need it. Subsequent `predict()` calls make zero LLM API calls and run at classifier speed (~0.1 ms / 1 000 rows for logistic regression). This matters for production pipelines where you want a stable, versioned model — not one that retrains on every invocation.

### Full observability

Inside a SQL function you cannot inspect intermediate state. thrifty-ml exposes everything:

- `EvalResult.proxy_f1` — the holdout F1 score that drove the fallback decision
- `EvalResult.use_proxy` — whether the proxy was used or the LLM fell back
- `EvalResult.holdout_size` — how many rows the evaluation was based on
- The trained proxy model itself — a standard sklearn or LightGBM object you can serialize, explain with SHAP, or hand to your model registry

You can also tune `fallback_threshold` explicitly to trade accuracy for cost in a way that is visible and reproducible, rather than relying on opaque platform defaults.

### Integration with the Python ML ecosystem

Because proxy models are sklearn or LightGBM objects, the full Python ML toolchain applies: feature importances, calibration curves, cross-validation, SHAP explanations, MLflow or W&B logging, and standard model registries. None of this is accessible through a SQL function.

### Tighter iteration loop

A data scientist tuning a prompt or trying a different embedding model gets immediate feedback in a notebook: run the cell, inspect the sample labels, check the proxy F1, adjust, re-run. The BigQuery equivalent is: upload data to a warehouse, write a SQL query, wait for a query job to complete, inspect a result table — then repeat. thrifty-ml collapses that loop to seconds.

### The wins transfer

The 100× or more cost reduction reported in the paper comes from the proxy technique itself, not from BigQuery. thrifty-ml uses the identical algorithm — random sampling, embedding-based proxy, holdout F1 evaluation, τ-threshold fallback — so the same efficiency gains apply to any DataFrame workload, without a Google account.
