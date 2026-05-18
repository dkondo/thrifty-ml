# frugal-ml Design

## What it does

frugal-ml replaces per-row LLM calls with lightweight ML classifiers ("proxy models") trained on text embeddings. Instead of asking an LLM to evaluate every row in a DataFrame, it:

1. Labels a small sample (~1 000 rows) using the LLM.
2. Embeds all rows with an embedding model.
3. Trains a fast classifier (logistic regression, SVM, or LightGBM) on the labeled sample.
4. Evaluates classifier quality on a holdout split.
5. If quality is good enough, uses the classifier to label the remaining rows instead of the LLM.

On large datasets this yields 100–1 000× cheaper and faster labeling with accuracy that matches the LLM within a configurable tolerance.

The technique is described in [arXiv 2603.15970](https://arxiv.org/html/2603.15970v6) and is used inside Google's BigQuery `AI.IF` and AlloyDB accelerated functions. frugal-ml ports it to any Python DataFrame.

---

## Architecture

```
frugal_ml/
├── __init__.py          # Public API: ml_filter, ml_classify, Proxy, EmbeddingBackend
├── engine.py            # Engine — the orchestration core
├── sampling.py          # random_sample(): splits df into sample + remainder
├── evaluator.py         # evaluate(), train_holdout_split()
├── cache.py             # diskcache wrappers for embeddings and labels
├── llm.py               # Async LLM labeling via LiteLLM
├── embeddings.py        # EmbeddingBackend ABC + LiteLLMEmbeddingBackend
├── cli.py               # Typer CLI (filter, classify, embed, label, cache clear)
└── proxy/
    ├── base.py          # ProxyModel ABC (fit, predict, save, load)
    ├── linear.py        # LogisticRegressionProxy, LinearSVCProxy
    └── trees.py         # LightGBMProxy
```

---

## Pipeline

Both online and offline modes share the same core pipeline inside `Engine`:

```
df
 │
 ├─ embed_texts(all rows)  ──────────── cache hit/miss per text ──── diskcache
 │
 ├─ random_sample(sample_size)
 │      │
 │      └─ label_texts(sample) ─────── LLM calls (async, batched) ── diskcache
 │
 ├─ train_holdout_split(labeled sample)
 │      │  80 % train / 20 % holdout (stratified when possible)
 │      │
 │      └─ proxy.fit(X_train, y_train)
 │
 ├─ evaluate(proxy, X_holdout, y_holdout)
 │      │  proxy_f1 >= 1.0 - τ  →  use_proxy = True
 │      │  otherwise emit UserWarning and fall back to LLM
 │      │
 │      └─ τ = fallback_threshold (default 0.1)
 │
 └─ predict remainder
        if use_proxy:  proxy.predict(X_remainder)
        else:          label_texts(remainder)   ← full LLM cost
```

**Online mode** (`ml_filter` / `ml_classify`) runs the whole pipeline and returns labels for every row.

**Offline mode** (`Proxy.fit` / `Proxy.predict`) runs up to and including `proxy.fit`, then serializes the trained model. `predict()` only embeds + classifies — no LLM calls in the hot path.

---

## Key components

### Engine (`engine.py`)

Accepts a prompt, LLM model string, embedding backend, and proxy type. Coordinates all pipeline stages. Two entry points:

- `run(df, text_column)` — online, returns a label array for the full DataFrame.
- `fit(df, text_column)` — offline, returns `(proxy_model, eval_result)` for serialization.

`_run_async(coro)` is a small helper that makes the async LLM calls work in any context — plain scripts use `asyncio.run()`, Jupyter notebooks (which already have a running loop) use `nest_asyncio` if available, or fall back to a `ThreadPoolExecutor` to avoid `RuntimeError: This event loop is already running`.

### Sampling (`sampling.py`)

`random_sample(df, n, seed)` returns `(sample_df, remainder_df)`. If `n >= len(df)` the entire DataFrame is the sample and remainder is empty (LLM labels are returned directly, no proxy needed).

### Evaluator (`evaluator.py`)

`train_holdout_split(X, y, holdout_fraction=0.2)` uses `StratifiedShuffleSplit` when all classes have ≥ 2 samples; falls back to a random permutation when the sample is too small to stratify.

`evaluate(proxy, X_train, y_train, X_holdout, y_holdout, fallback_threshold)`:

- Fits the proxy on the train split.
- Computes `proxy_f1` using `f1_score` with `average="binary"` for two-class problems and `"macro"` for three or more. The `average` is determined from the **union** of train and holdout labels, so a class that only appears in holdout does not cause a crash.
- Compares `proxy_f1 >= 1.0 - fallback_threshold`. Because LLM labels are used as ground truth, the LLM's own F1 is trivially 1.0.
- Emits a `UserWarning` and sets `use_proxy=False` on failure.

### LLM labeling (`llm.py`)

`label_texts(texts, prompt, model, classes, max_concurrency, cache_dir)` is an async function. It fires one coroutine per text, throttled by an `asyncio.Semaphore`. Each call:

1. Checks the label cache.
2. On a miss: calls `litellm.acompletion` with `response_format={"type": "json_object"}` and `temperature=0`.
3. For binary mode (no `classes`): parses `{"label": true/false}`.
4. For multiclass: parses `{"label": "<class>"}` and returns `"__unknown__"` if the value is not in the allowed list.
5. Writes the result to the label cache.

Retries on transient errors are handled by `_litellm_call_with_retry` in `_utils.py`.

### Embeddings (`embeddings.py`)

`EmbeddingBackend` is an ABC with two requirements:

- `model_id: str` — stable string used as the diskcache key.
- `embed(texts: list[str]) -> np.ndarray` — returns a float32 array of shape `(n, dim)`.

`LiteLLMEmbeddingBackend(model: str)` is the default implementation. It sends texts to any LiteLLM-supported embedding provider in chunks of 2 048.

`embed_texts(texts, backend, cache_dir)` does cache-then-fill: checks the cache for each text, collects misses, calls `backend.embed()` once for all misses, and writes results back. A warning is emitted for inputs exceeding 250 000 texts (memory risk).

Custom backends — sentence-transformers, pre-computed vectors, proprietary APIs — implement `EmbeddingBackend` and pass an instance anywhere a model string is accepted.

### Proxy models (`proxy/`)

`ProxyModel` ABC (`base.py`) defines `fit(X, y)`, `predict(X)`, optional `predict_proba(X)`, and default `save`/`load` via `joblib`.

| Class | Backend | Imbalance handling | `save`/`load` |
|---|---|---|---|
| `LogisticRegressionProxy` | sklearn `LogisticRegression` | `class_weight="balanced"` | joblib |
| `LinearSVCProxy` | sklearn `LinearSVC` | `class_weight="balanced"` | joblib |
| `LightGBMProxy` | LightGBM `LGBMClassifier` | `is_unbalance=True` | LightGBM booster text format |

`LightGBMProxy` overrides `save`/`load` to use the LightGBM native booster format (not joblib), since joblib-serialized LightGBM objects are not portable across LightGBM versions.

### Caching (`cache.py`)

All embeddings and labels are cached in a `diskcache.FanoutCache` (8 SQLite shards, 60 s timeout) at `~/.cache/frugal_ml/` by default.

Cache keys are namespaced by version: `frugal_ml/{VERSION}/...`

- **Embedding key**: `emb / sha256(text) / model_id`
- **Label key**: `lbl / sha256(text) / sha256(prompt) / model_id / classes_key`

`classes_key` is `"binary"` when no classes are provided, or `sha256(sorted(classes))` for multiclass. This ensures that cached binary labels are not reused for a multiclass query on the same text+prompt, and vice versa.

### `Proxy` save/load (`__init__.py`)

`Proxy.save(path)` writes two files:

- `path` — the serialized proxy model (joblib or LightGBM native).
- `path.meta.json` — a sidecar: `{"proxy_type": "lr"|"svc"|"lgbm", "embedding_model": "<model_id>"}`.

`Proxy.load(path, embedding_model=None)` reads the sidecar to determine the proxy type and embedding model, then dispatches to the correct backend's `load()`. If `embedding_model` is passed explicitly it overrides the sidecar value (required for custom `EmbeddingBackend` subclasses, since only a string model ID is stored in the sidecar).

---

## Public API

```python
from frugal_ml import ml_filter, ml_classify, Proxy, EmbeddingBackend

# Online binary filter
mask = ml_filter(
    df,
    prompt="Is this review positive?",
    text_column="review",
    llm="anthropic/claude-haiku-4-5",
    embedding_model="text-embedding-3-small",
    proxy="lr",               # "lr" | "svc" | "lgbm"
    sample_size=1000,
    fallback_threshold=0.1,   # τ: proxy F1 must be >= 1.0 - τ
)

# Online multiclass
labels = ml_classify(
    df,
    prompt="Classify support ticket intent",
    text_column="body",
    llm="anthropic/claude-haiku-4-5",
    embedding_model="text-embedding-3-small",
    classes=["billing", "tech", "other"],
)

# Offline sklearn-style
proxy = Proxy(prompt="...", llm="...", embedding_model="...", model="lgbm")
proxy.fit(train_df, "text")
proxy.save("proxy.lgbm")

loaded = Proxy.load("proxy.lgbm")
preds = loaded.predict(new_df, "text")   # no LLM calls

# Custom embedding backend
class MyBackend(EmbeddingBackend):
    model_id = "my-model-v1"
    def embed(self, texts):
        ...  # return np.ndarray of shape (len(texts), dim)

mask = ml_filter(df, ..., embedding_model=MyBackend())
```

---

## CLI

```
frugal-ml filter   input.parquet --prompt "..." --text-col review --out mask.parquet
frugal-ml classify input.parquet --prompt "..." --text-col review --classes a,b,c --out labels.parquet
frugal-ml embed    input.parquet --text-col review --model text-embedding-3-small --out embeds.npy
frugal-ml label    input.parquet --prompt "..." --text-col review --sample 1000 --out labels.parquet
frugal-ml cache clear
```

All commands accept `--llm`, `--embedding-model`, `--proxy`, `--cache-dir`, `--sample-size`, `--fallback-threshold`, `--max-concurrency`, `--seed`. Input formats: `.parquet`, `.csv`, `.json`, `.jsonl`.

---

## Design decisions

**Why logistic regression as the default proxy?** Embedding models are trained to produce linearly separable representations. The paper's own ablation finds that LR almost always matches or beats more complex classifiers when using modern embeddings. LR trains in seconds on 1 000 samples and infers in < 1 ms per batch.

**Why LightGBM for non-linear tasks?** Histogram-based splits are fast on dense float32 arrays, training is low-memory, and the install is lightweight. It handles class imbalance natively via `is_unbalance=True`.

**Why a separate sidecar file for `save`/`load`?** LightGBM and sklearn use different serialization formats (LightGBM native vs joblib). Storing proxy type and embedding model in a `.meta.json` sidecar lets `Proxy.load()` self-describe and dispatch correctly without requiring callers to remember or pass the original arguments.

**Why include `classes` in the label cache key?** A binary filter (`ml_filter`) and a multiclass classifier (`ml_classify`) can share the same prompt and model. Without a `classes` component in the key, a cached binary `True`/`False` label could be silently returned for a multiclass query expecting `"billing"` / `"tech"` / `"other"`.

**Why `_run_async` instead of `asyncio.run` everywhere?** `asyncio.run()` raises `RuntimeError: This event loop is already running` inside Jupyter notebooks and async frameworks like FastAPI. `_run_async` detects a running loop and either patches it with `nest_asyncio` or offloads to a thread, making the library usable without any setup from the caller.

**Fallback threshold τ.** The paper reports that τ = 0.1 (i.e., proxy F1 ≥ 0.9 of LLM F1) covers > 95% of production use cases. The default is 0.1 but it is user-configurable. Setting τ = 0.0 means the proxy must achieve perfect F1 on the holdout, which will almost always fall back to the LLM.

---

## Advantages over the SQL approach (BigQuery AI.IF / AlloyDB)

The paper's technique is implemented as SQL functions inside Google's data warehouse products — `AI.IF` in BigQuery and accelerated functions in AlloyDB. That surface imposes several constraints that frugal-ml removes.

**No infrastructure dependency.** The SQL approach requires data to be in BigQuery or AlloyDB, a GCP account, and quota. frugal-ml works on any DataFrame — pandas, a local parquet file, a CSV — with no cloud account required. The technique is available to anyone running Python.

**Any LLM and any embedding provider.** BigQuery and AlloyDB are wired to specific Google models (Vertex AI / Gemini). frugal-ml uses LiteLLM as the adapter layer, so the same code path works with Anthropic, OpenAI, Bedrock, Vertex, a local Ollama instance, or any LiteLLM-supported provider. The embedding backend is similarly pluggable via the `EmbeddingBackend` ABC — you can bring pre-computed vectors, a fine-tuned sentence-transformer, or a proprietary embedding API.

**Offline / deploy-once mode.** SQL functions re-run the sample-label-train pipeline on each query invocation. frugal-ml's `Proxy` class separates `fit` from `predict`: you train once, serialize to disk (`proxy.joblib` + `.meta.json` sidecar), and deploy the classifier independently. Subsequent predictions make zero LLM calls and run at classifier speed (~0.1 ms / 1 000 rows for logistic regression).

**Observability and control.** Inside a SQL function you cannot inspect intermediate results. In frugal-ml, `EvalResult` exposes `proxy_f1`, `llm_f1`, `use_proxy`, and `holdout_size` after every run. You can tune `fallback_threshold` explicitly, inspect the trained sklearn/LightGBM object, check holdout predictions, or step through the pipeline in a notebook. The SQL surface hides all of this.

**Integration with the Python ML ecosystem.** Because proxy models are sklearn or LightGBM objects, standard tooling applies directly — feature importances, calibration, cross-validation, SHAP explanations, model registries. None of that is available through a SQL function.

**Iterative development workflow.** A data scientist iterating on a prompt or embedding model has a much tighter feedback loop in Python — run in a notebook, inspect the sample labels, check the proxy F1, adjust, re-run — versus having to push data to a warehouse, re-run a SQL query, and parse results from a table each iteration.

The cost and latency wins described in the paper (300–1 000× reduction at 10M-row scale) transfer fully to frugal-ml because the underlying technique is identical. The difference is that frugal-ml makes those wins available without requiring a Google data warehouse.
