# BookReviews – Content-Based Book Recommender

A lightweight, offline-friendly recommender system that fetches books from **Google Books**, **Open Library**, or a local JSON file, then suggests similar titles—or recommends books based on free-text adjectives.

---

## Tech Stack

| Layer | Library / Tool | Why |
|-------|----------------|-----|
| Language | **Python 3.11** | Rapid prototyping & rich AI/NLP ecosystem |
| Data ops | **NumPy / SciPy** | Sparse matrices & fast linear algebra |
| NLP + ML | **scikit-learn** | TF-IDF vectoriser, cosine similarity |
| HTTP | **requests** | Simple, battle-tested REST client |
| Testing | **pytest + unittest** | Fast, readable unit tests |
| Packaging | Standard `__init__.py` packages | Keeps modules import-safe across scripts & tests |

---

## How the Pieces Connect (Execution Order)

┌────────────┐ 1 query / file path
│ script │──────────┐
└────────────┘ │
▼
┌───────────────────────┐ 2 fetch Books objects
│ Fetcher (Google / │
│ OpenLibrary / JSON) │
└───────────────────────┘
▼ 3 add()
┌────────────┐
│ Library │ stores every Book
└────────────┘
▼ 4 preprocess descriptions
┌──────────────────┐
│ Preprocessor │ clean, lowercase, strip URLs …
└──────────────────┘
▼ 5 vectorise (TF-IDF + tag one-hot)
┌──────────────────────┐
│ FeatureExtractor │ → sparse matrix
└──────────────────────┘
▼ 6 fit()
┌──────────────────────────┐
│ RecommendationEngine │ cosine similarity matrix
└──────────────────────────┘
▼ 7 recommend()
┌────────────┐
│ Library │ lookup IDs → Book objects
└────────────┘

scripts/
└─ demo_query.py ← sample CLI demo
server/
├─ features/ ← TF-IDF + tag encoder
│ └─ features.py
├─ fetcher/ ← Google, Open Library, local JSON
│ └─ fetcher.py
├─ models/
│ ├─ book.py ← Books dataclass
│ └─ library.py ← collection + CRUD/search
├─ preprocessing/
│ ├─ init.py ← re-exports Preprocessor
│ └─ text_processor.py ← simple cleaner
├─ recommender/
│ ├─ recommendation_engine.py
│ └─ recommender.py ← orchestrates full pipeline
└─ tests/ ← pytest & unittest suites



## Key Module & Method Overview

### `server/fetcher/fetcher.py`

| Method | Purpose |
|--------|---------|
| `fetch(query, max_results)` | Unified entry point; routes to local, Google, or Open Library |
| `_fetch_google_books()`     | REST call → JSON items |
| `_fetch_open_library()`     | REST call → docs |
| `_fetch_from_file()`        | Read local `books.json` |

### `server/models/library.py`

| Method | Purpose |
|--------|---------|
| `add(book)` / `remove(id)` | Mutate in-memory store |
| `get_by_id(id)`            | Fetch single Book |
| `find_by_title/author/tag` | Substring searches |
| `all()`                    | Return all books |

### `server/preprocessing/text_processor.py`

`process(text)` → Strip HTML & URLs, lowercase, remove punctuation, collapse spaces, optional stop-word drop.

### `server/features/features.py`

| Method | Purpose |
|--------|---------|
| `fit_transform(descs, tags)` | TF-IDF on descriptions + one-hot tags → sparse matrix & ID list |
| `transform(descs, tags)`     | Same vector space for new data / queries |

### `server/recommender/recommendation_engine.py`

| Method | Purpose |
|--------|---------|
| `fit(matrix, ids)`           | Store features, pre-compute cosine matrix |
| `recommend(book_id, k)`      | Top-*k* similar books |
| `recommend_for_vector(vec,k)`| Similar books for arbitrary vector |

### `server/recommender/recommender.py`

| Method | Purpose |
|--------|---------|
| `build(query=…, max_results=…)` | fetch → preprocess → vectorise → fit |
| `recommend(book_id, k)`         | Convenience wrapper |
| `recommend_by_text(text, k)`    | Free-text adjectives → recommendations |

---

## Running the Demo

```bash
# Install deps
python -m pip install -r requirements.txt   # numpy, scipy, scikit-learn, requests, pytest

# Live Google Books demo
python -m scripts.demo_query

# Run tests
python -m pytest