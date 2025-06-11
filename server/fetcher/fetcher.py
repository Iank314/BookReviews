"""
server/fetcher/fetcher.py
Fetch book data from a local JSON file, Google Books, or Open Library
and return them as `Books` instances.
"""

from __future__ import annotations

import json
from typing import List, Optional

# External-network imports are done lazily so tests that use DummyFetcher
# don’t require the requests package.
try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None  # will raise at runtime if remote fetch is used

from server.models.book import Books

GOOGLE_ENDPOINT   = "https://www.googleapis.com/books/v1/volumes"
OPENLIB_ENDPOINT  = "https://openlibrary.org/search.json"


class Fetcher:
    """
    Parameters
    ----------
    source : str
        Either a **filepath** (for local JSON) or one of the endpoint
        constants above.
    api_key : str | None
        Optional Google Books API key.  Unused for local/ Open Library.
    """

    def __init__(self, source: str, api_key: Optional[str] = None):
        self.source  = source
        self.api_key = api_key

    # ------------------------------------------------------------------ #
    # Public unified entrypoints
    # ------------------------------------------------------------------ #

    def fetch(self, query: str | None = None, max_results: int = 40) -> List[Books]:
        """
        Unified method:
        * If `self.source` is a file path → read JSON.
        * If it's GOOGLE_ENDPOINT      → hit Google Books.
        * If it's OPENLIB_ENDPOINT     → hit Open Library.

        For remote calls, `query` MUST be provided.
        """
        if self.source in (GOOGLE_ENDPOINT, OPENLIB_ENDPOINT):
            if not query:
                raise ValueError("`query` is required when fetching remotely.")
            if self.source == GOOGLE_ENDPOINT:
                return self._fetch_google_books(query, max_results)
            return self._fetch_open_library(query, max_results)

        # Otherwise treat `source` as a local file path
        return self._fetch_from_file()

    # ------------------------------------------------------------------ #
    # Local JSON
    # ------------------------------------------------------------------ #

    def _fetch_from_file(self) -> List[Books]:
        with open(self.source, "r", encoding="utf-8") as fh:
            raw = json.load(fh)  # expects a list[dict]
        return [self._from_local_dict(obj) for obj in raw]

    @staticmethod
    def _from_local_dict(raw: dict) -> Books:
        return Books(
            id=raw["id"],
            title=raw["title"],
            authors=raw.get("authors", []),
            description=raw.get("description", ""),
            tags=raw.get("tags", []),
            metadata=raw.get("metadata", {}),
        )

    # ------------------------------------------------------------------ #
    # Google Books
    # ------------------------------------------------------------------ #

    def _fetch_google_books(self, query: str, max_results: int) -> List[Books]:
        if requests is None:  # pragma: no cover
            raise ImportError("Install `requests` to use Google Books fetching.")
        params = {"q": query, "maxResults": max_results}
        if self.api_key:
            params["key"] = self.api_key
        resp = requests.get(GOOGLE_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [self._from_google_item(it) for it in items]

    @staticmethod
    def _from_google_item(item: dict) -> Books:
        info = item.get("volumeInfo", {})
        return Books(
            id=item.get("id", ""),
            title=info.get("title", ""),
            authors=info.get("authors", []),
            description=info.get("description", ""),
            tags=info.get("categories", []),
            metadata={
                "publishedDate": info.get("publishedDate"),
                "pageCount": info.get("pageCount"),
                "infoLink": info.get("infoLink"),
            },
        )

    # ------------------------------------------------------------------ #
    # Open Library
    # ------------------------------------------------------------------ #

    def _fetch_open_library(self, query: str, max_results: int) -> List[Books]:
        if requests is None:  # pragma: no cover
            raise ImportError("Install `requests` to use Open Library fetching.")
        params = {"q": query, "limit": max_results}
        resp = requests.get(OPENLIB_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        docs = resp.json().get("docs", [])
        return [self._from_openlib_doc(doc) for doc in docs]

    @staticmethod
    def _from_openlib_doc(doc: dict) -> Books:
        return Books(
            id=doc.get("key", ""),
            title=doc.get("title", ""),
            authors=doc.get("author_name", []),
            description=doc.get("first_sentence", ""),
            tags=doc.get("subject", [])[:5],  # keep tag list small
            metadata={"publish_year": doc.get("first_publish_year")},
        )
