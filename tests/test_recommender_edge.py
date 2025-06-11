"""Edge‑case tests for the Recommender pipeline."""

import numpy as np
from server.models.book import Books
from server.recommender.recommender import Recommender
from server.recommender.recommendation_engine import RecommendationEngine

# -----------------------------------------------------------------------------
# Dummy fetcher identical to the one in test_pipeline but defined locally so we
# don't introduce a shared dependency across tests.
# -----------------------------------------------------------------------------
class DummyFetcher:
    def __init__(self, books):
        self._books = books

    def fetch(self, source=None, **kwargs):
        return self._books


# -----------------------------------------------------------------------------
# Test: recommend() should return an empty list for an unknown book ID.
# -----------------------------------------------------------------------------

def test_recommend_invalid_book_id():
    book1 = Books(id="1", title="Alpha", authors=["A"], description="alpha", tags=[], metadata={})
    book2 = Books(id="2", title="Beta",  authors=["B"], description="beta",  tags=[], metadata={})
    books = [book1, book2]

    fetcher = DummyFetcher(books)
    engine  = RecommendationEngine(metric="cosine")
    rec     = Recommender(fetcher=fetcher, engine=engine)
    rec.build(source=None)

    # Ask for a non‑existent ID
    result = rec.recommend(book_id="NON_EXISTENT", top_n=3)
    assert result == [], "Expected empty list for unknown book ID"
