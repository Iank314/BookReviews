# tests/test_pipeline.py

import numpy as np
import pytest
from server.models.book import Books
from server.recommender.recommender import Recommender
from server.recommender.recommendation_engine import RecommendationEngine

class DummyFetcher:
    """Fetcher that returns a predefined list of Books instances."""
    def __init__(self, books):
        self._books = books

    def fetch(self, source=None, **kwargs):
        return self._books


def test_end_to_end_recommendation():
    # Create dummy books with distinct descriptions and tags
    book1 = Books(
        id='1', title='Alpha Tale', authors=['Author A'],
        description='alpha alpha', tags=['x'], metadata={}
    )
    book2 = Books(
        id='2', title='Beta Story', authors=['Author B'],
        description='beta beta', tags=['y'], metadata={}
    )
    book3 = Books(
        id='3', title='Alpha Beta Mix', authors=['Author C'],
        description='alpha beta', tags=['x', 'y'], metadata={}
    )

    books = [book1, book2, book3]
    fetcher = DummyFetcher(books)
    engine = RecommendationEngine(metric='cosine')
    recommender = Recommender(fetcher=fetcher, engine=engine)

    # Build the pipeline (source arg is ignored by DummyFetcher)
    recommender.build(source=None)

    # Get recommendations for book1
    recs = recommender.recommend(book_id='1', top_n=2)
    rec_ids = [b.id for b in recs]

    # Expect book3 (mixed content) first, then book2
    assert rec_ids == ['3', '2']
