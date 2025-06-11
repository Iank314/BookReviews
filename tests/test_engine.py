import numpy as np
import pytest
from server.recommender.recommendation_engine import RecommendationEngine

@pytest.fixture
def sample_engine():
    X = np.array([[1,0], [0,1], [1,1]])
    ids = ['book1', 'book2', 'book3']
    engine = RecommendationEngine()
    engine.fit(X, ids)
    return engine

def test_recommendations_for_book1(sample_engine):
    recs = sample_engine.recommend('book1', top_n=2)
    assert recs == ['book3', 'book2']
