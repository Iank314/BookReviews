from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import numpy as np

class TagEncoder:
    """One-hot encoder for list-of-tags data."""
    def __init__(self):
        self.vocab_: Dict[str, int] = {}

    # -------------------------------
    # Fit / Transform helpers
    # -------------------------------
    def fit(self, tags: List[List[str]]):
        """Build a vocabulary from nested tag lists."""
        unique = {t for sub in tags for t in sub}
        self.vocab_ = {tag: i for i, tag in enumerate(sorted(unique))}
        return self

    def transform(self, tags: List[List[str]]):
        rows, cols = len(tags), len(self.vocab_)
        mat = np.zeros((rows, cols), dtype=int)
        for r, taglist in enumerate(tags):
            for t in taglist:
                c = self.vocab_.get(t)
                if c is not None:
                    mat[r, c] = 1
        return csr_matrix(mat)

    def fit_transform(self, tags: List[List[str]]):
        self.fit(tags)
        return self.transform(tags)


class FeatureExtractor:
    """Vectorises descriptions (TF-IDF) + tags (one-hot) → sparse matrix."""
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.text_vect = TfidfVectorizer(max_features=max_features,
                                         ngram_range=ngram_range)
        self.tag_enc   = TagEncoder()

    # -------------------------------
    # Public API
    # -------------------------------
    def fit_transform(self,
                      descriptions: Dict[str, str],
                      tags: Dict[str, List[str]]):
        ids   = list(descriptions.keys())
        texts = [descriptions[i] for i in ids]
        tlists = [tags[i] for i in ids]

        X_text = self.text_vect.fit_transform(texts)
        X_tags = self.tag_enc.fit_transform(tlists)
        X      = hstack([X_text, X_tags])
        return X, ids

    def transform(self,
                  descriptions: Dict[str, str],
                  tags: Dict[str, List[str]]):
        ids   = list(descriptions.keys())
        texts = [descriptions[i] for i in ids]
        tlists = [tags[i] for i in ids]

        X_text = self.text_vect.transform(texts)
        X_tags = self.tag_enc.transform(tlists)
        return hstack([X_text, X_tags])