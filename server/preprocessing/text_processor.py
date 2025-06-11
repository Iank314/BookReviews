import re
import string
from typing import List, Optional

class Preprocessor:
    """Basic text cleaner.

    Steps:
    1. Remove HTML tags.
    2. Strip URLs.
    3. Lower-case.
    4. Remove punctuation.
    5. Collapse excess whitespace.
    6. Optionally drop extra stopwords supplied at construction time.
    """

    def __init__(self, extra_stopwords: Optional[List[str]] = None) -> None:
        self.extra_stopwords = set(extra_stopwords or [])

    def process(self, text: str) -> str:
        # 1) Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # 2) Strip URLs
        text = re.sub(r"http\S+", " ", text)
        # 3) Lower-case
        text = text.lower()
        # 4) Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # 5) Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # 6) Optional stop-word removal
        if self.extra_stopwords:
            tokens = [t for t in text.split() if t not in self.extra_stopwords]
            text = " ".join(tokens)
        return text