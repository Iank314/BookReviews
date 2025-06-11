from typing import List, Optional
from .book import Books

class Library:
    """
    Manages a collection of Books, providing CRUD and query methods.
    """
    def __init__(self):
        self._books: dict[str, Books] = {}

    def add(self, book: Books) -> None:
        """Add or replace a book in the library."""
        self._books[book.id] = book

    def remove(self, book_id: str) -> None:
        """Remove a book by its ID, if present."""
        self._books.pop(book_id, None)

    def all(self) -> List[Books]:
        """Return all books in the library."""
        return list(self._books.values())

    def get_by_id(self, book_id: str) -> Optional[Books]:
        """Fetch a single book by ID, or None if not found."""
        return self._books.get(book_id)

    def find_by_title(self, title_substr: str) -> List[Books]:
        """Return books whose titles contain the given substring (case-insensitive)."""
        return [b for b in self._books.values() if title_substr.lower() in b.title.lower()]

    def find_by_author(self, author_name: str) -> List[Books]:
        """Return books matching an author name (case-insensitive)."""
        return [b for b in self._books.values()
                if any(author_name.lower() in a.lower() for a in b.authors)]

    def find_by_tag(self, tag: str) -> List[Books]:
        """Return books tagged with the given tag."""
        return [b for b in self._books.values() if tag in b.tags]