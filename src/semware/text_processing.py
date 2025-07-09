"""
Text processing utilities for SemWare
"""

import re
from typing import List, Tuple
from difflib import SequenceMatcher


def split_into_chunks(text: str, words_per_chunk: int = 100) -> List[str]:
    """
    Split text into chunks of approximately words_per_chunk words.
    
    Args:
        text: The text to split
        words_per_chunk: Number of words per chunk (default: 100)
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split text into words
    words = text.split()
    
    if len(words) <= words_per_chunk:
        return [text]
    
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
    
    return chunks


def count_words(text: str) -> int:
    """
    Count the number of words in text.
    
    Args:
        text: The text to count words in
    
    Returns:
        Number of words
    """
    if not text:
        return 0
    return len(text.split())


def calculate_content_difference(old_text: str, new_text: str) -> Tuple[int, float]:
    """
    Calculate the difference between old and new content.
    
    Args:
        old_text: The old text content
        new_text: The new text content
    
    Returns:
        Tuple of (word_difference, similarity_ratio)
    """
    if not old_text and not new_text:
        return 0, 1.0
    
    if not old_text:
        return count_words(new_text), 0.0
    
    if not new_text:
        return count_words(old_text), 0.0
    
    # Calculate word difference
    old_words = set(old_text.lower().split())
    new_words = set(new_text.lower().split())
    
    word_difference = len(old_words.symmetric_difference(new_words))
    
    # Calculate similarity ratio using SequenceMatcher
    similarity_ratio = SequenceMatcher(None, old_text, new_text).ratio()
    
    return word_difference, similarity_ratio


def should_regenerate_embeddings(old_text: str, new_text: str, min_word_difference: int = 20) -> bool:
    """
    Determine if embeddings should be regenerated based on content change.
    
    Args:
        old_text: The old text content
        new_text: The new text content
        min_word_difference: Minimum word difference to trigger regeneration
    
    Returns:
        True if embeddings should be regenerated, False otherwise
    """
    word_difference, _ = calculate_content_difference(old_text, new_text)
    return word_difference >= min_word_difference


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: The text to clean
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text 