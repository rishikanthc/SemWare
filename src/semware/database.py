"""
Database module for SemWare
"""

import lancedb
import os
import pandas as pd
import pyarrow as pa
from datetime import datetime
from typing import List, Dict, Any, Optional
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry

from .embeddings import SentenceTransformerEmbeddings
from .text_processing import split_into_chunks, should_regenerate_embeddings, clean_text

DB_PATH = os.getenv("LANCEDB_URI", "db/semware_documents.lance")
TABLE_NAME = "documents"

registry = EmbeddingFunctionRegistry.get_instance()
embedding_function_creator = registry.get("sentence-transformers")
if not embedding_function_creator:
    raise RuntimeError("SentenceTransformerEmbeddings not found in registry.")
embedding_function = embedding_function_creator.create()


class DocumentChunkSchema(LanceModel):
    """Schema for individual document chunks with embeddings"""
    document_id: str
    chunk_index: int
    chunk_text: str = embedding_function.SourceField()
    vector: Vector(embedding_function.ndims()) = embedding_function.VectorField()
    word_count: int
    created_at: str
    updated_at: str


class DocumentMetadataSchema(LanceModel):
    """Schema for document metadata"""
    document_id: str
    full_content: str
    total_chunks: int
    total_words: int
    created_at: str
    updated_at: str


_db_conn = None
_chunks_table = None
_metadata_table = None


def init_db(db_path: str = DB_PATH, table_name: str = TABLE_NAME):
    """Initialize the database and tables"""
    global _db_conn, _chunks_table, _metadata_table
    
    if _db_conn is None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _db_conn = lancedb.connect(db_path)
    
    chunks_table_name = f"{table_name}_chunks"
    metadata_table_name = f"{table_name}_metadata"
    
    # Initialize chunks table
    if chunks_table_name not in _db_conn.table_names():
        _chunks_table = _db_conn.create_table(chunks_table_name, schema=DocumentChunkSchema)
        print(f"Table '{chunks_table_name}' created in '{db_path}'.")
    else:
        _chunks_table = _db_conn.open_table(chunks_table_name)
        print(f"Opened existing table '{chunks_table_name}' from '{db_path}'.")
    
    # Initialize metadata table
    if metadata_table_name not in _db_conn.table_names():
        _metadata_table = _db_conn.create_table(metadata_table_name, schema=DocumentMetadataSchema)
        print(f"Table '{metadata_table_name}' created in '{db_path}'.")
    else:
        _metadata_table = _db_conn.open_table(metadata_table_name)
        print(f"Table '{metadata_table_name}' created in '{db_path}'.")
    
    return _chunks_table, _metadata_table


def get_tables():
    """Get the database tables"""
    if _chunks_table is None or _metadata_table is None:
        return init_db()
    return _chunks_table, _metadata_table


def get_document_metadata(document_id: str) -> Optional[Dict[str, Any]]:
    """Get document metadata by ID"""
    chunks_table, metadata_table = get_tables()
    
    # Query metadata table
    metadata_df = metadata_table.search().where(f"document_id = '{document_id}'").to_pandas()
    
    if metadata_df.empty:
        return None
    
    return metadata_df.iloc[0].to_dict()


def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """Get all chunks for a document"""
    chunks_table, _ = get_tables()
    
    chunks_df = chunks_table.search().where(f"document_id = '{document_id}'").to_pandas()
    
    if chunks_df.empty:
        return []
    
    return chunks_df.to_dict('records')


def get_metadata_arrow_schema():
    return pa.schema([
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("full_content", pa.string(), nullable=False),
        pa.field("total_chunks", pa.int32(), nullable=False),
        pa.field("total_words", pa.int32(), nullable=False),
        pa.field("created_at", pa.string(), nullable=False),
        pa.field("updated_at", pa.string(), nullable=False),
    ])

def get_chunk_arrow_schema():
    return pa.schema([
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("chunk_index", pa.int32(), nullable=False),
        pa.field("chunk_text", pa.string(), nullable=False),
        pa.field("vector", pa.list_(pa.float32(), embedding_function.ndims()), nullable=False),
        pa.field("word_count", pa.int32(), nullable=False),
        pa.field("created_at", pa.string(), nullable=False),
        pa.field("updated_at", pa.string(), nullable=False),
    ])


def upsert_document(document_id: str, content: str) -> Dict[str, Any]:
    """
    Upsert a document with efficient embedding generation.
    Only regenerates embeddings if content has changed significantly.
    """
    chunks_table, metadata_table = get_tables()
    
    # Clean the content
    content = clean_text(content)
    
    # Check if document exists
    existing_metadata = get_document_metadata(document_id)
    
    if existing_metadata:
        # Check if we need to regenerate embeddings
        if not should_regenerate_embeddings(existing_metadata['full_content'], content):
            # Update metadata only
            now = datetime.utcnow().isoformat()
            metadata_data = {
                "document_id": [document_id],
                "full_content": [content],
                "total_chunks": [existing_metadata['total_chunks']],
                "total_words": [len(content.split())],
                "created_at": [existing_metadata['created_at']],
                "updated_at": [now]
            }
            
            metadata_arrow = pa.Table.from_pydict(metadata_data, schema=get_metadata_arrow_schema())
            metadata_table.merge_insert("document_id").when_matched_update_all().execute(metadata_arrow)
            
            return {
                "document_id": document_id,
                "action": "updated_metadata_only",
                "chunks_regenerated": False,
                "total_chunks": existing_metadata['total_chunks']
            }
    
    # Generate chunks and embeddings
    chunks = split_into_chunks(content)
    chunk_texts = [chunk for chunk in chunks if chunk.strip()]
    
    if not chunk_texts:
        raise ValueError("No valid text chunks found in content")
    
    # Generate embeddings for all chunks
    embeddings = embedding_function.generate_embeddings(chunk_texts)
    
    if len(embeddings) != len(chunk_texts):
        raise ValueError("Mismatch between number of chunks and embeddings")
    
    # Prepare chunk data
    now = datetime.utcnow().isoformat()
    chunk_data = []
    
    for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings)):
        chunk_data.append({
            "document_id": document_id,
            "chunk_index": i,
            "chunk_text": chunk_text,
            "vector": embedding,
            "word_count": len(chunk_text.split()),
            "created_at": now,
            "updated_at": now
        })
    
    # Delete existing chunks for this document
    chunks_table.delete(f"document_id = '{document_id}'")
    
    # Insert new chunks
    if chunk_data:
        chunks_arrow = pa.Table.from_pylist(chunk_data, schema=get_chunk_arrow_schema())
        chunks_table.add(chunks_arrow)
    
    # Upsert metadata
    metadata_data = {
        "document_id": [document_id],
        "full_content": [content],
        "total_chunks": [len(chunk_data)],
        "total_words": [len(content.split())],
        "created_at": [now if not existing_metadata else existing_metadata['created_at']],
        "updated_at": [now]
    }
    
    metadata_arrow = pa.Table.from_pydict(metadata_data, schema=get_metadata_arrow_schema())
    metadata_table.merge_insert("document_id").when_matched_update_all().when_not_matched_insert_all().execute(metadata_arrow)
    
    return {
        "document_id": document_id,
        "action": "upserted_with_embeddings",
        "chunks_regenerated": True,
        "total_chunks": len(chunk_data)
    }


def get_all_document_ids() -> List[str]:
    """Get all document IDs"""
    _, metadata_table = get_tables()
    
    metadata_df = metadata_table.search().to_pandas()
    
    if metadata_df.empty:
        return []
    
    return metadata_df['document_id'].tolist()


def get_document_content(document_id: str) -> Optional[str]:
    """Get document content by ID"""
    metadata = get_document_metadata(document_id)
    
    if metadata is None:
        return None
    
    return metadata['full_content']


def get_document_embeddings(document_id: str) -> List[Dict[str, Any]]:
    """Get all embeddings for a document"""
    chunks = get_document_chunks(document_id)
    
    if not chunks:
        return []
    
    return [
        {
            "chunk_index": chunk['chunk_index'],
            "vector": chunk['vector'],
            "text": chunk['chunk_text']
        }
        for chunk in chunks
    ]


def delete_document(document_id: str) -> bool:
    """Delete a document and all its chunks"""
    chunks_table, metadata_table = get_tables()
    
    # Check if document exists
    existing_metadata = get_document_metadata(document_id)
    
    if not existing_metadata:
        return False
    
    # Delete chunks
    chunks_table.delete(f"document_id = '{document_id}'")
    
    # Delete metadata
    metadata_table.delete(f"document_id = '{document_id}'")
    
    return True


def search_similar_documents(document_id: str, threshold: float = 0.7, limit: int = 10) -> List[Dict[str, Any]]:
    """Find documents similar to the given document"""
    chunks_table, _ = get_tables()
    
    # Get document chunks
    document_chunks = get_document_chunks(document_id)
    
    if not document_chunks:
        raise ValueError(f"Document with id '{document_id}' not found or has no chunks.")
    
    # Use the first chunk as the query vector (or average of all chunks)
    query_vector = document_chunks[0]['vector']
    
    # Search for similar chunks
    search_query = chunks_table.search(query_vector).metric("cosine")
    search_query = search_query.where(f"document_id != '{document_id}'").limit(limit * 2)  # Get more to filter by threshold
    
    results = search_query.to_list()
    
    # Group by document and calculate average similarity
    document_scores = {}
    
    for result in results:
        doc_id = result['document_id']
        similarity_score = result['_distance']
        
        if similarity_score >= threshold:
            if doc_id not in document_scores:
                document_scores[doc_id] = []
            document_scores[doc_id].append(similarity_score)
    
    # Calculate average scores and sort
    similar_docs = []
    for doc_id, scores in document_scores.items():
        avg_score = sum(scores) / len(scores)
        similar_docs.append({"id": doc_id, "score": avg_score})
    
    # Sort by score and limit results
    similar_docs.sort(key=lambda x: x['score'], reverse=True)
    return similar_docs[:limit]


def semantic_search_documents(
    query_text: str, 
    threshold: float = None, 
    top_k: int = None, 
    distance_metric: str = "cosine"
) -> List[Dict[str, Any]]:
    """
    Perform semantic search across all documents with configurable parameters.
    
    Args:
        query_text: Text to search for
        threshold: Minimum similarity score (0.0 to 1.0 for cosine, any positive value for euclidean)
        top_k: Maximum number of results to return
        distance_metric: Distance metric to use ("cosine" or "euclidean")
    
    Returns:
        List of documents with their similarity scores
    
    Raises:
        ValueError: If both threshold and top_k are specified, or invalid parameters
    """
    chunks_table, _ = get_tables()
    
    if not query_text:
        raise ValueError("Query text cannot be empty.")
    
    # Validate distance metric
    if distance_metric not in ["cosine", "l2"]:
        raise ValueError("Distance metric must be 'cosine' or 'l2' (euclidean)")
    
    # Validate threshold and top_k parameters
    if threshold is not None and top_k is not None:
        raise ValueError(
            "Cannot specify both 'threshold' and 'top_k' parameters. "
            "Please use either threshold-based filtering or top_k limiting, not both."
        )
    
    # Generate embedding for the query text
    query_embeddings = embedding_function.generate_embeddings(query_text)
    
    if not query_embeddings or not query_embeddings[0]:
        raise ValueError("Failed to generate embedding for the query text.")
    
    query_vector = query_embeddings[0]
    
    # Determine search limit based on parameters
    search_limit = 1000  # Large limit to get all potential matches
    if top_k is not None:
        search_limit = max(top_k * 5, 100)  # Get more results than needed for better aggregation
    
    # Search for similar chunks
    search_query = chunks_table.search(query_vector).metric(distance_metric).limit(search_limit)
    results = search_query.to_list()
    
    # Group by document and aggregate scores
    document_scores = {}
    
    for result in results:
        doc_id = result['document_id']
        distance = result['_distance']
        
        # Convert distance to similarity score
        if distance_metric == "cosine":
            # For cosine distance: similarity = 1 - distance
            similarity_score = 1.0 - distance
        else:  # l2 (euclidean)
            # For euclidean distance: convert to similarity using exponential decay
            # This gives higher scores for smaller distances
            similarity_score = 1.0 / (1.0 + distance)
        
        if doc_id not in document_scores:
            document_scores[doc_id] = []
        document_scores[doc_id].append(similarity_score)
    
    # Aggregate scores per document (using mean aggregation)
    aggregated_docs = []
    for doc_id, scores in document_scores.items():
        # Use mean aggregation for document-level similarity
        avg_score = sum(scores) / len(scores)
        aggregated_docs.append({"id": doc_id, "score": avg_score})
    
    # Sort by score (highest first)
    aggregated_docs.sort(key=lambda x: x['score'], reverse=True)
    
    # Apply filtering based on parameters
    if threshold is not None:
        # Filter by threshold
        filtered_docs = [doc for doc in aggregated_docs if doc['score'] >= threshold]
        return filtered_docs
    elif top_k is not None:
        # Return top_k results
        return aggregated_docs[:top_k]
    else:
        # Return all results (default behavior)
        return aggregated_docs 