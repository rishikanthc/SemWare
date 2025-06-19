# zendown_ai/database.py

import lancedb
import os
import pandas as pd
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
from .embeddings import SentenceTransformerEmbeddings

DB_PATH = os.getenv("LANCEDB_URI", ".lancedb")
TABLE_NAME = "zendown_documents"

registry = EmbeddingFunctionRegistry.get_instance()
embedding_function_creator = registry.get("sentence-transformers")
if not embedding_function_creator:
    raise RuntimeError("SentenceTransformerEmbeddings not found in registry.")
embedding_function = embedding_function_creator.create()


class DocumentSchema(LanceModel):
    id: str
    text: str = embedding_function.SourceField()
    vector: Vector(embedding_function.ndims()
                   ) = embedding_function.VectorField()


_db_conn = None
_table = None


def init_db(db_path: str = DB_PATH, table_name: str = TABLE_NAME):
    global _db_conn, _table
    if _db_conn is None:
        os.makedirs(db_path, exist_ok=True)
        _db_conn = lancedb.connect(db_path)
    if table_name not in _db_conn.table_names():
        _table = _db_conn.create_table(table_name, schema=DocumentSchema)
        print(f"Table '{table_name}' created in '{db_path}'.")
    else:
        _table = _db_conn.open_table(table_name)
        print(f"Opened existing table '{table_name}' from '{db_path}'.")
    return _table


def get_table():
    if _table is None:
        return init_db()
    return _table


def upsert_document(document_id: str, content: str):
    table = get_table()
    arrow_schema = pa.schema([
        pa.field('id', pa.string(), nullable=False),
        pa.field('text', pa.string(), nullable=False)
    ])
    data_for_arrow = {'id': [document_id], 'text': [content]}
    arrow_table_to_upsert = pa.Table.from_pydict(
        data_for_arrow, schema=arrow_schema)
    table.merge_insert("id") \
        .when_matched_update_all() \
        .when_not_matched_insert_all() \
        .execute(arrow_table_to_upsert)
    print(f"Document '{document_id}' upserted successfully.")


def search_similar_documents(document_id: str, threshold: float, limit: int = 10) -> list[dict[str, any]]:
    table = get_table()
    all_data_df = table.to_pandas()
    document_row_df = all_data_df[all_data_df['id'] == document_id]
    if document_row_df.empty:
        raise ValueError(f"Document with id '{document_id}' not found.")
    if 'vector' not in document_row_df.columns or document_row_df['vector'].iloc[0] is None:
        raise ValueError(f"Document with id '{
                         document_id}' found, but its vector is missing or null.")
    query_vector = document_row_df['vector'].iloc[0]
    search_query = table.search(query_vector).metric("cosine")
    search_query = search_query.where(f"id != '{document_id}'").limit(limit)
    results = search_query.to_list()
    similar_docs_with_scores = []
    for doc in results:
        cosine_distance = doc['_distance']
        # For cosine distance, similarity = 1 - distance
        # A smaller distance means higher similarity.
        # LanceDB's cosine distance is 1 - cosine_similarity.
        # So, similarity_score = 1 - cosine_distance (which is the actual cosine similarity)
        similarity_score = 1.0 - cosine_distance
        if similarity_score >= threshold:
            similar_docs_with_scores.append({
                "id": doc["id"],
                "score": similarity_score
            })
    return similar_docs_with_scores


def semantic_search_documents(query_text: str, threshold: float, limit: int = 10) -> list[dict[str, any]]:
    table = get_table()
    if not query_text:
        raise ValueError("Query text cannot be empty.")

    # Generate embedding for the query text
    # The embedding_function.generate_embeddings typically returns a list of embeddings
    # For a single query text, it will be a list containing one embedding.
    query_vector_list = embedding_function.generate_embeddings(query_text)
    if not query_vector_list or not query_vector_list[0]:
        raise ValueError("Failed to generate embedding for the query text.")
    query_vector = query_vector_list[0] # Get the first (and only) embedding

    # Perform search
    search_query = table.search(query_vector).metric("cosine").limit(limit)
    results = search_query.to_list()

    relevant_docs_with_scores = []
    for doc in results:
        cosine_distance = doc['_distance']
        # For cosine distance, similarity_score = 1 - cosine_distance
        similarity_score = 1.0 - cosine_distance
        if similarity_score >= threshold:
            relevant_docs_with_scores.append({
                "id": doc["id"],
                "score": similarity_score
            })
    return relevant_docs_with_scores
