"""
Main FastAPI application for SemWare
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import uvicorn

from .database import (
    init_db,
    upsert_document,
    get_all_document_ids,
    get_document_content,
    get_document_embeddings,
    delete_document,
    search_similar_documents,
    semantic_search_documents,
    DB_PATH,
    TABLE_NAME
)
from .auth import api_key_auth


# Request Models
class UpsertRequest(BaseModel):
    id: str = Field(
        ..., 
        description="Unique document identifier",
        example="doc_001"
    )
    content: str = Field(
        ..., 
        min_length=1, 
        description="Document content to be processed and stored",
        example="This is a sample document about artificial intelligence and machine learning."
    )
    
    class Config:
        schema_extra = {
            "example": {
                "id": "doc_001",
                "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. These machines can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation."
            }
        }


class SearchRequest(BaseModel):
    id: str = Field(
        ..., 
        description="Document ID to find similar documents for",
        example="doc_001"
    )
    threshold: Optional[float] = Field(
        None, 
        ge=0.0, 
        description="Minimum similarity score threshold. For cosine: 0.0-1.0 (higher = more similar). For l2: any positive value (lower = more similar)",
        example=0.7
    )
    top_k: Optional[int] = Field(
        None, 
        gt=0, 
        le=1000, 
        description="Maximum number of results to return",
        example=5
    )
    distance_metric: str = Field(
        "cosine", 
        description="Distance metric to use for similarity calculation",
        example="cosine"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "id": "doc_001",
                "threshold": 0.7,
                "top_k": 5,
                "distance_metric": "cosine"
            }
        }


class SemanticSearchRequest(BaseModel):
    query_text: str = Field(
        ..., 
        min_length=1, 
        description="Natural language query text to search for in documents",
        example="artificial intelligence and machine learning"
    )
    threshold: Optional[float] = Field(
        None, 
        ge=0.0, 
        description="Minimum similarity score threshold. For cosine: 0.0-1.0 (higher = more similar). For l2: any positive value (lower = more similar)",
        example=0.6
    )
    top_k: Optional[int] = Field(
        None, 
        gt=0, 
        le=1000, 
        description="Maximum number of results to return",
        example=10
    )
    distance_metric: str = Field(
        "cosine", 
        description="Distance metric to use for similarity calculation",
        example="cosine"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query_text": "artificial intelligence and machine learning",
                "threshold": 0.6,
                "top_k": 10,
                "distance_metric": "cosine"
            }
        }


# Response Models
class UpsertResponse(BaseModel):
    message: str = Field(description="Success message")
    document_id: str = Field(description="ID of the upserted document")
    action: str = Field(description="Action performed: 'created' or 'updated'")
    chunks_regenerated: bool = Field(description="Whether embeddings were regenerated")
    total_chunks: int = Field(description="Total number of chunks created")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Document upserted successfully",
                "document_id": "doc_001",
                "action": "created",
                "chunks_regenerated": True,
                "total_chunks": 3
            }
        }


class DocumentInfo(BaseModel):
    id: str = Field(description="Document identifier")
    content: str = Field(description="Full document content")
    total_chunks: int = Field(description="Number of chunks the document was split into")
    total_words: int = Field(description="Total word count in the document")
    created_at: str = Field(description="Document creation timestamp")
    updated_at: str = Field(description="Document last update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "doc_001",
                "content": "Artificial Intelligence (AI) is a branch of computer science...",
                "total_chunks": 3,
                "total_words": 150,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }


class EmbeddingInfo(BaseModel):
    chunk_index: int = Field(description="Index of the chunk (0-based)")
    text: str = Field(description="Text content of the chunk")
    vector: List[float] = Field(description="Embedding vector for the chunk")
    
    class Config:
        schema_extra = {
            "example": {
                "chunk_index": 0,
                "text": "Artificial Intelligence (AI) is a branch of computer science...",
                "vector": [0.1, 0.2, 0.3, ...]
            }
        }


class DocumentEmbeddingsResponse(BaseModel):
    document_id: str = Field(description="Document identifier")
    embeddings: List[EmbeddingInfo] = Field(description="List of chunk embeddings")
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc_001",
                "embeddings": [
                    {
                        "chunk_index": 0,
                        "text": "Artificial Intelligence (AI) is a branch...",
                        "vector": [0.1, 0.2, 0.3, ...]
                    }
                ]
            }
        }


class SimilarDocument(BaseModel):
    id: str = Field(description="Document identifier")
    score: float = Field(description="Similarity score (higher is more similar for cosine, lower for l2)")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "doc_002",
                "score": 0.85
            }
        }


class SearchResponse(BaseModel):
    query_id: str = Field(description="ID of the document used for similarity search")
    similar_results: List[SimilarDocument] = Field(description="List of similar documents with scores")
    count: int = Field(description="Number of results returned")
    
    class Config:
        schema_extra = {
            "example": {
                "query_id": "doc_001",
                "similar_results": [
                    {"id": "doc_002", "score": 0.85},
                    {"id": "doc_003", "score": 0.72}
                ],
                "count": 2
            }
        }


class SemanticSearchResponse(BaseModel):
    query_text: str = Field(description="Original query text used for search")
    similar_results: List[SimilarDocument] = Field(description="List of relevant documents with scores")
    count: int = Field(description="Number of results returned")
    
    class Config:
        schema_extra = {
            "example": {
                "query_text": "artificial intelligence and machine learning",
                "similar_results": [
                    {"id": "doc_001", "score": 0.92},
                    {"id": "doc_002", "score": 0.78}
                ],
                "count": 2
            }
        }


class DeleteResponse(BaseModel):
    message: str = Field(description="Success message")
    document_id: str = Field(description="ID of the deleted document")
    deleted: bool = Field(description="Whether the document was successfully deleted")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Document deleted successfully",
                "document_id": "doc_001",
                "deleted": True
            }
        }


class HealthResponse(BaseModel):
    status: str = Field(description="Health status of the service")
    database_path: str = Field(description="Path to the LanceDB database")
    total_documents: int = Field(description="Total number of documents in the database")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "database_path": "db/zendown_documents.lance",
                "total_documents": 5
            }
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    print(f"Application startup: Initializing SemWare database at '{DB_PATH}' with table '{TABLE_NAME}'...")
    try:
        init_db()
        print("SemWare database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise
    yield
    print("Application shutdown.")


app = FastAPI(
    title="SemWare API",
    description="""
    # SemWare - Self-Hosted Semantic Search Framework
    
    A powerful, self-hosted semantic search framework built with LanceDB, FastAPI, and Hugging Face models.
    
    ## Features
    
    - **Document Management**: Store, update, and retrieve documents with automatic chunking
    - **Semantic Search**: Find documents using natural language queries
    - **Similar Document Search**: Find documents similar to existing ones
    - **Efficient Embeddings**: Only regenerate embeddings when content changes significantly
    - **Vector Search**: Fast similarity search using LanceDB
    
    ## Authentication
    
    All API endpoints (except `/` and `/health`) require API key authentication.
    Set the `SEMWARE_API_KEY` environment variable and include it in requests:
    
    ```
    Authorization: Bearer your-api-key-here
    ```
    
    ## Quick Start
    
    1. **Add a document**:
       ```bash
       curl -X POST "http://localhost:8000/api/documents/upsert" \\
            -H "Authorization: Bearer your-api-key" \\
            -H "Content-Type: application/json" \\
            -d '{"id": "doc1", "content": "Your document content here..."}'
       ```
    
    2. **Search semantically**:
       ```bash
       curl -X POST "http://localhost:8000/api/search/semantic" \\
            -H "Authorization: Bearer your-api-key" \\
            -H "Content-Type: application/json" \\
            -d '{"query_text": "artificial intelligence", "top_k": 5}'
       ```
    
    3. **Find similar documents**:
       ```bash
       curl -X POST "http://localhost:8000/api/search/similar" \\
            -H "Authorization: Bearer your-api-key" \\
            -H "Content-Type: application/json" \\
            -d '{"id": "doc1", "top_k": 3}'
       ```
    
    ## Distance Metrics
    
    - **cosine**: Cosine similarity (0-1, higher is more similar)
    - **l2**: Euclidean distance (lower is more similar)
    
    ## Response Formats
    
    All responses are JSON with consistent error handling:
    - `200`: Success
    - `400`: Bad request (invalid parameters, empty content)
    - `401`: Unauthorized (missing or invalid API key)
    - `404`: Not found (document doesn't exist)
    - `500`: Internal server error
    """,
    version="0.1.0",
    contact={
        "name": "SemWare API Support",
        "url": "https://github.com/your-repo/semware",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint providing basic API information.
    
    Returns:
        Basic information about the SemWare API including version and documentation links.
    """
    return {
        "message": "Welcome to SemWare - Semantic Search Framework",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API and database status.
    
    This endpoint checks:
    - API service availability
    - Database connectivity
    - Total document count
    
    Returns:
        Health status information including database path and document count.
    """
    try:
        document_ids = get_all_document_ids()
        return HealthResponse(
            status="healthy",
            database_path=DB_PATH,
            total_documents=len(document_ids)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/api/documents/upsert", response_model=UpsertResponse)
async def api_upsert_document(
    request: UpsertRequest, 
    api_key: str = Depends(api_key_auth)
):
    """
    Create or update a document with automatic embedding generation.
    
    This endpoint processes documents by:
    - Splitting content into 100-word chunks for optimal semantic search
    - Generating embeddings for each chunk using the SentenceTransformer model
    - Storing both document metadata and chunk embeddings in LanceDB
    - Only regenerating embeddings if content changed significantly (>20 words different)
    
    The document is automatically chunked and embedded for efficient semantic search.
    If a document with the same ID already exists, it will be updated only if the content
    has changed significantly.
    
    Args:
        request: Document upsert request containing ID and content
        api_key: API key for authentication
    
    Returns:
        Upsert response with action details and chunk information
    
    Raises:
        400: Bad request (empty content, invalid parameters)
        401: Unauthorized (missing or invalid API key)
        500: Internal server error
    """
    try:
        result = upsert_document(document_id=request.id, content=request.content)
        return UpsertResponse(
            message="Document upserted successfully",
            document_id=result["document_id"],
            action=result["action"],
            chunks_regenerated=result["chunks_regenerated"],
            total_chunks=result["total_chunks"]
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upsert document: {str(e)}")


@app.get("/api/documents", response_model=List[str])
async def api_get_all_document_ids(api_key: str = Depends(api_key_auth)):
    """
    Retrieve all document IDs in the database.
    
    Returns a list of all document identifiers that have been stored in the system.
    This is useful for listing available documents or checking what documents exist.
    
    Args:
        api_key: API key for authentication
    
    Returns:
        List of document IDs as strings
    
    Raises:
        401: Unauthorized (missing or invalid API key)
        500: Internal server error
    """
    try:
        return get_all_document_ids()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document IDs: {str(e)}")


@app.get("/api/documents/{document_id}", response_model=DocumentInfo)
async def api_get_document_content(
    document_id: str, 
    api_key: str = Depends(api_key_auth)
):
    """
    Retrieve full document content and metadata by ID.
    
    Returns the complete document content along with metadata including:
    - Total number of chunks the document was split into
    - Total word count
    - Creation and update timestamps
    
    Args:
        document_id: Unique identifier of the document to retrieve
        api_key: API key for authentication
    
    Returns:
        Document information including content and metadata
    
    Raises:
        401: Unauthorized (missing or invalid API key)
        404: Document not found
        500: Internal server error
    """
    try:
        from .database import get_document_metadata
        
        metadata = get_document_metadata(document_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"Document with id '{document_id}' not found")
        
        return DocumentInfo(
            id=metadata["document_id"],
            content=metadata["full_content"],
            total_chunks=metadata["total_chunks"],
            total_words=metadata["total_words"],
            created_at=metadata["created_at"],
            updated_at=metadata["updated_at"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@app.get("/api/documents/{document_id}/embeddings", response_model=DocumentEmbeddingsResponse)
async def api_get_document_embeddings(
    document_id: str, 
    api_key: str = Depends(api_key_auth)
):
    """
    Retrieve all chunk embeddings for a specific document.
    
    Returns the embedding vectors for each chunk of the document, along with the
    corresponding chunk text. This is useful for debugging, analysis, or custom
    similarity calculations.
    
    Args:
        document_id: Unique identifier of the document
        api_key: API key for authentication
    
    Returns:
        Document embeddings response with chunk vectors and text
    
    Raises:
        401: Unauthorized (missing or invalid API key)
        404: Document not found or has no embeddings
        500: Internal server error
    """
    try:
        embeddings = get_document_embeddings(document_id)
        if not embeddings:
            raise HTTPException(status_code=404, detail=f"Document with id '{document_id}' not found or has no embeddings")
        
        embedding_infos = [
            EmbeddingInfo(
                chunk_index=emb["chunk_index"],
                text=emb["text"],
                vector=emb["vector"]
            )
            for emb in embeddings
        ]
        
        return DocumentEmbeddingsResponse(
            document_id=document_id,
            embeddings=embedding_infos
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document embeddings: {str(e)}")


@app.delete("/api/documents/{document_id}", response_model=DeleteResponse)
async def api_delete_document(
    document_id: str, 
    api_key: str = Depends(api_key_auth)
):
    """
    Delete a document and all its associated embeddings.
    
    Permanently removes a document from the database along with all its chunk
    embeddings. This operation cannot be undone.
    
    Args:
        document_id: Unique identifier of the document to delete
        api_key: API key for authentication
    
    Returns:
        Delete response confirming the operation
    
    Raises:
        401: Unauthorized (missing or invalid API key)
        404: Document not found
        500: Internal server error
    """
    try:
        deleted = delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document with id '{document_id}' not found")
        
        return DeleteResponse(
            message="Document deleted successfully",
            document_id=document_id,
            deleted=True
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.post("/api/search/similar", response_model=SearchResponse)
async def api_search_similar_documents(
    request: SearchRequest, 
    api_key: str = Depends(api_key_auth)
):
    """
    Find documents similar to a specific document.
    
    This endpoint uses the embeddings of the specified document to find similar
    documents in the database. It aggregates similarity scores across all chunks
    of the query document to provide document-level similarity scores.
    
    Features:
    - Supports both cosine similarity and L2 distance metrics
    - Configurable similarity threshold filtering
    - Top-k result limiting
    - Mean aggregation of chunk-level scores to document-level scores
    
    Args:
        request: Search request with document ID and search parameters
        api_key: API key for authentication
    
    Returns:
        Search response with similar documents and their scores
    
    Raises:
        400: Bad request (document not found, invalid parameters)
        401: Unauthorized (missing or invalid API key)
        500: Internal server error
    """
    try:
        similar_docs = search_similar_documents(
            document_id=request.id,
            threshold=request.threshold,
            top_k=request.top_k,
            distance_metric=request.distance_metric
        )
        
        similar_results = [
            SimilarDocument(id=doc["id"], score=doc["score"])
            for doc in similar_docs
        ]
        
        return SearchResponse(
            query_id=request.id,
            similar_results=similar_results,
            count=len(similar_results)
        )
    except ValueError as ve:
        print(f"ValueError during similar search for document '{request.id}': {ve}")
        # Return 400 for bad input (e.g., invalid parameters, document not found)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during similar search for document '{request.id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search for similar documents: {str(e)}")


@app.post("/api/search/semantic", response_model=SemanticSearchResponse)
async def api_semantic_search_documents(
    request: SemanticSearchRequest, 
    api_key: str = Depends(api_key_auth)
):
    """
    Perform semantic search across all documents using natural language queries.
    
    This endpoint generates embeddings for the provided query text and finds
    semantically similar documents in the database. It's ideal for natural
    language search queries and finding relevant content.
    
    Features:
    - Natural language query processing
    - Supports both cosine similarity and L2 distance metrics
    - Configurable similarity threshold filtering
    - Top-k result limiting
    - Real-time embedding generation for queries
    
    Args:
        request: Semantic search request with query text and parameters
        api_key: API key for authentication
    
    Returns:
        Semantic search response with relevant documents and scores
    
    Raises:
        400: Bad request (empty query, invalid parameters)
        401: Unauthorized (missing or invalid API key)
        500: Internal server error
    """
    try:
        relevant_docs = semantic_search_documents(
            query_text=request.query_text,
            threshold=request.threshold,
            top_k=request.top_k,
            distance_metric=request.distance_metric
        )
        
        similar_results = [
            SimilarDocument(id=doc["id"], score=doc["score"])
            for doc in relevant_docs
        ]
        
        return SemanticSearchResponse(
            query_text=request.query_text,
            similar_results=similar_results,
            count=len(similar_results)
        )
    except ValueError as ve:
        print(f"ValueError during semantic search for query '{request.query_text}': {ve}")
        # Return 400 for bad input (e.g., empty query text, failed embedding, invalid parameters)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during semantic search for query '{request.query_text}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform semantic search: {str(e)}")


if __name__ == "__main__":
    print("Starting SemWare API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 