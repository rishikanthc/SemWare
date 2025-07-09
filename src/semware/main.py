"""
Main FastAPI application for SemWare
"""

from fastapi import FastAPI, HTTPException
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


# Request Models
class UpsertRequest(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., min_length=1, description="Document content")


class SearchRequest(BaseModel):
    id: str = Field(..., description="Document ID to find similar documents for")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold (0.0 to 1.0)")
    limit: int = Field(10, gt=0, le=100, description="Maximum number of results to return")


class SemanticSearchRequest(BaseModel):
    query_text: str = Field(..., min_length=1, description="Text to search for")
    threshold: Optional[float] = Field(None, ge=0.0, description="Minimum similarity score (0.0 to 1.0 for cosine)")
    top_k: Optional[int] = Field(None, gt=0, le=1000, description="Maximum number of results to return")
    distance_metric: str = Field("cosine", description="Distance metric: 'cosine' or 'l2' (euclidean)")


# Response Models
class UpsertResponse(BaseModel):
    message: str
    document_id: str
    action: str
    chunks_regenerated: bool
    total_chunks: int


class DocumentInfo(BaseModel):
    id: str
    content: str
    total_chunks: int
    total_words: int
    created_at: str
    updated_at: str


class EmbeddingInfo(BaseModel):
    chunk_index: int
    text: str
    vector: List[float]


class DocumentEmbeddingsResponse(BaseModel):
    document_id: str
    embeddings: List[EmbeddingInfo]


class SimilarDocument(BaseModel):
    id: str
    score: float


class SearchResponse(BaseModel):
    query_id: str
    similar_results: List[SimilarDocument]
    count: int


class SemanticSearchResponse(BaseModel):
    query_text: str
    similar_results: List[SimilarDocument]
    count: int


class DeleteResponse(BaseModel):
    message: str
    document_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    database_path: str
    total_documents: int


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
    description="A semantic search framework for document embeddings with fast vector search capabilities.",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to SemWare - Semantic Search Framework",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
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
async def api_upsert_document(request: UpsertRequest):
    """
    Create or update a document with embeddings.
    
    - Splits content into 100-word chunks
    - Generates embeddings for each chunk
    - Only regenerates embeddings if content changed significantly (>20 words different)
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
async def api_get_all_document_ids():
    """Get all document IDs"""
    try:
        return get_all_document_ids()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document IDs: {str(e)}")


@app.get("/api/documents/{document_id}", response_model=DocumentInfo)
async def api_get_document_content(document_id: str):
    """Get document content by ID"""
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
async def api_get_document_embeddings(document_id: str):
    """Get all embeddings for a document"""
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
async def api_delete_document(document_id: str):
    """Delete a document and all its embeddings"""
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
async def api_search_similar_documents(request: SearchRequest):
    """
    Find documents similar to the given document.
    
    Uses the document's embeddings to find similar documents in the database.
    """
    try:
        similar_docs = search_similar_documents(
            document_id=request.id,
            threshold=request.threshold,
            limit=request.limit
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
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search for similar documents: {str(e)}")


@app.post("/api/search/semantic", response_model=SemanticSearchResponse)
async def api_semantic_search_documents(request: SemanticSearchRequest):
    """
    Perform semantic search across all documents.
    
    Generates embeddings for the query text and finds similar documents.
    Supports both threshold-based filtering and top_k limiting.
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