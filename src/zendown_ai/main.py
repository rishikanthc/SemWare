# zendown_ai/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn

from .database import (
    init_db,
    upsert_document,
    search_similar_documents,
    semantic_search_documents,  # Added import
    TABLE_NAME,
    DB_PATH
)


class UpsertRequest(BaseModel):
    id: str
    content: str


class SearchRequest(BaseModel):
    id: str
    thresh: float = Field(..., ge=0.0, le=1.0,
                          description="Similarity threshold (0.0 to 1.0)")
    limit: int = Field(
        10, gt=0, description="Maximum number of results to return")


# Request model for semantic search
class SemanticSearchRequest(BaseModel):
    query_text: str = Field(..., min_length=1, description="Text to search for.")
    thresh: float = Field(..., ge=0.0, le=1.0,
                          description="Similarity threshold (0.0 to 1.0)")
    limit: int = Field(
        10, gt=0, description="Maximum number of results to return")


class UpsertResponse(BaseModel):
    message: str
    id: str


class SimilarDocument(BaseModel):
    id: str
    score: float


class SearchResponse(BaseModel):
    query_id: str
    similar_results: list[SimilarDocument]
    count: int


# Response model for semantic search
class SemanticSearchResponse(BaseModel):
    query_text: str
    similar_results: list[SimilarDocument]
    count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Application startup: Initializing database at '{
          DB_PATH}' with table '{TABLE_NAME}'...")
    try:
        init_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    yield
    print("Application shutdown.")

app = FastAPI(
    title="Zendown AI API",
    description="API for document embedding and similarity search.",
    version="0.1.0",
    lifespan=lifespan
)


@app.post("/api/upsert/", response_model=UpsertResponse)
async def api_upsert_document(request: UpsertRequest):
    try:
        upsert_document(document_id=request.id, content=request.content)
        return UpsertResponse(message="Document upserted successfully", id=request.id)
    except Exception as e:
        print(f"Error during upsert for id {request.id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to upsert document: {str(e)}")


@app.post("/api/search/similar/", response_model=SearchResponse)
async def api_search_similar_documents(request: SearchRequest):
    try:
        similar_ids = search_similar_documents(
            document_id=request.id,
            threshold=request.thresh,
            limit=request.limit
        )
        # The search_similar_documents now returns List[Dict[str, Any]]
        # which Pydantic can automatically convert to List[SimilarDocument]
        return SearchResponse(
            query_id=request.id,
            similar_results=similar_ids, # similar_ids is now the list of dicts
            count=len(similar_ids)
        )
    except ValueError as ve:
        print(f"ValueError during search for id {request.id}: {ve}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        print(f"Error during search for id {request.id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to search for similar documents: {str(e)}")


@app.post("/api/search/semantic/", response_model=SemanticSearchResponse)
async def api_semantic_search_documents(request: SemanticSearchRequest):
    try:
        results = semantic_search_documents(
            query_text=request.query_text,
            threshold=request.thresh,
            limit=request.limit
        )
        return SemanticSearchResponse(
            query_text=request.query_text,
            similar_results=results,
            count=len(results)
        )
    except ValueError as ve:
        print(f"ValueError during semantic search for query '{
              request.query_text}': {ve}")
        # Return 400 for bad input (e.g., empty query text, failed embedding)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during semantic search for query '{
              request.query_text}': {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to perform semantic search: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Welcome to Zendown AI API. See /docs for API documentation."}

if __name__ == "__main__":
    print("Attempting to run with Uvicorn.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
