# SemWare - Self-Hosted Semantic Search Framework

A powerful, self-hosted semantic search framework built with LanceDB, FastAPI, and Hugging Face models. SemWare provides efficient document storage, automatic chunking, and fast vector search capabilities.

## Features

* **Document Management**: Store, update, and retrieve documents with automatic chunking
* **Semantic Search**: Find documents using natural language queries
* **Similar Document Search**: Find documents similar to existing ones
* **Vector Search**: Fast similarity search using LanceDB
* **API Key Authentication**: Secure access to all endpoints
* **Interactive Documentation**: Auto-generated OpenAPI/Swagger documentation

## Quick Start

### Prerequisites

* Python 3.12+
* uv or pip

### Installation

1. **Clone the repository**:
   
   ````bash
   git clone <repository-url>
   cd zendown-ai
   ````

1. **Install dependencies**:
   
   ````bash
   pip install -r requirements.txt
   # or using uv
   uv sync
   ````

1. **Set up environment variables**:
   
   ````bash
   # Fish shell
   set -x SEMWARE_API_KEY "your-secure-api-key-here"
   
   # Bash/Zsh
   export SEMWARE_API_KEY="your-secure-api-key-here"
   ````

1. **Start the server**:
   
   ````bash
   uv run python -m src.semware.main
   ````

The API will be available at `http://localhost:8000`

### Interactive Documentation

Once the server is running, you can access:

* **Swagger UI**: http://localhost:8000/docs
* **ReDoc**: http://localhost:8000/redoc
* **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Documentation

### Authentication

All API endpoints (except `/` and `/health`) require API key authentication. Include your API key in the Authorization header:

````
Authorization: Bearer your-api-key-here
````

### Endpoints

#### Health Check

````http
GET /health
````

Returns the health status of the API and database.

**Response**:

````json
{
  "status": "healthy",
  "database_path": "db/zendown_documents.lance",
  "total_documents": 5
}
````

#### Document Management

##### Create/Update Document

````http
POST /api/documents/upsert
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "id": "doc_001",
  "content": "Your document content here..."
}
````

**Response**:

````json
{
  "message": "Document upserted successfully",
  "document_id": "doc_001",
  "action": "created",
  "chunks_regenerated": true,
  "total_chunks": 3
}
````

##### Get All Document IDs

````http
GET /api/documents
Authorization: Bearer your-api-key
````

**Response**:

````json
["doc_001", "doc_002", "doc_003"]
````

##### Get Document Content

````http
GET /api/documents/{document_id}
Authorization: Bearer your-api-key
````

**Response**:

````json
{
  "id": "doc_001",
  "content": "Full document content...",
  "total_chunks": 3,
  "total_words": 150,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
````

##### Get Document Embeddings

````http
GET /api/documents/{document_id}/embeddings
Authorization: Bearer your-api-key
````

**Response**:

````json
{
  "document_id": "doc_001",
  "embeddings": [
    {
      "chunk_index": 0,
      "text": "Chunk text...",
      "vector": [0.1, 0.2, 0.3, ...]
    }
  ]
}
````

##### Delete Document

````http
DELETE /api/documents/{document_id}
Authorization: Bearer your-api-key
````

**Response**:

````json
{
  "message": "Document deleted successfully",
  "document_id": "doc_001",
  "deleted": true
}
````

#### Search

##### Semantic Search

````http
POST /api/search/semantic
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "query_text": "artificial intelligence and machine learning",
  "threshold": 0.6,
  "top_k": 10,
  "distance_metric": "cosine"
}
````

**Response**:

````json
{
  "query_text": "artificial intelligence and machine learning",
  "similar_results": [
    {
      "id": "doc_001",
      "score": 0.92
    },
    {
      "id": "doc_002",
      "score": 0.78
    }
  ],
  "count": 2
}
````

##### Similar Document Search

````http
POST /api/search/similar
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "id": "doc_001",
  "threshold": 0.7,
  "top_k": 5,
  "distance_metric": "cosine"
}
````

**Response**:

````json
{
  "query_id": "doc_001",
  "similar_results": [
    {
      "id": "doc_002",
      "score": 0.85
    },
    {
      "id": "doc_003",
      "score": 0.72
    }
  ],
  "count": 2
}
````

### Parameters

#### Distance Metrics

* **cosine**: Cosine similarity (0-1, higher is more similar)
* **l2**: Euclidean distance (lower is more similar)

#### Search Parameters

* **threshold**: Minimum similarity score (optional)
  * For cosine: 0.0-1.0 (higher = more similar)
  * For l2: any positive value (lower = more similar)
* **top_k**: Maximum number of results to return (optional, 1-1000)
* **distance_metric**: Distance calculation method (“cosine” or “l2”)

### Error Responses

All endpoints return consistent error responses:

````json
{
  "detail": "Error message here"
}
````

**Status Codes**:

* `200`: Success
* `400`: Bad request (invalid parameters, empty content)
* `401`: Unauthorized (missing or invalid API key)
* `404`: Not found (document doesn’t exist)
* `500`: Internal server error

## Usage Examples

### Python Client

````python
import requests

BASE_URL = "http://localhost:8000"
API_KEY = "your-api-key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Add a document
response = requests.post(
    f"{BASE_URL}/api/documents/upsert",
    headers=headers,
    json={
        "id": "ai_intro",
        "content": "Artificial Intelligence (AI) is a branch of computer science..."
    }
)

# Search semantically
response = requests.post(
    f"{BASE_URL}/api/search/semantic",
    headers=headers,
    json={
        "query_text": "machine learning applications",
        "top_k": 5
    }
)

results = response.json()
for doc in results["similar_results"]:
    print(f"Document {doc['id']}: {doc['score']:.3f}")
````

### cURL Examples

````bash
# Add a document
curl -X POST "http://localhost:8000/api/documents/upsert" \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"id": "doc1", "content": "Your document content here..."}'

# Search semantically
curl -X POST "http://localhost:8000/api/search/semantic" \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"query_text": "artificial intelligence", "top_k": 5}'

# Find similar documents
curl -X POST "http://localhost:8000/api/search/similar" \
     -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"id": "doc1", "top_k": 3}'
````

## Architecture

### Components

* **FastAPI**: Web framework for the REST API
* **LanceDB**: Vector database for storing embeddings
* **SentenceTransformers**: Hugging Face model for generating embeddings
* **PyArrow**: Data serialization and schema management

### Data Flow

1. **Document Ingestion**: Documents are chunked into 100-word segments
1. **Embedding Generation**: Each chunk is converted to a 384-dimensional vector
1. **Storage**: Chunks and embeddings are stored in LanceDB with metadata
1. **Search**: Queries are embedded and compared against stored vectors
1. **Aggregation**: Chunk-level scores are aggregated to document-level scores

### Database Schema

* **Documents Table**: Stores document metadata and full content
* **Chunks Table**: Stores chunk embeddings with document references

## Configuration

### Environment Variables

* `SEMWARE_API_KEY`: Required API key for authentication
* `LANCE_DB_PATH`: Database path (default: `db/zendown_documents.lance`)
* `EMBEDDING_MODEL`: Hugging Face model name (default: `all-MiniLM-L6-v2`)

### Model Information

* **Model**: `all-MiniLM-L6-v2`
* **Vector Dimension**: 384
* **Language**: English (multilingual support available)

## Development

### Running Tests

````bash
# Set API key
set -x SEMWARE_API_KEY "test-api-key-12345"

# Run comprehensive tests
python src/test_with_auth.py
````

### Project Structure

````
src/semware/
├── __init__.py
├── main.py          # FastAPI application and endpoints
├── database.py      # LanceDB operations and schema
├── embeddings.py    # SentenceTransformer wrapper
├── text_processing.py # Chunking and content comparison
└── auth.py          # API key authentication

tests/
├── test_with_auth.py           # Comprehensive API tests
├── test_enhanced_semantic.py   # Semantic search tests
└── test_enhanced_similar.py    # Similar document tests
````

## Contributing

1. Fork the repository
1. Create a feature branch
1. Make your changes
1. Add tests
1. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:

* Check the interactive documentation at `/docs`
* Review the API examples in this README
* Open an issue on GitHub



