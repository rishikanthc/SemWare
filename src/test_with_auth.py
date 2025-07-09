#!/usr/bin/env python3
"""
Comprehensive test script for SemWare API with authentication
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "test-api-key-12345"  # This should match your SEMWARE_API_KEY env var

# Test data
TEST_DOCUMENTS = [
    {
        "id": "doc1",
        "content": """
        Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. 
        These machines can perform tasks that typically require human intelligence, such as visual perception, 
        speech recognition, decision-making, and language translation. AI has applications in various fields 
        including healthcare, finance, transportation, and entertainment. Machine learning, a subset of AI, 
        enables computers to learn and improve from experience without being explicitly programmed.
        """
    },
    {
        "id": "doc2", 
        "content": """
        Machine Learning is a subset of artificial intelligence that focuses on the development of algorithms 
        and statistical models that enable computers to improve their performance on a specific task through 
        experience. It involves training models on large datasets to recognize patterns and make predictions. 
        Common types of machine learning include supervised learning, unsupervised learning, and reinforcement learning.
        """
    },
    {
        "id": "doc3",
        "content": """
        Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction 
        between computers and human language. It involves developing algorithms and models that can understand, 
        interpret, and generate human language. NLP applications include chatbots, language translation, 
        sentiment analysis, and text summarization. The field combines linguistics, computer science, and AI.
        """
    },
    {
        "id": "doc4",
        "content": """
        Computer Vision is a field of artificial intelligence that trains computers to interpret and understand 
        visual information from the world. It involves developing algorithms that can process, analyze, and 
        understand digital images and videos. Applications include facial recognition, autonomous vehicles, 
        medical image analysis, and augmented reality. Computer vision combines image processing, machine learning, 
        and pattern recognition techniques.
        """
    }
]


def get_headers() -> Dict[str, str]:
    """Get headers with API key authentication"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }


def test_health_check():
    """Test health check endpoint (no auth required)"""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Health: {data['status']}")
            print(f"Database: {data['database_path']}")
            print(f"Documents: {data['total_documents']}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")


def test_root():
    """Test root endpoint (no auth required)"""
    print("\n=== Testing Root Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Message: {data['message']}")
            print(f"Version: {data['version']}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")


def test_upsert_documents():
    """Test document upsert with authentication"""
    print("\n=== Testing Document Upsert ===")
    
    for doc in TEST_DOCUMENTS:
        try:
            response = requests.post(
                f"{BASE_URL}/api/documents/upsert",
                headers=get_headers(),
                json=doc
            )
            print(f"Upsert {doc['id']}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Action: {data['action']}")
                print(f"  Chunks: {data['total_chunks']}")
                print(f"  Regenerated: {data['chunks_regenerated']}")
            else:
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  Exception: {e}")


def test_get_all_documents():
    """Test getting all document IDs"""
    print("\n=== Testing Get All Documents ===")
    try:
        response = requests.get(
            f"{BASE_URL}/api/documents",
            headers=get_headers()
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            doc_ids = response.json()
            print(f"Document IDs: {doc_ids}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")


def test_get_document_content():
    """Test getting document content"""
    print("\n=== Testing Get Document Content ===")
    
    for doc_id in ["doc1", "doc2", "nonexistent"]:
        try:
            response = requests.get(
                f"{BASE_URL}/api/documents/{doc_id}",
                headers=get_headers()
            )
            print(f"Get {doc_id}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Content length: {len(data['content'])} chars")
                print(f"  Chunks: {data['total_chunks']}")
                print(f"  Words: {data['total_words']}")
            elif response.status_code == 404:
                print(f"  Not found (expected for {doc_id})")
            else:
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  Exception: {e}")


def test_get_document_embeddings():
    """Test getting document embeddings"""
    print("\n=== Testing Get Document Embeddings ===")
    
    for doc_id in ["doc1", "nonexistent"]:
        try:
            response = requests.get(
                f"{BASE_URL}/api/documents/{doc_id}/embeddings",
                headers=get_headers()
            )
            print(f"Get embeddings {doc_id}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Embeddings count: {len(data['embeddings'])}")
                if data['embeddings']:
                    print(f"  Vector dimension: {len(data['embeddings'][0]['vector'])}")
            elif response.status_code == 404:
                print(f"  Not found (expected for {doc_id})")
            else:
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  Exception: {e}")


def test_similar_search():
    """Test similar document search"""
    print("\n=== Testing Similar Document Search ===")
    
    test_cases = [
        {"id": "doc1", "threshold": 0.7, "top_k": 3, "distance_metric": "cosine"},
        {"id": "doc2", "threshold": None, "top_k": 2, "distance_metric": "l2"},
        {"id": "nonexistent", "threshold": 0.5, "top_k": 5, "distance_metric": "cosine"}
    ]
    
    for case in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/api/search/similar",
                headers=get_headers(),
                json=case
            )
            print(f"Similar search for {case['id']}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Results count: {data['count']}")
                for result in data['similar_results']:
                    print(f"    {result['id']}: {result['score']:.4f}")
            elif response.status_code == 400:
                print(f"  Bad request (expected for {case['id']}): {response.json()['detail']}")
            else:
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  Exception: {e}")


def test_semantic_search():
    """Test semantic search"""
    print("\n=== Testing Semantic Search ===")
    
    test_queries = [
        {"query_text": "artificial intelligence and machine learning", "threshold": 0.6, "top_k": 3, "distance_metric": "cosine"},
        {"query_text": "computer vision applications", "threshold": None, "top_k": 2, "distance_metric": "l2"},
        {"query_text": "", "threshold": 0.5, "top_k": 5, "distance_metric": "cosine"}  # Empty query
    ]
    
    for query in test_queries:
        try:
            response = requests.post(
                f"{BASE_URL}/api/search/semantic",
                headers=get_headers(),
                json=query
            )
            print(f"Semantic search for '{query['query_text'][:30]}...': {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Results count: {data['count']}")
                for result in data['similar_results']:
                    print(f"    {result['id']}: {result['score']:.4f}")
            elif response.status_code == 400:
                print(f"  Bad request: {response.json()['detail']}")
            else:
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  Exception: {e}")


def test_authentication_failures():
    """Test authentication failures"""
    print("\n=== Testing Authentication Failures ===")
    
    # Test without API key
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/upsert",
            headers={"Content-Type": "application/json"},
            json=TEST_DOCUMENTS[0]
        )
        print(f"No API key: {response.status_code}")
        if response.status_code == 401:
            print(f"  Error: {response.json()['detail']}")
    except Exception as e:
        print(f"  Exception: {e}")
    
    # Test with wrong API key
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/upsert",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer wrong-key"
            },
            json=TEST_DOCUMENTS[0]
        )
        print(f"Wrong API key: {response.status_code}")
        if response.status_code == 401:
            print(f"  Error: {response.json()['detail']}")
    except Exception as e:
        print(f"  Exception: {e}")
    
    # Test with malformed authorization header
    try:
        response = requests.post(
            f"{BASE_URL}/api/documents/upsert",
            headers={
                "Content-Type": "application/json",
                "Authorization": "InvalidFormat"
            },
            json=TEST_DOCUMENTS[0]
        )
        print(f"Malformed auth header: {response.status_code}")
        if response.status_code == 401:
            print(f"  Error: {response.json()['detail']}")
    except Exception as e:
        print(f"  Exception: {e}")


def test_delete_document():
    """Test document deletion"""
    print("\n=== Testing Document Deletion ===")
    
    # Delete a document
    try:
        response = requests.delete(
            f"{BASE_URL}/api/documents/doc4",
            headers=get_headers()
        )
        print(f"Delete doc4: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Message: {data['message']}")
            print(f"  Deleted: {data['deleted']}")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"  Exception: {e}")
    
    # Try to delete non-existent document
    try:
        response = requests.delete(
            f"{BASE_URL}/api/documents/nonexistent",
            headers=get_headers()
        )
        print(f"Delete nonexistent: {response.status_code}")
        if response.status_code == 404:
            print(f"  Not found (expected)")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"  Exception: {e}")


def main():
    """Run all tests"""
    print("Starting SemWare API Tests with Authentication")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    time.sleep(1)
    
    # Run tests
    test_health_check()
    test_root()
    test_upsert_documents()
    test_get_all_documents()
    test_get_document_content()
    test_get_document_embeddings()
    test_similar_search()
    test_semantic_search()
    test_authentication_failures()
    test_delete_document()
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    main() 