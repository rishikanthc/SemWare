#!/usr/bin/env python3
"""
Example usage of SemWare client library

This script demonstrates how to use the SemWare Python client
to interact with the SemWare API.
"""

import os
from semware.client import SemWareClient, quick_search, quick_upsert


def main():
    """Main example function"""
    
    # Get API key from environment
    api_key = os.getenv("SEMWARE_API_KEY", "test-api-key-12345")
    
    print("SemWare Client Example")
    print("=" * 50)
    
    # Create client
    client = SemWareClient(api_key=api_key)
    
    # Check health
    print("\n1. Health Check")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Documents: {health['total_documents']}")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Sample documents
    documents = [
        {
            "id": "ai_intro",
            "content": """
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines. These machines can perform tasks that typically require human 
            intelligence, such as visual perception, speech recognition, decision-making, and 
            language translation. AI has applications in various fields including healthcare, 
            finance, transportation, and entertainment.
            """
        },
        {
            "id": "ml_basics",
            "content": """
            Machine Learning is a subset of artificial intelligence that focuses on the 
            development of algorithms and statistical models that enable computers to improve 
            their performance on a specific task through experience. It involves training 
            models on large datasets to recognize patterns and make predictions.
            """
        },
        {
            "id": "nlp_overview",
            "content": """
            Natural Language Processing (NLP) is a field of artificial intelligence that 
            focuses on the interaction between computers and human language. It involves 
            developing algorithms and models that can understand, interpret, and generate 
            human language. NLP applications include chatbots, language translation, 
            sentiment analysis, and text summarization.
            """
        },
        {
            "id": "computer_vision",
            "content": """
            Computer Vision is a field of artificial intelligence that trains computers to 
            interpret and understand visual information from the world. It involves developing 
            algorithms that can process, analyze, and understand digital images and videos. 
            Applications include facial recognition, autonomous vehicles, medical image 
            analysis, and augmented reality.
            """
        }
    ]
    
    # Add documents
    print("\n2. Adding Documents")
    for doc in documents:
        try:
            result = client.upsert_document(doc["id"], doc["content"])
            print(f"   {doc['id']}: {result['action']} ({result['total_chunks']} chunks)")
        except Exception as e:
            print(f"   {doc['id']}: Error - {e}")
    
    # List all documents
    print("\n3. List All Documents")
    try:
        doc_ids = client.get_all_documents()
        print(f"   Found {len(doc_ids)} documents: {doc_ids}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Semantic search examples
    print("\n4. Semantic Search Examples")
    search_queries = [
        "artificial intelligence and machine learning",
        "computer vision applications",
        "natural language processing",
        "deep learning algorithms"
    ]
    
    for query in search_queries:
        try:
            results = client.semantic_search(query, top_k=3)
            print(f"\n   Query: '{query}'")
            print(f"   Results: {results['count']}")
            for doc in results['similar_results']:
                print(f"     {doc['id']}: {doc['score']:.3f}")
        except Exception as e:
            print(f"   Query '{query}': Error - {e}")
    
    # Similar document search
    print("\n5. Similar Document Search")
    try:
        similar = client.find_similar("ai_intro", top_k=3)
        print(f"   Documents similar to 'ai_intro':")
        for doc in similar['similar_results']:
            print(f"     {doc['id']}: {doc['score']:.3f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Get document details
    print("\n6. Document Details")
    try:
        doc_info = client.get_document("ai_intro")
        print(f"   Document: {doc_info['id']}")
        print(f"   Content length: {len(doc_info['content'])} chars")
        print(f"   Chunks: {doc_info['total_chunks']}")
        print(f"   Words: {doc_info['total_words']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Batch operations
    print("\n7. Batch Operations")
    batch_docs = [
        {"id": "batch_1", "content": "First batch document about data science."},
        {"id": "batch_2", "content": "Second batch document about statistics."},
        {"id": "batch_3", "content": "Third batch document about programming."}
    ]
    
    try:
        batch_results = client.batch_upsert(batch_docs)
        print(f"   Batch upsert results:")
        for result in batch_results:
            status = "✓" if result['success'] else "✗"
            print(f"     {status} {result['doc_id']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Multiple queries
    print("\n8. Multiple Query Search")
    queries = ["data science", "statistics", "programming"]
    try:
        multi_results = client.search_multiple_queries(queries, top_k=2)
        for result in multi_results:
            if result['success']:
                print(f"   Query '{result['query']}': {result['result']['count']} results")
            else:
                print(f"   Query '{result['query']}': Error - {result['error']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Quick functions
    print("\n9. Quick Functions")
    try:
        # Quick search
        quick_results = quick_search("machine learning", api_key, top_k=2)
        print(f"   Quick search results: {len(quick_results)} documents")
        
        # Quick upsert
        quick_result = quick_upsert("quick_doc", "Quick document content.", api_key)
        print(f"   Quick upsert: {quick_result['action']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Cleanup
    print("\n10. Cleanup")
    cleanup_docs = ["batch_1", "batch_2", "batch_3", "quick_doc"]
    for doc_id in cleanup_docs:
        try:
            client.delete_document(doc_id)
            print(f"   Deleted: {doc_id}")
        except Exception as e:
            print(f"   Failed to delete {doc_id}: {e}")
    
    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == "__main__":
    main() 