#!/usr/bin/env python3
"""
Test script for SemWare API endpoints
"""

import requests
import json
import time
from typing import Dict, Any


class SemWareTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        print("🔍 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_upsert_document(self, doc_id: str, content: str) -> bool:
        """Test document upsert"""
        print(f"📝 Testing upsert document: {doc_id}")
        try:
            payload = {
                "id": doc_id,
                "content": content
            }
            response = self.session.post(f"{self.base_url}/api/documents/upsert", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Document upserted: {data}")
                return True
            else:
                print(f"❌ Upsert failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Upsert error: {e}")
            return False
    
    def test_get_all_documents(self) -> bool:
        """Test get all document IDs"""
        print("📋 Testing get all documents...")
        try:
            response = self.session.get(f"{self.base_url}/api/documents")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ All documents: {data}")
                return True
            else:
                print(f"❌ Get documents failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Get documents error: {e}")
            return False
    
    def test_get_document(self, doc_id: str) -> bool:
        """Test get document content"""
        print(f"📄 Testing get document: {doc_id}")
        try:
            response = self.session.get(f"{self.base_url}/api/documents/{doc_id}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Document retrieved: {data['id']} - {len(data['content'])} chars, {data['total_chunks']} chunks")
                return True
            else:
                print(f"❌ Get document failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Get document error: {e}")
            return False
    
    def test_get_embeddings(self, doc_id: str) -> bool:
        """Test get document embeddings"""
        print(f"🧠 Testing get embeddings: {doc_id}")
        try:
            response = self.session.get(f"{self.base_url}/api/documents/{doc_id}/embeddings")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Embeddings retrieved: {len(data['embeddings'])} chunks")
                return True
            else:
                print(f"❌ Get embeddings failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Get embeddings error: {e}")
            return False
    
    def test_similar_search(self, doc_id: str, threshold: float = 0.7, limit: int = 5) -> bool:
        """Test similar document search"""
        print(f"🔍 Testing similar search for: {doc_id}")
        try:
            payload = {
                "id": doc_id,
                "threshold": threshold,
                "limit": limit
            }
            response = self.session.post(f"{self.base_url}/api/search/similar", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Similar search results: {data['count']} documents found")
                for result in data['similar_results']:
                    print(f"   - {result['id']}: {result['score']:.3f}")
                return True
            else:
                print(f"❌ Similar search failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Similar search error: {e}")
            return False
    
    def test_semantic_search(self, query: str, threshold: float = 0.7, limit: int = 5) -> bool:
        """Test semantic search"""
        print(f"🔍 Testing semantic search: '{query}'")
        try:
            payload = {
                "query_text": query,
                "threshold": threshold,
                "limit": limit
            }
            response = self.session.post(f"{self.base_url}/api/search/semantic", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Semantic search results: {data['count']} documents found")
                for result in data['similar_results']:
                    print(f"   - {result['id']}: {result['score']:.3f}")
                return True
            else:
                print(f"❌ Semantic search failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            return False
    
    def test_delete_document(self, doc_id: str) -> bool:
        """Test document deletion"""
        print(f"🗑️ Testing delete document: {doc_id}")
        try:
            response = self.session.delete(f"{self.base_url}/api/documents/{doc_id}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Document deleted: {data}")
                return True
            else:
                print(f"❌ Delete failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Delete error: {e}")
            return False
    
    def run_full_test(self):
        """Run a complete test suite"""
        print("🚀 Starting SemWare API Test Suite")
        print("=" * 50)
        
        # Test data
        test_documents = [
            {
                "id": "doc1",
                "content": """
                Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. 
                Some of the activities computers with artificial intelligence are designed for include speech recognition, learning, planning, and problem solving. 
                AI has been used in various applications such as virtual assistants, autonomous vehicles, medical diagnosis, and game playing. 
                Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.
                """
            },
            {
                "id": "doc2", 
                "content": """
                Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. 
                Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. 
                The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.
                """
            },
            {
                "id": "doc3",
                "content": """
                Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. 
                NLP combines computational linguistics with statistical, machine learning, and deep learning models. 
                These technologies enable computers to process human language in the form of text or voice data and understand its full meaning, complete with the speaker or writer's intent and sentiment.
                """
            }
        ]
        
        # Test 1: Health check
        if not self.test_health():
            print("❌ Health check failed, stopping tests")
            return
        
        print("\n" + "=" * 50)
        print("📚 Testing Document Operations")
        print("=" * 50)
        
        # Test 2: Upsert documents
        for doc in test_documents:
            self.test_upsert_document(doc["id"], doc["content"])
            time.sleep(1)  # Small delay between requests
        
        # Test 3: Get all documents
        self.test_get_all_documents()
        
        # Test 4: Get individual documents
        for doc in test_documents:
            self.test_get_document(doc["id"])
            time.sleep(0.5)
        
        # Test 5: Get embeddings
        for doc in test_documents:
            self.test_get_embeddings(doc["id"])
            time.sleep(0.5)
        
        print("\n" + "=" * 50)
        print("🔍 Testing Search Operations")
        print("=" * 50)
        
        # Test 6: Similar document search
        self.test_similar_search("doc1", threshold=0.6, limit=3)
        
        # Test 7: Semantic search
        self.test_semantic_search("artificial intelligence and machine learning", threshold=0.6, limit=3)
        self.test_semantic_search("natural language processing", threshold=0.6, limit=3)
        
        print("\n" + "=" * 50)
        print("🗑️ Testing Delete Operations")
        print("=" * 50)
        
        # Test 8: Delete documents
        for doc in test_documents:
            self.test_delete_document(doc["id"])
            time.sleep(0.5)
        
        # Test 9: Verify deletion
        self.test_get_all_documents()
        
        print("\n" + "=" * 50)
        print("✅ Test Suite Completed!")
        print("=" * 50)


def main():
    """Main test function"""
    tester = SemWareTester()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ SemWare server is running")
        else:
            print("❌ SemWare server is not responding correctly")
            return
    except requests.exceptions.ConnectionError:
        print("❌ SemWare server is not running. Please start it first:")
        print("   python -m src.semware.main")
        return
    
    # Run tests
    tester.run_full_test()


if __name__ == "__main__":
    main() 