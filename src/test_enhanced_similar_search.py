#!/usr/bin/env python3
"""
Enhanced Similar Documents Search Test Script for SemWare
Tests the new similar documents search functionality with threshold, top_k, and distance metrics.
"""

import requests
import json
import time
from typing import Dict, Any


class EnhancedSimilarSearchTester:
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
    
    def test_similar_search(self, doc_id: str, threshold: float = None, top_k: int = None, distance_metric: str = "cosine") -> bool:
        """Test enhanced similar documents search"""
        print(f"🔍 Testing similar search for document: '{doc_id}' (threshold={threshold}, top_k={top_k}, metric={distance_metric})")
        try:
            payload = {
                "id": doc_id,
                "distance_metric": distance_metric
            }
            
            if threshold is not None:
                payload["threshold"] = threshold
            if top_k is not None:
                payload["top_k"] = top_k
            
            response = self.session.post(f"{self.base_url}/api/search/similar", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Similar search results: {data['count']} documents found")
                for result in data['similar_results']:
                    print(f"   - {result['id']}: {result['score']:.4f}")
                return True
            else:
                print(f"❌ Similar search failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Similar search error: {e}")
            return False
    
    def test_error_cases(self) -> bool:
        """Test error cases for similar search"""
        print("🚨 Testing error cases...")
        
        # Test 1: Both threshold and top_k specified (should fail)
        print("  Testing: Both threshold and top_k specified")
        try:
            payload = {
                "id": "ai_doc",
                "threshold": 0.7,
                "top_k": 5
            }
            response = self.session.post(f"{self.base_url}/api/search/similar", json=payload)
            if response.status_code == 400:
                print("  ✅ Correctly rejected both threshold and top_k")
            else:
                print(f"  ❌ Expected 400, got {response.status_code}")
                return False
        except Exception as e:
            print(f"  ❌ Error testing both parameters: {e}")
            return False
        
        # Test 2: Invalid distance metric
        print("  Testing: Invalid distance metric")
        try:
            payload = {
                "id": "ai_doc",
                "distance_metric": "invalid_metric"
            }
            response = self.session.post(f"{self.base_url}/api/search/similar", json=payload)
            if response.status_code == 400:
                print("  ✅ Correctly rejected invalid distance metric")
            else:
                print(f"  ❌ Expected 400, got {response.status_code}")
                return False
        except Exception as e:
            print(f"  ❌ Error testing invalid metric: {e}")
            return False
        
        # Test 3: Non-existent document
        print("  Testing: Non-existent document")
        try:
            payload = {
                "id": "non_existent_doc"
            }
            response = self.session.post(f"{self.base_url}/api/search/similar", json=payload)
            if response.status_code == 400:
                print("  ✅ Correctly rejected non-existent document")
            else:
                print(f"  ❌ Expected 400, got {response.status_code}")
                return False
        except Exception as e:
            print(f"  ❌ Error testing non-existent document: {e}")
            return False
        
        return True
    
    def run_enhanced_test_suite(self):
        """Run comprehensive test suite for enhanced similar documents search"""
        print("🚀 Starting Enhanced Similar Documents Search Test Suite")
        print("=" * 70)
        
        # Test data with diverse content
        test_documents = [
            {
                "id": "ai_doc",
                "content": """
                Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. 
                Some of the activities computers with artificial intelligence are designed for include speech recognition, learning, planning, and problem solving. 
                AI has been used in various applications such as virtual assistants, autonomous vehicles, medical diagnosis, and game playing. 
                Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.
                Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns.
                """
            },
            {
                "id": "ml_doc", 
                "content": """
                Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. 
                Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. 
                The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.
                Supervised learning involves training a model on labeled data, while unsupervised learning finds hidden patterns in unlabeled data.
                """
            },
            {
                "id": "nlp_doc",
                "content": """
                Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. 
                NLP combines computational linguistics with statistical, machine learning, and deep learning models. 
                These technologies enable computers to process human language in the form of text or voice data and understand its full meaning, complete with the speaker or writer's intent and sentiment.
                Common NLP tasks include text classification, sentiment analysis, machine translation, and question answering.
                """
            },
            {
                "id": "cv_doc",
                "content": """
                Computer Vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. 
                Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects and then react to what they "see."
                Computer vision is used in applications such as facial recognition, autonomous vehicles, medical image analysis, and quality control in manufacturing.
                The field encompasses image processing, pattern recognition, and machine learning techniques.
                """
            },
            {
                "id": "robotics_doc",
                "content": """
                Robotics is an interdisciplinary field that combines computer science, engineering, and other disciplines to design, construct, operate, and use robots. 
                Robots are machines that can perform tasks automatically or with minimal human intervention. 
                They are used in manufacturing, healthcare, space exploration, and many other industries. 
                Modern robotics often incorporates artificial intelligence and machine learning to enable robots to adapt to changing environments and perform complex tasks.
                """
            }
        ]
        
        # Test 1: Health check
        if not self.test_health():
            print("❌ Health check failed, stopping tests")
            return
        
        print("\n" + "=" * 70)
        print("📚 Setting Up Test Documents")
        print("=" * 70)
        
        # Test 2: Upsert documents
        for doc in test_documents:
            self.test_upsert_document(doc["id"], doc["content"])
            time.sleep(1)  # Small delay between requests
        
        print("\n" + "=" * 70)
        print("🔍 Testing Enhanced Similar Documents Search Features")
        print("=" * 70)
        
        # Test 3: Threshold-based search (cosine)
        print("\n📊 Testing threshold-based search (cosine metric):")
        self.test_similar_search("ai_doc", threshold=0.6, distance_metric="cosine")
        self.test_similar_search("ml_doc", threshold=0.5, distance_metric="cosine")
        
        # Test 4: Top-k search (cosine)
        print("\n📊 Testing top-k search (cosine metric):")
        self.test_similar_search("nlp_doc", top_k=3, distance_metric="cosine")
        self.test_similar_search("cv_doc", top_k=2, distance_metric="cosine")
        
        # Test 5: Threshold-based search (l2/euclidean)
        print("\n📊 Testing threshold-based search (l2 metric):")
        self.test_similar_search("ai_doc", threshold=0.3, distance_metric="l2")
        self.test_similar_search("robotics_doc", threshold=0.4, distance_metric="l2")
        
        # Test 6: Top-k search (l2/euclidean)
        print("\n📊 Testing top-k search (l2 metric):")
        self.test_similar_search("ml_doc", top_k=4, distance_metric="l2")
        self.test_similar_search("nlp_doc", top_k=3, distance_metric="l2")
        
        # Test 7: Default behavior (no threshold, no top_k)
        print("\n📊 Testing default behavior (no parameters):")
        self.test_similar_search("ai_doc")
        self.test_similar_search("robotics_doc")
        
        # Test 8: Error cases
        print("\n" + "=" * 70)
        print("🚨 Testing Error Cases")
        print("=" * 70)
        self.test_error_cases()
        
        print("\n" + "=" * 70)
        print("✅ Enhanced Similar Documents Search Test Suite Completed!")
        print("=" * 70)


def main():
    """Main test function"""
    tester = EnhancedSimilarSearchTester()
    
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
        print("   uv run python -m src.semware.main")
        return
    
    # Run tests
    tester.run_enhanced_test_suite()


if __name__ == "__main__":
    main() 