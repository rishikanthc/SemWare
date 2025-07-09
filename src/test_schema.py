#!/usr/bin/env python3
"""
Test script to verify database schema
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semware.database import init_db, upsert_document, get_all_document_ids
from semware.embeddings import SentenceTransformerEmbeddings

def test_schema():
    """Test the database schema"""
    print("🧪 Testing database schema...")
    
    # Remove existing database
    import shutil
    if os.path.exists("db/semware_documents.lance"):
        shutil.rmtree("db/semware_documents.lance")
    
    try:
        # Initialize database
        print("📊 Initializing database...")
        init_db()
        
        # Test upsert
        print("📝 Testing document upsert...")
        test_content = "This is a test document about artificial intelligence and machine learning."
        result = upsert_document("test_doc", test_content)
        print(f"✅ Upsert result: {result}")
        
        # Test retrieval
        print("📋 Testing document retrieval...")
        doc_ids = get_all_document_ids()
        print(f"✅ Document IDs: {doc_ids}")
        
        print("✅ Schema test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_schema() 