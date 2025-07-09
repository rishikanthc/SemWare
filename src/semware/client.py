"""
SemWare Python Client

A simple client library for interacting with the SemWare API.
"""

import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SemWareConfig:
    """Configuration for SemWare client"""
    base_url: str = "http://localhost:8000"
    api_key: str = ""
    timeout: int = 30


class SemWareClient:
    """
    Python client for SemWare API
    
    Example usage:
        client = SemWareClient(api_key="your-api-key")
        
        # Add a document
        result = client.upsert_document("doc1", "Your content here...")
        
        # Search semantically
        results = client.semantic_search("artificial intelligence", top_k=5)
        
        # Find similar documents
        similar = client.find_similar("doc1", threshold=0.7)
    """
    
    def __init__(self, config: Optional[SemWareConfig] = None):
        """Initialize the SemWare client"""
        self.config = config or SemWareConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the SemWare API"""
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method, 
                url, 
                timeout=self.config.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json().get('detail', str(e))
                except:
                    error_detail = str(e)
                raise SemWareError(f"API request failed: {error_detail}")
            else:
                raise SemWareError(f"Network error: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of the SemWare API"""
        return self._make_request("GET", "/health")
    
    def upsert_document(self, doc_id: str, content: str) -> Dict[str, Any]:
        """
        Create or update a document
        
        Args:
            doc_id: Unique document identifier
            content: Document content
            
        Returns:
            Upsert response with action details
        """
        data = {"id": doc_id, "content": content}
        return self._make_request("POST", "/api/documents/upsert", json=data)
    
    def get_all_documents(self) -> List[str]:
        """Get all document IDs"""
        return self._make_request("GET", "/api/documents")
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get document content and metadata"""
        return self._make_request("GET", f"/api/documents/{doc_id}")
    
    def get_document_embeddings(self, doc_id: str) -> Dict[str, Any]:
        """Get all embeddings for a document"""
        return self._make_request("GET", f"/api/documents/{doc_id}/embeddings")
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document and all its embeddings"""
        return self._make_request("DELETE", f"/api/documents/{doc_id}")
    
    def semantic_search(
        self, 
        query_text: str, 
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        distance_metric: str = "cosine"
    ) -> Dict[str, Any]:
        """
        Perform semantic search across all documents
        
        Args:
            query_text: Natural language query
            threshold: Minimum similarity score
            top_k: Maximum number of results
            distance_metric: 'cosine' or 'l2'
            
        Returns:
            Search results with similar documents
        """
        data = {
            "query_text": query_text,
            "distance_metric": distance_metric
        }
        if threshold is not None:
            data["threshold"] = threshold
        if top_k is not None:
            data["top_k"] = top_k
            
        return self._make_request("POST", "/api/search/semantic", json=data)
    
    def find_similar(
        self, 
        doc_id: str, 
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        distance_metric: str = "cosine"
    ) -> Dict[str, Any]:
        """
        Find documents similar to a specific document
        
        Args:
            doc_id: Document ID to find similar documents for
            threshold: Minimum similarity score
            top_k: Maximum number of results
            distance_metric: 'cosine' or 'l2'
            
        Returns:
            Search results with similar documents
        """
        data = {
            "id": doc_id,
            "distance_metric": distance_metric
        }
        if threshold is not None:
            data["threshold"] = threshold
        if top_k is not None:
            data["top_k"] = top_k
            
        return self._make_request("POST", "/api/search/similar", json=data)
    
    def batch_upsert(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Upsert multiple documents in batch
        
        Args:
            documents: List of documents with 'id' and 'content' keys
            
        Returns:
            List of upsert responses
        """
        results = []
        for doc in documents:
            try:
                result = self.upsert_document(doc["id"], doc["content"])
                results.append({"success": True, "doc_id": doc["id"], "result": result})
            except Exception as e:
                results.append({"success": False, "doc_id": doc["id"], "error": str(e)})
        return results
    
    def search_multiple_queries(
        self, 
        queries: List[str], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search for multiple queries
        
        Args:
            queries: List of query texts
            top_k: Maximum number of results per query
            
        Returns:
            List of search results
        """
        results = []
        for query in queries:
            try:
                result = self.semantic_search(query, top_k=top_k)
                results.append({"success": True, "query": query, "result": result})
            except Exception as e:
                results.append({"success": False, "query": query, "error": str(e)})
        return results


class SemWareError(Exception):
    """Exception raised for SemWare API errors"""
    pass


# Convenience functions for quick usage
def create_client(api_key: str, base_url: str = "http://localhost:8000") -> SemWareClient:
    """Create a SemWare client with the given configuration"""
    config = SemWareConfig(base_url=base_url, api_key=api_key)
    return SemWareClient(config)


def quick_search(
    query: str, 
    api_key: str, 
    base_url: str = "http://localhost:8000",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Quick semantic search function
    
    Args:
        query: Search query
        api_key: API key
        base_url: API base URL
        top_k: Number of results
        
    Returns:
        List of similar documents
    """
    client = create_client(api_key, base_url)
    result = client.semantic_search(query, top_k=top_k)
    return result.get("similar_results", [])


def quick_upsert(
    doc_id: str, 
    content: str, 
    api_key: str, 
    base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Quick document upsert function
    
    Args:
        doc_id: Document ID
        content: Document content
        api_key: API key
        base_url: API base URL
        
    Returns:
        Upsert response
    """
    client = create_client(api_key, base_url)
    return client.upsert_document(doc_id, content) 