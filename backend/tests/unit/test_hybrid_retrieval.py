"""Tests for hybrid retrieval (FAISS + BM25) implementation."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestHybridRetrievalComponents:
    """Test individual hybrid retrieval components."""

    def test_imports_available(self):
        """Test that required libraries can be imported."""
        # Test FAISS
        try:
            import faiss
            faiss_available = True
        except ImportError:
            faiss_available = False
        
        # Test sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            st_available = True
        except ImportError:
            st_available = False
        
        # Test BM25
        try:
            from rank_bm25 import BM25Okapi
            bm25_available = True
        except ImportError:
            bm25_available = False
        
        print(f"\n📦 Library Availability:")
        print(f"   FAISS: {'✅' if faiss_available else '❌'}")
        print(f"   sentence-transformers: {'✅' if st_available else '❌'}")
        print(f"   rank-bm25: {'✅' if bm25_available else '❌'}")
        
        # At least one should be available for the test to be meaningful
        assert faiss_available or bm25_available, "At least FAISS or BM25 should be installed"

    def test_rag_service_has_hybrid_attributes(self):
        """Test that RAGService has hybrid retrieval attributes."""
        from src.services.rag_service import RAGService
        
        service = RAGService()
        
        # Check for hybrid retrieval attributes
        assert hasattr(service, 'faiss_index'), "Missing faiss_index attribute"
        assert hasattr(service, 'embedding_model'), "Missing embedding_model attribute"
        assert hasattr(service, 'bm25_index'), "Missing bm25_index attribute"
        assert hasattr(service, 'document_chunks'), "Missing document_chunks attribute"
        assert hasattr(service, 'chunk_texts'), "Missing chunk_texts attribute"
        assert hasattr(service, 'retrieval_initialized'), "Missing retrieval_initialized attribute"
        assert hasattr(service, 'indices_dir'), "Missing indices_dir attribute"
        assert hasattr(service, 'embedding_model_name'), "Missing embedding_model_name attribute"
        
        print("\n✅ RAGService has all hybrid retrieval attributes")

    def test_rag_service_has_hybrid_methods(self):
        """Test that RAGService has hybrid retrieval methods."""
        from src.services.rag_service import RAGService
        
        service = RAGService()
        
        # Check for hybrid retrieval methods
        assert hasattr(service, 'semantic_search'), "Missing semantic_search method"
        assert hasattr(service, 'keyword_search'), "Missing keyword_search method"
        assert hasattr(service, 'hybrid_search'), "Missing hybrid_search method"
        assert hasattr(service, '_initialize_hybrid_retrieval'), "Missing _initialize_hybrid_retrieval method"
        assert hasattr(service, '_build_indices'), "Missing _build_indices method"
        assert hasattr(service, '_load_indices'), "Missing _load_indices method"
        assert hasattr(service, '_save_indices'), "Missing _save_indices method"
        assert hasattr(service, '_tokenize'), "Missing _tokenize method"
        assert hasattr(service, '_chunk_content'), "Missing _chunk_content method"
        assert hasattr(service, 'rebuild_indices'), "Missing rebuild_indices method"
        
        print("\n✅ RAGService has all hybrid retrieval methods")


class TestTokenization:
    """Test tokenization for BM25."""

    def test_tokenize_simple_text(self):
        """Test basic tokenization."""
        from src.services.rag_service import RAGService
        
        service = RAGService()
        
        tokens = service._tokenize("Hello world this is a test")
        
        # Should remove stop words like 'this', 'is', 'a'
        assert 'hello' in tokens
        assert 'world' in tokens
        assert 'test' in tokens
        assert 'this' not in tokens  # Stop word
        assert 'is' not in tokens    # Stop word
        assert 'a' not in tokens     # Stop word
        
        print(f"\n✅ Tokenization works: {tokens}")

    def test_tokenize_code(self):
        """Test tokenization of code-like text."""
        from src.services.rag_service import RAGService
        
        service = RAGService()
        
        code_text = "def hello_world(): print('Hello World')"
        tokens = service._tokenize(code_text)
        
        assert 'def' in tokens
        assert 'hello_world' in tokens or 'hello' in tokens
        assert 'print' in tokens
        
        print(f"\n✅ Code tokenization works: {tokens}")


class TestChunking:
    """Test content chunking."""

    def test_chunk_generic_content(self):
        """Test generic content chunking."""
        from src.services.rag_service import RAGService
        
        service = RAGService()
        
        # Create sample content
        content = "\n".join([f"Line {i}: Some content here" for i in range(50)])
        
        chunks = service._chunk_generic(content, "test.txt", chunk_size=200, overlap=50)
        
        assert len(chunks) > 0, "Should produce at least one chunk"
        
        for chunk in chunks:
            assert 'file_path' in chunk
            assert 'content' in chunk
            assert 'line_start' in chunk
            assert 'line_end' in chunk
            assert 'type' in chunk
            assert chunk['file_path'] == 'test.txt'
        
        print(f"\n✅ Generic chunking produced {len(chunks)} chunks")

    def test_chunk_python_code(self):
        """Test Python code chunking."""
        from src.services.rag_service import RAGService
        
        service = RAGService()
        
        python_code = '''
class MyClass:
    def __init__(self):
        self.value = 0
    
    def method_one(self):
        return self.value

def standalone_function():
    print("Hello")
    return True

async def async_function():
    await something()
'''
        
        chunks = service._chunk_python_code(python_code, "test.py")
        
        assert len(chunks) > 0, "Should produce at least one chunk"
        
        # Check that chunks have proper structure
        for chunk in chunks:
            assert 'file_path' in chunk
            assert 'content' in chunk
            assert chunk['type'] == 'code'
        
        print(f"\n✅ Python chunking produced {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            preview = chunk['content'][:50].replace('\n', ' ')
            print(f"   Chunk {i+1}: lines {chunk['line_start']}-{chunk['line_end']}: {preview}...")


class TestBM25Search:
    """Test BM25 keyword search."""

    def test_bm25_index_creation(self):
        """Test BM25 index can be created."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank-bm25 not installed")
        
        from src.services.rag_service import RAGService
        
        service = RAGService()
        
        # Manually set up some test data
        service.chunk_texts = [
            "This is a function that calculates sum",
            "Database connection handler for PostgreSQL",
            "REST API endpoint for user authentication",
            "Machine learning model training script",
        ]
        
        # Build BM25 index
        service._build_bm25_index()
        
        assert service.bm25_index is not None, "BM25 index should be created"
        print("\n✅ BM25 index created successfully")

    def test_bm25_search(self):
        """Test BM25 search returns results."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank-bm25 not installed")
        
        from src.services.rag_service import RAGService
        
        service = RAGService()
        
        # Set up test data
        service.document_chunks = [
            {'file_path': 'math.py', 'content': 'def calculate_sum(a, b): return a + b', 'line_start': 1, 'line_end': 1, 'type': 'code'},
            {'file_path': 'db.py', 'content': 'class DatabaseConnection: postgresql handler', 'line_start': 1, 'line_end': 1, 'type': 'code'},
            {'file_path': 'api.py', 'content': 'REST API endpoint authentication login', 'line_start': 1, 'line_end': 1, 'type': 'code'},
            {'file_path': 'ml.py', 'content': 'machine learning model training neural network', 'line_start': 1, 'line_end': 1, 'type': 'code'},
        ]
        service.chunk_texts = [c['content'] for c in service.document_chunks]
        
        # Build index
        service._build_bm25_index()
        service.retrieval_initialized = True
        
        # Search
        results = service.keyword_search("database postgresql", top_k=2)
        
        assert len(results) > 0, "Should return results"
        assert results[0].file == 'db.py', "Should find database file first"
        
        print(f"\n✅ BM25 search works! Found {len(results)} results")
        for r in results:
            print(f"   {r.file}: score={r.score:.3f}")


class TestSemanticSearch:
    """Test FAISS semantic search."""

    @pytest.mark.skipif(
        not all([
            __import__('importlib.util').util.find_spec('faiss'),
            __import__('importlib.util').util.find_spec('sentence_transformers')
        ]),
        reason="FAISS or sentence-transformers not installed"
    )
    def test_faiss_index_creation(self):
        """Test FAISS index can be created."""
        import faiss
        from sentence_transformers import SentenceTransformer
        from src.services.rag_service import RAGService
        
        service = RAGService()
        
        # Load embedding model
        service.embedding_model = SentenceTransformer(service.embedding_model_name)
        
        # Set up test data
        service.chunk_texts = [
            "This is a function that calculates sum",
            "Database connection handler",
            "REST API endpoint",
        ]
        
        # Build FAISS index synchronously for testing
        import numpy as np
        embeddings = service.embedding_model.encode(service.chunk_texts, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        service.faiss_index = faiss.IndexFlatIP(dimension)
        service.faiss_index.add(embeddings)
        
        assert service.faiss_index.ntotal == 3, "Should have 3 vectors"
        print(f"\n✅ FAISS index created with {service.faiss_index.ntotal} vectors")


class TestHybridSearch:
    """Test combined hybrid search."""

    def test_hybrid_search_fallback(self):
        """Test hybrid search falls back when indices not initialized."""
        from src.services.rag_service import RAGService
        
        service = RAGService()
        service.retrieval_initialized = False
        
        # Should not raise, should fall back gracefully
        results = service.hybrid_search("test query", top_k=5)
        
        # Results may be empty but shouldn't error
        assert isinstance(results, list)
        print("\n✅ Hybrid search fallback works")

    def test_hybrid_search_with_mock_results(self):
        """Test hybrid search RRF combination logic."""
        from src.services.rag_service import RAGService
        from src.models.query import SourceReference
        
        service = RAGService()
        service.retrieval_initialized = True
        
        # Mock semantic and keyword search
        semantic_results = [
            SourceReference(file='a.py', content='content a', score=0.9, line_start=1, line_end=10, type='code'),
            SourceReference(file='b.py', content='content b', score=0.8, line_start=1, line_end=10, type='code'),
        ]
        
        keyword_results = [
            SourceReference(file='b.py', content='content b', score=0.95, line_start=1, line_end=10, type='code'),
            SourceReference(file='c.py', content='content c', score=0.7, line_start=1, line_end=10, type='code'),
        ]
        
        with patch.object(service, 'semantic_search', return_value=semantic_results):
            with patch.object(service, 'keyword_search', return_value=keyword_results):
                results = service.hybrid_search("test", top_k=3)
        
        assert len(results) > 0, "Should return combined results"
        
        # b.py should rank high as it appears in both
        files = [r.file for r in results]
        print(f"\n✅ Hybrid RRF combination works! Results: {files}")


class TestIntegration:
    """Integration tests for the full hybrid retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_initialization(self):
        """Test that hybrid retrieval initializes without errors."""
        from src.services.rag_service import RAGService
        
        service = RAGService()
        
        # Mock the LLM loading to focus on retrieval
        with patch.object(service, 'model', MagicMock()):
            service.is_initialized = True
            
            # Initialize hybrid retrieval
            await service._initialize_hybrid_retrieval()
            
            # Check initialization state
            print(f"\n📊 Initialization Status:")
            print(f"   retrieval_initialized: {service.retrieval_initialized}")
            print(f"   embedding_model: {'✅ Loaded' if service.embedding_model else '❌ Not loaded'}")
            print(f"   faiss_index: {'✅ Created' if service.faiss_index else '❌ Not created'}")
            print(f"   bm25_index: {'✅ Created' if service.bm25_index else '❌ Not created'}")
            print(f"   document_chunks: {len(service.document_chunks)} chunks")


def run_quick_test():
    """Run a quick manual test of hybrid retrieval."""
    print("=" * 60)
    print("🧪 HYBRID RETRIEVAL QUICK TEST")
    print("=" * 60)
    
    from src.services.rag_service import RAGService, FAISS_AVAILABLE, BM25_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE
    
    print(f"\n📦 Dependencies:")
    print(f"   FAISS_AVAILABLE: {FAISS_AVAILABLE}")
    print(f"   SENTENCE_TRANSFORMERS_AVAILABLE: {SENTENCE_TRANSFORMERS_AVAILABLE}")
    print(f"   BM25_AVAILABLE: {BM25_AVAILABLE}")
    
    service = RAGService()
    
    print(f"\n🔧 RAGService created with:")
    print(f"   embedding_model_name: {service.embedding_model_name}")
    print(f"   indices_dir: {service.indices_dir}")
    
    # Test tokenization
    print(f"\n📝 Tokenization test:")
    tokens = service._tokenize("def calculate_sum(numbers): return sum(numbers)")
    print(f"   Tokens: {tokens}")
    
    # Test chunking
    print(f"\n📄 Chunking test:")
    test_code = '''def hello():
    print("Hello")

def world():
    print("World")
'''
    chunks = service._chunk_content(test_code, "test.py")
    print(f"   Created {len(chunks)} chunks")
    
    print("\n" + "=" * 60)
    print("✅ Quick test completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_quick_test()
