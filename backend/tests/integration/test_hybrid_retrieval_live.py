"""
Live test for hybrid retrieval implementation.
Tests FAISS + BM25 hybrid search functionality.
"""
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required imports work."""
    print("=" * 60)
    print("Testing Imports...")
    print("=" * 60)
    
    results = {}
    
    try:
        import faiss
        results['faiss'] = f"✓ FAISS {faiss.__version__ if hasattr(faiss, '__version__') else 'available'}"
    except ImportError as e:
        results['faiss'] = f"✗ FAISS not available: {e}"
    
    try:
        from sentence_transformers import SentenceTransformer
        results['sentence_transformers'] = "✓ sentence-transformers available"
    except ImportError as e:
        results['sentence_transformers'] = f"✗ sentence-transformers not available: {e}"
    
    try:
        from rank_bm25 import BM25Okapi
        results['bm25'] = "✓ rank-bm25 available"
    except ImportError as e:
        results['bm25'] = f"✗ rank-bm25 not available: {e}"
    
    try:
        import numpy as np
        results['numpy'] = f"✓ numpy {np.__version__}"
    except ImportError as e:
        results['numpy'] = f"✗ numpy not available: {e}"
    
    for name, status in results.items():
        print(f"  {name}: {status}")
    
    return all("✓" in v for v in results.values())


def test_rag_service_imports():
    """Test RAG service module imports."""
    print("\n" + "=" * 60)
    print("Testing RAG Service Module...")
    print("=" * 60)
    
    try:
        from src.services.rag_service import (
            RAGService,
            FAISS_AVAILABLE,
            SENTENCE_TRANSFORMERS_AVAILABLE,
            BM25_AVAILABLE
        )
        print(f"  ✓ RAGService imported successfully")
        print(f"    - FAISS_AVAILABLE: {FAISS_AVAILABLE}")
        print(f"    - SENTENCE_TRANSFORMERS_AVAILABLE: {SENTENCE_TRANSFORMERS_AVAILABLE}")
        print(f"    - BM25_AVAILABLE: {BM25_AVAILABLE}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import RAGService: {e}")
        return False


def test_tokenization():
    """Test BM25 tokenization."""
    print("\n" + "=" * 60)
    print("Testing Tokenization...")
    print("=" * 60)
    
    from src.services.rag_service import RAGService
    
    service = RAGService()
    
    test_texts = [
        "How does the chunker work?",
        "def calculate_embeddings(text: str) -> List[float]:",
        "The embedding model uses all-MiniLM-L6-v2 for semantic search."
    ]
    
    for text in test_texts:
        tokens = service._tokenize(text)
        print(f"  Input: '{text[:50]}...' " if len(text) > 50 else f"  Input: '{text}'")
        print(f"  Tokens: {tokens}")
        print()
    
    return True


def test_chunking():
    """Test content chunking."""
    print("\n" + "=" * 60)
    print("Testing Content Chunking...")
    print("=" * 60)
    
    from src.services.rag_service import RAGService
    
    service = RAGService()
    
    # Test Python code chunking
    python_code = '''"""Sample module."""

class MyClass:
    """A sample class."""
    
    def __init__(self, value):
        self.value = value
    
    def process(self):
        return self.value * 2

def helper_function(x):
    """Helper function."""
    return x + 1

async def async_helper(y):
    """Async helper."""
    return y * 2
'''
    
    chunks = service._chunk_python_code(python_code, "test.py")
    print(f"  Python code chunking: {len(chunks)} chunks created")
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i+1}: lines {chunk['line_start']}-{chunk['line_end']}, {len(chunk['content'])} chars")
    
    # Test generic chunking
    generic_text = "Line " * 100 + "\n" + "Another line " * 50
    chunks = service._chunk_generic(generic_text, "test.md", chunk_size=200, overlap=50)
    print(f"\n  Generic text chunking: {len(chunks)} chunks created")
    
    return True


def test_embedding_model():
    """Test embedding model loading and encoding."""
    print("\n" + "=" * 60)
    print("Testing Embedding Model...")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        print("  Loading all-MiniLM-L6-v2 model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  ✓ Model loaded")
        
        # Test encoding
        texts = ["How does indexing work?", "The chunker splits code into segments."]
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        print(f"  ✓ Embeddings generated: shape {embeddings.shape}")
        print(f"    Dimension: {embeddings.shape[1]}")
        
        # Test similarity
        from numpy.linalg import norm
        similarity = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        print(f"    Cosine similarity between test texts: {similarity:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_faiss_index():
    """Test FAISS index creation and search."""
    print("\n" + "=" * 60)
    print("Testing FAISS Index...")
    print("=" * 60)
    
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        
        # Create test data
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [
            "The chunker splits documents into smaller pieces.",
            "FAISS provides efficient similarity search.",
            "BM25 is a keyword-based ranking function.",
            "Hybrid search combines semantic and keyword search.",
            "Embeddings are dense vector representations."
        ]
        
        print(f"  Creating embeddings for {len(texts)} documents...")
        embeddings = model.encode(texts, convert_to_numpy=True).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        print(f"  ✓ FAISS index created with {index.ntotal} vectors")
        
        # Test search
        query = "How does semantic search work?"
        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        k = 3
        distances, indices = index.search(query_embedding, k)
        
        print(f"\n  Query: '{query}'")
        print(f"  Top {k} results:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            print(f"    {i+1}. [{dist:.4f}] {texts[idx][:50]}...")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bm25_search():
    """Test BM25 search."""
    print("\n" + "=" * 60)
    print("Testing BM25 Search...")
    print("=" * 60)
    
    try:
        from rank_bm25 import BM25Okapi
        import numpy as np
        
        # Create test corpus
        corpus = [
            "The chunker splits documents into smaller pieces for processing",
            "FAISS provides efficient similarity search using vectors",
            "BM25 is a keyword-based ranking function for text retrieval",
            "Hybrid search combines semantic and keyword search methods",
            "Embeddings are dense vector representations of text"
        ]
        
        # Tokenize corpus
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"  ✓ BM25 index created with {len(corpus)} documents")
        
        # Test search
        query = "keyword search ranking"
        query_tokens = query.lower().split()
        
        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:3]
        
        print(f"\n  Query: '{query}'")
        print(f"  Top 3 results:")
        for i, idx in enumerate(top_indices):
            print(f"    {i+1}. [{scores[idx]:.4f}] {corpus[idx][:50]}...")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_hybrid_search_integration():
    """Test the full hybrid search in RAGService."""
    print("\n" + "=" * 60)
    print("Testing Hybrid Search Integration...")
    print("=" * 60)
    
    from src.services.rag_service import RAGService
    import numpy as np
    
    service = RAGService()
    
    # Manually set up test data
    test_chunks = [
        {"file_path": "src/chunker.py", "content": "def chunk_text(text): splits text into smaller pieces", "line_start": 1, "line_end": 10, "type": "code"},
        {"file_path": "src/indexer.py", "content": "class Indexer: handles document indexing with FAISS", "line_start": 1, "line_end": 20, "type": "code"},
        {"file_path": "src/search.py", "content": "def search(query): performs hybrid semantic and keyword search", "line_start": 1, "line_end": 15, "type": "code"},
        {"file_path": "README.md", "content": "# RAG System\nThis system uses hybrid retrieval with FAISS and BM25", "line_start": 1, "line_end": 5, "type": "doc"},
        {"file_path": "src/embeddings.py", "content": "def get_embeddings(text): uses sentence-transformers to create vectors", "line_start": 1, "line_end": 8, "type": "code"},
    ]
    
    service.document_chunks = test_chunks
    service.chunk_texts = [c['content'] for c in test_chunks]
    
    print(f"  Set up {len(test_chunks)} test chunks")
    
    # Build BM25 index
    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [service._tokenize(text) for text in service.chunk_texts]
        service.bm25_index = BM25Okapi(tokenized_corpus)
        print("  ✓ BM25 index built")
    except Exception as e:
        print(f"  ✗ BM25 index failed: {e}")
        return False
    
    # Build FAISS index
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        
        service.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = service.embedding_model.encode(service.chunk_texts, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        service.faiss_index = faiss.IndexFlatIP(dimension)
        service.faiss_index.add(embeddings)
        print(f"  ✓ FAISS index built with {service.faiss_index.ntotal} vectors")
    except Exception as e:
        print(f"  ✗ FAISS index failed: {e}")
        return False
    
    service.retrieval_initialized = True
    
    # Test hybrid search
    queries = [
        "How does text chunking work?",
        "What is the indexing process?",
        "How does semantic search work?"
    ]
    
    print("\n  Testing hybrid search:")
    for query in queries:
        print(f"\n  Query: '{query}'")
        results = service.hybrid_search(query, top_k=3)
        print(f"  Results ({len(results)}):")
        for i, r in enumerate(results):
            print(f"    {i+1}. [{r.score:.4f}] {r.file}: {r.content[:40]}...")
    
    return True


async def test_full_initialization():
    """Test full RAG service initialization with hybrid retrieval."""
    print("\n" + "=" * 60)
    print("Testing Full RAG Service Initialization...")
    print("=" * 60)
    
    from src.services.rag_service import RAGService
    
    # Check if there's indexed data
    metadata_dir = Path("./storage/metadata")
    if not metadata_dir.exists() or not list(metadata_dir.glob("*.json")):
        print("  ⚠ No indexed repository found. Skipping full initialization test.")
        print("    Index a repository first to test full hybrid retrieval.")
        return True
    
    service = RAGService()
    
    # Test initialization (this will load/build indices)
    print("  Initializing RAG service (this may take a moment)...")
    
    # Just test the hybrid retrieval initialization part
    try:
        await service._initialize_hybrid_retrieval()
        
        if service.retrieval_initialized:
            print(f"  ✓ Hybrid retrieval initialized")
            print(f"    - Document chunks: {len(service.document_chunks)}")
            print(f"    - FAISS index: {service.faiss_index.ntotal if service.faiss_index else 'None'} vectors")
            print(f"    - BM25 index: {'Ready' if service.bm25_index else 'None'}")
            
            # Test a search
            if service.document_chunks:
                query = "How does the system work?"
                results = service.hybrid_search(query, top_k=3)
                print(f"\n  Test query: '{query}'")
                print(f"  Results: {len(results)} documents found")
                for i, r in enumerate(results[:3]):
                    print(f"    {i+1}. [{r.score:.4f}] {r.file}")
        else:
            print("  ⚠ Hybrid retrieval not initialized (missing dependencies or data)")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("   HYBRID RETRIEVAL IMPLEMENTATION TESTS")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['rag_service'] = test_rag_service_imports()
    results['tokenization'] = test_tokenization()
    results['chunking'] = test_chunking()
    results['embedding_model'] = test_embedding_model()
    results['faiss_index'] = test_faiss_index()
    results['bm25_search'] = test_bm25_search()
    results['hybrid_integration'] = test_hybrid_search_integration()
    
    # Run async test
    results['full_init'] = asyncio.run(test_full_initialization())
    
    # Summary
    print("\n" + "=" * 60)
    print("   TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
