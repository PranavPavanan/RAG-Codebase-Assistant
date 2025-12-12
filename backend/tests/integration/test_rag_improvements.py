"""
Test script to verify RAG improvements for endpoint detection
"""
import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8000/api"

def test_endpoint_question():
    """Test the specific question about API endpoints"""
    
    print("=" * 80)
    print("TESTING RAG IMPROVEMENTS - Endpoint Detection")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # The question that was giving poor results
    question = "Can you list the main API endpoints and which files define them?"
    
    print(f"Question: {question}\n")
    print("-" * 80)
    
    try:
        # Make request to chat API
        response = requests.post(
            f"{API_BASE}/chat/query",
            json={"query": question},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Print the answer
            print("\n📝 ANSWER:")
            print("-" * 80)
            answer = data.get('response') or data.get('answer', 'No response')
            print(answer)
            print()
            
            # Check sources
            sources = data.get('sources', [])
            print("\n📚 SOURCES:")
            print("-" * 80)
            
            if sources:
                for i, source in enumerate(sources, 1):
                    file_path = source.get('file') or source.get('file_path', 'unknown')
                    line_start = source.get('line_start', '?')
                    line_end = source.get('line_end', '?')
                    score = source.get('score', 0)
                    content_preview = source.get('content', '')[:100]
                    
                    print(f"\nSource {i}:")
                    print(f"  📄 File: {file_path}")
                    print(f"  📍 Lines: {line_start}-{line_end}")
                    print(f"  ⭐ Relevance: {score:.2%}")
                    print(f"  📋 Preview: {content_preview}...")
            else:
                print("❌ No sources returned!")
            
            print("\n" + "=" * 80)
            print("QUALITY CHECKS:")
            print("=" * 80)
            
            # Quality checks
            checks = {
                "✅ Has answer": bool(answer and answer != 'No response'),
                "✅ Answer is not vague": "not explicitly listed" not in answer.lower() and "we would need to look" not in answer.lower(),
                "✅ Lists specific endpoints": any(x in answer.lower() for x in ['/crawl', '/index', '/ask', '/health']),
                "✅ Mentions HTTP methods": any(x in answer for x in ['GET', 'POST', 'PUT', 'DELETE']),
                "✅ Has sources": len(sources) > 0,
                "✅ Sources have file names": all(s.get('file') or s.get('file_path') for s in sources),
                "✅ Sources have line numbers": all(s.get('line_start') and s.get('line_end') for s in sources),
                "✅ Line numbers are not 'undefined'": all(s.get('line_start') != 'undefined' for s in sources),
                "✅ Mentions main.py": 'main.py' in answer.lower() or any('main.py' in str(s.get('file', '')) for s in sources),
            }
            
            passed = sum(1 for v in checks.values() if v)
            total = len(checks)
            
            for check, result in checks.items():
                status = "✅" if result else "❌"
                print(f"{status} {check.replace('✅ ', '')}")
            
            print(f"\n📊 Score: {passed}/{total} ({passed/total*100:.0f}%)")
            
            if passed == total:
                print("\n🎉 ALL CHECKS PASSED! RAG improvements are working correctly!")
            elif passed >= total * 0.7:
                print("\n✅ Most checks passed. Some minor issues remain.")
            else:
                print("\n❌ Multiple checks failed. Further improvements needed.")
            
            print("\n" + "=" * 80)
            
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_simple_question():
    """Test a simpler question to verify basic functionality"""
    
    print("\n" + "=" * 80)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 80)
    
    question = "What is this repository about?"
    print(f"\nQuestion: {question}\n")
    
    try:
        response = requests.post(
            f"{API_BASE}/chat/query",
            json={"query": question},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('response') or data.get('answer', '')
            sources = data.get('sources', [])
            
            print("Answer:", answer[:200] + "..." if len(answer) > 200 else answer)
            print(f"\nSources: {len(sources)} found")
            print("✅ Basic functionality working")
        else:
            print(f"❌ Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("\n🧪 RAG Improvement Test Suite\n")
    
    # Test health endpoint first
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5)
        if health.status_code == 200:
            print("✅ Backend server is running\n")
        else:
            print("❌ Backend health check failed")
            exit(1)
    except:
        print("❌ Cannot connect to backend server")
        print("Make sure the server is running on http://localhost:8000")
        exit(1)
    
    # Run tests
    test_endpoint_question()
    test_simple_question()
    
    print("\n✅ Test suite completed!\n")
