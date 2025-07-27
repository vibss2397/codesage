from rag_store import CodingPatternRag
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG (make sure GEMINI_API_KEY is set in environment)
    rag = CodingPatternRag()
    
    # Load patterns
    rag.add_patterns()
    
    # Test queries
    test_queries = [
        "find minimum element in rotated sorted array",
    ]
    
    print("\n🔍 TESTING SEARCH QUERIES:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = rag.search_similar(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['pattern']} ({result['chunk_type']})")
            print(f"     Similarity: {result['similarity_score']:.3f}")
            print(f"     Text: {result['text'][:80]}...")
    
    # Show available patterns
    print(f"\n📋 Available patterns: {', '.join(rag.list_available_patterns())}")
        
