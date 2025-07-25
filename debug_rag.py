def debug_rag():
    """Debug your RAG implementation step by step"""
    from coding_patterns import PATTERNS, process_pattern
    from rag_store import CodingPatternRag
    
    # Initialize RAG
    rag = CodingPatternRag()
    rag.add_patterns()
    
    print("=== DEBUGGING RAG ===")
    print(f"Number of patterns loaded: {len(PATTERNS)}")
    print(f"FAISS index size: {rag.index.ntotal}")
    print()
    
    # Check processed pattern strings
    print("=== PROCESSED PATTERNS ===")
    for i, pattern in enumerate(PATTERNS):
        processed = process_pattern(pattern)
        print(f"{i}: {pattern['name']}")
        print(f"   Processed: {processed[:100]}...")
        print()
    
    # Test specific queries
    test_queries = [
        "find two numbers that sum to target",  # Should be hash_map or two_pointers
        "check if string is palindrome",        # Should be two_pointers
        "longest substring without repeating",  # Should be sliding_window
        "find pairs in sorted array"           # Should be two_pointers
    ]
    
    print("=== QUERY RESULTS ===")
    for query in test_queries:
        results = rag.search_similar(query, k=3)
        print(f"Query: '{query}'")
        for i, (name, desc, distance) in enumerate(results):
            print(f"  {i+1}. {name} (distance: {distance:.3f})")
        print()

if __name__ == "__main__":
    debug_rag()