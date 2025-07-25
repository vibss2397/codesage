from rag_store import CodingPatternRag


def test_rag_retrieval():
    rag = CodingPatternRag()
    rag.add_patterns()
    
    query = "Two sum problem - find indices of numbers that add to target"
    results = rag.search_similar(query, k=1)
    print("Search Results for Query:", query)
    for name, description, distance in results:
        print(f"Pattern Name: {name}, Description: {description}, Distance: {distance:.4f} \n")

if __name__ == "__main__":
    test_rag_retrieval()