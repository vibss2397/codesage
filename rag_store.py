from sentence_transformers import SentenceTransformer
import faiss

from coding_patterns import PATTERNS, process_pattern

class CodingPatternRag:
    def __init__(self):
        # Initialize faiss and sentence-transformers
        # Use sentence-transformers/all-MiniLM-L6-v2 for embeddings
        self.transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(self.transformer.get_sentence_embedding_dimension())
        
    
    def add_patterns(self):
        # Add coding patterns to the FAISS index
        # Use embeddings from sentence-transformers for efficient similarity search.
        pattern_strings = [process_pattern(pattern) for pattern in PATTERNS]
        embeddings = self.transformer.encode(pattern_strings, convert_to_tensor=True)
        self.index.add(embeddings.cpu().numpy())

    
    def search_similar(self, query, k=3):
        # Search for similar coding patterns based on the query
        # Return top k similar patterns
        query_embedding = self.transformer.encode(query, convert_to_tensor=True)
        query_array = query_embedding.cpu().numpy().reshape(1, -1)
        distances, indices = self.index.search(query_array, k)
        return [(PATTERNS[i]['name'], PATTERNS[i]['description'], distances[0][j]) for j, i in enumerate(indices[0]) if i != -1]