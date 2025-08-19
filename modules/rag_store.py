from google import genai
import faiss
from typing import List, Dict, Any
import numpy as np
import os
from modules.coding_problems_knowledge_base import PROBLEM_KNOWLEDGE_BASE, create_retrieval_chunks

# ==============================================================================
# The following functions would be in a separate "builder" script,
# used for the one-time setup of the index.
# ==============================================================================
OUTPUT_DIM = 768  # Default output dimension for Gemini embeddings
def _get_batch_embeddings(api_key: str, texts: List[str]) -> np.ndarray:
    """Helper function to get embeddings for a list of texts."""
    client = genai.Client(api_key=api_key)
    # Note: The Gemini API currently processes one string at a time in embed_content.
    # A real production system might use batching if the API supports it in the future.
    embeddings = [
        np.array(
            client.models.embed_content(
                model="gemini-embedding-001", 
                contents=text,
                config=genai.types.EmbedContentConfig(output_dimensionality=OUTPUT_DIM) # Specify output dimensionality
            ).embeddings[0].values)
        for text in texts
    ]
    return np.array(embeddings, dtype=np.float32)

def build_rag_index(api_key: str, knowledge_base: dict) -> tuple:
    """
    Builds the FAISS index and mappings from the knowledge base.
    This is a one-time setup process.
    """
    print("Processing knowledge base into retrieval chunks...")
    chunks = create_retrieval_chunks(knowledge_base)
    
    chunk_texts = [chunk['text'] for chunk in chunks]
    index_to_key_map = [chunk['metadata']['problem_key'] for chunk in chunks]

    print(f"Generated {len(chunk_texts)} chunks for embedding.")
    
    print("Generating embeddings...")
    embeddings = _get_batch_embeddings(api_key, chunk_texts)
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    
    print(f"Successfully built FAISS index with {index.ntotal} vectors.")
    
    # In a real app, you would save the index and map to disk here
    # faiss.write_index(index, "problems.index")
    # with open("index_map.json", "w") as f:
    #     json.dump(index_to_key_map, f)

    return index, index_to_key_map


# ==============================================================================
class CodingProblemRagStore:
    """
    A retrieval store for coding problems using a FAISS index and Gemini embeddings.
    This class is responsible for searching for semantically similar problems.
    """
    def __init__(self, api_key: str, knowledge_base: dict, index, index_to_key_map: list):
        """
        Initializes the RAG store.

        Args:
            api_key: Your Gemini API key.
            knowledge_base: The full dictionary of problem data.
            index: A pre-built FAISS index of the problem embeddings.
            index_to_key_map: A list mapping a FAISS vector index to its problem_key.
        """
        self.client = genai.Client(api_key=api_key)
        self.knowledge_base = knowledge_base
        self.index = index
        self.index_to_key_map = index_to_key_map
        self.embedding_model = "gemini-embedding-001"
        self.output_dim = OUTPUT_DIM

    def _get_single_embedding(self, text: str) -> np.ndarray:
        """Generates an embedding for a single text query."""
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=genai.types.EmbedContentConfig(output_dimensionality=self.output_dim) # Specify output dimensionality
            )
            return np.array(result.embeddings[0].values, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            print(f"Error getting embedding for query: {text[:50]}...")
            print(f"Error: {e}")
            return np.zeros((1, self.output_dim), dtype=np.float32)

    def search(self, query: str, k: int = 3, similarity_threshold: float = 0.70) -> List[Dict[str, Any]]:
        """
        Searches for similar coding problems and returns their full data packets.

        Args:
            query: The user's problem description.
            k: The number of top results to return.
            similarity_threshold: The minimum similarity score for a result to be considered a match.

        Returns:
            A list of full problem packet dictionaries that meet the similarity threshold.
        """
        if self.index is None:
            raise ValueError("The FAISS index is not loaded.")

        query_embedding = self._get_single_embedding(query)
        faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            # Check for valid index and non-negative distance
            similarity = distances[0][i]
            if idx != -1 and similarity >= 0:
                if similarity >= similarity_threshold:
                    problem_key = self.index_to_key_map[idx]
                    problem_packet = self.knowledge_base.get(problem_key)
                    if problem_packet:
                        results.append({
                            "similarity_score": round(float(similarity), 4),
                            "problem": problem_packet
                        })
        
        if not results:
            print("No results found above the similarity threshold.")
        
        return results