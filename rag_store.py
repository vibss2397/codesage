from google import genai
import faiss
from typing import List, Dict, Any
import numpy as np
import os
from coding_patterns import PATTERNS, process_pattern

class CodingPatternRag:
    def __init__(self, api_key: str = None):
        """
        Initialize RAG with Gemini embeddings API
        
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var
        """
        # Configure Gemini API
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass api_key parameter")
        
        self.client = genai.Client(api_key=api_key)
        
        # Initialize FAISS index (we'll set dimension after first embedding call)
        self.index = None
        self.chunks = []  # Store chunk metadata
        self.embedding_model = "models/embedding-001"  # Gemini embedding model
        self.output_dim = 768  # Default output dimension for Gemini embeddings
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings from Gemini API for list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        embeddings = []
        
        for text in texts[:3]:
            try:
                result = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=genai.types.EmbedContentConfig(output_dimensionality=self.output_dim) # Specify output dimensionality
                )
                embeddings.append(result.embeddings[0].values)
            except Exception as e:
                print(f"Error getting embedding for text: {text[:50]}...")
                print(f"Error: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * self.output_dim)  # Hardcoded dimension, can be adjusted based on model
        return np.array(embeddings, dtype=np.float32)
    
    def _get_single_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text string
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy array of single embedding
        """
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=genai.types.EmbedContentConfig(output_dimensionality=self.output_dim) # Specify output dimensionality
            )
            return np.array([result.embeddings[0].values], dtype=np.float32)
        except Exception as e:
            print(f"Error getting embedding for query: {text[:50]}...")
            print(f"Error: {e}")
            return np.zeros((1, self.output_dim), dtype=np.float32)  # Use zero vector as fallback
    
    def add_patterns(self):
        """
        Process all patterns and add their chunks to FAISS index
        """
        print("Processing patterns into chunks...")
        
        # Generate chunks from all patterns
        for pattern in PATTERNS:
            chunks = process_pattern(pattern)
            self.chunks.extend(chunks)
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        
        print(f"Generated {len(chunk_texts)} chunks")
        print("Getting embeddings from Gemini API...")
        
        # Get embeddings using Gemini API
        embeddings = self._get_embeddings(chunk_texts)
        
        # Initialize FAISS index with correct dimension
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Add embeddings to FAISS
        self.index.add(embeddings)
        
        print(f"✅ Added {len(chunk_texts)} chunks to FAISS index")
        print(f"✅ Embedding dimension: {embedding_dim}")
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar pattern chunks based on query
        
        Args:
            query: Search query text
            k: Number of top results to return
            
        Returns:
            List of dictionaries containing chunk info and similarity scores
        """
        if self.index is None or not self.chunks:
            raise ValueError("No patterns loaded. Call add_patterns() first.")
        
        # Get query embedding
        query_embedding = self._get_single_embedding(query)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'pattern': chunk['pattern'],
                    'chunk_type': chunk['chunk_type'],
                    'text': chunk['text'], 
                    'distance': float(distances[0][i]),
                    'similarity_score': 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity
                })
        
        return results
    
    def get_pattern_info(self, pattern_name: str) -> Dict[str, Any]:
        """
        Get full information for a specific pattern
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            Full pattern dictionary from PATTERNS
        """
        return PATTERNS.get(pattern_name, {})
    
    def list_available_patterns(self) -> List[str]:
        """Get list of all available pattern names"""
        return list(PATTERNS.keys())
    
    def get_chunks_for_pattern(self, pattern_name: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific pattern
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            List of chunks for that pattern
        """
        return [chunk for chunk in self.chunks if chunk['pattern'] == pattern_name]
