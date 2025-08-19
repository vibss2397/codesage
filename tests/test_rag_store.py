import os
from dotenv import load_dotenv
from modules.rag_store import CodingProblemRagStore, build_rag_index

"""
--- Step 1: Define Knowledge Bases and Processing Functions ---
Knowledge Base for the NEW Problem-Focused RAG
"""

PROBLEM_KNOWLEDGE_BASE = {
"lc_3_longest_substring_no_repeats": {
"problem_id": "lc_3", "problem_title": "Longest Substring Without Repeating Characters", "patterns": ["sliding_window", "hash_map"], "difficulty": "Medium", "tags": ["string", "substring", "hash_set"], "problem_description": "Given a string s, find the length of the longest substring without repeating characters.", "canonical_solution": "..."
},
"lc_209_min_size_subarray_sum": {
"problem_id": "lc_209", "problem_title": "Minimum Size Subarray Sum", "patterns": ["sliding_window"], "difficulty": "Medium", "tags": ["array", "subarray", "sliding_window"], "problem_description": "Given an array of positive integers nums and a positive integer target, return the minimal length of a contiguous subarray whose sum is greater than or equal to target.", "canonical_solution": "..."
},
"lc_1_two_sum": {
"problem_id": "lc_1", "problem_title": "Two Sum", "patterns": ["hash_map"], "difficulty": "Easy", "tags": ["array", "hash_map"], "problem_description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.", "canonical_solution": "..."
},
"lc_98_validate_bst": {
"problem_id": "lc_98", "problem_title": "Validate Binary Search Tree", "patterns": ["tree_dfs"], "difficulty": "Medium", "tags": ["tree", "dfs", "bst"], "problem_description": "Given the root of a binary tree, determine if it is a valid binary search tree (BST).", "canonical_solution": "..."
},
}

# Knowledge Base for the OLD Pattern-Focused RAG
PATTERNS = {
"sliding_window": {
"when_to_use": ["Find longest/shortest substring with condition", "Maximum sum subarray of size K"],
"detection_patterns": {"inefficient": ["nested loops for subarrays"], "optimal": ["two pointers (left, right)"]},
"time_complexity": "O(n)", "space_complexity": "O(k)"
},
"two_pointers": {
"when_to_use": ["Find pairs with target sum in sorted array", "Palindrome checking"],
"detection_patterns": {"inefficient": ["nested loops for pair finding"], "optimal": ["left/right pointers"]},
"time_complexity": "O(n)", "space_complexity": "O(1)"
},
"tree_dfs": {
"when_to_use": ["Tree traversals", "Path sum problems", "Tree validation"],
"detection_patterns": {"inefficient": ["multiple tree traversals"], "optimal": ["single recursive pass"]},
"time_complexity": "O(n)", "space_complexity": "O(h)"
}
}


# --- Step 3: Main Test Script Execution ---
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable in a .env file.")

# --- Setup for NEW Problem-Focused RAG ---
print("\n" + "="*50)
print("🚀 SETTING UP NEW PROBLEM-FOCUSED RAG 🚀")
print("="*50)
problem_faiss_index, problem_index_map = build_rag_index(api_key, PROBLEM_KNOWLEDGE_BASE)
problem_rag_store = CodingProblemRagStore(api_key=api_key, knowledge_base=PROBLEM_KNOWLEDGE_BASE, index=problem_faiss_index, index_to_key_map=problem_index_map)

# --- Run Test Queries ---
print("\n" + "="*50)
print("🧪 RUNNING COMPARISON QUERIES 🧪")
print("="*50)

test_queries = [
    ("Strong Match", "Find the longest substring that doesn't have any duplicate characters."),
    ("Semantic Match", "Given a list of numbers, find the shortest continuous slice that sums up to at least a certain value."),
    ("Fallback Match", "How do I check if a binary tree is balanced?")
]

for name, query in test_queries:
    print(f"\n\n--- TEST: {name} ---")
    print(f"QUERY: \"{query}\"")

    # Test New Problem-Focused RAG
    print("\n--- RESULTS (New Problem-Focused RAG) ---")
    problem_results = problem_rag_store.search(query, k=1, similarity_threshold=0.7)
    if problem_results:
        print(problem_results)
    else:
        print("No confident match found.")

print("\n" + "="*50)
print("✅ TEST SCRIPT COMPLETE ✅")
print("="*50)