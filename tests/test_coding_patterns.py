# Test script for process_pattern function
from coding_patterns import process_pattern
# Sample patterns data for testing
TEST_PATTERNS = {
    "two_pointers": {
        "category": "Array & String",
        "description": "Use two pointers moving towards each other or in same direction",
        "when_to_use": [
            "Find pairs with target sum in sorted array",
            "Remove duplicates from sorted array",
            "Palindrome checking"
        ],
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
        "detection_patterns": {
            "inefficient": ["nested loops for pair finding", "O(n²) enumeration"],
            "optimal": ["left/right pointers", "slow/fast pointers"]
        },
        "common_problems": ["Two Sum II", "3Sum", "Valid Palindrome"]
    },
    
    "sliding_window": {
        "category": "Array & String", 
        "description": "Maintain a window of elements and slide it to find optimal subarray",
        "when_to_use": [
            "Find longest/shortest substring with condition",
            "Maximum sum subarray of size K"
        ],
        "time_complexity": "O(n)",
        "detection_patterns": {
            "inefficient": ["nested loops for subarrays", "O(n²) substring checking"],
            "optimal": ["two pointers (left, right)", "single pass with window tracking"]
        }
        # Note: missing space_complexity and common_problems to test robustness
    },
    
    "hash_map": {
        "category": "Array & String",
        "description": "Use hash table for O(1) lookups",
        "when_to_use": ["Two Sum problems", "Check for duplicates"],
        "time_complexity": "O(n)",
        "space_complexity": "O(n)"
        # Note: missing detection_patterns to test edge case
    }
}

def test_single_pattern():
    """Test processing a single pattern"""
    print("=== TESTING SINGLE PATTERN: two_pointers ===")
    
    chunks = process_pattern("two_pointers", TEST_PATTERNS)
    
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({chunk['chunk_type']}):")
        print(f"  Pattern: {chunk['pattern']}")
        print(f"  Text: {chunk['text']}")
    
    return chunks

def test_missing_fields():
    """Test pattern with missing fields"""
    print("\n=== TESTING MISSING FIELDS: sliding_window ===")
    
    chunks = process_pattern("sliding_window", TEST_PATTERNS)
    
    print(f"Generated {len(chunks)} chunks (some fields missing):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({chunk['chunk_type']}):")
        print(f"  Text: {chunk['text']}")

def test_minimal_pattern():
    """Test pattern with minimal fields"""
    print("\n=== TESTING MINIMAL PATTERN: hash_map ===")
    
    chunks = process_pattern("hash_map", TEST_PATTERNS)
    
    print(f"Generated {len(chunks)} chunks (minimal fields):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({chunk['chunk_type']}):")
        print(f"  Text: {chunk['text']}")

def test_all_patterns():
    """Test processing all patterns at once"""
    print("\n=== TESTING ALL PATTERNS ===")
    
    all_chunks = process_all_patterns(TEST_PATTERNS)
    
    print(f"Total chunks generated: {len(all_chunks)}")
    
    # Group by pattern for summary
    pattern_counts = {}
    for chunk in all_chunks:
        pattern = chunk['pattern']
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print("\nChunks per pattern:")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} chunks")

def test_chunk_types():
    """Test what chunk types are generated"""
    print("\n=== TESTING CHUNK TYPES ===")
    
    all_chunks = process_all_patterns(TEST_PATTERNS)
    
    chunk_types = set(chunk['chunk_type'] for chunk in all_chunks)
    print(f"Chunk types generated: {', '.join(sorted(chunk_types))}")
    
    # Show examples of each type
    for chunk_type in sorted(chunk_types):
        example = next(chunk for chunk in all_chunks if chunk['chunk_type'] == chunk_type)
        print(f"\n{chunk_type} example:")
        print(f"  {example['text'][:80]}...")

def test_error_handling():
    """Test error cases"""
    print("\n=== TESTING ERROR HANDLING ===")
    
    try:
        process_pattern("nonexistent_pattern", TEST_PATTERNS)
        print("❌ Should have raised error for nonexistent pattern")
    except (KeyError, ValueError) as e:
        print(f"✅ Correctly handled missing pattern: {e}")

def run_all_tests():
    """Run all tests"""
    print("🧪 TESTING PROCESS_PATTERN FUNCTION\n")
    
    # Test individual functions
    test_single_pattern()
    test_missing_fields() 
    test_minimal_pattern()
    test_all_patterns()
    test_chunk_types()
    test_error_handling()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    # Copy the process_pattern functions here or import them
    # For now, including simplified versions for testing
    
    def process_pattern(pattern_name, patterns_dict):
        """Simplified version for testing"""
        pattern = patterns_dict[pattern_name]
        chunks = []
        
        chunk_configs = [
            {
                "type": "problem_recognition",
                "fields": ["when_to_use", "common_problems"],
                "template": f"{pattern_name} solves: {{content}}"
            },
            {
                "type": "inefficient_detection", 
                "fields": ["detection_patterns.inefficient"],
                "template": f"Inefficient patterns suggesting {pattern_name}: {{content}}"
            },
            {
                "type": "optimal_approach",
                "fields": ["detection_patterns.optimal"], 
                "template": f"{pattern_name} optimal approach: {{content}}"
            },
            {
                "type": "complexity_info",
                "fields": ["time_complexity", "space_complexity"],
                "template": f"{pattern_name} complexity: {{content}}"
            }
        ]
        
        for config in chunk_configs:
            content_parts = []
            for field in config["fields"]:
                value = _get_nested_field(pattern, field)
                if value:
                    content_parts.extend(value if isinstance(value, list) else [value])
            
            if content_parts:
                chunks.append({
                    "pattern": pattern_name,
                    "chunk_type": config["type"],
                    "text": config["template"].format(content=", ".join(content_parts))
                })
        
        return chunks

    def _get_nested_field(data, field_path):
        """Helper to get nested dict values"""
        keys = field_path.split('.')
        value = data
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                break
        return value

    def process_all_patterns(patterns_dict):
        """Process all patterns into chunks"""
        return [chunk for name in patterns_dict 
                for chunk in process_pattern(name, patterns_dict)]
    
    # Run tests
    run_all_tests()