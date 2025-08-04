PATTERNS =  {
    # ============ ARRAY & STRING PATTERNS ============
    "two_pointers": {
        "category": "Array & String",
        "description": "Use two pointers moving towards each other or in same direction to solve array/string problems efficiently",
        "subcategories": ["opposite_direction", "same_direction", "fast_slow"],
        
        "when_to_use": [
            "Find pairs with target sum in sorted array",
            "Remove duplicates from sorted array",
            "Reverse array/string in place",
            "Palindrome checking",
            "Merge sorted arrays",
            "Partition problems",
            "3Sum, 4Sum problems"
        ],
        
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
        
        "code_template": """
def two_pointers_opposite(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

def two_pointers_same_direction(arr):
    slow = fast = 0
    while fast < len(arr):
        if condition:
            arr[slow] = arr[fast]
            slow += 1
        fast += 1
    return slow
        """,
        
        "detection_patterns": {
            "inefficient": ["nested loops for pair finding", "O(n²) enumeration"],
            "optimal": ["left/right pointers", "slow/fast pointers", "convergence pattern"]
        },
        
        "progressive_hints": [
            "Can you avoid checking every pair?",
            "What if you used pointers from opposite ends?",
            "How can sorted order help eliminate possibilities?",
            "Consider using two pointers moving at different speeds"
        ],
        
        "common_problems": [
            "Two Sum II", "3Sum", "Container With Most Water", 
            "Remove Duplicates", "Valid Palindrome"
        ]
    },

    "sliding_window": {
        "category": "Array & String",
        "description": "Maintain a window of elements and slide it to find optimal subarray/substring",
        "subcategories": ["fixed_size", "variable_size", "shrinkable"],
        
        "when_to_use": [
            "Find longest/shortest substring with condition",
            "Maximum sum subarray of size K",
            "Count distinct elements in window",
            "String permutation problems",
            "Minimum window covering substring"
        ],
        
        "time_complexity": "O(n)",
        "space_complexity": "O(k) where k is window size or distinct elements",
        
        "code_template": """
def sliding_window_fixed(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    return max_sum

def sliding_window_variable(s, condition):
    left = 0
    result = 0
    window_data = {}
    
    for right in range(len(s)):
        # Expand window
        window_data[s[right]] = window_data.get(s[right], 0) + 1
        
        # Shrink window if needed
        while not condition_met(window_data):
            window_data[s[left]] -= 1
            if window_data[s[left]] == 0:
                del window_data[s[left]]
            left += 1
        
        result = max(result, right - left + 1)
    return result
        """,
        
        "detection_patterns": {
            "inefficient": ["nested loops for subarrays", "O(n²) substring checking"],
            "optimal": ["two pointers (left, right)", "single pass with window tracking"]
        },
        
        "progressive_hints": [
            "Can you avoid recalculating from scratch for each position?",
            "What if you maintained a 'window' and only adjusted boundaries?",
            "Think about what to add/remove when window slides"
        ],
        
        "common_problems": [
            "Longest Substring Without Repeating Characters",
            "Minimum Window Substring", "Max Consecutive Ones",
            "Longest Repeating Character Replacement"
        ]
    },

    "hash_map": {
        "category": "Array & String",
        "description": "Use hash table for O(1) lookups to replace expensive searches",
        "subcategories": ["lookup", "frequency_count", "complement_search"],
        
        "when_to_use": [
            "Two Sum problems",
            "Check for duplicates",
            "Frequency counting",
            "Anagram detection",
            "Group elements by property"
        ],
        
        "time_complexity": "O(n)",
        "space_complexity": "O(n)",
        
        "code_template": """
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

def frequency_count(arr):
    freq = {}
    for item in arr:
        freq[item] = freq.get(item, 0) + 1
    return freq
        """,
        
        "detection_patterns": {
            "inefficient": ["nested loops for searching", "multiple passes for counting"],
            "optimal": ["single pass with hash lookups", "dict.get() or 'in' checks"]
        },
        
        "progressive_hints": [
            "Can you check existence in constant time?",
            "What if you stored what you've seen so far?",
            "Think about trading space for time"
        ],
        
        "common_problems": [
            "Two Sum", "Group Anagrams", "Valid Anagram",
            "First Non-Repeating Character"
        ]
    },

    # ============ BINARY SEARCH PATTERNS ============
    "binary_search": {
        "category": "Search",
        "description": "Efficiently search sorted data by eliminating half the search space each step",
        "subcategories": ["classic", "first_occurrence", "last_occurrence", "peak_finding"],
        
        "when_to_use": [
            "Search in sorted array",
            "Find first/last occurrence",
            "Search in rotated sorted array",
            "Find peak element",
            "Search in 2D matrix"
        ],
        
        "time_complexity": "O(log n)",
        "space_complexity": "O(1)",
        
        "code_template": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def find_first_occurrence(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result
        """,
        
        "detection_patterns": {
            "inefficient": ["linear search in sorted data", "O(n) scanning"],
            "optimal": ["divide and conquer", "mid calculation", "boundary adjustment"]
        },
        
        "progressive_hints": [
            "Can you eliminate half the possibilities each step?",
            "What property of sorted data can you exploit?",
            "Think about the invariant you want to maintain"
        ],
        
        "common_problems": [
            "Binary Search", "Search in Rotated Sorted Array",
            "Find First and Last Position", "Search a 2D Matrix"
        ]
    },

    # ============ TREE PATTERNS ============
    "tree_dfs": {
        "category": "Tree",
        "description": "Traverse tree depth-first using recursion or stack",
        "subcategories": ["preorder", "inorder", "postorder", "path_problems"],
        
        "when_to_use": [
            "Tree traversals",
            "Path sum problems",
            "Tree validation",
            "Serialize/deserialize trees",
            "Find paths with conditions"
        ],
        
        "time_complexity": "O(n)",
        "space_complexity": "O(h) where h is tree height",
        
        "code_template": """
def dfs_recursive(root):
    if not root:
        return
    
    # Preorder: process current, then children
    process(root.val)
    dfs_recursive(root.left)
    dfs_recursive(root.right)

def path_sum(root, target):
    def dfs(node, current_sum):
        if not node:
            return False
        
        current_sum += node.val
        
        # Leaf node check
        if not node.left and not node.right:
            return current_sum == target
        
        return dfs(node.left, current_sum) or dfs(node.right, current_sum)
    
    return dfs(root, 0)
        """,
        
        "detection_patterns": {
            "inefficient": ["multiple tree traversals", "level-by-level for path problems"],
            "optimal": ["single recursive pass", "backtracking pattern"]
        },
        
        "progressive_hints": [
            "Can you solve this by processing current node and recursing?",
            "What information do you need to pass down to children?",
            "Think about base case and recursive case"
        ],
        
        "common_problems": [
            "Binary Tree Paths", "Path Sum", "Maximum Depth",
            "Validate Binary Search Tree"
        ]
    },

    "tree_bfs": {
        "category": "Tree",
        "description": "Traverse tree level by level using queue",
        "subcategories": ["level_order", "level_processing", "rightmost_view"],
        
        "when_to_use": [
            "Level order traversal",
            "Find nodes at specific level",
            "Tree serialization",
            "Find rightmost/leftmost nodes",
            "Level-based processing"
        ],
        
        "time_complexity": "O(n)",
        "space_complexity": "O(w) where w is maximum width",
        
        "code_template": """
from collections import deque

def level_order(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_nodes = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level_nodes.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_nodes)
    return result
        """,
        
        "detection_patterns": {
            "inefficient": ["recursive solution for level problems", "DFS for level order"],
            "optimal": ["queue-based iteration", "level size tracking"]
        },
        
        "progressive_hints": [
            "Do you need to process nodes level by level?",
            "What data structure processes elements in order added?",
            "How can you track when a level ends?"
        ],
        
        "common_problems": [
            "Binary Tree Level Order Traversal",
            "Binary Tree Right Side View", "Minimum Depth"
        ]
    },

    # ============ GRAPH PATTERNS ============
    "graph_dfs": {
        "category": "Graph",
        "description": "Explore graph paths completely using DFS",
        "subcategories": ["connected_components", "cycle_detection", "path_finding"],
        
        "when_to_use": [
            "Find connected components",
            "Detect cycles",
            "Topological sorting",
            "Path finding problems",
            "Island problems"
        ],
        
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        
        "code_template": """
def dfs_graph(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_graph(graph, neighbor, visited)
    
    return visited

def count_islands(grid):
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    visited = set()
    islands = 0
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            (r, c) in visited or grid[r][c] == '0'):
            return
        
        visited.add((r, c))
        # Visit all 4 directions
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            dfs(r + dr, c + dc)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                dfs(r, c)
                islands += 1
    
    return islands
        """,
        
        "detection_patterns": {
            "inefficient": ["BFS for deep path problems", "multiple traversals"],
            "optimal": ["recursive DFS", "visited set", "backtracking"]
        },
        
        "progressive_hints": [
            "Do you need to explore all connected nodes?",
            "Can you mark visited nodes to avoid cycles?",
            "Think about when to use DFS vs BFS"
        ],
        
        "common_problems": [
            "Number of Islands", "Clone Graph", "Course Schedule",
            "Word Search"
        ]
    },

    "graph_bfs": {
        "category": "Graph",
        "description": "Explore graph level by level for shortest paths",
        "subcategories": ["shortest_path", "level_exploration", "minimum_steps"],
        
        "when_to_use": [
            "Shortest path in unweighted graph",
            "Minimum steps problems",
            "Level-based exploration",
            "Word ladder problems"
        ],
        
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        
        "code_template": """
from collections import deque

def bfs_shortest_path(graph, start, target):
    if start == target:
        return 0
    
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        node, distance = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == target:
                return distance + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return -1
        """,
        
        "detection_patterns": {
            "inefficient": ["DFS for shortest path", "multiple path exploration"],
            "optimal": ["queue-based BFS", "distance tracking"]
        },
        
        "progressive_hints": [
            "Do you need the shortest path?",
            "What explores nodes in order of distance?",
            "Think about level-by-level exploration"
        ],
        
        "common_problems": [
            "Word Ladder", "Shortest Path in Binary Matrix",
            "Minimum Knight Moves"
        ]
    },

    # ============ DYNAMIC PROGRAMMING PATTERNS ============
    "dynamic_programming": {
        "category": "Dynamic Programming",
        "description": "Break down problems into overlapping subproblems and store solutions",
        "subcategories": ["1d_dp", "2d_dp", "knapsack", "subsequence", "palindrome"],
        
        "when_to_use": [
            "Optimization problems (min/max)",
            "Counting problems",
            "Decision problems (possible/impossible)",
            "Problems with overlapping subproblems",
            "Sequence/string problems"
        ],
        
        "time_complexity": "O(n) to O(n³) depending on dimensions",
        "space_complexity": "O(n) to O(n²) (can often be optimized)",
        
        "code_template": """
# 1D DP - Fibonacci
def fibonacci(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# 2D DP - Unique Paths
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]

# Knapsack Pattern
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],  # don't take item
                    dp[i-1][w-weights[i-1]] + values[i-1]  # take item
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]
        """,
        
        "detection_patterns": {
            "inefficient": ["recursive with repeated calculations", "exponential time complexity"],
            "optimal": ["memoization", "bottom-up table", "state transitions"]
        },
        
        "progressive_hints": [
            "Are you solving the same subproblems multiple times?",
            "Can you break this into smaller similar problems?",
            "What's the recurrence relation?",
            "What state do you need to track?"
        ],
        
        "common_problems": [
            "Climbing Stairs", "Coin Change", "Longest Common Subsequence",
            "Edit Distance", "House Robber"
        ]
    },

    # ============ BACKTRACKING PATTERNS ============
    "backtracking": {
        "category": "Backtracking",
        "description": "Explore all possible solutions by trying choices and backtracking on failure",
        "subcategories": ["permutations", "combinations", "subsets", "constraint_satisfaction"],
        
        "when_to_use": [
            "Generate all permutations/combinations",
            "Sudoku/N-Queens problems",
            "Word search in grid",
            "Generate all subsets",
            "Path finding with constraints"
        ],
        
        "time_complexity": "Exponential (varies by problem)",
        "space_complexity": "O(depth of recursion)",
        
        "code_template": """
def backtrack_template(nums):
    result = []
    
    def backtrack(current_path, remaining_choices):
        # Base case
        if is_complete(current_path):
            result.append(current_path[:])  # Make a copy
            return
        
        for choice in remaining_choices:
            if is_valid(choice):
                # Make choice
                current_path.append(choice)
                
                # Recurse
                backtrack(current_path, get_new_choices(remaining_choices, choice))
                
                # Backtrack
                current_path.pop()
    
    backtrack([], nums)
    return result

def generate_subsets(nums):
    result = []
    
    def backtrack(start, current_subset):
        result.append(current_subset[:])
        
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()
    
    backtrack(0, [])
    return result
        """,
        
        "detection_patterns": {
            "inefficient": ["iterative generation", "duplicate work"],
            "optimal": ["recursive choice making", "backtracking pattern", "pruning"]
        },
        
        "progressive_hints": [
            "Do you need to try all possibilities?",
            "Can you make a choice, recurse, then undo the choice?",
            "What's your base case for complete solutions?"
        ],
        
        "common_problems": [
            "Subsets", "Permutations", "Combination Sum",
            "N-Queens", "Word Search"
        ]
    },

    # ============ GREEDY PATTERNS ============
    "greedy": {
        "category": "Greedy",
        "description": "Make locally optimal choices at each step",
        "subcategories": ["activity_selection", "huffman_coding", "minimum_spanning_tree"],
        
        "when_to_use": [
            "Activity selection problems",
            "Interval scheduling",
            "Minimum spanning tree",
            "Fractional knapsack",
            "Job scheduling"
        ],
        
        "time_complexity": "Usually O(n log n) due to sorting",
        "space_complexity": "O(1) to O(n)",
        
        "code_template": """
def activity_selection(activities):
    # Sort by end time
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0]]
    last_end_time = activities[0][1]
    
    for start, end in activities[1:]:
        if start >= last_end_time:
            selected.append((start, end))
            last_end_time = end
    
    return selected

def gas_station(gas, cost):
    total_tank = current_tank = start = 0
    
    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        current_tank += gas[i] - cost[i]
        
        if current_tank < 0:
            start = i + 1
            current_tank = 0
    
    return start if total_tank >= 0 else -1
        """,
        
        "detection_patterns": {
            "inefficient": ["trying all combinations", "dynamic programming where greedy works"],
            "optimal": ["sorting by criteria", "local optimal choices"]
        },
        
        "progressive_hints": [
            "Can you make the best choice at each step?",
            "What property should you sort by?",
            "Will local optimal lead to global optimal?"
        ],
        
        "common_problems": [
            "Activity Selection", "Gas Station", "Jump Game",
            "Meeting Rooms II"
        ]
    },

    # ============ HEAP PATTERNS ============
    "heap": {
        "category": "Heap",
        "description": "Use heap for efficient min/max operations and top-k problems",
        "subcategories": ["min_heap", "max_heap", "top_k", "merge_k"],
        
        "when_to_use": [
            "Find top K elements",
            "Merge K sorted lists",
            "Running median",
            "Task scheduling with priority",
            "Dijkstra's algorithm"
        ],
        
        "time_complexity": "O(n log k) for top-k, O(log n) for insert/delete",
        "space_complexity": "O(k) for top-k problems",
        
        "code_template": """
import heapq

def top_k_frequent(nums, k):
    frequency = {}
    for num in nums:
        frequency[num] = frequency.get(num, 0) + 1
    
    heap = []
    for num, freq in frequency.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]

def merge_k_sorted_lists(lists):
    heap = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, list_idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, list_idx, node.next))
    
    return dummy.next
        """,
        
        "detection_patterns": {
            "inefficient": ["sorting entire array for top-k", "linear search for min/max"],
            "optimal": ["heapq operations", "maintaining heap of size k"]
        },
        
        "progressive_hints": [
            "Do you need the top/bottom K elements?",
            "Can you maintain a data structure for min/max?",
            "What if you only kept track of the K elements you care about?"
        ],
        
        "common_problems": [
            "Top K Frequent Elements", "Merge k Sorted Lists",
            "Find Median from Data Stream", "Kth Largest Element"
        ]
    },

    # ============ ADVANCED PATTERNS ============
    "union_find": {
        "category": "Advanced",
        "description": "Efficiently handle dynamic connectivity queries",
        "subcategories": ["path_compression", "union_by_rank"],
        
        "when_to_use": [
            "Connected components in graph",
            "Detect cycles in undirected graph",
            "Kruskal's MST algorithm",
            "Friend groups problems",
            "Percolation problems"
        ],
        
        "time_complexity": "Nearly O(1) with optimizations",
        "space_complexity": "O(n)",
        
        "code_template": """
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
        """,
        
        "detection_patterns": {
            "inefficient": ["DFS/BFS for connectivity queries", "adjacency list maintenance"],
            "optimal": ["parent array", "path compression", "union by rank"]
        },
        
        "progressive_hints": [
            "Do you need to track connected components efficiently?",
            "Can you represent each component by its representative?",
            "How can you flatten the tree structure for faster queries?"
        ],
        
        "common_problems": [
            "Number of Connected Components", "Redundant Connection",
            "Accounts Merge", "Most Stones Removed"
        ]
    },

    "trie": {
        "category": "Advanced",
        "description": "Tree-like data structure for efficient string storage and retrieval",
        "subcategories": ["prefix_matching", "word_search", "autocomplete"],
        
        "when_to_use": [
            "Prefix matching",
            "Word search/dictionary",
            "Autocomplete features",
            "Find words with common prefix",
            "Phone number/IP address validation"
        ],
        
        "time_complexity": "O(m) where m is word length",
        "space_complexity": "O(ALPHABET_SIZE * N * M)",
        
        "code_template": """
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_word = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_word
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
        """,
        
        "detection_patterns": {
            "inefficient": ["linear search through word list", "repeated prefix computation"],
            "optimal": ["tree-like character storage", "path represents words"]
        },
        
        "progressive_hints": [
            "Do you need efficient prefix operations?",
            "Can you store strings in a tree-like structure?",
            "What if each path from root to leaf represents a word?"
        ],
        
        "common_problems": [
            "Implement Trie", "Word Search II", "Add and Search Word",
            "Replace Words"
        ]
    }
}


def process_pattern(pattern_name):
    """Convert pattern into RAG chunks"""
    pattern = PATTERNS[pattern_name]
    chunks = []
    
    # Define chunk templates
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
    
    # Generate chunks
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
    """Helper to get nested dict values like 'detection_patterns.inefficient'"""
    keys = field_path.split('.')
    value = data
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            break
    return value


if __name__ == "__main__":
    # Example usage
    chunks = process_pattern("two_pointers")
    pattern_name = "two_pointers"
    print(f"Processed chunks for {pattern_name}:")
    for chunk in chunks:
        print(chunk)
        print('\n')