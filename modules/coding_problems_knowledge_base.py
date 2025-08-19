PROBLEM_KNOWLEDGE_BASE = {
    # ==================== Array & String ====================
    "lc_125_valid_palindrome": {
        "problem_id": "lc_125",
        "problem_title": "Valid Palindrome",
        "patterns": ["two_pointers"],
        "difficulty": "Easy",
        "tags": ["string", "two_pointers"],
        "problem_description": "Given a string s, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.",
        "canonical_solution": """
        def isPalindrome(s: str) -> bool:
        left, right = 0, len(s) - 1
        while left < right:
        while left < right and not s[left].isalnum():
        left += 1
        while left < right and not s[right].isalnum():
        right -= 1
        if s[left].lower() != s[right].lower():
        return False
        left, right = left + 1, right - 1
        return True
    """
    },
    "lc_15_3sum": {
        "problem_id": "lc_15",
        "problem_title": "3Sum",
        "patterns": ["two_pointers"],
        "difficulty": "Medium",
        "tags": ["array", "two_pointers", "sorting"],
        "problem_description": "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.",
        "canonical_solution": """
        def threeSum(nums: list[int]]) -> list[list[int]]:
        res = []
        nums.sort()
        for i, a in enumerate(nums):
        if i > 0 and a == nums[i - 1]:
        continue
        l, r = i + 1, len(nums) - 1
        while l < r:
        threeSum = a + nums[l] + nums[r]
        if threeSum > 0:
        r -= 1
        elif threeSum < 0:
        l += 1
        else:
        res.append([a, nums[l], nums[r]])
        l += 1
        while nums[l] == nums[l - 1] and l < r:
        l += 1
        return res
    """
    },
    "lc_3_longest_substring_no_repeats": {
        "problem_id": "lc_3",
        "problem_title": "Longest Substring Without Repeating Characters",
        "patterns": ["sliding_window", "hash_map"],
        "difficulty": "Medium",
        "tags": ["string", "substring", "hash_set"],
        "problem_description": "Given a string s, find the length of the longest substring without repeating characters.",
        "canonical_solution": """
        def lengthOfLongestSubstring(s: str) -> int:
        char_set = set()
        left = 0
        max_len = 0
        for right in range(len(s)):
        while s[right] in char_set:
        char_set.remove(s[left])
        left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
        return max_len
        """
    },
    "lc_209_min_size_subarray_sum": {
        "problem_id": "lc_209",
        "problem_title": "Minimum Size Subarray Sum",
        "patterns": ["sliding_window"],
        "difficulty": "Medium",
        "tags": ["array", "subarray", "sliding_window"],
        "problem_description": "Given an array of positive integers nums and a positive integer target, return the minimal length of a contiguous subarray whose sum is greater than or equal to target.",
        "canonical_solution": """
        def minSubArrayLen(target: int, nums: list[int]) -> int:
        left, total = 0, 0
        res = float("inf")
        for right in range(len(nums)):
        total += nums[right]
        while total >= target:
        res = min(res, right - left + 1)
        total -= nums[left]
        left += 1
        return 0 if res == float("inf") else res
    """
    },
    "lc_1_two_sum": {
        "problem_id": "lc_1",
        "problem_title": "Two Sum",
        "patterns": ["hash_map"],
        "difficulty": "Easy",
        "tags": ["array", "hash_map"],
        "problem_description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        "canonical_solution": """
        def twoSum(nums: list[int], target: int) -> list[int]:
        prevMap = {}  # val -> index
        for i, n in enumerate(nums):
        diff = target - n
        if diff in prevMap:
        return [prevMap[diff], i]
        prevMap[n] = i
        """
    },
    "lc_49_group_anagrams": {
        "problem_id": "lc_49",
        "problem_title": "Group Anagrams",
        "patterns": ["hash_map"],
        "difficulty": "Medium",
        "tags": ["string", "hash_map", "sorting"],
        "problem_description": "Given an array of strings strs, group the anagrams together. You can return the answer in any order.",
        "canonical_solution": """
        from collections import defaultdict

        def groupAnagrams(strs: list[str]) -> list[list[str]]:
        ans = defaultdict(list)
        for s in strs:
        count = [0] * 26
        for c in s:
        count[ord(c) - ord("a")] += 1
        ans[tuple(count)].append(s)
        return list(ans.values())
        """
    },
    # ==================== Search ====================
    "lc_704_binary_search": {
        "problem_id": "lc_704",
        "problem_title": "Binary Search",
        "patterns": ["binary_search"],
        "difficulty": "Easy",
        "tags": ["array", "binary_search"],
        "problem_description": "Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.",
        "canonical_solution": """
        def search(nums: list[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
        m = (l + r) // 2
        if nums[m] > target:
        r = m - 1
        elif nums[m] < target:
        l = m + 1
        else:
        return m
        return -1
        """
    },
    "lc_33_search_rotated_sorted_array": {
        "problem_id": "lc_33",
        "problem_title": "Search in Rotated Sorted Array",
        "patterns": ["binary_search"],
        "difficulty": "Medium",
        "tags": ["array", "binary_search"],
        "problem_description": "There is an integer array nums sorted in ascending order (with distinct values). Prior to being passed to your function, nums is possibly rotated at an unknown pivot index. Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.",
        "canonical_solution": """
        def search(nums: list[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
        mid = (l + r) // 2
        if target == nums[mid]:
        return mid
        # left sorted portion
        if nums[l] <= nums[mid]:
        if target > nums[mid] or target < nums[l]:
        l = mid + 1
        else:
        r = mid - 1
        # right sorted portion
        else:
        if target < nums[mid] or target > nums[r]:
        r = mid - 1
        else:
        l = mid + 1
        return -1
        """
    },
    # ==================== Tree ====================
    "lc_104_max_depth_binary_tree": {
        "problem_id": "lc_104",
        "problem_title": "Maximum Depth of Binary Tree",
        "patterns": ["tree_dfs"],
        "difficulty": "Easy",
        "tags": ["tree", "dfs", "recursion"],
        "problem_description": "Given the root of a binary tree, return its maximum depth.",
        "canonical_solution": """

        Definition for a binary tree node.
        class TreeNode:
        def init(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        def maxDepth(root) -> int:
        if not root:
        return 0
        return 1 + max(maxDepth(root.left), maxDepth(root.right))
        """
    },
    "lc_98_validate_bst": {
        "problem_id": "lc_98",
        "problem_title": "Validate Binary Search Tree",
        "patterns": ["tree_dfs"],
        "difficulty": "Medium",
        "tags": ["tree", "dfs", "bst"],
        "problem_description": "Given the root of a binary tree, determine if it is a valid binary search tree (BST).",
        "canonical_solution": """

        Definition for a binary tree node.
        class TreeNode:
        def init(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        def isValidBST(root) -> bool:
        def valid(node, left, right):
        if not node:
        return True
        if not (node.val < right and node.val > left):
        return False
        return valid(node.left, left, node.val) and valid(node.right, node.val, right)
        return valid(root, float("-inf"), float("inf"))
        """
    },
    "lc_102_level_order_traversal": {
        "problem_id": "lc_102",
        "problem_title": "Binary Tree Level Order Traversal",
        "patterns": ["tree_bfs"],
        "difficulty": "Medium",
        "tags": ["tree", "bfs", "queue"],
        "problem_description": "Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).",
        "canonical_solution": """
        from collections import deque

        Definition for a binary tree node.
        class TreeNode:
        def init(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        def levelOrder(root) -> list[list[int]]:
        res = []
        q = deque()
        if root:
        q.append(root)
        while q:
        val = []
        for i in range(len(q)):
        node = q.popleft()
        val.append(node.val)
        if node.left:
        q.append(node.left)
        if node.right:
        q.append(node.right)
        res.append(val)
        return res
        """
    },
    "lc_199_binary_tree_right_side_view": {
        "problem_id": "lc_199",
        "problem_title": "Binary Tree Right Side View",
        "patterns": ["tree_bfs"],
        "difficulty": "Medium",
        "tags": ["tree", "bfs", "queue"],
        "problem_description": "Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.",
        "canonical_solution": """
        from collections import deque

        Definition for a binary tree node.
        class TreeNode:
        def init(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        def rightSideView(root) -> list[int]:
        res = []
        q = deque()
        if root:
        q.append(root)
        while q:
        rightSide = None
        qLen = len(q)
        for i in range(qLen):
        node = q.popleft()
        if node:
        rightSide = node
        if node.left:
        q.append(node.left)
        if node.right:
        q.append(node.right)
        if rightSide:
        res.append(rightSide.val)
        return res
    """
    },
    # ==================== Graph ====================
    "lc_200_number_of_islands": {
        "problem_id": "lc_200",
        "problem_title": "Number of Islands",
        "patterns": ["graph_dfs", "graph_bfs"],
        "difficulty": "Medium",
        "tags": ["graph", "dfs", "bfs", "matrix"],
        "problem_description": "Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.",
        "canonical_solution": """
        def numIslands(grid: list[list[str]]) -> int:
        if not grid:
        return 0
        rows, cols = len(grid), len(grid[0])
        visited = set()
        islands = 0
        def dfs(r, c):
        if (r not in range(rows) or
        c not in range(cols) or
        grid[r][c] == "0" or
        (r, c) in visited):
        return
        visited.add((r, c))
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for dr, dc in directions:
        dfs(r + dr, c + dc)
        for r in range(rows):
        for c in range(cols):
        if grid[r][c] == "1" and (r, c) not in visited:
        islands += 1
        dfs(r, c)
        return islands
        """
    },
    "lc_133_clone_graph": {
    "problem_id": "lc_133",
    "problem_title": "Clone Graph",
    "patterns": ["graph_dfs", "graph_bfs"],
    "difficulty": "Medium",
    "tags": ["graph", "dfs", "bfs", "hash_map"],
    "problem_description": "Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph.",
    "canonical_solution": """

        Definition for a Node.
        class Node:
        def init(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
        def cloneGraph(node):
        oldToNew = {}
        def dfs(node):
        if node in oldToNew:
        return oldToNew[node]
        copy = Node(node.val)
        oldToNew[node] = copy
        for nei in node.neighbors:
        copy.neighbors.append(dfs(nei))
        return copy
        return dfs(node) if node else None
        """
    },
    "lc_994_rotting_oranges": {
        "problem_id": "lc_994",
        "problem_title": "Rotting Oranges",
        "patterns": ["graph_bfs"],
        "difficulty": "Medium",
        "tags": ["graph", "bfs", "matrix", "queue"],
        "problem_description": "You are given an m x n grid where each cell can have one of three values: 0 representing an empty cell, 1 representing a fresh orange, or 2 representing a rotten orange. Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten. Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.",
        "canonical_solution": """
        from collections import deque
        def orangesRotting(grid: list[list[int]]) -> int:
        q = deque()
        fresh, time = 0, 0
        ROWS, COLS = len(grid), len(grid[0])
        for r in range(ROWS):
        for c in range(COLS):
        if grid[r][c] == 1:
        fresh += 1
        if grid[r][c] == 2:
        q.append((r, c))
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        while fresh > 0 and q:
        length = len(q)
        for i in range(length):
        r, c = q.popleft()
        for dr, dc in directions:
        row, col = r + dr, c + dc
        if (row in range(ROWS) and
        col in range(COLS) and
        grid[row][col] == 1):
        grid[row][col] = 2
        q.append((row, col))
        fresh -= 1
        time += 1
        return time if fresh == 0 else -1
        """
    },
    "lc_127_word_ladder": {
        "problem_id": "lc_127",
        "problem_title": "Word Ladder",
        "patterns": ["graph_bfs"],
        "difficulty": "Hard",
        "tags": ["graph", "bfs", "string"],
        "problem_description": "A transformation sequence from word beginWord to endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that every adjacent pair of words differs by a single letter. Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence, or 0 if no such sequence exists.",
        "canonical_solution": """
        from collections import deque
        def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:
        if endWord not in wordList:
        return 0
        nei = collections.defaultdict(list)
        wordList.append(beginWord)
        for word in wordList:
        for j in range(len(word)):
        pattern = word[:j] + "" + word[j + 1:]
        nei[pattern].append(word)
        visit = {beginWord}
        q = deque([beginWord])
        res = 1
        while q:
        for i in range(len(q)):
        word = q.popleft()
        if word == endWord:
        return res
        for j in range(len(word)):
        pattern = word[:j] + "" + word[j + 1:]
        for neiWord in nei[pattern]:
        if neiWord not in visit:
        visit.add(neiWord)
        q.append(neiWord)
        res += 1
        return 0
        """
    },
    # ==================== Advanced ====================
    "lc_70_climbing_stairs": {
        "problem_id": "lc_70",
        "problem_title": "Climbing Stairs",
        "patterns": ["dynamic_programming"],
        "difficulty": "Easy",
        "tags": ["dynamic_programming", "memoization"],
        "problem_description": "You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?",
        "canonical_solution": """
        def climbStairs(n: int) -> int:
        one, two = 1, 1
        for i in range(n - 1):
        temp = one
        one = one + two
        two = temp
        return one
        """
    },
    "lc_322_coin_change": {
        "problem_id": "lc_322",
        "problem_title": "Coin Change",
        "patterns": ["dynamic_programming"],
        "difficulty": "Medium",
        "tags": ["dynamic_programming", "array"],
        "problem_description": "You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.",
        "canonical_solution": """
        def coinChange(coins: list[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for a in range(1, amount + 1):
        for c in coins:
        if a - c >= 0:
        dp[a] = min(dp[a], 1 + dp[a - c])
        return dp[amount] if dp[amount] != amount + 1 else -1
        """
    },
    "lc_78_subsets": {
        "problem_id": "lc_78",
        "problem_title": "Subsets",
        "patterns": ["backtracking"],
        "difficulty": "Medium",
        "tags": ["backtracking", "recursion", "array"],
        "problem_description": "Given an integer array nums of unique elements, return all possible subsets (the power set).",
        "canonical_solution": """
        def subsets(nums: list[int]) -> list[list[int]]:
        res = []
        subset = []
        def dfs(i):
        if i >= len(nums):
        res.append(subset.copy())
        return
        # decision to include nums[i]
        subset.append(nums[i])
        dfs(i + 1)
        # decision NOT to include nums[i]
        subset.pop()
        dfs(i + 1)
        dfs(0)
        return res
        """
    },
    "lc_39_combination_sum": {
        "problem_id": "lc_39",
        "problem_title": "Combination Sum",
        "patterns": ["backtracking"],
        "difficulty": "Medium",
        "tags": ["backtracking", "recursion", "array"],
        "problem_description": "Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target.",
        "canonical_solution": """
        def combinationSum(candidates: list[int], target: int) -> list[list[int]]:
        res = []
        def dfs(i, cur, total):
        if total == target:
        res.append(cur.copy())
        return
        if i >= len(candidates) or total > target:
        return
        cur.append(candidates[i])
        dfs(i, cur, total + candidates[i])
        cur.pop()
        dfs(i + 1, cur, total)
        dfs(0, [], 0)
        return res
        """
    },
    "lc_55_jump_game": {
        "problem_id": "lc_55",
        "problem_title": "Jump Game",
        "patterns": ["greedy"],
        "difficulty": "Medium",
        "tags": ["greedy", "array"],
        "problem_description": "You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position. Return true if you can reach the last index, or false otherwise.",
        "canonical_solution": """
        def canJump(nums: list[int]) -> bool:
        goal = len(nums) - 1
        for i in range(len(nums) - 2, -1, -1):
        if i + nums[i] >= goal:
        goal = i
        return goal == 0
        """
        },
    "lc_134_gas_station": {
        "problem_id": "lc_134",
        "problem_title": "Gas Station",
        "patterns": ["greedy"],
        "difficulty": "Medium",
        "tags": ["greedy", "array"],
        "problem_description": "There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i]. You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next station (i + 1). You begin the journey with an empty tank at one of the gas stations. Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1.",
        "canonical_solution": """
        def canCompleteCircuit(gas: list[int], cost: list[int]) -> int:
        if sum(gas) < sum(cost):
        return -1
        total = 0
        res = 0
        for i in range(len(gas)):
        total += gas[i] - cost[i]
        if total < 0:
        total = 0
        res = i + 1
        return res
        """
    },
    "lc_347_top_k_frequent": {
        "problem_id": "lc_347",
        "problem_title": "Top K Frequent Elements",
        "patterns": ["heap", "hash_map"],
        "difficulty": "Medium",
        "tags": ["array", "heap", "hash_map", "bucket_sort"],
        "problem_description": "Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.",
        "canonical_solution": """
        def topKFrequent(nums: list[int], k: int) -> list[int]:
        count = {}
        freq = [[] for i in range(len(nums) + 1)]
        for n in nums:
        count[n] = 1 + count.get(n, 0)
        for n, c in count.items():
        freq[c].append(n)
        res = []
        for i in range(len(freq) - 1, 0, -1):
        for n in freq[i]:
        res.append(n)
        if len(res) == k:
        return res
        """
    },
    "lc_23_merge_k_sorted_lists": {
        "problem_id": "lc_23",
        "problem_title": "Merge K Sorted Lists",
        "patterns": ["heap"],
        "difficulty": "Hard",
        "tags": ["linked_list", "heap", "divide_and_conquer"],
        "problem_description": "You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.",
        "canonical_solution": """
        import heapq

        Definition for singly-linked list.
        class ListNode:
        def init(self, val=0, next=None):
        self.val = val
        self.next = next
        def mergeKLists(lists: list) -> list:
        if not lists:
        return None

        heap = []
        for i, l in enumerate(lists):
            if l:
                # Add a unique identifier (index i) to handle nodes with same values
                heapq.heappush(heap, (l.val, i, l))
                
        dummy = ListNode(0)
        curr = dummy

        while heap:
            val, i, node = heapq.heappop(heap)
            curr.next = node
            curr = curr.next
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
                
        return dummy.next
        """
    },
    "lc_323_connected_components": {
        "problem_id": "lc_323",
        "problem_title": "Number of Connected Components in an Undirected Graph",
        "patterns": ["union_find", "graph_dfs", "graph_bfs"],
        "difficulty": "Medium",
        "tags": ["graph", "union_find", "dfs", "bfs"],
        "problem_description": "You have a graph of n nodes. You are given an integer n and an array edges where edges[i] = [ai, bi] indicates that there is an edge between ai and bi in the graph. Return the number of connected components in the graph.",
        "canonical_solution": """
        def countComponents(n: int, edges: list[list[int]]) -> int:
        par = [i for i in range(n)]
        rank = [1] * n

        def find(n1):
            res = n1
            while res != par[res]:
                par[res] = par[par[res]]
                res = par[res]
            return res
            
        def union(n1, n2):
            p1, p2 = find(n1), find(n2)
            if p1 == p2:
                return 0
            if rank[p2] > rank[p1]:
                par[p1] = p2
                rank[p2] += rank[p1]
            else:
                par[p2] = p1
                rank[p1] += rank[p2]
            return 1
            
        res = n
        for n1, n2 in edges:
            res -= union(n1, n2)
        return res
        """
    },
    "lc_684_redundant_connection": {
        "problem_id": "lc_684",
        "problem_title": "Redundant Connection",
        "patterns": ["union_find"],
        "difficulty": "Medium",
        "tags": ["graph", "union_find"],
        "problem_description": "In this problem, a tree is an undirected graph that is connected and has no cycles. You are given a graph that started as a tree with n nodes labeled from 1 to n, with one additional edge added. The added edge has two different vertices chosen from 1 to n, and was not an edge that already existed. The graph is represented as an array edges of length n where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the graph. Return an edge that can be removed so that the resulting graph is a tree of n nodes.",
        "canonical_solution": """
        def findRedundantConnection(edges: list[list[int]]) -> list[int]:
        par = [i for i in range(len(edges) + 1)]
        rank = [1] * (len(edges) + 1)
        def find(n):
        p = par[n]
        while p != par[p]:
        par[p] = par[par[p]]
        p = par[p]
        return p
        # return False if already unioned
        def union(n1, n2):
        p1, p2 = find(n1), find(n2)
        if p1 == p2:
        return False
        if rank[p1] > rank[p2]:
        par[p2] = p1
        rank[p1] += rank[p2]
        else:
        par[p1] = p2
        rank[p2] += rank[p1]
        return True
        for n1, n2 in edges:
        if not union(n1, n2):
        return [n1, n2]
        """
    },
    "lc_208_implement_trie": {
        "problem_id": "lc_208",
        "problem_title": "Implement Trie (Prefix Tree)",
        "patterns": ["trie"],
        "difficulty": "Medium",
        "tags": ["trie", "string", "design"],
        "problem_description": "A trie (pronounced as 'try') or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. Implement the Trie class.",
        "canonical_solution": """
        class TrieNode:
        def init(self):
        self.children = {}
        self.endOfWord = False

        class Trie:
        def init(self):
        self.root = TrieNode()
        def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
        if c not in cur.children:
        cur.children[c] = TrieNode()
        cur = cur.children[c]
        cur.endOfWord = True
        def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
        if c not in cur.children:
        return False
        cur = cur.children[c]
        return cur.endOfWord
        def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
        if c not in cur.children:
        return False
        cur = cur.children[c]
        return True
        """
    },
    "lc_212_word_search_ii": {
        "problem_id": "lc_212",
        "problem_title": "Word Search II",
        "patterns": ["trie", "backtracking"],
        "difficulty": "Hard",
        "tags": ["trie", "backtracking", "matrix", "graph"],
        "problem_description": "Given an m x n board of characters and a list of strings words, return all words on the board. Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.",
        "canonical_solution": """
        class TrieNode:
        def init(self):
        self.children = {}
        self.isWord = False
        def addWord(self, word):
        cur = self
        for c in word:
        if c not in cur.children:
        cur.children[c] = TrieNode()
        cur = cur.children[c]
        cur.isWord = True

        def findWords(board: list[list[str]], words: list[str]) -> list[str]:
        root = TrieNode()
        for w in words:
        root.addWord(w)
        ROWS, COLS = len(board), len(board[0])
        res, visit = set(), set()
        def dfs(r, c, node, word):
        if (
        r < 0
        or c < 0
        or r == ROWS
        or c == COLS
        or (r, c) in visit
        or board[r][c] not in node.children
        ):
        return

            visit.add((r, c))
            node = node.children[board[r][c]]
            word += board[r][c]
            if node.isWord:
                res.add(word)
            dfs(r + 1, c, node, word)
            dfs(r - 1, c, node, word)
            dfs(r, c + 1, node, word)
            dfs(r, c - 1, node, word)
            visit.remove((r, c))

        for r in range(ROWS):
            for c in range(COLS):
                dfs(r, c, root, "")
        return list(res)
        """
    },
}

    
def create_retrieval_chunks(knowledge_base: dict) -> list[dict]:
    """
    Converts the problem knowledge base into a list of chunks for embedding.

    Each chunk contains the text to be embedded and metadata to reference
    the full problem details.

    Args:
        knowledge_base: A dictionary containing the problem packets.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk.
    """
    chunks = []
    for problem_key, problem_data in knowledge_base.items():
        # 1. Construct the text for embedding (The "Retrieval Chunk")
        # This combines the most important semantic information into one string.
        retrieval_text = (
            f"problem_title: {problem_data['problem_title']}\n"
            f"problem_description: {problem_data['problem_description']}\n"
            f"tags: {', '.join(problem_data['tags'])}"
        )

        # 2. Prepare the metadata
        # This holds all other information for later use (the "Payload" reference).
        # We store the problem_key to easily look up the full packet later.
        metadata = {
            "problem_key": problem_key,
            "problem_id": problem_data["problem_id"],
            "patterns": problem_data["patterns"],
            "difficulty": problem_data["difficulty"]
        }

        # 3. Append the final chunk to our list
        chunks.append({
            "text": retrieval_text,
            "metadata": metadata
        })

    return chunks
