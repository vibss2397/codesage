# CodeSage: Multi-Agent LeetCode Coaching System

CodeSage is a multi-agent AI system that provides educational coaching for LeetCode solutions. Instead of giving direct answers, it analyzes code through multiple specialized agents and guides users toward optimization discoveries through progressive Socratic questioning.

## 🎯 Core Concept

CodeSage transforms technical analysis into educational experiences by:
- **Multi-perspective analysis**: 3 specialized agents examine code from different angles
- **Educational guidance**: Progressive hints that guide discovery, not direct solutions
- **Real code execution**: Tests user solutions against comprehensive test cases
- **Smart prioritization**: Focuses on most impactful improvements first

## 🏗️ System Architecture

```
Input (Problem + User Code) → Orchestrator Agent → Educational Coaching
                                      ↓
    [Analyze Agent] → [QA Agent] → [CodeExecutor] → [Synthesis Agent]
                          ↓
         RAG Knowledge Base (10 Algorithmic Patterns) + BaseAgent Infrastructure
```

## 🤖 Agent Specifications

### **AnalyzeAgent** 🔍
**Purpose:** Technical algorithmic analysis
- Detects time/space complexity (O(n²), O(n), etc.)
- Identifies current algorithmic approach
- Suggests optimal patterns using RAG + intelligent fallback
- Generates edge cases for testing

### **QAAgent** 🧹  
**Purpose:** Code quality and test generation
- Analyzes code quality (naming, structure, documentation)
- Generates comprehensive test cases with expected outputs
- Provides actionable improvement suggestions
- Validates problem assumptions and constraints

### **CodeExecutor** ⚡
**Purpose:** Safe code execution and validation
- Executes user code against generated test cases
- Isolated environment with limited built-ins
- Structured pass/fail results for each test
- Graceful error handling for crashes/exceptions

### **SynthesisAgent** 🎯
**Purpose:** Educational coaching synthesis
- **Priority system**: Correctness > Algorithmic > Quality
- Generates progressive Socratic hints for discovery learning
- Combines all analysis into focused coaching guidance
- Suggests learning paths and practice problems

## 🧠 Technical Stack

- **LLM**: Google Gemini 2.5-Flash (upgraded from 1.5 for better reasoning)
- **RAG**: FAISS vector store with Gemini embeddings
- **Knowledge Base**: 10 core algorithmic patterns (56 searchable chunks)
- **Architecture**: BaseAgent inheritance pattern for shared infrastructure
- **Execution**: Safe code sandbox with isolated namespaces
- **Frontend**: Streamlit + FastAPI + shared DB (planned)

## 📚 Algorithmic Patterns Database

**Core Patterns (10 total):**
- **Array & String**: two_pointers, sliding_window, hash_map
- **Search**: binary_search
- **Tree**: tree_dfs, tree_bfs  
- **Graph**: graph_dfs, graph_bfs
- **Advanced**: dynamic_programming, backtracking, greedy, heap, union_find, trie

Each pattern includes:
- Problem recognition criteria
- Complexity information
- Code templates and examples
- Progressive learning hints

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Google Gemini API Key

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repo-url>
   cd codesage
   pip install -r requirements.txt
   ```

2. **Environment configuration:**
   ```bash
   # Create .env file
   GEMINI_API_KEY="your-gemini-api-key"
   ```

### Basic Usage
<fill in later>

## Improvements:
- rag system can be improved, currently a lot of cases override
- code executor can be improved to make the system more robust
- prompt can generate a full weird response like hindi in the middle

## 🎪 Demo Examples

### Example 1: Two Sum Optimization
**Input:** O(n²) nested loop solution
**Process:** 
- AnalyzeAgent identifies complexity gap
- QAAgent generates edge case tests  
- CodeExecutor validates correctness
- SynthesisAgent provides hash_map discovery hints

**Output:** Progressive coaching toward O(n) optimization

### Example 2: Working but Unoptimized Code
**Input:** Correct palindrome check with O(n) space
**Process:**
- All tests pass ✅
- Complexity suboptimal ⚠️  
- SynthesisAgent focuses on space optimization
