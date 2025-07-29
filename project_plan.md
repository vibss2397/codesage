# Multi-Agent LeetCode Coaching System - Project Plan

## 🎯 **Project Overview**
Build an AI coaching system where 3 specialized agents analyze user's LeetCode solutions and provide educational guidance. System takes problem description + user's code, analyzes it through multiple lenses, and provides progressive hints rather than direct answers.

## 🏗️ **System Architecture**
```
Input (Problem + User Code) → Agent Orchestrator → Educational Response
                                      ↓
    [Analyze Agent] [Quality Agent] [Strategy Coach Agent]
                                      ↓
RAG Knowledge Base (Patterns, Optimizations, Edge Cases) + Code Execution Tools
```

## 🤖 **Agent Specifications**

### **1. Analyze Agent** 🔍
**Purpose:** Technical analysis of solution
- **Current complexity:** Detect O(n²), O(n), etc. from code structure
- **Pattern detection:** Identify user's current approach (brute force, hash map, etc.)
- **Optimal pattern:** Determine best pattern for this problem type
- **Edge cases:** List boundary conditions based on problem (empty array, single element, etc.)

### **2. Quality Agent** 🧹
**Purpose:** Code quality and style review
- **Style analysis:** Variable naming, formatting, comments
- **Readability:** Function structure, logic clarity
- **Organization:** Code structure and best practices
- **Maintainability:** Suggestions for cleaner code

### **3. Strategy Coach Agent** 🎯
**Purpose:** Educational guidance and testing
- **Progressive hints:** Based on Analyze Agent findings, provide Socratic guidance
- **Quality coaching:** Based on Quality Agent findings, suggest improvements
- **Edge case testing:** Execute user code against edge cases using local sandbox
- **Learning path:** Suggest next problems and patterns to practice

## 🔧 **Technical Stack**
- **LLM:** Google Gemini Pro (free tier, high rate limits)
- **Orchestration:** Custom sequential agent execution
- **RAG:** Local FAISS vector store with sentence-transformers embeddings
- **Knowledge Base:** 8-10 core LeetCode patterns (Two Pointers, Hash Map, Sliding Window, etc.)
- **Code Execution:** Local subprocess sandbox for edge case testing
- **Frontend:** Streamlit web interface
- **Languages:** Python, supports analysis of Python/Java/JavaScript solutions

## 📚 **RAG Knowledge Base Structure**
```json
filling in later
```

## 🔄 **System Workflow**
1. **Input:** User provides LeetCode problem description + their solution code
2. **Analysis:** Analyze Agent determines complexity, patterns, optimal approach
3. **Quality Review:** Quality Agent reviews code style and structure
4. **Coaching:** Strategy Coach synthesizes findings into educational hints
5. **Testing:** Strategy Coach runs code against edge cases and reports failures
6. **Output:** Comprehensive coaching report with progressive hints and specific improvements

## 🎨 **User Interface Features**
- **Input forms:** Problem description + code editor
- **Real-time progress:** Agent execution status indicators
- **Expandable results:** Each agent's analysis in separate sections
- **Progressive hints:** Click-to-reveal hint system
- **Code testing:** Live edge case execution results
- **Pattern learning:** Identification of problem patterns and next steps

## ⚡ **Implementation Priority**
### **Core MVP (Weekend Goal):**
1. **RAG Setup:** Vector store with 6 core patterns
2. **Agent Framework:** Basic orchestrator + 3 specialized agents
3. **Code Analysis:** Complexity detection and pattern matching
4. **Edge Case Testing:** Local sandbox execution with common test cases
5. **Web Interface:** Basic Streamlit app with input/output
6. **Demo:** Working end-to-end flow on 3 different LeetCode problems

### **Advanced Features (Future):**
- Support for more programming languages
- Expanded pattern database
- Performance benchmarking
- Learning progress tracking
- Integration with actual LeetCode API

## 🎯 **Success Criteria**
- **Functional:** System analyzes solutions in 3+ programming languages
- **Educational:** Provides progressive hints rather than direct answers
- **Accurate:** Correctly identifies complexity and suggests appropriate patterns
- **Safe:** Code execution sandbox works reliably without security issues
- **Fast:** Complete analysis in under 30 seconds
- **Demo-Ready:** Clean interface showing agent collaboration

## 📁 **Project Structure**
```
codesage/
├── agents/
│   ├── analyze_agent.py
│   └── base_agent.py
├── prompts/
│   └── analyze_agent_prompt.py
├── tests/
│   ├── test_agent.py
│   ├── test_coding_patterns.py
│   └── test_rag_store.py
├── .gitignore
├── coding_patterns.py
├── project_plan.md
├── rag_store.py
├── requirements.txt
└── test_analyze_agent.py
```

## 💡 **Key Innovations**
- **Multi-agent collaboration** for comprehensive code review
- **Educational focus** on progressive hints vs direct answers
- **Real code execution** for concrete edge case feedback
- **Pattern-based learning** with clear progression paths
- **RAG-enhanced coaching** with relevant context retrieval

## 🎪 **Demo Script**
1. **Input:** Brute force Two Sum solution with nested loops
2. **Show:** Real-time agent analysis and collaboration
3. **Highlight:** Progressive hint system guiding toward hash map optimization
4. **Demonstrate:** Live edge case testing revealing code failures
5. **Result:** User learns optimization pattern through guided discovery

---

**This system combines advanced AI techniques (multi-agent, RAG, tool use) with practical educational value, creating a sophisticated yet immediately useful coding interview preparation tool.**