# Implementation Plan: AI Learning Developer Productivity (Success-Focused Prototype)

## Overview

This implementation plan creates a functional CLI tool that explains code files using local RAG architecture with Tree-sitter semantic parsing, ChromaDB vector storage, and FastAPI backend. The focus is on delivering a working prototype with context-aware explanations and source attribution.

## Tasks

- [ ] 1. Set up project structure and dependencies
  - Create Python project with poetry or pip requirements
  - Install core dependencies: FastAPI, Tree-sitter, ChromaDB, Click, OpenAI SDK
  - Set up project directory structure (cli/, api/, parser/, indexer/, retriever/, generator/)
  - Configure environment variables for API keys
  - _Requirements: 5.1, 5.2_

- [ ] 2. Implement Tree-sitter parser for semantic code chunking
  - [ ] 2.1 Set up Tree-sitter with language grammars
    - Download and compile Tree-sitter grammars for Python, JavaScript, TypeScript, Java, Go, Rust
    - Create TreeSitterParser class with language detection
    - _Requirements: 4.1, 4.5, 9.2_
  
  - [ ] 2.2 Implement semantic chunk extraction
    - Write functions to extract functions, classes, and methods from AST
    - Capture line numbers, parameters, return types, and docstrings
    - Identify dependencies between code elements
    - _Requirements: 2.2, 4.2, 4.3, 4.4_
  
  - [ ]* 2.3 Write property test for semantic chunking
    - **Property 11: Chunks align with semantic boundaries**
    - **Validates: Requirements 4.2**
  
  - [ ]* 2.4 Write property test for chunk metadata
    - **Property 12: Chunks preserve context and dependencies**
    - **Validates: Requirements 4.3, 4.4**

- [ ] 3. Implement ChromaDB integration for local vector storage
  - [ ] 3.1 Create VectorDBManager class
    - Initialize ChromaDB with local persistence
    - Create collections for code chunks and file metadata
    - Implement add, query, and delete operations
    - _Requirements: 5.2, 5.3, 5.4_
  
  - [ ] 3.2 Implement embedding generation
    - Integrate OpenAI embeddings API or local sentence-transformers
    - Generate embeddings for semantic chunks
    - Handle embedding errors gracefully
    - _Requirements: 6.3_
  
  - [ ]* 3.3 Write property test for local storage
    - **Property 13: Embeddings stored locally**
    - **Validates: Requirements 5.3**
  
  - [ ]* 3.4 Write property test for persistence
    - **Property 15: Vector store persists across sessions**
    - **Validates: Requirements 5.5**

- [ ] 4. Checkpoint - Ensure parsing and storage work
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement code indexer
  - [ ] 5.1 Create CodeIndexer class
    - Implement directory traversal for supported file types
    - Use Tree-sitter parser to extract chunks from each file
    - Generate embeddings and store in ChromaDB
    - Track file hashes to detect changes
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [ ] 5.2 Implement incremental indexing
    - Check file hashes before re-indexing
    - Skip unchanged files
    - Update only modified files
    - _Requirements: 6.5_
  
  - [ ]* 5.3 Write property test for indexing
    - **Property 16: All supported files indexed**
    - **Validates: Requirements 6.1**
  
  - [ ]* 5.4 Write property test for incremental indexing
    - **Property 19: Unchanged files skipped on re-index**
    - **Validates: Requirements 6.5**
  
  - [ ]* 5.5 Write unit tests for error handling
    - Test parse errors, unsupported languages, file not found
    - _Requirements: 10.1, 10.2_

- [ ] 6. Implement context retriever
  - [ ] 6.1 Create ContextRetriever class
    - Implement semantic similarity search using ChromaDB
    - Retrieve top-k relevant chunks for a query
    - Always include chunks from target file
    - Deduplicate and rank results
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [ ]* 6.2 Write property test for retrieval
    - **Property 20: Relevant chunks retrieved**
    - **Validates: Requirements 7.1**
  
  - [ ]* 6.3 Write property test for target file inclusion
    - **Property 24: Target file chunks always included**
    - **Validates: Requirements 7.5**

- [ ] 7. Implement explanation generator with source attribution
  - [ ] 7.1 Create ExplanationGenerator class
    - Build prompts with retrieved context and query
    - Include instructions for source attribution (cite line numbers)
    - Call LLM API (OpenAI GPT-4 or compatible)
    - Parse response to extract source attributions
    - _Requirements: 8.1, 8.2, 8.3_
  
  - [ ] 7.2 Implement source attribution parsing
    - Extract line number references from explanation text
    - Validate format "Line X" or "Lines X-Y"
    - Create SourceAttribution objects with file paths and snippets
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [ ] 7.3 Implement retry logic for LLM API
    - Retry failed requests with exponential backoff (3 attempts)
    - Handle rate limits and timeouts
    - Display clear error messages
    - _Requirements: 10.3, 10.4_
  
  - [ ]* 7.4 Write property test for source attribution
    - **Property 8: Line numbers included in explanations**
    - **Validates: Requirements 3.1, 3.4**
  
  - [ ]* 7.5 Write property test for context usage
    - **Property 25: Context and query in prompt**
    - **Validates: Requirements 8.1**
  
  - [ ]* 7.6 Write property test for retry logic
    - **Property 32: LLM failures trigger retry**
    - **Validates: Requirements 10.3**

- [ ] 8. Checkpoint - Ensure retrieval and generation work
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement FastAPI backend
  - [ ] 9.1 Create FastAPI application
    - Define /explain endpoint with ExplainRequest/Response models
    - Define /index endpoint with IndexRequest/Response models
    - Define /status endpoint for indexing statistics
    - Add error handling middleware
    - _Requirements: 1.1, 1.3_
  
  - [ ] 9.2 Wire components together
    - Connect parser, indexer, retriever, and generator
    - Handle file loading and validation
    - Return formatted explanations with source attributions
    - _Requirements: 2.1, 2.3, 2.5_
  
  - [ ]* 9.3 Write integration tests for API endpoints
    - Test /explain with various file types and queries
    - Test /index with sample codebase
    - Test error responses
    - _Requirements: 1.1, 1.3, 1.5_

- [ ] 10. Implement CLI interface
  - [ ] 10.1 Create Click CLI application
    - Implement 'explain' command with file path and --query option
    - Implement 'index' command with --force option
    - Implement 'status' command for index statistics
    - Add --verbose flag for detailed output
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [ ] 10.2 Implement output formatting
    - Use Rich library for terminal formatting
    - Add syntax highlighting for code snippets
    - Format source attributions clearly
    - Display errors in user-friendly format
    - _Requirements: 1.4, 1.5_
  
  - [ ] 10.3 Connect CLI to FastAPI backend
    - Make HTTP requests to local FastAPI server
    - Handle API errors and display to user
    - Stream responses for better UX
    - _Requirements: 1.1, 1.3_
  
  - [ ]* 10.4 Write unit tests for CLI
    - Test argument parsing
    - Test error messages for invalid inputs
    - Test output formatting
    - _Requirements: 1.2, 1.5_

- [ ] 11. Add configuration and environment setup
  - [ ] 11.1 Create configuration file support
    - Support .env file for API keys
    - Support config file for default settings (top-k, model, etc.)
    - Validate configuration on startup
    - _Requirements: 5.1_
  
  - [ ] 11.2 Add setup script
    - Script to download and compile Tree-sitter grammars
    - Script to initialize ChromaDB
    - Script to test API key configuration
    - _Requirements: 4.1, 5.2_

- [ ] 12. Implement comprehensive error handling
  - [ ] 12.1 Add error handlers for all error categories
    - Parse errors with line numbers
    - File not found errors
    - Unsupported language errors
    - Vector DB errors with troubleshooting
    - LLM API errors with retry logic
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  
  - [ ]* 12.2 Write unit tests for error handling
    - Test each error category
    - Test retry logic
    - Test error message formatting
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 13. Create documentation and examples
  - [ ] 13.1 Write README with setup instructions
    - Installation steps
    - Configuration guide
    - Usage examples
    - Supported languages
  
  - [ ] 13.2 Create example codebase for testing
    - Sample files in each supported language
    - Example queries and expected outputs
    - Test cases for edge cases

- [ ] 14. Final checkpoint - End-to-end testing
  - Test complete workflow: index → explain → display
  - Test with real codebases in each supported language
  - Verify source attribution accuracy
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Focus on getting a working prototype quickly, then iterate
