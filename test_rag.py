"""
Test RAG system components
"""

import sys
import os
sys.path.append('src')

# Test imports
try:
    from document_processor import SimpleDocumentProcessor
    from embeddings import OpenAIEmbeddingSystem
    from vector_db import MilvusVectorDB
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test document processor
print("\n🧪 Testing Document Processor...")
processor = SimpleDocumentProcessor()

# Create test document if it doesn't exist
test_content = """RAG (Retrieval-Augmented Generation) is a powerful AI technique.

It combines information retrieval with text generation.

Key components include:
1. Document processing
2. Embedding generation  
3. Vector storage
4. Language model integration

RAG provides up-to-date information without retraining models."""

os.makedirs('data/documents', exist_ok=True)
with open('data/documents/test_rag.txt', 'w') as f:
    f.write(test_content)

print("✅ Created test document")

# Process document
chunks = processor.process_text_file('data/documents/test_rag.txt')
print(f"✅ Created {len(chunks)} chunks")

# Test embeddings
print("\n🧪 Testing Embeddings...")
embedder = OpenAIEmbeddingSystem()

# Test with first chunk
if chunks:
    test_text = chunks[0]['text']
    embedding = embedder.generate_single_embedding(test_text)
    if embedding:
        print(f"✅ Generated embedding: {len(embedding)} dimensions")
    else:
        print("❌ Embedding generation failed")

# Test vector database
print("\n🧪 Testing Vector Database...")
vector_db = MilvusVectorDB()
vector_db.create_collection()

# Test storing documents
if chunks and embedding:
    embeddings = [embedding]  # Just test with one
    vector_db.insert_documents([chunks[0]], embeddings)
    print("✅ Stored test document")
    
    # Test search
    results = vector_db.search_similar(embedding, top_k=1)
    if results:
        print(f"✅ Search returned {len(results)} results")
    else:
        print("⚠️ Search returned no results")

print("\n🎉 RAG component testing completed!")
