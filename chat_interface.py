"""
RAG Knowledge Assistant - Main Chat Interface
Integrates document processing, embeddings, vector search, and OpenAI API
"""

import sys
import os
sys.path.append('src')

from document_processor import SimpleDocumentProcessor
from embeddings import OpenAIEmbeddingSystem
from vector_db import MilvusVectorDB
import openai
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict

load_dotenv()

class RAGKnowledgeAssistant:
    def __init__(self):
        """Initialize the RAG system with all components"""
        print("🚀 Initializing RAG Knowledge Assistant...")
        
        # Initialize components
        self.document_processor = SimpleDocumentProcessor()
        self.embeddings = OpenAIEmbeddingSystem()
        self.vector_db = MilvusVectorDB()
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create vector database collection
        self.vector_db.create_collection()
        
        print("🎉 All components initialized!")
    
    def ingest_documents(self, documents_folder: str = "data/documents"):
        """Process and ingest all documents from a folder"""
        print(f"📂 Ingesting documents from {documents_folder}...")
        
        documents_path = Path(documents_folder)
        if not documents_path.exists():
            print(f"❌ Documents folder not found: {documents_folder}")
            return
        
        all_chunks = []
        
        # Process all PDF and text files
        for file_path in documents_path.iterdir():
            if file_path.suffix.lower() == '.pdf':
                chunks = self.document_processor.process_pdf(str(file_path))
                all_chunks.extend(chunks)
            elif file_path.suffix.lower() == '.txt':
                chunks = self.document_processor.process_text_file(str(file_path))
                all_chunks.extend(chunks)
        
        if not all_chunks:
            print("❌ No documents found to process")
            return
        
        print(f"📄 Processing {len(all_chunks)} chunks...")
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.embeddings.generate_embeddings(chunk_texts)
        
        if not embeddings:
            print("❌ Failed to generate embeddings")
            return
        
        # Store in vector database
        self.vector_db.insert_documents(all_chunks, embeddings)
        print(f"✅ Successfully ingested {len(all_chunks)} document chunks!")
    
    def search_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search the knowledge base for relevant information"""
        print(f"🔍 Searching for: '{query}'")
        
        # Generate embedding for the query
        query_embedding = self.embeddings.generate_single_embedding(query)
        if not query_embedding:
            print("❌ Failed to generate query embedding")
            return []
        
        # Search vector database
        results = self.vector_db.search_similar(query_embedding, top_k=top_k)
        
        if results:
            print(f"✅ Found {len(results)} relevant documents")
        else:
            print("⚠️  No relevant documents found")
        
        return results
    
    def generate_enhanced_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer with improved prompting"""
        
        if not context_docs:
            return "⚠️ I couldn't find relevant information in your documents to answer this question."
        
        # Build context
        context = "\n\n".join([
            f"Source {i+1} ({doc['source']}): {doc['text']}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Much more permissive prompt
        system_prompt = """You are a helpful assistant that answers questions using provided context. 

Instructions:
- Use the context to answer the user's question thoroughly
- Extract and synthesize relevant information from all sources
- Provide detailed explanations when the context supports it
- Only say information is missing if absolutely nothing relevant is found
- Be comprehensive and helpful"""

        user_prompt = f"""Context: {context}

Question: {query}

Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Error generating response: {e}"
    
    def chat(self, query: str) -> str:
        """Main chat function - search and generate answer"""
        print(f"\n💬 User Query: {query}")
        
        # Search for relevant context
        context_docs = self.search_knowledge_base(query, top_k=3)
        
        # Generate answer
        answer = self.generate_enhanced_answer(query, context_docs)
        
        return answer
    
    def interactive_chat(self):
        """Start interactive chat session"""
        print("\n🤖 RAG Knowledge Assistant Ready!")
        print("💡 Type 'quit' to exit, 'ingest' to process documents")
        print("-" * 50)
        
        while True:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif query.lower() == 'ingest':
                self.ingest_documents()
                continue
            elif not query:
                continue
            
            # Get and display answer
            answer = self.chat(query)
            print(f"\n🤖 Assistant: {answer}")

def main():
    """Main function to run the RAG system"""
    
    # Initialize the system
    rag_assistant = RAGKnowledgeAssistant()
    
    # Check if documents need to be ingested
    print("\n" + "="*60)
    print("📚 DOCUMENT INGESTION")
    print("="*60)
    
    choice = input("Do you want to ingest documents now? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        rag_assistant.ingest_documents()
    
    # Start interactive chat
    print("\n" + "="*60)
    print("💬 INTERACTIVE CHAT")
    print("="*60)
    
    rag_assistant.interactive_chat()

if __name__ == "__main__":
    main()