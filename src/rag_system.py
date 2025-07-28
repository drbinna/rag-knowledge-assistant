"""
Complete RAG system combining all components
"""

from document_processor import SimpleDocumentProcessor
from embeddings import OpenAIEmbeddingSystem
from vector_db import MilvusVectorDB
import openai
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self):
        """Initialize complete RAG system"""
        print("🚀 Initializing RAG System...")
        
        # Initialize components
        self.doc_processor = SimpleDocumentProcessor()
        self.embedder = OpenAIEmbeddingSystem()
        self.vector_db = MilvusVectorDB()
        
        # Initialize OpenAI client for chat
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create vector collection
        self.vector_db.create_collection()
        
        print("✅ RAG System initialized!")
    
    def add_document(self, file_path: str):
        """Add a document to the RAG system"""
        print(f"📄 Adding document: {file_path}")
        
        # Process document
        chunks = self.doc_processor.process_pdf(file_path)
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.generate_embeddings(texts)
        
        # Store in vector database
        self.vector_db.insert_documents(chunks, embeddings)
        
        print(f"✅ Document added: {len(chunks)} chunks processed")
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Query the RAG system"""
        print(f"🔍 Querying: {question}")
        
        # Generate query embedding
        query_embedding = self.embedder.generate_single_embedding(question)
        
        # Search for relevant chunks
        relevant_chunks = self.vector_db.search_similar(query_embedding, top_k)
        
        if not relevant_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Create context from relevant chunks
        context = "\n\n".join([
            f"Source: {chunk['source']}\nContent: {chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        # Generate answer using OpenAI
        prompt = f"""Based on the following context, answer the question. If you can't find the answer in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            print(f"✅ Answer generated")
            return answer
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

# Test the complete RAG system
if __name__ == "__main__":
    print("🧪 Testing Complete RAG System...")
    
    rag = RAGSystem()
    
    print("✅ RAG System ready!")
    print("💡 Use rag.add_document('path/to/document.pdf') to add documents")
    print("💡 Use rag.query('your question') to ask questions")