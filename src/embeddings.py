"""
OpenAI-based embedding system (no local model dependencies)
"""

import openai
import os
from typing import List
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class OpenAIEmbeddingSystem:
    def __init__(self):
        """Initialize OpenAI embedding system"""
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "text-embedding-ada-002"
        print("✅ OpenAI embedding system initialized")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        
        if not texts:
            return []
        
        print(f"🧠 Generating embeddings for {len(texts)} texts...")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [embedding.embedding for embedding in response.data]
            print(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            return []
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        result = self.generate_embeddings([text])
        return result[0] if result else []

# Test the embedding system
if __name__ == "__main__":
    embedder = OpenAIEmbeddingSystem()
    
    # Test with sample text
    test_text = "This is a test sentence for embedding generation."
    embedding = embedder.generate_single_embedding(test_text)
    
    if embedding:
        print(f"✅ Test embedding generated: {len(embedding)} dimensions")
    else:
        print("❌ Test failed - check your OpenAI API key in .env file")