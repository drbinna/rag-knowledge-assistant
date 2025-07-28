"""
Vector database for RAG system with real similarity search
No Milvus dependencies - pure NumPy implementation
"""

import numpy as np
from typing import List, Dict
import json
import pickle
from pathlib import Path

class MilvusVectorDB:
    def __init__(self, storage_path="data/vector_storage.pkl"):
        """Initialize vector database with real similarity search"""
        self.storage_path = storage_path
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        
        # Load existing data if available
        self.load_from_disk()
        
        print("✅ Vector database initialized with real similarity search")
    
    def create_collection(self, embedding_dim=1536):
        """Create/initialize the vector storage"""
        print("✅ Vector database ready for document storage")
        return True
    
    def insert_documents(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Insert documents and embeddings with real vector storage"""
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch")
        
        try:
            # Convert embeddings to numpy arrays for efficient computation
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Store data
            self.chunks.extend(chunks)
            self.embeddings.extend(embeddings_array)
            self.metadata.extend([
                {
                    'chunk_id': chunk['chunk_id'],
                    'source': chunk['source'],
                    'char_start': chunk.get('char_start', 0),
                    'char_end': chunk.get('char_end', 0)
                }
                for chunk in chunks
            ])
            
            # Save to disk for persistence
            self.save_to_disk()
            
            print(f"✅ Stored {len(chunks)} documents with real embeddings")
            print(f"📊 Total documents in database: {len(self.chunks)}")
            
        except Exception as e:
            print(f"❌ Storage insertion failed: {e}")
    
    def search_similar(self, query_embedding: List[float], top_k=5, similarity_threshold=0.1) -> List[Dict]:
        """Search for similar documents using real cosine similarity"""
        
        if not self.embeddings:
            print("⚠️ No documents in database")
            return []
        
        try:
            # Convert query to numpy array
            query_vector = np.array(query_embedding, dtype=np.float32)
            stored_embeddings = np.array(self.embeddings, dtype=np.float32)
            
            # Calculate cosine similarity
            # Normalize vectors for cosine similarity
            query_norm = query_vector / np.linalg.norm(query_vector)
            stored_norms = stored_embeddings / np.linalg.norm(stored_embeddings, axis=1, keepdims=True)
            
            # Compute similarities (dot product of normalized vectors = cosine similarity)
            similarities = np.dot(stored_norms, query_norm)
            
            # Get top k results sorted by similarity
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter by similarity threshold and build results
            results = []
            for idx in top_indices:
                similarity_score = float(similarities[idx])
                
                if similarity_score >= similarity_threshold:
                    results.append({
                        'text': self.chunks[idx]['text'],
                        'source': self.chunks[idx]['source'],
                        'score': similarity_score,
                        'chunk_id': self.chunks[idx]['chunk_id'],
                        'metadata': self.metadata[idx]
                    })
            
            print(f"🔍 Found {len(results)} relevant documents (threshold: {similarity_threshold})")
            for i, result in enumerate(results[:3]):
                print(f"   {i+1}. Score: {result['score']:.3f} - {Path(result['source']).name}")
            
            return results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def save_to_disk(self):
        """Save vector database to disk for persistence"""
        try:
            storage_dir = Path(self.storage_path).parent
            storage_dir.mkdir(exist_ok=True)
            
            data = {
                'chunks': self.chunks,
                'embeddings': [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in self.embeddings],
                'metadata': self.metadata
            }
            
            with open(self.storage_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"💾 Database saved to {self.storage_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save to disk: {e}")
    
    def load_from_disk(self):
        """Load vector database from disk"""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.chunks = data.get('chunks', [])
                self.embeddings = data.get('embeddings', [])
                self.metadata = data.get('metadata', [])
                
                print(f"📂 Loaded {len(self.chunks)} documents from disk")
            else:
                print("📂 Starting with empty database")
                
        except Exception as e:
            print(f"⚠️ Failed to load from disk: {e}")
            self.chunks = []
            self.embeddings = []
            self.metadata = []
    
    def get_status(self) -> Dict:
        """Get database status"""
        return {
            'total_chunks': len(self.chunks),
            'total_embeddings': len(self.embeddings),
            'sources': list(set([chunk['source'] for chunk in self.chunks])) if self.chunks else [],
            'storage_path': self.storage_path
        }
    
    def clear_database(self):
        """Clear all data from database"""
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        
        # Remove storage file
        if Path(self.storage_path).exists():
            Path(self.storage_path).unlink()
        
        print("🗑️ Database cleared")

# Test the vector database
if __name__ == "__main__":
    print("🧪 Testing Vector Database...")
    
    vector_db = MilvusVectorDB()
    vector_db.create_collection()
    
    status = vector_db.get_status()
    print(f"📊 Status: {status['total_chunks']} chunks loaded")
    print("✅ Vector database ready with real similarity search!")