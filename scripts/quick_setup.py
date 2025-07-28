#!/usr/bin/env python3
"""
Quick setup script for RAG Knowledge Assistant
Sets up environment, installs dependencies, and verifies configuration
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'data/documents', 'data/processed']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def install_dependencies():
    """Install required packages"""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = Path('.env')
    if not env_path.exists():
        print("⚠️  .env file not found. Creating template...")
        with open('.env', 'w') as f:
            f.write("OPENAI_API_KEY=your_api_key_here\n")
        print("📝 Please add your OpenAI API key to .env file")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
        if 'your_api_key_here' in content or not content.strip():
            print("⚠️  Please update .env file with your actual OpenAI API key")
            return False
    
    print("✅ Environment file configured")
    return True

def main():
    """Main setup function"""
    print("🚀 RAG Knowledge Assistant - Quick Setup")
    print("=" * 50)
    
    success = True
    success &= check_python_version()
    create_directories()
    success &= install_dependencies()
    success &= check_env_file()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add documents to data/documents/ folder")
        print("2. Run: python chat_interface.py")
    else:
        print("⚠️  Setup completed with warnings. Please check above.")

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Batch document processing script for RAG Knowledge Assistant
Processes all documents and updates vector database
"""

import sys
import os
sys.path.append('src')

from pathlib import Path
from document_processor import SimpleDocumentProcessor
from embeddings import OpenAIEmbeddingSystem
from vector_db import MilvusVectorDB
from dotenv import load_dotenv

load_dotenv()

def process_all_documents(documents_folder="data/documents", force_reprocess=False):
    """Process all documents in the specified folder"""
    
    print("📂 RAG Document Processor")
    print("=" * 40)
    
    # Initialize components
    processor = SimpleDocumentProcessor(chunk_size=400, overlap=50)
    embeddings = OpenAIEmbeddingSystem()
    vector_db = MilvusVectorDB()
    
    # Check for existing data
    if not force_reprocess:
        status = vector_db.get_status()
        if status['total_chunks'] > 0:
            print(f"📊 Found {status['total_chunks']} existing chunks")
            choice = input("Reprocess all documents? (y/n): ").strip().lower()
            if choice not in ['y', 'yes']:
                print("✅ Using existing processed documents")
                return
    
    # Clear existing data if reprocessing
    if force_reprocess:
        vector_db.clear_database()
        print("🗑️  Cleared existing vector database")
    
    # Process documents
    documents_path = Path(documents_folder)
    if not documents_path.exists():
        print(f"❌ Documents folder not found: {documents_folder}")
        return
    
    all_chunks = []
    processed_files = []
    
    # Process all supported files
    for file_path in documents_path.iterdir():
        if file_path.suffix.lower() in ['.pdf', '.txt']:
            try:
                if file_path.suffix.lower() == '.pdf':
                    chunks = processor.process_pdf(str(file_path))
                else:
                    chunks = processor.process_text_file(str(file_path))
                
                all_chunks.extend(chunks)
                processed_files.append(f"📄 {file_path.name}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
    
    if not all_chunks:
        print("❌ No documents found to process")
        return
    
    print(f"\n📊 Processing Summary:")
    for file_info in processed_files:
        print(f"  {file_info}")
    print(f"  📈 Total chunks: {len(all_chunks)}")
    
    # Generate embeddings
    print(f"\n🧠 Generating embeddings...")
    chunk_texts = [chunk['text'] for chunk in all_chunks]
    
    # Process in batches for large datasets
    batch_size = 50
    all_embeddings = []
    
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        batch_embeddings = embeddings.generate_embeddings(batch)
        
        if batch_embeddings:
            all_embeddings.extend(batch_embeddings)
            print(f"  ✅ Processed batch {i//batch_size + 1}/{(len(chunk_texts)-1)//batch_size + 1}")
        else:
            print(f"  ❌ Failed to generate embeddings for batch {i//batch_size + 1}")
            return
    
    # Store in vector database
    print(f"\n💾 Storing in vector database...")
    vector_db.insert_documents(all_chunks, all_embeddings)
    
    # Final status
    final_status = vector_db.get_status()
    print(f"\n🎉 Processing Complete!")
    print(f"📊 Total chunks in database: {final_status['total_chunks']}")
    print(f"📚 Sources: {len(final_status['sources'])}")
    
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process documents for RAG system')
    parser.add_argument('--folder', default='data/documents', 
                       help='Documents folder path (default: data/documents)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing of all documents')
    
    args = parser.parse_args()
    
    try:
        process_all_documents(args.folder, args.force)
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()