#!/usr/bin/env python3
"""
Setup verification script for RAG Knowledge Assistant
Verifies all components are working correctly
"""

import sys
import os
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.append('src')

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility"""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        return True, f"✅ Python {current_version[0]}.{current_version[1]} (compatible)"
    else:
        return False, f"❌ Python {current_version[0]}.{current_version[1]} (requires 3.8+)"

def check_required_packages() -> Tuple[bool, List[str]]:
    """Check if all required packages are installed"""
    required_packages = [
        'openai',
        'numpy', 
        'dotenv',
        'pathlib'
    ]
    
    results = []
    all_good = True
    
    for package in required_packages:
        try:
            if package == 'dotenv':
                importlib.import_module('dotenv')
            else:
                importlib.import_module(package)
            results.append(f"✅ {package}")
        except ImportError:
            results.append(f"❌ {package} (missing)")
            all_good = False
    
    return all_good, results

def check_project_structure() -> Tuple[bool, List[str]]:
    """Check if project structure is correct"""
    required_structure = [
        'src/',
        'src/document_processor.py',
        'src/embeddings.py', 
        'src/vector_db.py',
        'data/',
        'data/documents/',
        'chat_interface.py',
        'requirements.txt',
        '.env'
    ]
    
    results = []
    all_good = True
    
    for item in required_structure:
        path = Path(item)
        if path.exists():
            results.append(f"✅ {item}")
        else:
            results.append(f"❌ {item} (missing)")
            all_good = False
    
    return all_good, results

def check_environment_variables() -> Tuple[bool, List[str]]:
    """Check environment variables"""
    from dotenv import load_dotenv
    load_dotenv()
    
    results = []
    all_good = True
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key and openai_key != 'your_api_key_here':
        results.append("✅ OPENAI_API_KEY (configured)")
    else:
        results.append("❌ OPENAI_API_KEY (missing or placeholder)")
        all_good = False
    
    return all_good, results

def check_src_modules() -> Tuple[bool, List[str]]:
    """Check if source modules can be imported and initialized"""
    results = []
    all_good = True
    
    # Test document processor
    try:
        from document_processor import SimpleDocumentProcessor
        processor = SimpleDocumentProcessor()
        results.append("✅ DocumentProcessor (importable)")
    except Exception as e:
        results.append(f"❌ DocumentProcessor ({str(e)[:50]}...)")
        all_good = False
    
    # Test embeddings (requires API key)
    try:
        from embeddings import OpenAIEmbeddingSystem
        embedder = OpenAIEmbeddingSystem()
        results.append("✅ EmbeddingSystem (importable)")
    except Exception as e:
        results.append(f"❌ EmbeddingSystem ({str(e)[:50]}...)")
        all_good = False
    
    # Test vector database
    try:
        from vector_db import MilvusVectorDB
        vector_db = MilvusVectorDB()
        results.append("✅ VectorDB (importable)")
    except Exception as e:
        results.append(f"❌ VectorDB ({str(e)[:50]}...)")
        all_good = False
    
    return all_good, results

def check_document_folder() -> Tuple[bool, List[str]]:
    """Check documents folder and contents"""
    results = []
    
    docs_path = Path('data/documents')
    if not docs_path.exists():
        return False, ["❌ Documents folder doesn't exist"]
    
    # Count documents
    pdf_files = list(docs_path.glob('*.pdf'))
    txt_files = list(docs_path.glob('*.txt'))
    
    total_docs = len(pdf_files) + len(txt_files)
    
    if total_docs == 0:
        results.append("⚠️  No documents found (add files to data/documents/)")
    else:
        results.append(f"✅ Found {total_docs} documents ({len(pdf_files)} PDF, {len(txt_files)} TXT)")
    
    # Check for vector storage
    vector_storage = Path('data/vector_storage.pkl')
    if vector_storage.exists():
        size = vector_storage.stat().st_size
        results.append(f"✅ Vector storage exists ({size:,} bytes)")
    else:
        results.append("⚠️  No vector storage found (run document processing)")
    
    return True, results

def test_embeddings_api() -> Tuple[bool, str]:
    """Test OpenAI embeddings API connection"""
    try:
        from embeddings import OpenAIEmbeddingSystem
        embedder = OpenAIEmbeddingSystem()
        
        # Test with small text
        test_embedding = embedder.generate_single_embedding("test")
        
        if test_embedding and len(test_embedding) > 0:
            return True, f"✅ OpenAI API working (embedding dim: {len(test_embedding)})"
        else:
            return False, "❌ OpenAI API returned empty embedding"
            
    except Exception as e:
        return False, f"❌ OpenAI API error: {str(e)[:100]}..."

def check_vector_database_status() -> Tuple[bool, List[str]]:
    """Check vector database status"""
    results = []
    
    try:
        from vector_db import MilvusVectorDB
        vector_db = MilvusVectorDB()
        status = vector_db.get_status()
        
        results.append(f"✅ Vector DB initialized")
        results.append(f"📊 Total chunks: {status['total_chunks']}")
        results.append(f"📚 Sources: {len(status['sources'])}")
        
        if status['total_chunks'] > 0:
            results.append("✅ Ready for queries")
        else:
            results.append("⚠️  No documents processed yet")
        
        return True, results
        
    except Exception as e:
        return False, [f"❌ Vector DB error: {str(e)[:100]}..."]

def run_full_verification():
    """Run complete system verification"""
    print("🔍 RAG Knowledge Assistant - System Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Environment Variables", check_environment_variables),
        ("Source Modules", check_src_modules),
        ("Document Folder", check_document_folder),
        ("Vector Database", check_vector_database_status)
    ]
    
    overall_status = True
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        print("-" * 30)
        
        try:
            success, details = check_func()
            
            if isinstance(details, list):
                for detail in details:
                    print(f"  {detail}")
            else:
                print(f"  {details}")
            
            if not success:
                overall_status = False
                
        except Exception as e:
            print(f"  ❌ Check failed: {e}")
            overall_status = False
    
    # Optional API test (only if basic checks pass)
    if overall_status:
        print(f"\n📋 API Connection Test:")
        print("-" * 30)
        
        try:
            api_success, api_msg = test_embeddings_api()
            print(f"  {api_msg}")
            if not api_success:
                overall_status = False
        except:
            print("  ⚠️  API test skipped (add API key to test)")
    
    # Final summary
    print("\n" + "=" * 60)
    if overall_status:
        print("🎉 SYSTEM VERIFICATION PASSED")
        print("✅ Your RAG Knowledge Assistant is ready to use!")
        print("\nTo start: python chat_interface.py")
    else:
        print("⚠️  SYSTEM VERIFICATION ISSUES FOUND")
        print("Please fix the issues above before running the system.")
    
    print("=" * 60)
    
    return overall_status

def main():
    """Main verification function"""
    try:
        run_full_verification()
    except KeyboardInterrupt:
        print("\n⚠️  Verification interrupted by user")
    except Exception as e:
        print(f"\n❌ Verification error: {e}")

if __name__ == "__main__":
    main()