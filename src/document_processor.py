"""
Simple document processor using PyPDF2 (Updated with Better Chunking)
"""

import PyPDF2
import os
from pathlib import Path
from typing import List, Dict
import re

class SimpleDocumentProcessor:
    def __init__(self, chunk_size=400, overlap=50):
        """Initialize document processor with optimized chunking parameters"""
        self.chunk_size = chunk_size  # Smaller chunks for better precision
        self.overlap = overlap
        print("✅ Simple document processor initialized")
        print(f"📊 Optimized chunking - Size: {chunk_size}, Overlap: {overlap}")
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning while preserving structure"""
        # Remove extra whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
        text = re.sub(r'\n[ \t]+', '\n', text)   # Remove leading spaces on lines
        
        # Remove common PDF artifacts while keeping important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\^\&\*\+\=\<\>\|\\`~]', ' ', text)
        
        return text.strip()
    
    def smart_chunk_text(self, text: str, source: str) -> List[Dict]:
        """Improved chunking strategy that preserves context and meaning"""
        
        # First, try to split by paragraphs for better context preservation
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph fits in current chunk, add it
            if len(current_chunk) + len(para) < self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, chunk_id, source))
                    chunk_id += 1
                
                # If paragraph is too long, split it further
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_long_text(para, source, chunk_id)
                    chunks.extend(sub_chunks)
                    chunk_id += len(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, chunk_id, source))
        
        return chunks
    
    def _split_long_text(self, text: str, source: str, start_id: int) -> List[Dict]:
        """Split long text while preserving sentence boundaries"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_id = start_id
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence fits in current chunk
            if len(current_chunk) + len(sentence) < self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, chunk_id, source))
                    chunk_id += 1
                
                # If sentence is still too long, force split with overlap
                if len(sentence) > self.chunk_size:
                    force_chunks = self._force_split_text(sentence, source, chunk_id)
                    chunks.extend(force_chunks)
                    chunk_id += len(force_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, chunk_id, source))
        
        return chunks
    
    def _force_split_text(self, text: str, source: str, start_id: int) -> List[Dict]:
        """Force split very long text with overlap to preserve context"""
        chunks = []
        chunk_id = start_id
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            if chunk_text.strip():
                chunks.append(self._create_chunk(chunk_text, chunk_id, source))
                chunk_id += 1
        
        return chunks
    
    def _create_chunk(self, text: str, chunk_id: int, source: str) -> Dict:
        """Create a standardized chunk dictionary with metadata"""
        return {
            'text': text.strip(),
            'chunk_id': f"{Path(source).stem}_chunk_{chunk_id}",
            'source': source,
            'chunk_number': chunk_id,
            'char_count': len(text),
            'word_count': len(text.split()),
            'char_start': 0,  # Will be updated if needed
            'char_end': len(text)
        }
    
    def process_pdf(self, file_path: str) -> List[Dict]:
        """Process a PDF file with improved chunking strategy"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"📄 Processing PDF: {Path(file_path).name}")
        
        # Extract text from PDF
        full_text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    full_text += f"\n\n--- Page {page_num + 1} ---\n\n"
                    full_text += page_text
        
        # Clean the extracted text
        clean_text = self.clean_text(full_text)
        
        # Create smart chunks with better strategy
        chunks = self.smart_chunk_text(clean_text, file_path)
        
        print(f"✅ Created {len(chunks)} optimized chunks from {len(pdf_reader.pages)} pages")
        
        # Show chunk statistics for debugging
        if chunks:
            avg_chars = sum(chunk['char_count'] for chunk in chunks) / len(chunks)
            min_chars = min(chunk['char_count'] for chunk in chunks)
            max_chars = max(chunk['char_count'] for chunk in chunks)
            print(f"📊 Chunk stats - Avg: {avg_chars:.0f}, Min: {min_chars}, Max: {max_chars} chars")
        
        return chunks
    
    def process_text_file(self, file_path: str) -> List[Dict]:
        """Process a text file with improved chunking"""
        
        print(f"📝 Processing text file: {Path(file_path).name}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Clean the text
        clean_text = self.clean_text(text)
        
        # Create smart chunks
        chunks = self.smart_chunk_text(clean_text, file_path)
        
        print(f"✅ Created {len(chunks)} optimized chunks")
        
        # Show statistics
        if chunks:
            avg_chars = sum(chunk['char_count'] for chunk in chunks) / len(chunks)
            print(f"📊 Average chunk size: {avg_chars:.0f} characters")
        
        return chunks

# Test the processor
if __name__ == "__main__":
    processor = SimpleDocumentProcessor(chunk_size=400, overlap=50)
    print("🧪 Improved document processor ready!")
    print("📝 Optimized for better search accuracy with smaller, context-aware chunks")
    print("💡 Place your documents in data/documents/ folder")