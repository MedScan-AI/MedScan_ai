"""Chunking module for RAG pipeline."""
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import tiktoken

# Setup logging
LOG_DIR = Path("/opt/airflow/logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"chunking_{datetime.now().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HeaderDetector:
    """Detects headings in poorly-structured markdown."""

    def __init__(self):
        self.heading_keywords = {
            'introduction', 'overview', 'background', 'summary', 'conclusion',
            'abstract', 'methods', 'results', 'discussion', 'references',
            'symptoms', 'treatment', 'diagnosis', 'causes', 'prevention',
            'what is', 'how to', 'why', 'when', 'where', 'section'
        }

    def is_likely_heading(
        self, line: str, next_line: str = ""
    ) -> Tuple[bool, int]:
        """Determine if a line is likely a heading.

        Returns:
            Tuple of (is_heading, level)
        """
        stripped = line.strip()
        
        if not stripped or re.match(r'^#{1,6}\s', stripped):
            return False, 0
        
        # Bold text on its own line
        bold_match = re.match(r'^\*\*(.+?)\*\*$|^__(.+?)__$', stripped)
        if bold_match:
            text = bold_match.group(1) or bold_match.group(2)
            if len(text.split()) <= 10:
                return True, 2
        
        # ALL CAPS (including single hyphenated words like WITH-HYPHENS)
        word_count = len(stripped.split())
        is_all_caps_multi = word_count >= 2 and word_count <= 12
        is_single_hyphenated = word_count == 1 and '-' in stripped
        if stripped.isupper() and (is_all_caps_multi or is_single_hyphenated):
            return True, 2
        
        # Short line followed by longer content (use word count)
        first_line_words = len(stripped.split())
        next_line_words = len(next_line.strip().split())
        if first_line_words < 20 and next_line_words > first_line_words:
            if not stripped.endswith(('.', ',', ';', '!', '?')):
                if any(keyword in stripped.lower() for keyword in self.heading_keywords):
                    return True, 1  # Main section headers are level 1
                if re.match(r'^\d+\.?\d*\.?\s+[A-Z]', stripped):
                    return True, 2
        
        # Numbered sections (supports sub-numbering like 1., 2.1., 3.1.2.)
        if re.match(r'^\d+(\.\d+)*\.\s+[A-Z]', stripped) and len(stripped) < 80:
            return True, 2
        
        # Question format headings
        if stripped.endswith('?') and len(stripped.split()) <= 12:
            if stripped[0].isupper():
                return True, 3
        
        # Common heading words in title case (check ANY word for keywords)
        words = stripped.split()
        if len(words) >= 2 and len(words) <= 10:
            # Check if any word (after removing punctuation) is a keyword
            has_keyword = any(w.lower().rstrip(':') in self.heading_keywords for w in words)
            if has_keyword:
                title_case_count = sum(1 for w in words if w[0].isupper())
                if title_case_count >= len(words) * 0.6:
                    return True, 1  # Main section headers are level 1
        
        return False, 0
    
    def process_markdown(self, content: str) -> str:
        """Add proper headers where detected."""
        lines = content.split('\n')
        processed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            
            is_heading, level = self.is_likely_heading(line, next_line)
            
            if is_heading:
                cleaned = re.sub(r'^\*\*(.+?)\*\*$|^__(.+?)__$', r'\1\2', line.strip())
                header = '#' * level + ' ' + cleaned
                processed_lines.append(header)
            else:
                processed_lines.append(line)
            
            i += 1
        
        return '\n'.join(processed_lines)


class RAGChunker:
    """Chunks markdown documents for RAG with header detection."""

    def __init__(self):
        self.detector = HeaderDetector()
        self.token_counts = []  # Track token counts for statistics

    def count_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken."""
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)
    
    def extract_metadata(self, content: str) -> Tuple[Dict, str]:
        """
        Extract YAML frontmatter metadata from markdown.
        
        Returns:
            (metadata_dict, content_without_metadata)
        """
        metadata = {}
        
        # Check for YAML frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                yaml_content = parts[1].strip()
                remaining_content = parts[2].strip()
                
                # Parse simple YAML (key: value pairs)
                for line in yaml_content.split('\n'):
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        metadata[key] = value
                
                return metadata, remaining_content
        
        return metadata, content
    
    def has_headers(self, content: str) -> bool:
        """Check if content already has conventional # headers."""
        lines = content.split('\n')
        header_count = sum(1 for line in lines if re.match(r'^#{1,6}\s', line.strip()))
        return header_count > 0
    
    def find_min_header_level(self, content: str) -> int:
        """
        Find the minimum (highest) header level in the content.
        E.g., if doc has ## and ###, returns 2.
        """
        lines = content.split('\n')
        min_level = 6  # Start with max possible level
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                level = len(header_match.group(1))
                min_level = min(min_level, level)
        
        return min_level if min_level < 6 else 1
    
    def chunk_by_headers(self, content: str) -> List[Dict[str, str]]:
        """
        Chunk content by markdown headers, dynamically using the highest header level.
        
        Returns:
            List of chunks with {header, content, level}
        """
        lines = content.split('\n')
        chunks = []
        
        # Find the minimum header level to use for chunking
        chunk_level = self.find_min_header_level(content)
        
        current_chunk = {
            'header': 'Introduction',
            'content': [],
            'level': 0
        }
        
        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            
            if header_match:
                level = len(header_match.group(1))
                header_text = header_match.group(2)
                
                # Only create new chunk if this is at or above our chunking level
                if level <= chunk_level:
                    # Save previous chunk if it has content
                    if current_chunk['content']:
                        current_chunk['content'] = '\n'.join(current_chunk['content']).strip()
                        if current_chunk['content']:  # Only add non-empty chunks
                            chunks.append(current_chunk)
                    
                    # Start new chunk
                    current_chunk = {
                        'header': header_text,
                        'content': [],
                        'level': level
                    }
                else:
                    # This is a sub-header, include it in content
                    current_chunk['content'].append(line)
            else:
                current_chunk['content'].append(line)
            
            token_count = self.count_tokens('\n'.join(current_chunk['content']))
            self.token_counts.append(token_count)
            current_chunk['token_count'] = token_count
        
        # Add final chunk
        if current_chunk['content']:
            current_chunk['content'] = '\n'.join(current_chunk['content']).strip()
            if current_chunk['content']:
                chunks.append(current_chunk)
        
        return chunks
    
    def process_file(self, record: dict) -> List[Dict]:
        """
        Process a markdown file and return chunks with metadata.
        
        Returns:
            List of dictionaries with metadata and content chunks
        """
        try:
            # Step 1 & 6: Extract metadata
            content_body = record.get('markdown_text', record.get('text', ''))
            excluded_keys = ['markdown_title', 'markdown_text', 'text', 'word_count', 'token_count']
            metadata = {k: v for k,v in record.items() if k not in excluded_keys}
            doc_link = record.get('link', 'N/A')
            if content_body == "":
                logger.error(f"No text found for {doc_link}")
                return []
            
            # Step 2: Check if headers exist
            has_headers = self.has_headers(content_body)
            
            # Step 3: Add headers if missing
            if not has_headers:
                logger.info(f"Adding headers to: {doc_link}")
                content_body = self.detector.process_markdown(content_body)
            else:
                logger.info(f"  ✓ Headers found in: {doc_link}")
            
            # Step 4: Chunk by headers
            chunks = self.chunk_by_headers(content_body)
            
            # Step 5: Create JSON with metadata and chunks (flattened)
            result = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'chunk_id': i,
                    'section_header': chunk['header'],
                    'section_level': chunk['level'],
                    'content': chunk['content'],
                    'chunk_token_count': chunk.get('token_count', 0),
                    **metadata  # Expand metadata into the main structure
                }
                result.append(chunk_data)

            
            logger.info(f"Generated {len(result)} chunks")
            return result
            
        except Exception as e:
            doc_link = record.get('link', 'N/A')
            logger.error(f"Error processing {doc_link}: {e}")
            return []
    
    def process_jsonl(self, input_path: Path) -> List[Dict]:
        """
        Process JSONL file and return all chunks.
        
        Args:
            input_path: Path to input JSONL file
            
        Returns:
            List of all chunks
        """
        logger.info(f"Starting chunking process...")
        logger.info(f"Input file: {input_path}")
        
        all_chunks = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            logger.info(f"Found {len(lines)} records\n")
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                record = json.loads(line)
                logger.info(f"Processing: {record.get('link', 'N/A')}")
                chunks = self.process_file(record)
                all_chunks.extend(chunks)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETE!")
        logger.info(f"Processed {len(all_chunks)} chunks from {len(lines)} records")
        logger.info(f"{'='*60}")
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict], output_path: Path):
        """
        Save chunks to JSON file.
        
        Args:
            chunks: List of chunk dictionaries
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved to: {output_path}")


def main():
    """Main execution function - for standalone testing only."""
    INPUT_FILE = Path("/opt/airflow/data/RAG/raw_data/raw_data.jsonl")
    OUTPUT_FILE = Path("/opt/airflow/data/RAG/chunked_data/chunks.json")
    
    chunker = RAGChunker()
    
    if INPUT_FILE.exists():
        chunks = chunker.process_jsonl(INPUT_FILE)
        chunker.save_chunks(chunks, OUTPUT_FILE)
        
        # Print sample chunk for verification
        if chunks:
            logger.info("\nSample chunk:")
            logger.info(json.dumps(chunks[0], indent=2))
            token_counts = chunker.token_counts
            logger.info(f"\nToken count statistics across chunks:")
            logger.info(f"  - Min tokens: {min(token_counts)}")
            logger.info(f"  - Max tokens: {max(token_counts)}")
            logger.info(f"  - Avg tokens: {sum(token_counts)/len(token_counts):.2f}")
    else:
        logger.error(f"Input File not found: {INPUT_FILE}")


if __name__ == "__main__":
    main()