import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import tiktoken

# Configure paths
INPUT_DIR = Path(__file__).parent.parent.parent / "data" / "RAG" / "raw_data"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "RAG" / "chunked_data" / "chunks.json"

# Setup logging
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"chunking_{datetime.now().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
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
            'what is', 'how to', 'why', 'when', 'where'
        }
    
    def is_likely_heading(self, line: str, next_line: str = "") -> Tuple[bool, int]:
        """Determines if a line is likely a heading and returns (is_heading, level)."""
        stripped = line.strip()
        
        if not stripped or re.match(r'^#{1,6}\s', stripped):
            return False, 0
        
        # Bold text on its own line
        bold_match = re.match(r'^\*\*(.+?)\*\*$|^__(.+?)__$', stripped)
        if bold_match:
            text = bold_match.group(1) or bold_match.group(2)
            if len(text.split()) <= 10:
                return True, 2
        
        # ALL CAPS
        if stripped.isupper() and 2 <= len(stripped.split()) <= 12:
            if not re.match(r'^[A-Z0-9\-]+$', stripped.replace(' ', '')):
                return True, 2
        
        # Short line followed by longer content
        if len(stripped) < 60 and len(next_line.strip()) > 60:
            if not stripped.endswith(('.', ',', ';', ':', '!', '?')):
                if any(keyword in stripped.lower() for keyword in self.heading_keywords):
                    return True, 2
                if re.match(r'^\d+\.?\d*\.?\s+[A-Z]', stripped):
                    return True, 2
        
        # Numbered sections
        if re.match(r'^\d+\.\s+[A-Z]', stripped) and len(stripped) < 80:
            return True, 2
        
        # Question format headings
        if stripped.endswith('?') and len(stripped.split()) <= 12:
            if stripped[0].isupper():
                return True, 3
        
        # Common heading words in title case
        words = stripped.split()
        if len(words) >= 2 and len(words) <= 10:
            first_word = words[0].lower().rstrip(':')
            if first_word in self.heading_keywords:
                title_case_count = sum(1 for w in words if w[0].isupper())
                if title_case_count >= len(words) * 0.6:
                    return True, 2
        
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
        """
        Estimate token count using tiktoken.
        """
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
    
    def process_file(self, filepath: Path) -> List[Dict]:
        """
        Process a markdown file and return chunks with metadata.
        
        Returns:
            List of dictionaries with metadata and content chunks
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Step 1 & 6: Extract metadata
            metadata, content_body = self.extract_metadata(content)
            
            # Step 2: Check if headers exist
            has_headers = self.has_headers(content_body)
            
            # Step 3: Add headers if missing
            if not has_headers:
                logger.info(f"  ⚡ Adding headers to: {filepath.name}")
                content_body = self.detector.process_markdown(content_body)
            else:
                logger.info(f"  ✓ Headers found in: {filepath.name}")
            
            # Step 4: Chunk by headers
            chunks = self.chunk_by_headers(content_body)
            
            # Step 5: Create JSON with metadata and chunks (flattened)
            result = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'chunk_id': i,
                    'source_file': str(filepath.name),
                    'section_header': chunk['header'],
                    'section_level': chunk['level'],
                    'content': chunk['content'],
                    'token_count': chunk.get('token_count', 0),
                    **metadata  # Expand metadata into the main structure
                }
                result.append(chunk_data)

            
            logger.info(f"  → Generated {len(result)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {filepath.name}: {e}")
            return []
    
    def process_directory(self, input_dir: Path, output_file: Path):
        """
        Process all markdown files in a directory and save to JSON.
        
        Args:
            input_dir: Directory containing .md files
            output_file: Path to output JSON file
        """
        logger.info(f"Starting chunking process...")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output file: {output_file}")
        
        all_chunks = []
        md_files = sorted(input_dir.glob('*.md'))
        
        if not md_files:
            logger.warning(f"No .md files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(md_files)} markdown files\n")
        
        for md_file in md_files:
            logger.info(f"Processing: {md_file.name}")
            chunks = self.process_file(md_file)
            all_chunks.extend(chunks)
        
        # Save to JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETE!")
        logger.info(f"Processed {len(all_chunks)} chunks from {len(md_files)} files")
        logger.info(f"Saved to: {output_file}")
        logger.info(f"{'='*60}")
        
        return all_chunks


def main():
    """Main execution function."""
    # Initialize chunker
    chunker = RAGChunker()
    
    # Process all files
    if INPUT_DIR.exists():
        chunks = chunker.process_directory(INPUT_DIR, OUTPUT_FILE)
        
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
        logger.error(f"Input directory not found: {INPUT_DIR}")
        logger.error("Please create the directory and add markdown files.")


if __name__ == "__main__":
    main()