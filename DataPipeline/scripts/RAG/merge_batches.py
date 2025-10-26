"""Merge batches - used AFTER validation."""
import json
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_jsonl(file_path: Path) -> list:
    """Load JSONL file."""
    records = []
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return records
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                logger.warning(f"Line {line_num}: Invalid JSON")
    
    logger.info(f"Loaded {len(records)} from {file_path.name}")
    return records


def save_jsonl(records: list, file_path: Path):
    """Save to JSONL."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(records)} records")


def deduplicate_records(records: list) -> list:
    """Remove duplicates by link."""
    seen = set()
    unique = []
    dupes = 0
    
    for record in records:
        link = record.get('link', '')
        if link and link not in seen:
            seen.add(link)
            unique.append(record)
        else:
            dupes += 1
    
    if dupes > 0:
        logger.info(f"Removed {dupes} duplicates")
    
    return unique


def merge_jsonl_files(baseline_file: Path, batch_file: Path, output_file: Path, deduplicate: bool = True) -> bool:
    """Merge baseline and validated batch."""
    logger.info("=" * 60)
    logger.info("MERGE OPERATION")
    logger.info("=" * 60)
    
    try:
        baseline_records = load_jsonl(baseline_file)
        batch_records = load_jsonl(batch_file)
        
        if not batch_records:
            logger.error("No batch records")
            return False
        
        all_records = baseline_records + batch_records
        logger.info(f"Combined: {len(all_records)} total")
        
        if deduplicate:
            all_records = deduplicate_records(all_records)
            logger.info(f"After dedup: {len(all_records)} records")
        
        save_jsonl(all_records, output_file)
        
        logger.info(" Merge complete")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f" Merge failed: {e}")
        return False


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-dedupe", action="store_true")
    
    args = parser.parse_args()
    
    success = merge_jsonl_files(
        Path(args.baseline),
        Path(args.batch),
        Path(args.output),
        deduplicate=not args.no_dedupe
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()