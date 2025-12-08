#!/usr/bin/env python3
"""
Helper script to temporarily adjust monitoring thresholds for testing.

This makes it easier to trigger retraining decisions during testing.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

monitor_file = project_root / "Monitoring" / "RAG" / "rag_monitor.py"


def adjust_thresholds(multiplier: float = 0.5):
    """
    Adjust thresholds by a multiplier to make them easier to trigger.
    
    Args:
        multiplier: Multiply thresholds by this value (0.5 = half, 2.0 = double)
    """
    with open(monitor_file, 'r') as f:
        content = f.read()
    
    # Backup original
    backup_file = monitor_file.with_suffix('.py.backup')
    if not backup_file.exists():
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"✓ Created backup: {backup_file}")
    
    # Adjust thresholds
    lines = content.split('\n')
    new_lines = []
    in_thresholds = False
    
    for line in lines:
        if 'THRESHOLDS = {' in line:
            in_thresholds = True
            new_lines.append(line)
        elif in_thresholds and line.strip().startswith('}'):
            in_thresholds = False
            new_lines.append(line)
        elif in_thresholds and ':' in line and any(keyword in line for keyword in ['max_', 'min_']):
            # Extract threshold value
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    key_part = parts[0].strip()
                    value_part = parts[1].strip().rstrip(',')
                    
                    # Try to extract numeric value
                    try:
                        # Handle comments
                        if '#' in value_part:
                            value_str, comment = value_part.split('#', 1)
                            value_str = value_str.strip()
                            comment = ' #' + comment
                        else:
                            value_str = value_part
                            comment = ''
                        
                        # Extract number
                        if value_str.replace('.', '').replace('-', '').isdigit():
                            old_value = float(value_str)
                            new_value = old_value * multiplier
                            
                            # Format appropriately
                            if old_value >= 1.0:
                                new_value_str = f"{new_value:.1f}"
                            else:
                                new_value_str = f"{new_value:.3f}"
                            
                            new_line = f"        {key_part}: {new_value_str},{comment}"
                            new_lines.append(new_line)
                            print(f"  Adjusted {key_part}: {old_value} → {new_value_str}")
                        else:
                            new_lines.append(line)
                    except:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Write modified content
    with open(monitor_file, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"\n✓ Adjusted thresholds by {multiplier}x")
    print(f"  Original backed up to: {backup_file}")


def restore_thresholds():
    """Restore original thresholds from backup."""
    backup_file = monitor_file.with_suffix('.py.backup')
    
    if not backup_file.exists():
        print("No backup found. Cannot restore.")
        return
    
    with open(backup_file, 'r') as f:
        content = f.read()
    
    with open(monitor_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Restored original thresholds from {backup_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adjust monitoring thresholds for testing")
    parser.add_argument("--multiplier", type=float, default=0.5,
                       help="Multiply thresholds by this value (default: 0.5 = half)")
    parser.add_argument("--restore", action="store_true",
                       help="Restore original thresholds from backup")
    
    args = parser.parse_args()
    
    if args.restore:
        restore_thresholds()
    else:
        adjust_thresholds(args.multiplier)

