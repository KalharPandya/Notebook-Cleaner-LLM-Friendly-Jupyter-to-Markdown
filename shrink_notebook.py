#!/usr/bin/env python3
"""
Notebook Size Reduction Script

Converts Jupyter notebooks (.ipynb) to clean markdown files optimized for LLM consumption.
Removes images, metadata, and verbose outputs while preserving code and essential content.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')


class NotebookCleaner:
    """Handles the conversion of Jupyter notebooks to clean markdown."""
    
    def __init__(self, max_text_length: int = 1000, max_error_lines: int = 8, 
                 max_html_length: int = 500):
        """
        Initialize the notebook cleaner.
        
        Args:
            max_text_length: Maximum length for text outputs before truncation
            max_error_lines: Maximum number of error traceback lines to keep
            max_html_length: Maximum HTML length before removal
        """
        self.max_text_length = max_text_length
        self.max_error_lines = max_error_lines
        self.max_html_length = max_html_length
        self.stats = {
            'images_removed': 0,
            'outputs_truncated': 0,
            'html_removed': 0,
            'js_removed': 0
        }
    
    def load_notebook(self, path: Path) -> Optional[Dict]:
        """
        Load and validate a Jupyter notebook file.
        
        Args:
            path: Path to the .ipynb file
            
        Returns:
            Notebook dictionary or None if invalid
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            # Validate basic notebook structure
            if not isinstance(notebook, dict):
                print(f"[WARNING] Invalid notebook structure: {path}")
                return None
            
            if 'cells' not in notebook:
                print(f"[WARNING] No cells found in notebook: {path}")
                return None
            
            return notebook
        
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON decode error in {path}: {e}")
            return None
        except Exception as e:
            print(f"[WARNING] Error loading {path}: {e}")
            return None
    
    def truncate_text(self, text: str) -> str:
        """
        Truncate long text, keeping beginning and end.
        
        Args:
            text: Text to truncate
            
        Returns:
            Truncated text or original if short enough
        """
        if len(text) <= self.max_text_length:
            return text
        
        self.stats['outputs_truncated'] += 1
        keep_chars = (self.max_text_length - 100) // 2  # Leave room for truncation message
        
        truncated = (
            text[:keep_chars] + 
            f"\n\n[... truncated {len(text) - 2 * keep_chars} characters ...]\n\n" + 
            text[-keep_chars:]
        )
        return truncated
    
    def truncate_error(self, traceback_lines: List[str]) -> List[str]:
        """
        Truncate error tracebacks, keeping beginning and end.
        
        Args:
            traceback_lines: List of traceback lines
            
        Returns:
            Truncated traceback lines
        """
        if len(traceback_lines) <= self.max_error_lines:
            return traceback_lines
        
        self.stats['outputs_truncated'] += 1
        keep_start = (self.max_error_lines + 1) // 2
        keep_end = self.max_error_lines - keep_start
        
        return (
            traceback_lines[:keep_start] + 
            [f"... ({len(traceback_lines) - self.max_error_lines} lines truncated) ..."] + 
            traceback_lines[-keep_end:]
        )
    
    def clean_output_item(self, output: Dict) -> Optional[str]:
        """
        Clean a single output item from a code cell.
        
        Args:
            output: Output dictionary from notebook
            
        Returns:
            Cleaned output string or None if removed
        """
        output_type = output.get('output_type', '')
        
        # Handle stream outputs (stdout/stderr)
        if output_type == 'stream':
            text = ''.join(output.get('text', []))
            if isinstance(output.get('text'), str):
                text = output['text']
            return self.truncate_text(text)
        
        # Handle error outputs
        if output_type == 'error':
            traceback = output.get('traceback', [])
            truncated = self.truncate_error(traceback)
            return '\n'.join(truncated)
        
        # Handle display_data and execute_result
        if output_type in ('display_data', 'execute_result'):
            data = output.get('data', {})
            result_parts = []
            
            # Check for images and remove them
            image_types = ['image/png', 'image/jpeg', 'image/jpg', 'image/svg+xml']
            for img_type in image_types:
                if img_type in data:
                    self.stats['images_removed'] += 1
                    result_parts.append('[Image output removed]')
                    # Remove this type from data to avoid processing again
                    data = {k: v for k, v in data.items() if k not in image_types}
                    break
            
            # Handle text/plain output
            if 'text/plain' in data:
                text = data['text/plain']
                if isinstance(text, list):
                    text = ''.join(text)
                result_parts.append(self.truncate_text(text))
            
            # Handle HTML (conditionally)
            if 'text/html' in data:
                html = data['text/html']
                if isinstance(html, list):
                    html = ''.join(html)
                
                if len(html) > self.max_html_length:
                    self.stats['html_removed'] += 1
                    result_parts.append('[Large HTML output removed]')
                else:
                    # Keep short HTML (like tables)
                    result_parts.append(f"HTML Output:\n{html}")
            
            # Remove JavaScript
            if 'application/javascript' in data:
                self.stats['js_removed'] += 1
                result_parts.append('[JavaScript output removed]')
            
            return '\n\n'.join(result_parts) if result_parts else None
        
        return None
    
    def process_markdown_cell(self, cell: Dict, cell_num: int) -> str:
        """
        Process a markdown cell.
        
        Args:
            cell: Cell dictionary
            cell_num: Cell number (1-indexed)
            
        Returns:
            Formatted markdown string
        """
        source = cell.get('source', [])
        if isinstance(source, list):
            content = ''.join(source)
        else:
            content = source
        
        # Ensure content is valid string
        if not isinstance(content, str):
            content = str(content)
        
        return f"## Markdown Cell {cell_num}\n\n{content}\n\n"
    
    def process_code_cell(self, cell: Dict, cell_num: int) -> str:
        """
        Process a code cell.
        
        Args:
            cell: Cell dictionary
            cell_num: Cell number (1-indexed)
            
        Returns:
            Formatted markdown string with code and outputs
        """
        # Extract source code
        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source
        
        # Ensure code is valid string
        if not isinstance(code, str):
            code = str(code)
        
        result = f"## Code Cell {cell_num}\n\n```python\n{code}\n```\n\n"
        
        # Process outputs
        outputs = cell.get('outputs', [])
        if outputs:
            output_texts = []
            for output in outputs:
                cleaned = self.clean_output_item(output)
                if cleaned:
                    output_texts.append(cleaned)
            
            if output_texts:
                result += "**Output:**\n\n```\n" + '\n\n'.join(output_texts) + "\n```\n\n"
        
        return result
    
    def convert_to_markdown(self, notebook: Dict, input_path: Path) -> str:
        """
        Convert entire notebook to markdown format.
        
        Args:
            notebook: Notebook dictionary
            input_path: Original file path
            
        Returns:
            Complete markdown string
        """
        # Reset stats for this notebook
        self.stats = {
            'images_removed': 0,
            'outputs_truncated': 0,
            'html_removed': 0,
            'js_removed': 0
        }
        
        # Build header
        markdown_parts = [
            f"# Notebook: {input_path.name}\n",
            f"Generated from: {input_path}\n",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n---\n\n"
        ]
        
        # Process cells
        cells = notebook.get('cells', [])
        code_cell_num = 1
        markdown_cell_num = 1
        
        for cell in cells:
            cell_type = cell.get('cell_type', '')
            
            if cell_type == 'markdown':
                markdown_parts.append(self.process_markdown_cell(cell, markdown_cell_num))
                markdown_cell_num += 1
            elif cell_type == 'code':
                markdown_parts.append(self.process_code_cell(cell, code_cell_num))
                code_cell_num += 1
            # Skip raw cells and other types
        
        return ''.join(markdown_parts)
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics for the last notebook."""
        return self.stats.copy()


def find_notebooks(path: Path, skip_existing: bool = False) -> List[Tuple[Path, Path]]:
    """
    Recursively find all .ipynb files in directory tree.
    
    Args:
        path: File or directory path to search
        skip_existing: Skip files that already have _cleaned.md versions
        
    Returns:
        List of (input_path, output_path) tuples
    """
    notebooks = []
    
    if path.is_file():
        if path.suffix == '.ipynb':
            output_path = path.parent / f"{path.stem}_cleaned.md"
            if not skip_existing or not output_path.exists():
                notebooks.append((path, output_path))
        return notebooks
    
    # Directory traversal
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        
        # Skip hidden directories and Mac metadata directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX']
        
        for file in files:
            # Skip Mac resource fork files and process only .ipynb
            if file.endswith('.ipynb') and not file.startswith('._'):
                input_path = root_path / file
                output_path = root_path / f"{input_path.stem}_cleaned.md"
                
                if not skip_existing or not output_path.exists():
                    notebooks.append((input_path, output_path))
    
    return notebooks


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def process_notebooks(input_path: Path, output_path: Optional[Path] = None,
                      max_text_length: int = 1000, max_error_lines: int = 8,
                      skip_existing: bool = False, dry_run: bool = False,
                      verbose: bool = False) -> Dict[str, Any]:
    """
    Process notebooks and convert to markdown.
    
    Args:
        input_path: Input file or directory
        output_path: Custom output path (for single files only)
        max_text_length: Maximum text length before truncation
        max_error_lines: Maximum error lines to keep
        skip_existing: Skip files with existing cleaned versions
        dry_run: Show what would be processed without writing
        verbose: Show detailed progress
        
    Returns:
        Statistics dictionary
    """
    cleaner = NotebookCleaner(max_text_length, max_error_lines)
    
    # Find all notebooks to process
    if output_path and input_path.is_file():
        # Custom output for single file
        notebooks = [(input_path, output_path)]
    else:
        notebooks = find_notebooks(input_path, skip_existing)
    
    if not notebooks:
        print("No notebooks found to process.")
        return {}
    
    print(f"Processing: {input_path}\n")
    print(f"Found {len(notebooks)} notebook(s)")
    print("-" * 60)
    
    total_stats = {
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'original_size': 0,
        'cleaned_size': 0,
        'images_removed': 0,
        'outputs_truncated': 0
    }
    
    for input_file, output_file in notebooks:
        try:
            # Load notebook
            notebook = cleaner.load_notebook(input_file)
            if notebook is None:
                total_stats['errors'] += 1
                continue
            
            # Get original size
            original_size = input_file.stat().st_size
            total_stats['original_size'] += original_size
            
            # Convert to markdown
            markdown_content = cleaner.convert_to_markdown(notebook, input_file)
            
            if dry_run:
                print(f"[DRY RUN] Would create: {output_file}")
                if verbose:
                    stats = cleaner.get_stats()
                    print(f"  - Would remove {stats['images_removed']} images")
                    print(f"  - Would truncate {stats['outputs_truncated']} outputs")
            else:
                # Write output with UTF-8 encoding and error handling
                try:
                    with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
                        f.write(markdown_content)
                except Exception as write_error:
                    print(f"[ERROR] Error writing {output_file}: {write_error}")
                    total_stats['errors'] += 1
                    continue
                
                cleaned_size = output_file.stat().st_size
                total_stats['cleaned_size'] += cleaned_size
                
                reduction = 100 * (1 - cleaned_size / original_size) if original_size > 0 else 0
                
                stats = cleaner.get_stats()
                total_stats['images_removed'] += stats['images_removed']
                total_stats['outputs_truncated'] += stats['outputs_truncated']
                
                print(f"[OK] {input_file.name} -> {output_file.name}")
                print(f"  {format_size(original_size)} → {format_size(cleaned_size)} ({reduction:.1f}% reduction)")
                if verbose or stats['images_removed'] > 0 or stats['outputs_truncated'] > 0:
                    details = []
                    if stats['images_removed'] > 0:
                        details.append(f"removed {stats['images_removed']} images")
                    if stats['outputs_truncated'] > 0:
                        details.append(f"truncated {stats['outputs_truncated']} outputs")
                    if stats['html_removed'] > 0:
                        details.append(f"removed {stats['html_removed']} HTML outputs")
                    if stats['js_removed'] > 0:
                        details.append(f"removed {stats['js_removed']} JS outputs")
                    if details:
                        print(f"  {', '.join(details)}")
            
            total_stats['processed'] += 1
            
        except Exception as e:
            print(f"[ERROR] Error processing {input_file.name}: {e}")
            total_stats['errors'] += 1
    
    # Print summary
    print("-" * 60)
    print("\nSummary:")
    print(f"  Processed: {total_stats['processed']} file(s)")
    if total_stats['skipped'] > 0:
        print(f"  Skipped: {total_stats['skipped']} file(s) (already cleaned)")
    if total_stats['errors'] > 0:
        print(f"  Errors: {total_stats['errors']} file(s)")
    
    if not dry_run and total_stats['processed'] > 0:
        total_reduction = (
            100 * (1 - total_stats['cleaned_size'] / total_stats['original_size'])
            if total_stats['original_size'] > 0 else 0
        )
        print(f"  Total size: {format_size(total_stats['original_size'])} → "
              f"{format_size(total_stats['cleaned_size'])} "
              f"({total_reduction:.1f}% reduction)")
        print(f"  Images removed: {total_stats['images_removed']}")
        print(f"  Outputs truncated: {total_stats['outputs_truncated']}")
    
    return total_stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Convert Jupyter notebooks to clean markdown files optimized for LLM consumption.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python shrink_notebook.py sample.ipynb
  
  # Process entire directory recursively
  python shrink_notebook.py ./notebooks/
  
  # Custom output location
  python shrink_notebook.py sample.ipynb -o custom_output.md
  
  # Batch processing with options
  python shrink_notebook.py ./hw_submissions/ --max-text-length 1500 --skip-existing
        """
    )
    
    parser.add_argument(
        'input_path',
        type=Path,
        help='Input notebook file or directory to process'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Custom output path (for single files only)'
    )
    
    parser.add_argument(
        '--max-text-length',
        type=int,
        default=1000,
        help='Maximum text length before truncation (default: 1000)'
    )
    
    parser.add_argument(
        '--max-error-lines',
        type=int,
        default=8,
        help='Maximum error traceback lines to keep (default: 8)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already have _cleaned.md versions'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without writing files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress output'
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not args.input_path.exists():
        print(f"Error: Input path does not exist: {args.input_path}")
        sys.exit(1)
    
    # Validate custom output (only for single files)
    if args.output and not args.input_path.is_file():
        print("Error: Custom output path can only be used with single input files")
        sys.exit(1)
    
    # Process notebooks
    try:
        start_time = datetime.now()
        process_notebooks(
            args.input_path,
            args.output,
            args.max_text_length,
            args.max_error_lines,
            args.skip_existing,
            args.dry_run,
            args.verbose
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"  Time: {elapsed:.1f} seconds")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
