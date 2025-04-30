import os
import argparse
from fnmatch import fnmatch

def print_file_contents(path, exclude_patterns=None):
    """
    Recursively prints contents of all files under a directory with the specified format.
    
    Args:
        path: Root directory to start from (defaults to current directory)
        exclude_patterns: List of glob patterns for files/folders to exclude
    """
    exclude_patterns = exclude_patterns or []
    
    # Get all files recursively
    all_files = []
    for root, dirs, files in os.walk(path):
        # Remove excluded directories from dirs to prevent os.walk from traversing them
        dirs[:] = [d for d in dirs if not any(fnmatch(d, pat) for pat in exclude_patterns)]
        
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, path)
            
            # Skip excluded files
            if any(fnmatch(rel_path, pat) for pat in exclude_patterns):
                continue
                
            all_files.append(rel_path)
    
    # Sort files for consistent output
    all_files.sort()
    
    # Print contents of each file
    for i, rel_path in enumerate(all_files):
        if i > 0:
            print("\n")  # Add extra newline between files
            
        full_path = os.path.join(path, rel_path)
        print(rel_path)
        print("-" * len(rel_path))
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print(f"Error reading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print contents of files recursively.')
    parser.add_argument('--path', default='.', help='Root directory to start from')
    parser.add_argument('--exclude', nargs='+', default=[], 
                        help='Patterns to exclude (e.g. "*.pyc" "__pycache__")')
    
    args = parser.parse_args()
    
    print_file_contents(args.path, args.exclude)