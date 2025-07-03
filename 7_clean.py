import json
import os
import shutil
import argparse
from pathlib import Path
from typing import List

def read_config(config_path: str = 'config.json') -> dict:
    """Read configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration data
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def count_files_in_folder(folder_path: str) -> int:
    """Count the number of files in a folder (excluding subdirectories).
    
    Args:
        folder_path: Path to the folder to count files in
        
    Returns:
        Number of files in the folder
    """
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

def has_subdirectories(folder_path: str) -> bool:
    """Check if a directory contains any subdirectories.
    
    Args:
        folder_path: Path to the directory to check
        
    Returns:
        True if the directory contains subdirectories, False otherwise
    """
    try:
        return any(os.path.isdir(os.path.join(folder_path, item)) for item in os.listdir(folder_path))
    except (OSError, PermissionError):
        return False

def is_empty_folder(folder_path: str) -> bool:
    """Check if a directory is empty (no files and no subdirectories).
    
    Args:
        folder_path: Path to the directory to check
        
    Returns:
        True if the directory is empty, False otherwise
    """
    try:
        return len(os.listdir(folder_path)) == 0
    except (OSError, PermissionError):
        return False

def find_orphaned_log_folders(models_dir: str, logs_dir: str) -> List[str]:
    """Find log folders that don't have corresponding model folders.
    
    Args:
        models_dir: Path to the models directory
        logs_dir: Path to the logs directory
        
    Returns:
        List of orphaned log folder paths
    """
    orphaned_logs = []
    
    if not os.path.exists(logs_dir):
        return orphaned_logs
    
    # Get all model folders (including subdirectories)
    model_folders = set()
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            # Get relative path from models directory
            rel_path = os.path.relpath(root, models_dir)
            model_folders.add(rel_path)
    
    # Check all log folders
    for root, dirs, files in os.walk(logs_dir):
        # Get relative path from logs directory
        rel_path = os.path.relpath(root, logs_dir)
        
        # If this log folder doesn't have a corresponding model folder, it's orphaned
        if rel_path not in model_folders:
            orphaned_logs.append(root)
    
    return orphaned_logs

def delete_orphaned_log_folders(orphaned_logs: List[str], dry_run: bool = False) -> None:
    """Delete orphaned log folders that don't have corresponding model folders.
    
    Args:
        orphaned_logs: List of orphaned log folder paths
        dry_run: If True, only show what would be deleted without actually deleting
    """
    if not orphaned_logs:
        print("No orphaned log folders found.")
        return
    
    print(f"\n{'Deleting' if not dry_run else 'Would delete'} {len(orphaned_logs)} orphaned log folders...")
    
    for log_folder in orphaned_logs:
        if dry_run:
            print(f"[DRY RUN] Would delete orphaned log folder: {log_folder}")
        else:
            try:
                shutil.rmtree(log_folder)
                print(f"Deleted orphaned log folder: {log_folder}")
            except Exception as e:
                print(f"Error deleting orphaned log folder {log_folder}: {e}")

def delete_folder_and_logs(folder_path: str, models_dir: str, logs_dir: str, dry_run: bool = False) -> None:
    """Delete a folder and its corresponding log folder.
    
    Args:
        folder_path: Path to the folder to delete
        models_dir: Path to the models directory
        logs_dir: Path to the logs directory
        dry_run: If True, only show what would be deleted without actually deleting
    """
    # Get the relative path from models directory
    rel_path = os.path.relpath(folder_path, models_dir)
    
    if dry_run:
        print(f"[DRY RUN] Would delete model folder: {folder_path}")
    else:
        # Delete model folder
        try:
            shutil.rmtree(folder_path)
            print(f"Deleted model folder: {folder_path}")
        except Exception as e:
            print(f"Error deleting model folder {folder_path}: {e}")
    
    # Delete corresponding log folder
    log_folder = os.path.join(logs_dir, rel_path)
    if os.path.exists(log_folder):
        if dry_run:
            print(f"[DRY RUN] Would delete log folder: {log_folder}")
        else:
            try:
                shutil.rmtree(log_folder)
                print(f"Deleted log folder: {log_folder}")
            except Exception as e:
                print(f"Error deleting log folder {log_folder}: {e}")
    else:
        if dry_run:
            print(f"[DRY RUN] Log folder does not exist: {log_folder}")
        else:
            print(f"Log folder does not exist: {log_folder}")

def main(dry_run: bool = False) -> None:
    """Main function to clean up model and log folders.
    
    Args:
        dry_run: If True, only show what would be deleted without actually deleting
    """
    # Read config
    config = read_config()
    save_dir = config['trainer']['save_dir']
    
    # Get the models directory path
    models_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'logs')
    
    print(f"Models directory: {models_dir}")
    print(f"Logs directory: {logs_dir}")
    
    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN MODE - No files will be deleted")
        print(f"{'='*60}")
    
    # List to store folders to be deleted
    folders_to_delete: List[str] = []
    
    # First, check folders at the models directory level
    if os.path.exists(models_dir):
        print(f"\nChecking folders at models directory level...")
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                if is_empty_folder(item_path):
                    folders_to_delete.append(item_path)
                    print(f"Marked for deletion (empty folder): {item_path}")
                elif not has_subdirectories(item_path):
                    folders_to_delete.append(item_path)
                    print(f"Marked for deletion (no subdirectories): {item_path}")
    else:
        print(f"Models directory does not exist: {models_dir}")
        exit(1)
    
    # Walk through all subdirectories in models directory
    print(f"\nChecking subdirectories for incomplete models...")
    for root, dirs, files in os.walk(models_dir):
        # Skip the root directory itself and model name directories
        if root == models_dir or os.path.dirname(root) == models_dir:
            continue
            
        # Count files in current directory
        file_count = count_files_in_folder(root)
        
        # If less than 5 files, mark for deletion
        if file_count < 5:
            folders_to_delete.append(root)
            print(f"Marked for deletion: {root} (contains {file_count} files)")
    
    # Delete marked folders and their corresponding log folders
    print(f"\n{'Deleting' if not dry_run else 'Would delete'} {len(folders_to_delete)} marked folders...")
    for folder in folders_to_delete:
        delete_folder_and_logs(folder, models_dir, logs_dir, dry_run)
    
    # Find and delete orphaned log folders
    print(f"\nChecking for orphaned log folders...")
    orphaned_logs = find_orphaned_log_folders(models_dir, logs_dir)
    delete_orphaned_log_folders(orphaned_logs, dry_run)
    
    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN COMPLETED - No files were actually deleted")
        print(f"{'='*60}")
    else:
        print(f"\nCleanup completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean up incomplete model and log folders')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be deleted without actually deleting anything')
    
    args = parser.parse_args()
    main(dry_run=args.dry_run)

