import json
import os
import shutil
from pathlib import Path

def read_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def count_files_in_folder(folder_path):
    """Count the number of files in a folder (excluding subdirectories)"""
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

if __name__ == '__main__':
    # Read config
    config = read_config()
    save_dir = config['trainer']['save_dir']
    
    # Get the models directory path
    models_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'logs')
    
    # List to store folders to be deleted
    folders_to_delete = []
    
    # Walk through all subdirectories in models directory
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
    for folder in folders_to_delete:
        # Get the relative path from models directory
        rel_path = os.path.relpath(folder, models_dir)
        
        # Delete model folder
        try:
            shutil.rmtree(folder)
            print(f"Deleted model folder: {folder}")
        except Exception as e:
            print(f"Error deleting model folder {folder}: {e}")
        
        # Delete corresponding log folder
        log_folder = os.path.join(logs_dir, rel_path)
        if os.path.exists(log_folder):
            try:
                shutil.rmtree(log_folder)
                print(f"Deleted log folder: {log_folder}")
            except Exception as e:
                print(f"Error deleting log folder {log_folder}: {e}")

