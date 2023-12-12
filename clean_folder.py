import os
import shutil

def remove_ipynb_checkpoints(root_dir):
    """Recursively remove .ipynb_checkpoints directories from the specified root directory."""
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname == '.ipynb_checkpoints':
                full_path = os.path.join(dirpath, dirname)
                print(f"Removing: {full_path}")
                shutil.rmtree(full_path)

if __name__ == "__main__":
    dataset_directory = '/workspace/arknights_op/gallery-dl/danbooru/'  # Replace with the path to your dataset directory
    remove_ipynb_checkpoints(dataset_directory)
    print("Cleanup completed.")
