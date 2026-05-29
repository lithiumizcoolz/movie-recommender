"""
Program to update the MovieLens Latest dataset.
"""

import os
import sys
import shutil
import zipfile
import argparse
import urllib.request

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    
    # Custom block-by-block download to show progress
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100.0, read_so_far * 100 / total_size)
            # Print a neat text progress bar
            bar_len = 30
            filled_len = int(bar_len * percent / 100)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            sys.stdout.write(f"\r[{bar}] {percent:.1f}% ({read_so_far / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")
        else:
            sys.stdout.write(f"\rDownloaded {read_so_far / (1024*1024):.1f}MB")
        sys.stdout.flush()
        
    urllib.request.urlretrieve(url, dest_path, reporthook=report_hook)
    print("\nDownload complete.")

def main():
    parser = argparse.ArgumentParser(description="Update MovieLens Latest Dataset")
    parser.add_argument("--small", action="store_true", help="Download the small dataset (~100k ratings, 1MB) instead of full dataset")
    args = parser.parse_args()

    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "Dataset"))
    os.makedirs(DATASET_ROOT, exist_ok=True)

    # Determine dataset URL, zip name, and target extraction paths
    if args.small:
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        zip_name = "ml-latest-small.zip"
        extracted_folder_name = "ml-latest-small"
        target_folder = os.path.join(DATASET_ROOT, "ml-latest-small")
    else:
        url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
        zip_name = "ml-latest.zip"
        extracted_folder_name = "ml-latest"
        target_folder = os.path.join(DATASET_ROOT, "ml-latest")

    zip_path = os.path.join(DATASET_ROOT, zip_name)
    
    # Download dataset
    try:
        download_file(url, zip_path)
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        sys.exit(1)

    # Extract dataset
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_ROOT)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        sys.exit(1)
    finally:
        # Clean up zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # Move extracted files to target_folder if they are not already there
    extracted_path = os.path.join(DATASET_ROOT, extracted_folder_name)
    if os.path.exists(extracted_path):
        if os.path.abspath(extracted_path) != os.path.abspath(target_folder):
            os.makedirs(os.path.dirname(target_folder), exist_ok=True)
            if os.path.exists(target_folder):
                if os.path.isdir(target_folder):
                    shutil.rmtree(target_folder)
                else:
                    os.remove(target_folder)
            shutil.move(extracted_path, target_folder)
            print(f"Successfully saved dataset in the location: {target_folder}")
        else:
            print(f"Successfully updated dataset in the location: {target_folder}")
    else:
        print(f"Error: Extracted folder {extracted_path} not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()
