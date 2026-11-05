import os
import zipfile
import shutil

def extract_dataset(zip_path, extract_to="dataset"):
    """
    Extracts a downloaded Kaggle dataset zip file (e.g., plant-identification.zip)
    into a structured dataset directory for model training.
    """
    if not os.path.exists(zip_path):
        print(f"File '{zip_path}' not found.")
        print("Please download the zip file from Kaggle and place it in this workspace folder.")
        return
        
    print(f"Extracting '{zip_path}' into '{extract_to}/'...")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    print(f"Extraction complete! Dataset is ready at '{os.path.abspath(extract_to)}'")
    
    # Inspect dataset folder structure
    for root, dirs, files in os.walk(extract_to):
        depth = root.replace(extract_to, '').count(os.sep)
        if depth <= 2:
            print(f"{'  ' * depth}📁 {os.path.basename(root)}/ ({len(files)} files)")

if __name__ == "__main__":
    # Look for any zip file in the current working directory
    zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]
    if zip_files:
        extract_dataset(zip_files[0])
    else:
        print("No .zip file found in the workspace yet.")
        print("Download the zip from https://www.kaggle.com/datasets/aminahanna/plant-identification and run this script.")
