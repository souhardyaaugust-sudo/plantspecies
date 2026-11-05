import zipfile
from collections import defaultdict, Counter
import os

zip_path = 'archive (2).zip'

with zipfile.ZipFile(zip_path, 'r') as z:
    all_files = [f for f in z.namelist() if not f.endswith('/')]
    print(f"Total files: {len(all_files)}")
    
    splits = defaultdict(lambda: defaultdict(int))
    augmented_count = defaultdict(int)
    non_augmented_count = defaultdict(int)
    extensions = Counter()
    
    for f in all_files:
        parts = f.split('/')
        ext = os.path.splitext(f)[1].lower()
        extensions[ext] += 1
        
        # typically: split_ttv_dataset_type_of_plants / <split> / <class_name> / <filename>
        if len(parts) >= 4:
            split_name = parts[1]
            class_name = parts[2]
            filename = parts[3]
            splits[split_name][class_name] += 1
            if filename.startswith('aug_'):
                augmented_count[split_name] += 1
            else:
                non_augmented_count[split_name] += 1
        else:
            print("Other file:", f)

print("\n--- FILE EXTENSIONS ---")
for ext, count in extensions.items():
    print(f"  {ext}: {count}")

print("\n--- SPLITS & CLASS COUNTS ---")
for split_name, class_dict in splits.items():
    total_in_split = sum(class_dict.values())
    print(f"\nSplit: '{split_name}' -> Total Images: {total_in_split}")
    print(f"  Classes ({len(class_dict)}):")
    for cls, cnt in sorted(class_dict.items()):
        print(f"    - {cls}: {cnt} images")
    print(f"  Augmented images ('aug_*'): {augmented_count[split_name]}")
    print(f"  Original/Non-augmented: {non_augmented_count[split_name]}")
