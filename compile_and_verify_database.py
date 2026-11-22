# Script to compile, validate, and write indian_plant_database.json
import json
import os

from data_trees import trees_data
from data_flowers import flowers_data
from data_fruits import fruits_data
from data_crops_veg import crops_veg_data
from data_herbs_aquatic import herbs_aquatic_data

output_path = r"c:\Users\ASUS\OneDrive\Desktop\Major Project\indian_plant_database.json"

REQUIRED_FIELDS = [
    "species", "scientific_name", "family", "order", "category",
    "native_to_india", "common_in_india", "regional_names", "description",
    "habitat", "health_status", "active_compounds", "sunlight", "watering",
    "soil", "toxicity_pets", "uses", "ayurvedic_uses", "conservation_status"
]

ALLOWED_CATEGORIES = {"tree", "flower", "fruit", "vegetable", "crop", "herb", "grass", "aquatic"}

combined_db = {}
category_counts = {}

all_modules = [
    ("Trees", trees_data),
    ("Flowers", flowers_data),
    ("Fruits", fruits_data),
    ("Crops & Vegetables", crops_veg_data),
    ("Herbs & Aquatic", herbs_aquatic_data)
]

for group_name, dataset in all_modules:
    print(f"\nProcessing group: {group_name} ({len(dataset)} items)")
    for key, data in dataset.items():
        if key in combined_db:
            raise ValueError(f"Duplicate key detected across groups: '{key}'")
        
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in data:
                raise ValueError(f"Entry '{key}' missing required field: '{field}'")
            if data[field] is None:
                raise ValueError(f"Entry '{key}' has None for field: '{field}'")
        
        # Check category
        cat = data["category"]
        if cat not in ALLOWED_CATEGORIES:
            raise ValueError(f"Entry '{key}' has invalid category: '{cat}'. Must be one of {ALLOWED_CATEGORIES}")
        
        # Check native_to_india & common_in_india
        if not isinstance(data["native_to_india"], bool):
            raise TypeError(f"Entry '{key}' native_to_india must be bool, got {type(data['native_to_india'])}")
        if not isinstance(data["common_in_india"], bool) or not data["common_in_india"]:
            raise ValueError(f"Entry '{key}' common_in_india must be True")
            
        # Check regional names
        reg = data["regional_names"]
        if not isinstance(reg, dict):
            raise TypeError(f"Entry '{key}' regional_names must be a dict")
        if "hindi" not in reg or not reg["hindi"]:
            raise ValueError(f"Entry '{key}' missing required Hindi name")
        if "tamil" not in reg or not reg["tamil"]:
            raise ValueError(f"Entry '{key}' missing required Tamil name")
            
        combined_db[key] = data
        category_counts[cat] = category_counts.get(cat, 0) + 1

total_count = len(combined_db)
print(f"\n==========================================")
print(f"DATABASE COMPILATION SUMMARY")
print(f"==========================================")
print(f"Total Species Compiled: {total_count}")
print("Category Breakdown:")
for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {cat.capitalize()}: {count}")

assert total_count >= 100, f"Total count {total_count} is less than required 100!"

# Save database with UTF-8 encoding
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined_db, f, ensure_ascii=False, indent=2)

print(f"\nSuccessfully wrote database to: {output_path}")

# Verification check: Read file back and verify JSON validity and size
with open(output_path, "r", encoding="utf-8") as f:
    verified_db = json.load(f)

file_size_kb = os.path.getsize(output_path) / 1024
print(f"Verification successful: {len(verified_db)} entries loaded back correctly.")
print(f"File size: {file_size_kb:.2f} KB")
