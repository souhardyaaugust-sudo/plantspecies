import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_plant_dataset(json_path="sample_plant_metadata.json"):
    print(f"--- PLANT DATASET ANALYSIS REPORT ---")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    df = pd.json_normalize(data)
    print(f"Total Species Analyzed: {len(df)}")
    print(f"Columns present in metadata:\n{list(df.columns)}\n")
    
    # 1. Image Distribution per Species & Organ Type
    organ_cols = [c for c in df.columns if c.startswith('dataset_image_distribution.') and not c.endswith('.total')]
    organ_summary = {}
    for col in organ_cols:
        organ_name = col.split('.')[-1]
        organ_summary[organ_name] = df[col].sum()
        
    print("--- ORGAN IMAGE DISTRIBUTION SUMMARY ---")
    for organ, count in organ_summary.items():
        print(f"  - {organ.capitalize()}: {count} images")
    print(f"  - Total Images Across Sample: {df['dataset_image_distribution.total'].sum()}\n")
    
    # 2. Missing Feature Matrix Check
    required_features = [
        'scientific_name', 'taxonomy.family', 'taxonomy.genus', 
        'care_and_environment.sunlight', 'care_and_environment.watering', 
        'care_and_environment.soil_type', 'safety_and_uses.toxicity_pets', 
        'safety_and_uses.uses'
    ]
    
    missing_counts = {}
    for feature in required_features:
        if feature in df.columns:
            missing_counts[feature] = df[feature].isna().sum()
        else:
            missing_counts[feature] = len(df) # 100% missing
            
    print("--- MISSING FEATURE CHECK IN RAW DATASET ---")
    for feat, missing in missing_counts.items():
        pct = (missing / len(df)) * 100
        print(f"  - {feat}: {missing}/{len(df)} missing ({pct:.1f}%)")
        
    # 3. Plotting Charts for Presentation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Organ Distribution
    axes[0].bar(organ_summary.keys(), organ_summary.values(), color=['#2ca02c', '#8c564b', '#e377c2', '#ff7f0e', '#1f77b4'])
    axes[0].set_title("Image Count by Organ Type (Leaf, Bark, Flower, etc.)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Total Number of Images")
    axes[0].set_xlabel("Organ Tag")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Total Images Per Species (Class Imbalance)
    species_names = df['scientific_name']
    image_totals = df['dataset_image_distribution.total']
    axes[1].barh(species_names, image_totals, color='#17becf')
    axes[1].set_title("Total Images per Plant Species (Class Imbalance Analysis)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Image Count")
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    chart_output = "dataset_analysis_chart.png"
    plt.savefig(chart_output, dpi=300)
    print(f"\nSaved analysis visual chart to '{chart_output}'")
    
    return df

if __name__ == "__main__":
    analyze_plant_dataset()
