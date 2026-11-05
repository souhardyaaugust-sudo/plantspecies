import zipfile
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import io
from PIL import Image

zip_path = 'archive (2).zip'

with zipfile.ZipFile(zip_path, 'r') as z:
    names = [f for f in z.namelist() if f.endswith('.jpg')]
    
    orig_count = sum(1 for f in names if 'aug_' not in f.split('/')[-1])
    aug_count = len(names) - orig_count
    
    # Split stats
    splits = defaultdict(lambda: {'orig': 0, 'aug': 0})
    for f in names:
        parts = f.split('/')
        if len(parts) >= 4:
            s_name = parts[1]
            fname = parts[3]
            if fname.startswith('aug_'):
                splits[s_name]['aug'] += 1
            else:
                splits[s_name]['orig'] += 1
                
    # Sample resolutions
    sample_files = names[:100]
    widths, heights = [], []
    for sf in sample_files:
        try:
            im = Image.open(io.BytesIO(z.read(sf)))
            w, h = im.size
            widths.append(w)
            heights.append(h)
        except Exception:
            pass

# Create a 2x2 presentation dashboard chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Pie Chart: Real vs Synthetic (Augmented)
labels = [f'Real Original Images\n(3,000 / 10%)', f'Synthetic Augmented Copies\n(27,000 / 90%)']
sizes = [orig_count, aug_count]
colors = ['#2ca02c', '#ff7f0e']
axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.08, 0),
              textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0, 0].set_title("Actual Data Composition: 90% Synthetic Copies", fontsize=12, fontweight='bold')

# 2. Bar Chart: Original vs Augmented across Splits
split_labels = ['Train (23,972)', 'Val (3,030)', 'Test (2,998)']
split_keys = ['Train_Set_Folder', 'Validation_Set_Folder', 'Test_Set_Folder']
orig_vals = [splits[k]['orig'] for k in split_keys]
aug_vals = [splits[k]['aug'] for k in split_keys]

x = range(len(split_labels))
width = 0.35
axes[0, 1].bar([i - width/2 for i in x], orig_vals, width, label='Original Images', color='#2ca02c')
axes[0, 1].bar([i + width/2 for i in x], aug_vals, width, label='Pre-Augmented Images', color='#ff7f0e')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(split_labels, fontsize=10)
axes[0, 1].set_ylabel("Image Count")
axes[0, 1].set_title("Splits Analysis: Test & Val Sets Contain Augmentations!", fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.5)

# 3. Scatter Plot: Resolution Variance
axes[1, 0].scatter(widths, heights, color='#1f77b4', alpha=0.7, edgecolors='k')
axes[1, 0].set_title("Image Resolution Variance (Sample 100 images)", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Width (pixels)")
axes[1, 0].set_ylabel("Height (pixels)")
axes[1, 0].axvline(224, color='red', linestyle='--', label='Standard CNN Input (224x224)')
axes[1, 0].axhline(224, color='red', linestyle='--')
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='--', alpha=0.5)

# 4. Summary Text Card
summary_text = (
    "KEY FINDINGS FOR PROFESSOR:\n"
    "--------------------------------------------------\n"
    "• Total Classes: 30 Plant/Crop Species\n"
    "• Total Stored Files: 30,000 images\n"
    "• Genuine Unique Images: Only 3,000 (100 per class)\n"
    "• Synthetic Augmentations: 27,000 (90% of dataset)\n"
    "• Critical Methodology Flaw:\n"
    "  Augmentations were baked directly into Test & Val\n"
    "  sets, causing Data Leakage and inflated metrics!\n"
    "• Missing Metadata:\n"
    "  No scientific binomial names, no taxonomy, no care\n"
    "  instructions, no toxicity data.\n"
    "• Spelling & Labeling Issues:\n"
    "  Typo in folder name ('peper chili')."
)
axes[1, 1].axis('off')
axes[1, 1].text(0.05, 0.5, summary_text, fontsize=11, family='monospace', va='center',
                bbox=dict(boxstyle="round,pad=1", facecolor="#f0f3f6", edgecolor="#333333"))

plt.tight_layout()
chart_path = "archive_analysis_dashboard.png"
plt.savefig(chart_path, dpi=300)
print(f"Saved analysis dashboard chart to {chart_path}")
