# Plant & Tree Species Identification - Semester Major Project

This workspace contains the initial **Dataset Analysis, Gap Identification, and Feature Enhancement** setup for the Plant Species Identification App.

---

## 📁 Workspace Files

1. `plant_dataset_analysis_report.md` (Artifact): Full report ready to present/submit to your professor.
2. `sample_plant_metadata.json`: Standardized JSON metadata catalog combining image labels with plant care, taxonomy, and safety features.
3. `fetch_botanical_metadata.py`: Automated Python script fetching taxonomy from **GBIF API** and descriptions from **Wikipedia API**.
4. `dataset_analysis.py`: Exploratory Data Analysis (EDA) script generating class distribution and organ breakdown charts.
5. `dataset_analysis_chart.png`: Visual chart showing organ breakdown and class imbalance.

---

## 🚀 How to Run the Scripts

### 1. Test Botanical API Metadata Fetcher
Fetch taxonomy and summary details for any plant's scientific name:
```bash
python fetch_botanical_metadata.py "Azadirachta indica" "Quercus robur"
```

### 2. Run Dataset Exploratory Analysis (EDA)
Generate analysis metrics and save visual charts (`dataset_analysis_chart.png`):
```bash
python dataset_analysis.py
```

---

## 📊 Summary of Recommendations for Your Professor

1. **Recommended Base Dataset**: **Pl@ntNet-300K** (or **Pl@ntNet-Mini** for initial model training) because it includes real-world outdoor images with organ tags (`leaf`, `flower`, `fruit`, `bark`, `habit`).
2. **Key Data Gaps Identified**:
   - Raw vision datasets only provide class labels (e.g. `Quercus_robur`), missing common names, family, genus, sunlight/watering needs, toxicity, and medicinal/ecological uses.
   - Organ imbalances (leaves dominate, while bark/flowers are fewer).
   - Class imbalance across rare vs. common species.
3. **Proposed Dataset Enhancements**:
   - **Data Fusion Engine**: Combine image dataset with REST APIs (GBIF & Wikipedia) to build a rich JSON plant catalog.
   - **Organ-Aware Training**: Support organ tags (`leaf`, `flower`, `bark`) so users can upload photos of leaves OR tree trunks.
   - **Structured Metadata Schema**: Standardized JSON format linking vision model predictions to rich UI care cards.
