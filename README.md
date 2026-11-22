# Indian Flora AI — Plant Species Identification System

🌿 **AI-powered plant identification system specialized for Indian flora** — trees, flowers, fruits, crops, herbs & medicinal plants.

## 🎯 Features

- **Multi-Engine AI Pipeline**: Combines Gemini Vision API + CLIP ViT-B/32 + Custom Deep CNN for maximum accuracy
- **100+ Indian Plant Species**: Comprehensive database covering trees, flowers, fruits, crops, herbs & medicinal plants
- **Disease Diagnosis**: Identifies 43 crop leaf diseases with treatment recommendations
- **Botanical Encyclopedia**: Browse all species with scientific names, phytochemicals, and care guides
- **Ayurvedic Information**: Traditional medicinal uses for Indian plants
- **Regional Names**: Hindi, Tamil, and Bengali names for all species
- **Explainable AI**: Saliency heatmap visualization showing AI attention regions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Upload                       │
│            (Photo of any Indian plant)              │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│          Smart Identification Pipeline              │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Engine 1    │  │  Engine 2     │  │ Engine 3  │  │
│  │ Gemini       │  │ CLIP ViT     │  │ Custom    │  │
│  │ Vision API   │  │ Zero-Shot    │  │ CNN       │  │
│  │ (Primary)    │  │ (Fallback)   │  │ (Crops)   │  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬─────┘  │
│         │                │               │          │
│         ▼                ▼               ▼          │
│  ┌──────────────────────────────────────────────┐   │
│  │          Confidence Gate & Fusion             │   │
│  │   (Rejects low-confidence misidentifications) │   │
│  └──────────────────────┬───────────────────────┘   │
└─────────────────────────┼───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│          Indian Flora Knowledge Base                 │
│    (100+ species with botanical & Ayurvedic data)   │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              Identification Report                   │
│  • Species & Scientific Name                        │
│  • Regional Names (Hindi, Tamil, Bengali)           │
│  • Phytochemicals & Bioactive Compounds             │
│  • Ayurvedic & Traditional Uses                     │
│  • Cultivation & Care Guide                         │
│  • Toxicity & Conservation Status                   │
└─────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow/Keras (MobileNetV2 transfer learning)
- **Vision AI**: Google Gemini API, OpenAI CLIP (ViT-B/32)
- **Computer Vision**: OpenCV (saliency heatmaps)
- **Language**: Python 3.13+

## 🚀 Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Optional: Enable Gemini Vision API
For the most accurate identification, add your Gemini API key:
```bash
export GEMINI_API_KEY=your_key_here
```
Or enter it in the sidebar of the app.

## 📊 Plant Database

The system includes **100+ species** organized into:
- 🌳 **Trees**: Banyan, Peepal, Neem, Mango, Coconut, Teak, Ashoka, etc.
- 🌸 **Flowers**: Lotus, Marigold, Jasmine, Hibiscus, Rose, Sunflower, etc.
- 🍎 **Fruits**: Mango, Banana, Apple, Guava, Pomegranate, Papaya, etc.
- 🌾 **Crops**: Rice, Wheat, Sugarcane, Tomato, Potato, Cotton, etc.
- 🌿 **Herbs**: Tulsi, Ashwagandha, Brahmi, Aloe Vera, Turmeric, etc.
- 💧 **Aquatic**: Lotus, Water Lily, Mangrove, etc.

## 📄 License

University Major Project — Plant Species Identification System
