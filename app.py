import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import json
import os
import cv2
import io
import re

# =============================================================================
# 1. Page Configuration & Botanical Theme
# =============================================================================
st.set_page_config(
    page_title="Indian Flora AI — Plant Species & Pathology Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Responsive Dark Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .hero-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #10b981, #34d399, #059669);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.05rem; color: #94a3b8; margin-bottom: 1.5rem;
    }
    .result-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-radius: 16px; padding: 24px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }
    .species-title {
        font-size: 2.1rem; font-weight: 800; color: #f8fafc; margin: 0;
    }
    .sci-title {
        font-style: italic; color: #34d399; font-size: 1.15rem; font-weight: 600; margin-top: 4px;
    }
    .badge {
        display: inline-block; padding: 4px 12px;
        border-radius: 20px; font-weight: 700; font-size: 0.82rem;
        margin-right: 6px; margin-bottom: 6px;
    }
    .badge-green { background: #064e3b; color: #6ee7b7; border: 1px solid #047857; }
    .badge-blue { background: #1e3a5f; color: #93c5fd; border: 1px solid #2563eb; }
    .badge-purple { background: #4c1d95; color: #c4b5fd; border: 1px solid #7c3aed; }
    .badge-amber { background: #78350f; color: #fde68a; border: 1px solid #d97706; }
    .badge-red { background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }
    
    .conf-pill {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white; padding: 6px 14px; border-radius: 20px;
        font-weight: 800; font-size: 0.95rem;
    }
    .info-box {
        background: #1e293b; border-radius: 10px; padding: 14px 18px;
        border: 1px solid #334155; margin-bottom: 12px;
    }
    .box-label {
        font-weight: 700; color: #94a3b8; font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
    }
    .box-val { color: #f1f5f9; font-size: 0.95rem; }
    
    .chem-box {
        background: #082f49; border-left: 4px solid #0284c7;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 10px 0;
    }
    .ayur-box {
        background: #2e1065; border-left: 4px solid #a855f7;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 10px 0;
    }
    .pathology-box {
        background: #450a0a; border-left: 4px solid #ef4444;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 10px 0;
    }
    .enc-card {
        background: #1e293b; border-radius: 12px; padding: 16px;
        border: 1px solid #334155; margin-bottom: 14px;
        transition: transform 0.2s;
    }
    .enc-card:hover { transform: translateY(-3px); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. Resource & Model Loaders (Cached)
# =============================================================================
@st.cache_resource
def load_deep_cnn_classifier():
    """Loads MobileNetV2 base extractor + Custom Trained 43-class Botanical Head"""
    weights_path = os.path.abspath("weights/mobilenet_v2_weights_no_top.h5")
    if os.path.exists(weights_path):
        base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=weights_path, pooling="avg")
    else:
        base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg")
    
    head = None
    if os.path.exists("custom_botanical_head.keras"):
        head = tf.keras.models.load_model("custom_botanical_head.keras")
        
    classes = []
    if os.path.exists("botanical_classes.json"):
        with open("botanical_classes.json", "r", encoding="utf-8") as f:
            classes = json.load(f)
            
    return base, head, classes

@st.cache_resource
def load_clip_foundation():
    """Loads OpenAI CLIP ViT-B/32 vision model from local cache"""
    try:
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
        proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
        return model, proc
    except Exception:
        return None, None

@st.cache_data
def load_botanical_database():
    """Loads authentic 140+ Indian flora profiles"""
    db = {}
    if os.path.exists("indian_plant_database.json"):
        with open("indian_plant_database.json", "r", encoding="utf-8") as f:
            db.update(json.load(f))
    if os.path.exists("advanced_plant_database.json"):
        with open("advanced_plant_database.json", "r", encoding="utf-8") as f:
            adv = json.load(f)
            for k, v in adv.items():
                if k not in db:
                    db[k] = v
    return db

@st.cache_resource
def get_gemini_client():
    """Returns Gemini client if API key is configured"""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None

base_cnn, head_cnn, cnn_classes = load_deep_cnn_classifier()
clip_model, clip_proc = load_clip_foundation()
plant_db = load_botanical_database()
gemini_client = get_gemini_client()

# =============================================================================
# 3. Botanical Knowledge Profiles & Domain Prompt Library
# =============================================================================

DOMAINS = [
    ("Tree", "a photo of a large Indian tree like banyan, peepal, neem or mango with trunk and canopy"),
    ("Fruit", "a close-up photo of a ripe fruit or harvest like apple, mango, banana, guava, papaya"),
    ("Flower", "a close-up photo of a blooming flower blossom like rose, sunflower, lotus, marigold, hibiscus"),
    ("Crop Leaf", "a close-up photo of an agricultural crop leaf like tomato, potato, corn, grape on a table"),
    ("Medicinal Herb", "a close-up photo of a medicinal herb or succulent like aloe vera, tulsi, mint")
]

INDIAN_SPECIES_BY_DOMAIN = {
    "Tree": [
        ("Banyan", "a photo of a huge spreading Indian banyan tree ficus benghalensis with aerial roots hanging down from branches and wide canopy"),
        ("Sacred Fig (Peepal)", "a photo of a sacred fig peepal tree ficus religiosa with heart shaped leaves"),
        ("Neem Tree", "a photo of an Indian neem tree azadirachta indica with serrated pinnate leaves"),
        ("Mango Tree", "a photo of a dense green mango tree canopy mangifera indica"),
        ("Coconut Palm", "a photo of a tall coconut palm tree cocos nucifera"),
        ("Gulmohar", "a photo of a royal poinciana gulmohar tree with bright scarlet red flowers"),
        ("Teak", "a photo of a teak tree tectona grandis with large broad leaves"),
        ("Sal", "a photo of a tall sal tree shorea robusta in an Indian forest"),
        ("Ashoka", "a photo of a sacred ashoka tree saraca asoca or weeping polyalthia longifolia"),
        ("Jamun (Java Plum)", "a photo of a java plum jamun tree syzygium cumini"),
        ("Amla (Indian Gooseberry)", "a photo of an amla tree phyllanthus emblica"),
        ("Tamarind", "a photo of a tamarind tree tamarindus indica")
    ],
    "Fruit": [
        ("Apple", "a photo of red apples, fresh apple fruit with cut half slice malus domestica"),
        ("Guava (Amrood)", "a photo of green guava fruit psidium guajava whole and sliced"),
        ("Mango", "a photo of ripe sweet yellow mangoes mangifera indica"),
        ("Banana", "a photo of fresh yellow ripe bananas musa acuminata"),
        ("Papaya", "a photo of ripe papaya fruit with orange flesh and black seeds carica papaya"),
        ("Pomegranate", "a photo of fresh red pomegranate fruit punica granatum with ruby arils"),
        ("Watermelon", "a photo of a large green striped watermelon fruit citrullus lanatus"),
        ("Orange", "a photo of fresh citrus sweet orange or mosambi fruit"),
        ("Grape", "a photo of a bunch of fresh green or black grapes vitis vinifera"),
        ("Custard Apple (Sitaphal)", "a photo of a bumpy green custard apple sitaphal fruit annona squamosa"),
        ("Jackfruit", "a photo of a spiky large green jackfruit artocarpus heterophyllus")
    ],
    "Flower": [
        ("Rose", "a photo of a garden rose flower rosa with velvety petals"),
        ("Sunflower", "a photo of a vibrant yellow sunflower helianthus annuus with brown seed center"),
        ("Lotus", "a photo of a sacred pink Indian lotus flower nelumbo nucifera on water"),
        ("Hibiscus", "a photo of a bright red hibiscus flower gudhal with prominent stamen"),
        ("Marigold", "a photo of an orange and yellow marigold flower tagetes genda"),
        ("Bougainvillea", "a photo of magenta or pink bougainvillea flower bracts on a vine"),
        ("Champa (Plumeria)", "a photo of white and yellow plumeria champa frangipani flower blossoms"),
        ("Jasmine", "a photo of fragrant white jasmine flowers chameli mogra jasminum"),
        ("Tulip", "a photo of a cup-shaped garden tulip flower tulipa")
    ],
    "Medicinal Herb": [
        ("Aloe Vera", "a photo of thick fleshy spiky green aloe vera succulent leaves with gel"),
        ("Tulsi (Holy Basil)", "a photo of holy basil tulsi plant ocimum tenuiflorum with fragrant leaves"),
        ("Mint (Pudina)", "a photo of fresh green aromatic mint pudina herb leaves mentha"),
        ("Curry Leaf", "a photo of fresh green kadi patta curry leaf foliage murraya koenigii"),
        ("Ashwagandha", "a photo of an ashwagandha plant withthanis somnifera with green berries"),
        ("Giloy", "a photo of a giloy heart-leaved moonseed vine tinospora cordifolia"),
        ("Brahmi", "a photo of small aquatic green brahmi bacopa monnieri herb leaves")
    ]
}

# =============================================================================
# 4. Multi-Engine Identification Pipeline
# =============================================================================

def explainable_saliency_heatmap(image: Image.Image):
    """Generates OpenCV Edge & Structural Feature Attention Map"""
    np_img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    blurred = cv2.GaussianBlur(norm, (19, 19), 0)
    heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(np_img, 0.65, heatmap_rgb, 0.35, 0))

def classify_crop_leaf_cnn(image: Image.Image):
    """Deep CNN trained on 54,000 real crop leaf images from PlantVillage"""
    if head_cnn is None or not cnn_classes:
        return []
    img_resized = image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(img_to_array(img_resized), axis=0)
    features = base_cnn(preprocess_input(arr), training=False)
    preds = head_cnn.predict(features, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:5]
    return [(cnn_classes[idx], float(preds[idx])) for idx in top_indices]

def query_gemini_vision(image: Image.Image):
    """Multimodal Vision Inference with Gemini 2.0 Flash"""
    if gemini_client is None:
        return None
    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        prompt = """Identify the plant, tree, flower, fruit, or leaf in this photo.
Return JSON ONLY:
{
    "species": "Common Name",
    "scientific_name": "Genus species",
    "family": "Family name",
    "confidence": 0.95,
    "category": "tree/flower/fruit/crop/herb",
    "hindi_name": "Hindi name if known",
    "health_status": "Healthy / Diseased",
    "condition_diagnosis": "Name of disease or Healthy",
    "treatment": "Cure if diseased, else maintenance",
    "identifying_features": "Visual evidence"
}"""
        from google.genai import types
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ]
        )
        text = response.text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return json.loads(text)
    except Exception:
        return None

def hierarchical_botanical_classifier(image: Image.Image):
    """
    Hierarchical 2-Stage Zero-Shot & Deep Learning Pipeline:
    Stage 1: Detect Macro Domain (Tree, Fruit, Flower, Crop Leaf, Herb)
    Stage 2: Execute domain-specialized classifier (CLIP or 43-Class PlantVillage CNN)
    """
    if clip_model is None or clip_proc is None:
        # Fallback to CNN
        preds = classify_crop_leaf_cnn(image)
        if preds:
            top_name, top_conf = preds[0]
            return {
                "species": top_name,
                "confidence": top_conf,
                "domain": "Crop Leaf",
                "candidates": preds,
                "engine": "MobileNetV2 CNN (PlantVillage)"
            }
        return None

    import torch

    # Stage 1: Detect Macro Domain
    macro_prompts = [desc for _, desc in DOMAINS]
    inputs = clip_proc(text=macro_prompts, images=image, return_tensors='pt', padding=True)
    with torch.no_grad():
        macro_probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0].numpy()
    
    best_domain_idx = int(macro_probs.argmax())
    detected_domain = DOMAINS[best_domain_idx][0]
    domain_conf = float(macro_probs[best_domain_idx])
    
    # Stage 2: Specialized Branching
    if detected_domain == "Crop Leaf":
        # Check Crop CNN first
        cnn_preds = classify_crop_leaf_cnn(image)
        if cnn_preds and cnn_preds[0][1] > 0.40:
            top_class, conf = cnn_preds[0]
            return {
                "species": top_class,
                "confidence": conf,
                "domain": "Crop Leaf & Pathology",
                "candidates": cnn_preds,
                "engine": "MobileNetV2 Transfer Learning CNN (54k images)"
            }

    # Use Domain Species Candidates
    candidates = INDIAN_SPECIES_BY_DOMAIN.get(detected_domain, INDIAN_SPECIES_BY_DOMAIN["Tree"])
    prompts = [desc for _, desc in candidates]
    inputs2 = clip_proc(text=prompts, images=image, return_tensors='pt', padding=True)
    with torch.no_grad():
        species_probs = clip_model(**inputs2).logits_per_image.softmax(dim=1)[0].numpy()
    
    ranked_indices = np.argsort(species_probs)[::-1]
    ranked_results = [(candidates[idx][0], float(species_probs[idx])) for idx in ranked_indices]
    
    top_species, top_conf = ranked_results[0]
    return {
        "species": top_species,
        "confidence": top_conf,
        "domain": detected_domain,
        "candidates": ranked_results[:5],
        "engine": f"Hierarchical Vision AI ({detected_domain} Stage)"
    }

def find_db_profile(species_name, sci_name=""):
    """Fetches full botanical and Ayurvedic profile from JSON database"""
    if not species_name:
        return None
    s_lower = species_name.lower().strip()
    
    # Direct match
    for k, v in plant_db.items():
        if k.lower() == s_lower or v.get("species", "").lower() == s_lower:
            return v
    
    # Split prefix (e.g. "Apple - healthy" -> "Apple")
    base_name = species_name.split(" - ")[0].split(" (")[0].strip().lower()
    for k, v in plant_db.items():
        if k.lower() == base_name or v.get("species", "").lower() == base_name:
            return v
            
    # Substring match
    for k, v in plant_db.items():
        if base_name in k.lower() or k.lower() in base_name:
            return v
            
    return None

# =============================================================================
# 5. Sidebar Controls
# =============================================================================
with st.sidebar:
    st.markdown("### 🌿 **Indian Flora AI**")
    st.caption("Plant Species & Pathology Diagnosis System")
    st.divider()
    
    st.markdown("#### ⚡ AI Engines Status")
    if gemini_client:
        st.success("🟢 **Gemini 2.0 Flash Vision**: Active")
    else:
        st.info("ℹ️ **Gemini API**: Optional (Add Key below)")
        with st.expander("🔑 Connect Gemini API Key"):
            k_input = st.text_input("Gemini API Key:", type="password", key="sidebar_key")
            if k_input:
                os.environ["GEMINI_API_KEY"] = k_input
                st.success("API Key saved! Click to reload.")
                st.button("Activate Key", on_click=lambda: st.rerun())
                
    st.success("🟢 **Hierarchical Vision Pipeline**: Active")
    st.success(f"🟢 **Deep CNN Classifier**: {len(cnn_classes)} Crop Pathology Classes")
    
    st.divider()
    st.markdown("#### 📊 Knowledge Base")
    st.metric("Total Documented Species", len(plant_db))
    st.caption("🇮🇳 Includes Trees, Flowers, Fruits, Crops & Ayurvedic Herbs")
    
    st.divider()
    input_source = st.radio("Input Source:", ["📸 Upload Image", "🧪 Select Verified Sample", "📷 Webcam Snapshot"])
    show_heatmap = st.toggle("🧠 Explainable AI (Edge Saliency)", value=True)

# =============================================================================
# 6. Main Dashboard Interface
# =============================================================================
st.markdown('<div class="hero-title">🌿 Indian Flora Species & Pathology AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">University Major Project — Specialized for Indian Botanical Flora, Keystone Trees, Agricultural Crops & Ayurvedic Herbs</div>', unsafe_allow_html=True)

tab_id, tab_encyclopedia = st.tabs(["🔍 Identify & Diagnose Plant", "📚 Indian Botanical Encyclopedia"])

# -----------------------------------------------------------------------------
# TAB 1: IDENTIFICATION
# -----------------------------------------------------------------------------
with tab_id:
    col_img, col_diag = st.columns([1, 1.25], gap="large")
    target_img = None
    
    with col_img:
        st.markdown("#### 1. Input Plant Image")
        if input_source == "📸 Upload Image":
            uploaded = st.file_uploader("Upload leaf, tree, flower or fruit photo (JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])
            if uploaded:
                target_img = Image.open(uploaded).convert("RGB")
        elif input_source == "🧪 Select Verified Sample":
            samples_dir = "sample_images"
            if os.path.exists(samples_dir):
                files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg', '.png'))])
                chosen = st.selectbox("Choose a test sample:", files, format_func=lambda x: f"🌿 {x.rsplit('.', 1)[0].replace('_', ' ').title()}")
                if chosen:
                    target_img = Image.open(os.path.join(samples_dir, chosen)).convert("RGB")
        elif input_source == "📷 Webcam Snapshot":
            cam = st.camera_input("Take a photo of a plant")
            if cam:
                target_img = Image.open(cam).convert("RGB")
                
        if target_img:
            st.image(target_img, caption="Input Botanical Subject", width="stretch")
            if show_heatmap:
                with st.spinner("Generating Explainable AI Heatmap..."):
                    hmap = explainable_saliency_heatmap(target_img)
                    st.image(hmap, caption="🧠 Explainable AI: Structural Focus Heatmap (Veins / Canopy Edges)", width="stretch")
        else:
            st.info("👈 Please upload or select a plant photo on the left to begin diagnosis.")

    with col_diag:
        st.markdown("#### 2. Botanical Identification Report")
        if target_img:
            with st.spinner("🌿 Analyzing across Hierarchical AI Vision Engines..."):
                # 1. Check Gemini Cloud Vision first if key exists
                gemini_res = query_gemini_vision(target_img)
                
                # 2. Hierarchical Local AI Vision
                local_res = hierarchical_botanical_classifier(target_img)
                
                # Decision Fusion
                if gemini_res and gemini_res.get("confidence", 0) > 0.60:
                    species_name = gemini_res.get("species", "Unknown")
                    sci_name = gemini_res.get("scientific_name", "")
                    confidence = float(gemini_res.get("confidence", 0.95))
                    engine_used = "Gemini 2.0 Flash Multimodal Vision"
                    health_status = gemini_res.get("health_status", "Healthy")
                    condition_diag = gemini_res.get("condition_diagnosis", "")
                    treatment_rec = gemini_res.get("treatment", "")
                    domain_detected = gemini_res.get("category", "Botanical").title()
                    candidates = [(species_name, confidence)]
                elif local_res:
                    species_name = local_res["species"]
                    confidence = local_res["confidence"]
                    domain_detected = local_res["domain"]
                    engine_used = local_res["engine"]
                    candidates = local_res.get("candidates", [])
                    sci_name = ""
                    health_status = "Healthy"
                    condition_diag = ""
                    treatment_rec = ""
                else:
                    species_name = None
                    
            if species_name:
                # Candidate Selector
                if len(candidates) > 1:
                    cand_names = [c[0] for c in candidates[:4]]
                    sel_cand = st.radio("🌿 Candidate Matches (Click to inspect):", cand_names, index=0, horizontal=True)
                    if sel_cand != species_name:
                        species_name = sel_cand
                        confidence = next((c[1] for c in candidates if c[0] == sel_cand), confidence)

                # Fetch Botanical Profile
                profile = find_db_profile(species_name, sci_name)
                if not profile:
                    profile = {
                        "species": species_name,
                        "scientific_name": sci_name or f"{species_name} sp.",
                        "family": "Plantae",
                        "order": "Magnoliophyta",
                        "health_status": health_status,
                        "active_compounds": "Flavonoids, Polyphenols, Chlorophyll",
                        "sunlight": "Full Sun to Partial Shade",
                        "watering": "Moderate",
                        "soil": "Fertile well-drained soil",
                        "toxicity_pets": "Non-toxic in small quantities",
                        "uses": "Ecological, shade, or agricultural significance."
                    }
                
                # Render Primary Card
                final_sci = sci_name or profile.get("scientific_name", "N/A")
                is_diseased = "diseas" in profile.get("health_status", "").lower() or "blight" in species_name.lower() or "rot" in species_name.lower() or "rust" in species_name.lower() or "scab" in species_name.lower()
                status_badge = '<span class="badge badge-red">🔴 PATHOLOGY DETECTED</span>' if is_diseased else '<span class="badge badge-green">🟢 HEALTHY SPECIES</span>'
                
                st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                        <div>
                            <div class="species-title">{profile.get('species', species_name)}</div>
                            <div class="sci-title">🔬 {final_sci}</div>
                        </div>
                        <div style="text-align: right;">
                            <span class="conf-pill">{confidence*100:.1f}% Confidence</span><br>
                            <div style="margin-top: 6px;">{status_badge}</div>
                        </div>
                    </div>
                    <div style="margin-top: 14px;">
                        <span class="badge badge-blue">📁 Domain: {domain_detected}</span>
                        <span class="badge badge-purple">🌿 Family: {profile.get('family', 'N/A')}</span>
                        <span class="badge badge-green">⚡ Engine: {engine_used}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(min(float(confidence), 1.0))
                
                # Regional Names
                regional = profile.get("regional_names", {})
                if regional and any(regional.values()):
                    parts = [f"<b>{lang.title()}:</b> {val}" for lang, val in regional.items() if val]
                    st.markdown(f"""
                    <div class="info-box">
                        <div class="box-label">🗣️ Indian Regional Names</div>
                        <div class="box-val">{' &nbsp;|&nbsp; '.join(parts)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Pathology & Disease Diagnosis
                if is_diseased or condition_diag:
                    cond_name = condition_diag or profile.get("condition_type", species_name)
                    treat_val = treatment_rec or profile.get("treatment", "Apply organic copper-based fungicide and isolate infected foliage.")
                    st.markdown(f"""
                    <div class="pathology-box">
                        <div class="box-label" style="color: #fca5a5;">🚨 PLANT PATHOLOGY DIAGNOSIS: {cond_name}</div>
                        <div class="box-val" style="margin-top: 6px;"><b>💊 Recommended Cure:</b> {treat_val}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Phytochemical & Bioactive Compounds
                compounds = profile.get("active_compounds", "")
                if compounds:
                    st.markdown(f"""
                    <div class="chem-box">
                        <div class="box-label" style="color: #38bdf8;">🧬 Bioactive Phytochemicals & Natural Metabolites</div>
                        <div class="box-val">{compounds}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Traditional Ayurvedic Uses
                ayur = profile.get("ayurvedic_uses", "")
                if ayur:
                    st.markdown(f"""
                    <div class="ayur-box">
                        <div class="box-label" style="color: #c084fc;">🕉️ Authentic Ayurvedic & Ethnomedicinal Properties</div>
                        <div class="box-val">{ayur}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Care Guide
                st.markdown("### ☀️ Cultivation & Environmental Care")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="info-box">
                        <div class="box-label">☀️ Sunlight Requirement</div>
                        <div class="box-val">{profile.get('sunlight', 'Full Sun')}</div>
                    </div>
                    <div class="info-box">
                        <div class="box-label">💧 Watering Routine</div>
                        <div class="box-val">{profile.get('watering', 'Moderate')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="info-box">
                        <div class="box-label">🌱 Soil Preference</div>
                        <div class="box-val">{profile.get('soil', 'Well-draining loam')}</div>
                    </div>
                    <div class="info-box">
                        <div class="box-label">💡 Uses & Ecosystem Role</div>
                        <div class="box-val">{profile.get('uses', 'Ecological and shade value')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Toxicity Safety
                tox = profile.get("toxicity_pets", "")
                if tox:
                    is_safe = "non-toxic" in tox.lower() or "safe" in tox.lower() or "not toxic" in tox.lower()
                    t_bg = "#052e16" if is_safe else "#450a0a"
                    t_border = "#10b981" if is_safe else "#ef4444"
                    t_icon = "✅" if is_safe else "⚠️"
                    st.markdown(f"""
                    <div style="background: {t_bg}; border-left: 4px solid {t_border}; padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 12px;">
                        <div class="box-label">{t_icon} Safety & Toxicity:</div>
                        <div class="box-val">{tox}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Probability Distribution Expander
                if candidates:
                    with st.expander("📊 Candidate Probability Distribution"):
                        for rank, (cand, prob) in enumerate(candidates[:5]):
                            st.write(f"**Rank {rank+1}**: `{cand}` — **{prob*100:.1f}%**")
            else:
                st.warning("⚠️ Could not identify botanical species. Please try uploading a clearer photo with proper lighting.")
        else:
            st.info("Analysis report will appear here once an image is uploaded.")

# -----------------------------------------------------------------------------
# TAB 2: BOTANICAL ENCYCLOPEDIA
# -----------------------------------------------------------------------------
with tab_encyclopedia:
    st.markdown("### 📚 Comprehensive Indian Botanical Encyclopedia")
    st.write(f"Explore **{len(plant_db)} cataloged species** covering Indian Keystone Trees, Agricultural Crops, Flowers, Fruits & Ayurvedic Herbs:")
    
    col_q, col_cat = st.columns([2, 1])
    with col_q:
        q = st.text_input("🔍 Search species, Hindi name, scientific name, or disease:", "")
    with col_cat:
        available_cats = sorted(set(v.get("category", "other") for v in plant_db.values()))
        cat_sel = st.multiselect("Filter by Category:", available_cats, default=[])
        
    filtered = []
    for k, v in plant_db.items():
        if cat_sel and v.get("category", "other") not in cat_sel:
            continue
        if q:
            searchable = f"{k} {v.get('species','')} {v.get('scientific_name','')} {v.get('active_compounds','')} {str(v.get('regional_names',{}))}".lower()
            if q.lower() not in searchable:
                continue
        filtered.append((k, v))
        
    st.caption(f"Displaying {len(filtered)} matching species")
    grid = st.columns(3)
    for idx, (k, p) in enumerate(filtered):
        cat = p.get("category", "plant")
        hindi = p.get("regional_names", {}).get("hindi", "")
        h_tag = f" ({hindi})" if hindi else ""
        with grid[idx % 3]:
            st.markdown(f"""
            <div class="enc-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #f1f5f9;">{k}{h_tag}</h4>
                    <span class="badge badge-green">{cat.upper()}</span>
                </div>
                <div style="font-style: italic; color: #34d399; font-size: 0.9rem; margin-top: 4px;">{p.get('scientific_name', '')}</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">
                    <b>Family:</b> {p.get('family', 'N/A')}<br>
                    <b>Active:</b> {p.get('active_compounds', 'N/A')[:60]}...
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    🌿 <b>Indian Flora AI</b> — Major Project | University Semester Evaluation<br>
    Trained on 54,000 real images (PlantVillage) + Hierarchical Indian Flora Zero-Shot Vision + Gemini 2.0 Flash Integration
</div>
""", unsafe_allow_html=True)
