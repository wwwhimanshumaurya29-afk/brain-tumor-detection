
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from src.models.model import create_model
from src.config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_REPO = "hmtech2005/brain-tumor-resnet18-v2"
MODEL_FILENAME = "best_model_v2.pth"

DEVICE = torch.device("cpu")


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="NeuroScan AI | Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #09111f;
    --panel: #111e30;
    --panel2: #16263b;
    --border: #263950;
    --teal: #51dbc5;
    --text: #edf5ff;
    --muted: #92a5bd;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #18334a 0%, transparent 38%),
        radial-gradient(ellipse at 95% 15%, #123b42 0%, transparent 30%),
        var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: white;
    letter-spacing: -0.5px;
}

.brand span {
    color: var(--teal);
}

.top-tag {
    color: var(--teal);
    border: 1px solid #326d70;
    background: #143239;
    padding: 7px 13px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero {
    padding: 35px 0 30px 0;
}

.eyebrow {
    color: var(--teal);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(35px, 5vw, 56px);
    line-height: 1.12;
    letter-spacing: -2px;
    color: #f4f8ff;
    font-weight: 800;
    margin-top: 12px;
}

.hero-title span {
    color: var(--teal);
}

.hero-description {
    color: var(--muted);
    font-size: 15px;
    max-width: 700px;
    line-height: 1.8;
    margin-top: 15px;
}

.feature-card {
    background: #14243a;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 17px;
    min-height: 105px;
}

.feature-icon {
    color: var(--teal);
    font-size: 23px;
}

.feature-title {
    font-size: 14px;
    color: white;
    font-weight: 700;
    margin-top: 9px;
}

.feature-desc {
    font-size: 11px;
    color: var(--muted);
    margin-top: 5px;
}

.section-card {
    background: linear-gradient(145deg, #14243a, #101c2e);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
    min-height: 160px;
}

.card-heading {
    font-size: 19px;
    font-weight: 800;
    color: #f0f6ff;
    margin-bottom: 5px;
}

.card-subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 15px;
}

.result-card {
    background: linear-gradient(135deg, #12363b, #14283a);
    border: 1px solid #34716c;
    padding: 22px;
    border-radius: 16px;
    margin-top: 12px;
}

.result-label {
    color: var(--teal);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.result-value {
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 800;
    color: white;
    margin-top: 8px;
}

.metric-box {
    background: #14243a;
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 15px;
    margin-top: 12px;
}

.metric-label {
    color: var(--muted);
    font-size: 11px;
}

.metric-value {
    color: white;
    font-size: 22px;
    font-weight: 800;
    margin-top: 5px;
}

.stButton > button {
    background: linear-gradient(100deg, #40cdb8, #68e3c8) !important;
    color: #071a20 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    padding: 12px 18px !important;
    min-height: 48px;
    transition: 0.2s;
}

.stButton > button:hover {
    box-shadow: 0 5px 25px #51dbc544;
    transform: translateY(-2px);
}

[data-testid="stFileUploader"] {
    background: #102036;
    border: 1px dashed #4a7585;
    border-radius: 13px;
    padding: 12px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stImage"] img {
    border-radius: 12px;
}

.stProgress > div > div > div > div {
    background: var(--teal);
}

.footer {
    border-top: 1px solid var(--border);
    margin-top: 35px;
    padding-top: 20px;
    color: var(--muted);
    font-size: 11px;
    text-align: center;
    line-height: 1.8;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL FROM HUGGING FACE
# =====================================================

@st.cache_resource
def load_trained_model():

    # Download model checkpoint from Hugging Face.
    # The downloaded file is cached automatically.

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )

    # Create architecture without downloading
    # additional pretrained weights.

    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_uploaded_image(uploaded_file):

    image = Image.open(uploaded_file).convert("L")

    image_array = np.array(
        image,
        dtype=np.float32
    )

    min_value = image_array.min()
    max_value = image_array.max()

    if max_value == min_value:
        raise ValueError(
            "Uploaded image has constant intensity."
        )

    # Min-max intensity normalization

    image_array = (
        (image_array - min_value) /
        (max_value - min_value)
    )

    image_array = (
        image_array * 255
    ).clip(0, 255).astype(np.uint8)

    image = Image.fromarray(image_array)

    # Resize to model input size

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image = image.convert("RGB")

    # ImageNet normalization

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image).unsqueeze(0)


# =====================================================
# HEADER
# =====================================================

header_left, header_right = st.columns([3, 1])

with header_left:

    st.markdown(
        '<div class="brand">✳ NeuroScan<span> AI</span></div>',
        unsafe_allow_html=True
    )

with header_right:

    st.markdown(
        '<div style="text-align:right">'
        '<span class="top-tag">● RESEARCH PROTOTYPE</span>'
        '</div>',
        unsafe_allow_html=True
    )


# =====================================================
# HERO SECTION
# =====================================================

st.html("""
<div class="hero">

    <div class="eyebrow">
        DEEP LEARNING · MEDICAL IMAGE RESEARCH
    </div>

    <div class="hero-title">
        Brain MRI<br>
        <span>Analysis Platform.</span>
    </div>

    <div class="hero-description">
        Explore brain MRI image classification through a
        ResNet18 deep learning model. Upload an image and
        view the model's three-class prediction probabilities.
    </div>

</div>
""")


# =====================================================
# FEATURE CARDS
# =====================================================

feature1, feature2, feature3 = st.columns(3)

features = [
    ("◈", "MRI Image Input", "Upload JPG, JPEG or PNG images"),
    ("✳", "ResNet18 Engine", "Trained three-class classification model"),
    ("▤", "Prediction Insights", "Class probabilities and confidence")
]

for col, feature in zip(
    [feature1, feature2, feature3],
    features
):

    with col:

        st.html(f"""
        <div class="feature-card">

            <div class="feature-icon">
                {feature[0]}
            </div>

            <div class="feature-title">
                {feature[1]}
            </div>

            <div class="feature-desc">
                {feature[2]}
            </div>

        </div>
        """)


st.write("")


# =====================================================
# MAIN WORKSPACE
# =====================================================

left, right = st.columns([1, 1], gap="large")


# -----------------------------
# LEFT: IMAGE UPLOAD
# -----------------------------

with left:

    st.html("""
    <div class="section-card">

        <div class="card-heading">
            01 &nbsp; Image Upload
        </div>

        <div class="card-subtitle">
            Select a brain MRI image for model analysis.
        </div>

    </div>
    """)

    uploaded_file = st.file_uploader(
        "Upload MRI image",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
        key="mri_upload"
    )

    if uploaded_file is not None:

        try:

            preview = Image.open(uploaded_file)

            st.image(
                preview,
                caption="Uploaded MRI Image",
                use_container_width=True
            )

        except Exception:

            st.error("Unable to preview this image.")


# -----------------------------
# RIGHT: AI ANALYSIS
# -----------------------------

with right:

    st.html("""
    <div class="section-card">

        <div class="card-heading">
            02 &nbsp; AI Analysis
        </div>

        <div class="card-subtitle">
            Run the trained model and inspect its output.
        </div>

    </div>
    """)

    st.write("")

    if uploaded_file is None:

        st.info(
            "Your prediction results will appear here "
            "after uploading an image."
        )

    else:

        if st.button(
            "✳  Analyze MRI Image",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Loading model and analyzing image..."
                ):

                    model = load_trained_model()

                    uploaded_file.seek(0)

                    input_tensor = preprocess_uploaded_image(
                        uploaded_file
                    ).to(DEVICE)

                    with torch.no_grad():

                        outputs = model(input_tensor)

                        probabilities = F.softmax(
                            outputs,
                            dim=1
                        )[0]

                    predicted_index = torch.argmax(
                        probabilities
                    ).item()

                    predicted_class = CLASS_NAMES[
                        predicted_index
                    ]

                    confidence = (
                        probabilities[predicted_index].item()
                        * 100
                    )

                    st.session_state["prediction"] = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "probabilities": probabilities.tolist(),
                        "filename": uploaded_file.name
                    }

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )


# =====================================================
# DISPLAY RESULTS
# =====================================================

prediction = st.session_state.get("prediction")

if prediction is not None:

    st.write("")
    st.divider()

    st.html("""
    <div class="eyebrow">
        03 · MODEL OUTPUT
    </div>

    <h2 style="color:white">
        Analysis Results
    </h2>
    """)

    result_left, result_right = st.columns([1, 1])

    with result_left:

        st.html(f"""
        <div class="result-card">

            <div class="result-label">
                PREDICTED CLASS
            </div>

            <div class="result-value">
                {prediction["class"]}
            </div>

        </div>
        """)

        st.html(f"""
        <div class="metric-box">

            <div class="metric-label">
                MODEL CONFIDENCE
            </div>

            <div class="metric-value">
                {prediction["confidence"]:.2f}%
            </div>

        </div>
        """)

    with result_right:

        st.html("""
        <div class="card-heading">
            Class Probability Distribution
        </div>

        <div class="card-subtitle">
            Softmax probabilities produced by the model.
        </div>
        """)

        for index, class_name in enumerate(CLASS_NAMES):

            score = prediction["probabilities"][index]

            st.write(
                f"**{class_name}** — {score * 100:.2f}%"
            )

            st.progress(score)


# =====================================================
# RESEARCH DISCLAIMER
# =====================================================

st.write("")

st.warning("""
**Research Prototype Disclaimer**

This application is developed exclusively for educational
and research purposes. It is not a medical diagnostic system
and must not be used to make healthcare decisions.
Model predictions may be incorrect.
Always consult qualified healthcare professionals
for medical assessment.
""")


# =====================================================
# FOOTER
# =====================================================

st.html("""
<div class="footer">

    <b>NEUROSCAN AI</b> · Brain Tumor Detection Research Project

    <br>

    ResNet18 Architecture &nbsp;|&nbsp;
    224 × 224 Input &nbsp;|&nbsp;
    3 Classification Classes

    <br>

    Built for Educational and Research Purposes

</div>
""")
