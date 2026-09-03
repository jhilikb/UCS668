import streamlit as st
import numpy as np
import albumentations as A
from PIL import Image, ImageOps, ImageEnhance
import io
import math
import torch
import torch.nn.functional as F


# --- PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="Data Alchemist Lab",
    page_icon="🧪",
    layout="wide",
)

# Custom CSS for a slight gamified vibe
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .score-box {
        background-color: #1E1E24;
        color: #00FFCC;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        border: 2px solid #00FFCC;
    }
    .hint-box {
        background-color: #FFF3CD;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.title("🧪 The Data Alchemist Lab: Image Augmentation")
st.caption("Mission: Forge indestructible datasets to train unbeatable AI models so that they work even if the data is noisy, blur, unaligned.")

# --- SIDEBAR: MISSION CONTROL ---
st.sidebar.header("🛸 Mission Control")
mode = st.sidebar.radio("Select Lab Module:", ["1. Training Grounds (Explore)", "2. The Stress Test (Challenge)"])

# Sample image loader
@st.cache_data
def load_default_image():
    # Creating a simple colorful placeholder image if user doesn't upload one
    img = Image.new('RGB', (400, 400), color = (73, 109, 137))
    # Draw a simple shape so rotation/flipping is obvious
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([(100, 100), (300, 300)], fill=(255, 255, 0))
    d.polygon([(200, 50), (150, 100), (250, 100)], fill=(255, 0, 0))
    return img

uploaded_file = st.sidebar.file_uploader("Upload a subject image (Optional)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    base_image = Image.open(uploaded_file).convert("RGB")
else:
    base_image = load_default_image()

# --- MODULE 1: TRAINING GROUNDS ---
if mode == "1. Training Grounds (Explore)":
    st.subheader("🎛️ Experiment Chamber")
    st.write("Adjust the sliders below to alter the image. Watch how the underlying code updates in real-time!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🛠️ Augmentation Controls")
        
        # Spatial Transforms
        st.markdown("**Spatial Changes (Geometry)**")
        angle = st.slider("Rotate Image (Degrees)", -180, 180, 0)
        flip = st.checkbox("Horizontal Flip")
        
        # Pixel-level Transforms
        st.markdown("---")
        st.markdown("**Pixel Changes (Color & Quality)**")
        brightness = st.slider("Brightness Adjustment", 0.5, 2.0, 1.0)
        blur = st.slider("Blur Intensity (Kernel Size)", 1, 15, 1, step=2)
        noise = st.slider("Add Glitch Noise", 0, 100, 0)

    # Building the dynamic Albumentations pipeline based on UI
    transform_list = []
    code_list = ["import albumentations as A", "transform = A.Compose(["]

    if angle != 0:
        transform_list.append(A.Rotate(limit=(angle, angle), p=1.0))
        code_list.append(f"    A.Rotate(limit=({angle}, {angle}), p=1.0),  # Rotates the canvas")
    if flip:
        transform_list.append(A.HorizontalFlip(p=1.0))
        code_list.append("    A.HorizontalFlip(p=1.0),  # Flips left-to-right")
    if brightness != 1.0:
        # Albumentations uses limit format around 0 (-0.5 to 0.5 for example)
        b_limit = brightness - 1.0
        transform_list.append(A.RandomBrightnessContrast(brightness_limit=(b_limit, b_limit), contrast_limit=0, p=1.0))
        code_list.append(f"    A.RandomBrightnessContrast(brightness_limit=({b_limit:.2f}, {b_limit:.2f}), contrast_limit=0, p=1.0),")
    if blur > 1:
        transform_list.append(A.Blur(blur_limit=(blur, blur), p=1.0))
        code_list.append(f"    A.Blur(blur_limit=({blur}, {blur}), p=1.0),  # Softens edges")
    if noise > 0:
        transform_list.append(A.GaussNoise(var_limit=(noise*2, noise*5), p=1.0))
        code_list.append(f"    A.GaussNoise(var_limit=({noise*2}, {noise*5}), p=1.0),  # Simulates camera static")

    code_list.append("])")
    
    # Apply transformation
    image_np = np.array(base_image)
    if transform_list:
        transform = A.Compose(transform_list)
        augmented = transform(image=image_np)
        output_image = Image.fromarray(augmented['image'])
    else:
        output_image = base_image

    with col2:
        st.markdown("### 👁️ Visual Output")
        sub_col1, sub_col2 = st.columns(2)
        sub_col1.image(base_image, caption="Original Data", use_container_width=True)
        sub_col2.image(output_image, caption="Augmented Variant", use_container_width=True)

    # --- CODE & EXPLANATION SECTION ---
    st.markdown("---")
    st.subheader("💻 The Matrix (Generated Code)")
    
    code_col, text_col = st.columns([1, 1])
    
    with code_col:
        st.code("\n".join(code_list), language="python")
    
    with text_col:
        st.markdown("**What's happening behind the scenes?**")
        if not transform_list:
            st.info("Move the sliders to start generating production-ready data science code!")
        else:
            st.markdown("""
            * **`A.Compose`**: Think of this as an assembly line. It stitches all your chosen modifications together so they execute sequentially on every image.
            * **`p=1.0`**: This stands for *Probability*. In production training, you'd set this to something like `0.5`, meaning only 50% of your images get this specific change, creating a healthy mix of clean and messy data.
            """)

# --- MODULE 2: THE STRESS TEST (GAMIFIED CHALLENGE) ---
elif mode == "2. The Stress Test (Challenge)":
    st.subheader("🏆 The Dataset Resilience Challenge")
    st.markdown("""
    Your AI model is deployment-ready, but it's prone to failing in real-world messy conditions. 
    **Your Goal:** Create a pipeline that makes the dataset diverse enough without making it completely unrecognizable.
    """)

    # Gamified Score Logic
    st.markdown("### 📊 Live Pipeline Performance Profile")
    
    c1, c2, c3 = st.columns(3)
    
    # User selects options
    with c1:
        st.markdown("**Choose Rotations**")
        rot_choice = st.selectbox("Max Angle Allowable", [0, 15, 45, 90, 180])
    with c2:
        st.markdown("**Choose Distortions**")
        noise_choice = st.checkbox("Inject Night-time Grain (Noise)")
        blur_choice = st.checkbox("Inject Rainy/Windshield Blur")
    with c3:
        st.markdown("**Choose Color Shifts**")
        color_choice = st.slider("Color Shift Intensity", 0.0, 1.0, 0.0)

    # Calculate Game Scores dynamically
    robustness_score = 0
    readability_score = 100
    
    if rot_choice > 0:
        robustness_score += 25 if rot_choice <= 45 else 40
        readability_score -= 10 if rot_choice <= 45 else 25
    if noise_choice:
        robustness_score += 20
        readability_score -= 15
    if blur_choice:
        robustness_score += 25
        readability_score -= 20
    if color_choice > 0:
        robustness_score += int(color_choice * 30)
        readability_score -= int(color_choice * 35)

    # Clamping scores between 0 and 100
    robustness_score = min(max(robustness_score, 0), 100)
    readability_score = min(max(readability_score, 0), 100)
    
    # Render "Gamified Dashboard"
    score_col1, score_col2 = st.columns(2)
    with score_col1:
        st.markdown(f"<div class='score-box'>🛡️ AI ROBUSTNESS SCORE: {robustness_score}/100</div>", unsafe_allow_html=True)
        st.caption("How well your model handles unexpected real-world chaotic variations.")
    with score_col2:
        st.markdown(f"<div class='score-box'>👁️ HUMAN READABILITY: {readability_score}/100</div>", unsafe_allow_html=True)
        st.caption("Can a human still recognize the object? If this hits too low, your data is just garbage text.")

    # Win/Lose Condition Feedback
    st.markdown("#### Mission Evaluation Status:")
    if robustness_score >= 60 and readability_score >= 50:
        st.success("🎉 **SUCCESS:** You found the sweet spot! Your model can survive bad weather and weird angles without losing its mind.")
    elif readability_score < 50:
        st.error("🚨 **CRITICAL FAILURE:** Human readability is too low. You over-augmented the images to the point where they look like broken static. Tone it down!")
    else:
        st.warning("⚠️ **WEAK SECURITY:** Your robustness score is too low. The model will pass perfect lab tests but fail completely the minute a cloud blocks the sun.")




    # Utility: convert PIL to tensor
    def pil_to_tensor(img):
        return torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.

    # Mixup
    def mixup(x1, y1, x2, y2, alpha=1.0):
        lam = torch.distributions.Beta(alpha, alpha).sample()
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        return x, y

    # CutMix
    def cutmix(x1, y1, x2, y2):
        lam = torch.rand(1).item()
        _, H, W = x1.shape
        
        cut_rat = math.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = torch.randint(W, (1,)), torch.randint(H, (1,))
        x1[:, cy:cy+cut_h, cx:cx+cut_w] = x2[:, cy:cy+cut_h, cx:cx+cut_w]
        lam = 1 - (cut_w * cut_h) / (W * H)
        y = lam * y1 + (1 - lam) * y2
        return x1, y

    # --- Streamlit UI ---
    st.subheader("Advanced Augmentation Demo (Sample Images)")

    # Load two sample images
    img1 = Image.open("park.jpeg").resize((128,128))
    img2 = Image.open("traffic.jpeg").resize((128,128))

    x1 = pil_to_tensor(img1)
    x2 = pil_to_tensor(img2)

    # One-hot labels (example: Cat=0, Dog=1)
    y1 = F.one_hot(torch.tensor(0), num_classes=2).float()
    y2 = F.one_hot(torch.tensor(1), num_classes=2).float()

    # Mixup
    x_mix, y_mix = mixup(x1, y1, x2, y2)
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(x_mix.permute(1,2,0).numpy(), caption="Mixup Image")
    with col2:
        st.subheader("Mixup Explanation")
        st.write(f"**Label distribution:** {y_mix.numpy()}")
        st.write("Mixup blends two images and their labels. "
                 "Here the two images are combined with a weighted ratio, "
                 "so the model learns smooth decision boundaries.")

    # CutMix demo
    x_cut, y_cut = cutmix(x1.clone(), y1, x2, y2)
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(x_cut.permute(1,2,0).numpy(), caption="CutMix Image")
    with col2:
        st.subheader("CutMix Explanation")
        st.write(f"**Label distribution:** {y_cut.numpy()}")
        st.write("CutMix pastes a patch from one image into another. "
                 "Labels are mixed proportionally to the area replaced, "
                 "making the model robust to occlusion and partial information.")
