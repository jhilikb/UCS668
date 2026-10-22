import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Deep Learning Segmentation ", layout="wide")

# Custom CSS to elevate UI
st.html("""
<style>
    .theory-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-bottom: 20px; }
    .math-text { font-family: monospace; font-size: 1.1rem; background-color: #262730; color: #fffd82; padding: 2px 6px; border-radius: 4px; }
</style>
""")

st.title("🔬 Deep Learning Image Segmentation")

# Dynamic Lesson Navigation
module = st.sidebar.radio("Navigate Modules", [
    "1. Semantic vs. Instance Segmentation",
    "2. Deep Learning Architectures (U-Net)",
    "3. Loss Functions Math Sandbox",
    "4. Metrics & Evaluation Sandbox"
])

# 🛠️ Helper Data Generator for Modules
@st.cache_data
def get_mock_images():
    # Generate a mock photo containing two distinct objects (e.g., cell nuclei)
    img = np.ones((300, 600, 3), dtype=np.uint8) * 40
    cv2.circle(img, (200, 150), 50, (100, 200, 100), -1) # Object A
    cv2.circle(img, (400, 150), 50, (100, 200, 100), -1) # Object B
    
    # Semantic ground truth (Both objects share label 1)
    sem_mask = np.zeros((300, 600), dtype=np.uint8)
    cv2.circle(sem_mask, (200, 150), 50, 1, -1)
    cv2.circle(sem_mask, (400, 150), 50, 1, -1)
    
    # Instance ground truth (Object A = 1, Object B = 2)
    inst_mask = np.zeros((300, 600), dtype=np.uint8)
    cv2.circle(inst_mask, (200, 150), 50, 1, -1)
    cv2.circle(inst_mask, (400, 150), 50, 2, -1)
    
    return img, sem_mask, inst_mask

img, sem_mask, inst_mask = get_mock_images()

# ==========================================
# MODULE 1: SEMANTIC VS INSTANCE
# ==========================================
if module == "1. Semantic vs. Instance Segmentation":
    st.header("Module 1: Defining the Boundaries")
    
    st.html("""
    <div class="theory-box">
    <b>What is Image Segmentation?</b><br>
    Classification tells you <i>what</i> is in an image. Object detection tells you <i>where</i> it is via bounding boxes. 
    <b>Segmentation</b> predicts labels at the <b>individual pixel level</b>, creating a precise mask.
    </div>
    """)
    
    st.subheader("Interactive Visualizer: Spot the Paradigm Shift")
    st.write("Toggle the views below to observe how identical visual objects are treated by different systems.")

    mode = st.radio("Choose Segmentation Strategy:", ["Original Image", "Semantic Segmentation", "Instance Segmentation"])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if mode == "Original Image":
            st.image(img, caption="Two distinct biological items under a microscope", use_container_width=True)
        elif mode == "Semantic Segmentation":
            # Color both items identically
            viz = np.zeros_like(img)
            viz[sem_mask == 1] = [0, 255, 0]
            st.image(viz, caption="Semantic Mask: Grouped purely by class ('Cell')", use_container_width= True)
        else:
            # Color items distinctly
            viz = np.zeros_like(img)
            viz[inst_mask == 1] = [0, 255, 0]   # Cell #1 Green
            viz[inst_mask == 2] = [255, 0, 128] # Cell #2 Pink
            st.image(viz, caption="Instance Mask: Individual tracks for distinct entities ('Cell #1' vs 'Cell #2')", use_container_width= True)
            
    with col2:
        st.markdown("### Core Differences")
        if mode == "Semantic Segmentation":
            st.info("**Rule:** Treats all pixels of the same class as a single amorphous blob. Great for background styling, sky/road identification, and medical tissue tracking.")
        elif mode == "Instance Segmentation":
            st.success("**Rule:** Detects objects first, then separates individual pixel tracks inside each boundary. Critical for autonomous driving (tracking individual pedestrians) and crowd counting.")
        else:
            st.write("Select a mode to unpack the concepts.")
    st.image("seg.jpeg", caption="A sample example...")
# ==========================================
# MODULE 2: DEEP LEARNING STRUCTURES (CODE & DIMENSIONS)
# ==========================================
elif module == "2. Deep Learning Architectures (U-Net)":
    st.header("Module 2: Inside the Code Architecture — The Symmetrical U-Net")
    
    st.html("""
    <div class="theory-box">
    <b>The Architecture Blueprint:</b><br>
    A U-Net consists of an <b>Encoder</b> (to compress features) and a <b>Decoder</b> (to expand features back to the original pixel map). 
    Below is the literal PyTorch structure. Use the simulator to pass a tensor through the network and track its shapes.
    </div>
    """)
    
    # Live Interactive Controls
    st.subheader("⚙️ Input Tensor Dimension Simulator")
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        batch_size = st.slider("Batch Size (N)", min_value=1, max_value=16, value=4)
        in_channels = st.selectbox("Input Channels (C) [e.g., RGB vs Grayscale]", options=[1, 3], index=1)
    with col_input2:
        img_dim = st.selectbox("Base Image Resolution (H x W)", options=[128, 256, 512], index=1)

    st.markdown("---")
    
    # Dynamic Math & Shapes calculation based on student inputs
    enc1 = img_dim // 2
    enc2 = enc1 // 2
    dec1 = enc2 * 2
    out_dim = dec1 * 2

    # Two columns: Code on the left, Visual Tensor States on the right
    code_col, visual_col = st.columns([1, 1])
    
    with code_col:
        st.subheader("🧑‍💻 PyTorch Architectural Code")
        st.code(f"""
import torch
import torch.nn as nn

class UNetCore(nn.Module):
    def __init__(self):
        super().__init__()
        # --- ENCODER (Downsampling) ---
        self.enc_conv = nn.Conv2d({in_channels}, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- BOTTLENECK ---
        self.bottleneck = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # --- DECODER (Upsampling) ---
        self.upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_conv = nn.Conv2d(64, 1, kernel_size=1) # 1 Output Mask Class

    def forward(self, x):
        # Tracking internal shapes interactively...
        x=self.enc_conv(x)
        x=self.pool(x)
        x=self.bottleneck(x)
        x=self.pool(x)
        x=self.upconv(x)
        x=self.upconv2(x)
        x=self.dec_conv(x)
        return x
        """, language="python")

    with visual_col:
        st.subheader("📦 Live Tensor Shape Tracking")
        st.write("Watch how the shape array `[Batch, Channels, Height, Width]` morphs at each programmatic layer:")
        
        # Stage 1
        st.info(f"**1. Raw Input Tensor passed to Model:**\n`[{batch_size}, {in_channels}, {img_dim}, {img_dim}]`")
        
        # Stage 2
        st.markdown("⬇️ *Executed: `self.enc_conv(x)` followed by `self.pool(x)`*")
        st.warning(f"**2. After Encoder Downsampling:**\n`[{batch_size}, 64, {enc1}, {enc1}]`  \n*(Notice spatial size cut in half, channels increased!)*")
        
        # Stage 3
        st.markdown("⬇️ *Executed: `self.bottleneck(x)` followed by `self.pool(x)`*")
        st.error(f"**3. At Deepest Latent Bottleneck:**\n`[{batch_size}, 128, {enc2}, {enc2}]`  \n*(Resolution is at its lowest point. Spatial traits are abstract.)*")
        
        # Stage 4
        st.markdown("⬇️ *Executed: `self.upconv(x)` [Transposed Convolution]*")
        st.success(f"**4. After first Decoder Upsampling:**\n`[{batch_size}, 64, {dec1}, {dec1}]`  \n*(Resolution is climbing back up dynamically!)*")
        
        # Stage 5
        st.markdown("⬇️ *Executed: `self.dec_conv(x)` [1x1 Convolution mapping to binary mask]*")
        st.metric(label="🏁 Final Target Mask Output Shape", value=f"[{batch_size}, 1, {out_dim}, {out_dim}]")
        st.caption("Perfect match! The output spatial shape perfectly maps back to the student's initial input dimensions.")

    st.header(" The Symmetrical U-Net with Skip Connections")
    
    st.html("""
    <div class="theory-box">
    <b>The Skip Connection Breakthrough:</b><br>
    As an image passes through the contracting encoder, spatial features (like crisp edges) are systematically destroyed. 
    A <b>Skip Connection</b> copies the high-resolution feature maps from the Encoder and <b>concatenates</b> them directly with the upsampled feature maps in the Decoder. 
    This allows the model to retain pixel-perfect localization!
    </div>
    """)
    st.image("net.png", caption="A sample network...")
    # Live Interactive Controls for Tensor Shapes
    st.subheader("⚙️ Input Tensor Dimension Simulator")
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        batch_size = st.slider("Batch Size for U-Net with skip (N)", min_value=1, max_value=16, value=4)
        in_channels = st.selectbox("Input Channels for U-Net with skip(C)", options=[1, 3], index=1)
    with col_input2:
        img_dim = st.selectbox("Base Image Resolution for U-Net with skip (H x W)", options=[128, 256, 512], index=1)

    st.markdown("---")
    
    # Calculate dimensional mechanics dynamically based on student slider input
    half_dim = img_dim // 2
    
    # Column splits for clear code-to-dimension matching
    code_col, visual_col = st.columns([1.1, 0.9])
    
    with code_col:
        st.subheader("🧑‍💻 PyTorch U-Net with Concatenation")
        st.code(f"""
import torch
import torch.nn as nn

class UNetWithSkips(nn.Module):
    def __init__(self):
        super().__init__()
        # --- Encoder Stage ---
        self.enc_conv = nn.Conv2d({in_channels}, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Bottleneck Stage ---
        self.bottleneck = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # --- Decoder Stage ---
        self.upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1) 

    def forward(self, x):
        # 1. Raw Input Tensor
        # x shape: [{batch_size}, {in_channels}, {img_dim}, {img_dim}]
        
        # 2. Save this feature map BEFORE pooling! (Skip Connection Source)
        x_skip = self.enc_conv(x) 
        
        # 3. Downsample to the Bottleneck
        x_pool = self.pool(x_skip)
        x_bottle = self.bottleneck(x_pool)
        
        # 4. Upsample the low-res feature map
        x_up = self.upconv(x_bottle)
        
        # 5. THE SKIP CONNECTION: Concatenate along Channel Dim (dim=1)
        x_concat = torch.cat([x_up, x_skip], dim=1)
        
        # 6. Final Convolution to target mask output
        out = self.dec_conv(x_concat)
        return out
        """, language="python")

    with visual_col:
        st.subheader("📦 Live Tensor Shape Tracking")
        st.write("Track how channels double instantly when the skip connection merges:")
        
        # Step 1
        st.info(f"**1. Input Tensor:**\n`[{batch_size}, {in_channels}, {img_dim}, {img_dim}]`")
        
        # Step 2
        st.warning(f"**2. Saved Skip Feature Map (`x_skip`):**\n`[{batch_size}, 64, {img_dim}, {img_dim}]`  \n*Kept in memory before downsampling.*")
        
        # Step 3
        st.error(f"**3. Bottleneck Layer (`x_bottle`):**\n`[{batch_size}, 128, {half_dim}, {half_dim}]`  \n*Resolution is halved, channel richness is deep.*")
        
        # Step 4
        st.success(f"**4. Upsampled Map (`x_up`):**\n`[{batch_size}, 64, {img_dim}, {img_dim}]`  \n*Spatial size is back to original, but details are blurry.*")
        
        # Step 5 - The Core Concept
        st.markdown("🧩 **Executing `torch.cat([x_up, x_skip], dim=1)`**")
        st.html(
            f"<div style='background-color:#ffe6e6; padding:12px; border-radius:5px; border-left: 5px solid #ff4b4b; font-family:monospace;'>"
            f"<b>5. Concatenated Tensor (x_concat):</b><br>"
            f"[{batch_size}, 64 + 64, {img_dim}, {img_dim}]<br>"
            f"➔ <b>[{batch_size}, 128, {img_dim}, {img_dim}]</b>"
            f"</div>"
        )
        st.caption("Look at the channels! The 64 abstract channels from the decoder are stitched side-by-side with the 64 crisp spatial channels from the encoder.")
        
        # Step 6
        st.metric(label="🏁 Final Output Segmentation Mask Shape", value=f"[{batch_size}, 1, {img_dim}, {img_dim}]")
        st.caption("The final conv filters the combined 128 channels down to a single binary mask mapping individual pixels.")
    # ==========================================

# ==========================================
# MODULE 3: LOSS FUNCTIONS & CLASS IMBALANCE (VISUAL BREAKDOWN)
# ==========================================
elif module == "3. Loss Functions Math Sandbox":
    st.header("Module 3: Fighting Class Imbalance via Loss Design")
    
    st.html("""
    <div class="theory-box">
    <b>The Classroom Challenge: The Lazy AI Trap</b><br>
    Imagine a tiny cellular structure that takes up only 1% of a microscope scan (100 pixels out of 10,000 total). 
    If a model goes rogue and predicts 100% background, pixel accuracy reads a deceptive <b>99% accuracy</b>! 
    Let's watch how different loss functions process this error mathematically.
    </div>
    """)
    
    # Mathematical Formulas displayed clearly for the lecture
    st.subheader("📚 The Mathematical Core")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("**Pixel-by-Pixel: Binary Cross-Entropy**")
        st.latex(r"\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]")
    with f_col2:
        st.markdown("**Region-Overlap: Dice Loss**")
        st.latex(r"\mathcal{L}_{Dice} = 1 - \frac{2 \sum y_i \hat{y}_i + \epsilon}{\sum y_i + \sum \hat{y}_i + \epsilon}")

    st.markdown("---")
    
    # Interactive Controls Section
    st.subheader("🎛️ Live Error Simulator Controls")
    st.write("Simulate a grid layout containing **100 true target pixels** ($y_i=1$) and **9,900 background pixels** ($y_i=0$).")
    
    missed_pixels = st.slider(
        "Select how many target pixels the model missed (False Negatives):", 
        min_value=0, max_value=100, value=70, step=10
    )
    
    # Dynamic Simulation Computations
    total_pixels = 10000
    gt_target = 100
    gt_bg = 9900
    
    tp = gt_target - missed_pixels  # True Positives (Found target pixels)
    fn = missed_pixels              # False Negatives (Missed target pixels)
    fp = 0                          # Assume no false alarms for a clean isolation test
    tn = gt_bg                      # True Negatives (Correctly identified background)
    
    # 1. Compute BCE Math Step-by-Step
    eps = 1e-7  # Prevent log(0)
    # Target pixel loss contributions (where y=1)
    target_loss_sum = -(tp * np.log(1.0 - eps) + fn * np.log(eps))
    # Background pixel loss contributions (where y=0)
    bg_loss_sum = -(tn * np.log(1.0 - eps) + fp * np.log(eps))
    
    raw_bce = (target_loss_sum + bg_loss_sum) / total_pixels
    # Scaled gracefully for classroom display visibility
    display_bce = max(0.005, float(raw_bce * 0.05 + (fn * 0.003)))
    
    # 2. Compute Dice Math Step-by-Step
    intersection = tp
    total_predicted = tp + fp
    total_ground_truth = gt_target
    
    dice_coefficient = (2.0 * intersection) / (total_ground_truth + total_predicted + eps)
    dice_loss = 1.0 - dice_coefficient

    st.markdown("---")
    
    # The Live Computational Screen Output
    st.subheader("🖥️ Real-Time Computational Grid Feedback")
    
    # Split layout: Visual Grid Simulation vs Live Mathematical Readout
    viz_col, calculation_col = st.columns([1, 1])
    
    with viz_col:
        st.markdown("### Simulated Mask Canvas")
        # Draw a quick visual representation matrix for the student
        canvas = np.zeros((100, 100, 3), dtype=np.uint8) + 40  # Dark gray background
        
        # Draw ground truth region boundaries as a reference outline
        cv2.rectangle(canvas, (35, 35), (65, 65), (255, 255, 255), 1) 
        
        # Color fill based on the current model error profile slider state
        if tp > 0:
            # Color correctly found pixels green
            cv2.rectangle(canvas, (36, 36), (36 + int(tp // 3.5), 64), (46, 204, 113), -1)
        if fn > 0:
            # Color missed target pixels bright red
            cv2.rectangle(canvas, (64 - int(fn // 3.5), 36), (64, 64), (231, 76, 60), -1)
            
        st.image(canvas, width=320, caption="Live Pixel Map (Green = Hit, Red = Missed Target, Black = Background)")
        
        # Print active counts so students follow the variable numbers
        st.markdown(f"""
        **Current Pixel Quantities:**
        *   🟩 **True Positives (Hits):** `{tp}` pixels
        *   🟥 **False Negatives (Misses):** `{fn}` pixels
        *   ⬛ **True Negatives (Background):** `{tn}` pixels
        """)

    with calculation_col:
        st.markdown("### The Live Scoreboard")
        
        c1, c2 = st.columns(2)
        c1.metric(label="Binary Cross-Entropy Loss", value=f"{display_bce:.4f}")
        c2.metric(label="Dice Loss Score", value=f"{dice_loss:.4f}", delta="🚨 High Error Penalty" if dice_loss > 0.5 else None, delta_color="inverse")
        
        st.markdown("### 🔍 The Math Breakdown Explanation")
        
        # Dynamic commentary box that adjusts text context based on student slider value
        if missed_pixels >= 70:
            st.error(f"""
            **Why is BCE failing here?**  
            Even though the model missed **{missed_pixels}%** of the target, BCE remains tiny (`{display_bce:.4f}`) because the 9,900 background pixels are drowning out the error calculation. The model feels no incentive to change.
            
            **Why is Dice working?**  
            Dice looks at region volume. The intersection is down to `{intersection}`, causing the loss to jump up to **`{dice_loss:.4f}`**. This forces strong gradients during training backpropagation!
            """)
        elif missed_pixels == 0:
            st.success("Perfect spatial fit! Both loss metrics hit optimal alignment scores because there are zero red pixels on the grid canvas.")
        else:
            st.warning(f"Partial fit state. As the red missing region area scales down, notice that Dice loss drops in clean, linear proportion to the actual shape mismatch.")
    with st.expander("How are multi class labels handled"):
        st.write("**While binary labels are 0 and 1 and allows computing dice scores. "
                 "y will be 0,1,2,3 etc in clase of multi class labels.\n\n"
                 "We hence need to convert our target into one-hot encoding i.e. for a 25 x 25 image where each pixel tells the label 0-4, we now have 5 x 25 x 25. "
                 "So pixel 1,1 will now look like 1,0,0,0,0 if it belong to class 0 and 0,0,1,0,0 if it belongs to class 2")

# ==========================================
# MODULE 4: METRICS & EVALUATION SANDBOX
# ==========================================
elif module == "4. Metrics & Evaluation Sandbox":
    st.header("Module 4: Calculating Success — Accuracy vs IoU")
    
    st.markdown("""
    We use two primary metrics to judge an AI segmentation pipeline:
    *   **Pixel Accuracy:** Total correctly predicted pixels divided by overall pixels.
    *   **Intersection over Union (IoU / Jaccard Index):** The overlap area divided by the total combined area.
    """)
    
    st.latex(r"\text{IoU} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives} + \text{False Negatives}}")
    
    st.subheader("Interactive Confusion Matrix Simulator")
    st.write("Simulate a prediction grid. Tweak the model error profiles below to see why raw accuracy is a dangerous metric in computer vision.")
    
    tp = st.slider("True Positives (Correctly Found Target Pixels):", 10, 100, 80)
    fp = st.slider("False Positives (Background Mistakenly Labeled as Target):", 0, 100, 20)
    fn = st.slider("False Negatives (Actual Target Missed by Model):", 0, 100, 10)
    tn = 1000  # Assume massive background
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    iou = tp / (tp + fp + fn) * 100
    
    m1, m2 = st.columns(2)
    m1.metric("Overall Pixel Accuracy Score", f"{accuracy:.2f}%")
    m2.metric("True Intersection over Union (IoU)", f"{iou:.2f}%")
    
    if accuracy > 95 and iou < 70:
        st.error("🚨 **Classic Trap Caught!** Notice your overall accuracy looks nearly perfect (95%+), but your IoU reveals that the predicted mask shape is heavily distorted. Your students can interactively see why IoU is mandatory for robust evaluation.")