import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
import torch
import torch.nn as nn


st.title("📦 Object Detection ")

# Tabs for different concepts

tab1, tab2, tab3 = st.tabs([
    "Feature Extraction", "Bounding Boxes", "Losses",
    ])
# ---------------- Feature Extraction ----------------
with tab1:
    st.header("Feature Extraction with any backbone (ResNet)")
    

    uploaded2 = st.file_uploader("Upload image", type=["png","jpg"])
    if uploaded2:
        img = Image.open(uploaded2).convert("RGB")
        st.image(img, caption="Original Image")

        # Preprocessing (ImageNet normalization)
        transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ])
        x = transform(img).unsqueeze(0)  # (1,3,224,224)

        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        resnet.eval()

        # Extract features from 2nd or 3rd layer
        # Let's hook into layer2 (third block)
        # Truncate model to layer1 (early features)
        # Define sequential modules up to layer2 and layer4
        model_layer1 = torch.nn.Sequential(
    resnet.conv1,
    resnet.bn1,
    resnet.relu,
    resnet.maxpool,
    resnet.layer1,
    
)

        model_layer4 = torch.nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4)

            # Forward pass
        feat2 = model_layer1(x)   # after layer2
        feat4 = model_layer4(x)   # after layer4 # shape (1, C, H, W)

        # st.write("Extracted feature shape:", feat.shape)
    # else:
    #     feat = torch.randn(1, 64, 160, 160)
        

        
        
        

        
        st.subheader("Feature Map Visualization (Channel 0)")

        fig, ax = plt.subplots(2,4, figsize=(12,6))

        for i in range(4):
            fmap2 = feat2[0,i].detach().cpu().numpy()
            fmap2 = (fmap2 - fmap2.min()) / (fmap2.max() - fmap2.min() + 1e-5)
            ax[0,i].imshow(fmap2, cmap='viridis')
            ax[0,i].set_title(f"Layer1 Ch {i}")
            ax[0,i].axis("off")

        for i in range(4):
            fmap4 = feat4[0,i].detach().cpu().numpy()
            fmap4 = (fmap4 - fmap4.min()) / (fmap4.max() - fmap4.min() + 1e-5)
            ax[1,i].imshow(fmap4, cmap='viridis')
            ax[1,i].set_title(f"Layer4 Ch {i}")
            ax[1,i].axis("off")

        st.pyplot(fig)


            # # Augmented feature map
            # fmap1 = y[0,i].detach().cpu().numpy()
            # fmap1 = (fmap1 - fmap1.min()) / (fmap1.max() - fmap1.min() + 1e-5)
            # axes[i,1].imshow(fmap1, cmap="viridis")
            # axes[i,1].set_title(f"Channel {i} - Augmented")
            # axes[i,1].axis("off")

        # st.pyplot(fig) # <-- Streamlit display
   

    st.subheader("Backbone Comparison")

    # Show table
    st.markdown("""
    | Backbone | Design Goal | Feature Extraction | Strengths | Weaknesses |
    |----------|-------------|--------------------|-----------|-------------|
    | YOLO (CSPDarknet) | Real-time speed | Lightweight CNN | Fast, efficient | Misses small objects |
    | RetinaNet (ResNet+FPN) | Accuracy | Multi-scale pyramid | Robust small-object detection | Slower |
    | Faster R-CNN (ResNet+RPN) | High accuracy | Two-stage proposals | Precise localization | Heavy compute |
    | EfficientDet (EfficientNet+BiFPN) | Balanced | Bi-directional FPN | Scalable, efficient | Complex design |
    | DETR (ResNet+Transformer) | End-to-end | CNN + Transformer | No anchors | Slow convergence |
    """)

    # Expanders for detailed explanations + images
    with st.expander("YOLO Backbone"):
        
        st.write("YOLO uses CSPDarknet, optimized for speed. Features are directly fed into detection heads.")

    with st.expander("RetinaNet Backbone"):
        
        st.write("RetinaNet uses ResNet + FPN to build multi-scale features. Better accuracy for small objects.")

    with st.expander("Faster R-CNN Backbone"):
        
        st.write("Faster R-CNN uses a two-stage process: proposals then classification. High accuracy but slower.")

    with st.expander("EfficientDet Backbone"):
       
        st.write("EfficientDet combines EfficientNet + BiFPN for balanced speed and accuracy.")

    with st.expander("DETR Backbone"):
       
        st.write("DETR uses ResNet + Transformer. End-to-end detection without anchors, but slower training.")

# ---------------- Bounding Boxes ----------------
with tab2:
    st.header("Bounding Boxes & IoU")
    st.markdown("""
    Bounding boxes are defined by coordinates \((x_{min}, y_{min}, x_{max}, y_{max})\).

    **Intersection over Union (IoU):**
    

\[
    IoU = \frac{Area_{intersection}}{Area_{union}}
    \]


    """)
    x1 = st.slider("Box1 x_min",0,100,10)
    y1 = st.slider("Box1 y_min",0,100,10)
    x2 = st.slider("Box1 x_max",0,100,50)
    y2 = st.slider("Box1 y_max",0,100,50)

    x3 = st.slider("Box2 x_min",0,100,20)
    y3 = st.slider("Box2 y_min",0,100,20)
    x4 = st.slider("Box2 x_max",0,100,60)
    y4 = st.slider("Box2 y_max",0,100,60)

    inter_area = max(0,min(x2,x4)-max(x1,x3)) * max(0,min(y2,y4)-max(y1,y3))
    union_area = (x2-x1)*(y2-y1)+(x4-x3)*(y4-y3)-inter_area
    iou = inter_area/union_area if union_area>0 else 0
    st.metric("IoU", f"{iou:.2f}")

# ---------------- Loss Functions ----------------
with tab3:
    st.header("Loss Functions")
    

    st.markdown("Object detection uses **multi-task loss**:")

    st.markdown("- **Localization loss** (Smooth L1):")
    st.latex(r"""
    L_{loc}(x) = 
    \begin{cases} 
    0.5x^2 & |x| < 1 \\ 
    |x|-0.5 & \text{otherwise} 
    \end{cases}
    """)
    st.markdown(
        r"Where **$x = t_i - v_i$** is the difference (error) between the predicted "
        r"bounding box offset ($t_i$) and the ground-truth offset ($v_i$)."
    )

    st.markdown("- **Classification loss** (Cross-Entropy):")
    st.latex(r"L_{cls} = -\sum y_i \log(\hat{y}_i)")
    st.markdown(
        r"Where **$y_i$** is the ground-truth target label ($1$ for the correct class, $0$ otherwise) "
        r"and **$\hat{y}_i$** is the predicted probability for class $i$."
    )
    x = np.linspace(-3,3,100)
    smooth_l1 = np.where(np.abs(x)<1,0.5*x**2,np.abs(x)-0.5)
    l2 = x**2

    fig, ax = plt.subplots()
    ax.plot(x,smooth_l1,label="Smooth L1")
    ax.plot(x,l2,label="L2")
    ax.legend(); st.pyplot(fig)

