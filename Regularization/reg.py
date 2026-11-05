import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import pandas as pd


st.set_page_config(page_title="Interactive DL Concepts", layout="wide")

st.title("Deep Learning Concepts Explorer")
def propagate_variance(init, layers=10, fan_in=128, fan_out=64):
    variances = []
    x = torch.randn(1000, fan_in)  # input batch
    for l in range(layers):
        layer = nn.Linear(fan_in, fan_out)
        if init == "Xavier":
            nn.init.xavier_uniform_(layer.weight)
        elif init == "He":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        elif init == "Uniform":
            nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        else:
            nn.init.normal_(layer.weight, mean=0, std=0.05)
        x = layer(x)
        variances.append(x.var().item())
        x = torch.relu(x)  # apply nonlinearity
        fan_in=fan_out
    return variances
tabs = st.tabs([
    "Weight Initialization",
    "Image Normalization",
    "Feature Normalization",
    "Feature Augmentation",
    "Regularization"
])

# --- Tab 1: Weight Initialization ---
with tabs[0]:




    details = {
        "Xavier": """**Concept**: Proposed by Glorot & Bengio (2010), Xavier initialization is designed for sigmoid/tanh activations.
    **Mechanism**: Balances variance of inputs and outputs by scaling weights with both fan_in and fan_out.
    **Benefit**: Prevents vanishing/exploding gradients, enabling deeper networks with smooth activations.""",

        "He": """**Concept**: Introduced by He et al. (2015), optimized for ReLU activations.
    **Mechanism**: Scales variance as 2/fan_in to account for half of inputs being zeroed by ReLU.
    **Benefit**: Keeps signals stable, reduces risk of dying ReLUs, and speeds up convergence.""",

        "Uniform": """**Concept**: Simple baseline, weights sampled uniformly from a fixed range.
    **Mechanism**: Typically [-a, a], chosen manually without variance scaling.
    **Benefit**: Easy to implement, but can cause unstable training in deeper networks if ranges aren’t tuned.""",

        "Normal": """**Concept**: Weights drawn from a Gaussian distribution with chosen mean and std.
    **Mechanism**: Often Normal(0, 0.05), not scaled to layer depth or activation type.
    **Benefit**: Provides randomness, but without scaling can lead to exploding/vanishing gradients."""
    }

    
    st.header("Weight Initialization")

    init_type = st.selectbox("Choose initialization", ["Xavier", "He", "Uniform", "Normal"])
    # Sliders for fan_in and fan_out
    fan_in = st.slider("Number of input units (fan_in)", min_value=16, max_value=512, value=128, step=16)
    fan_out = st.slider("Number of output units (fan_out)", min_value=16, max_value=512, value=64, step=16)
    layers = st.slider("Number of layers (layers)", min_value=1, max_value=15, value=4, step=1)

    # Create layer with chosen dimensions
    layer = nn.Linear(fan_in, fan_out)

    # Apply initialization
    if init_type == "Xavier":
        nn.init.xavier_uniform_(layer.weight)
        formula = r"Var(w) = 2 / (fan\_in + fan\_out)"
        code = "nn.init.xavier_uniform_(layer.weight)"
    elif init_type == "He":
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        formula = r"Var(w) = 2 / fan\_in"
        code = 'nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")'
    elif init_type == "Uniform":
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        formula = r"weights drawn from U(-a, a)"
        code = "nn.init.uniform_(layer.weight, a=-0.1, b=0.1)"
    else:
        nn.init.normal_(layer.weight, mean=0, std=0.05)
        formula = r"weights drawn from N(0, 0.05^2)"
        code = "nn.init.normal_(layer.weight, mean=0, std=0.05)"

    # Show math + code
    st.subheader("Initialization Formula")
    st.latex(formula)

    st.subheader("PyTorch Code")
    st.code(code, language="python")
    col1, col2 = st.columns(2)

    # Histogram in col1
    with col1:
        weights = layer.weight.detach().numpy()
        fig, ax = plt.subplots(figsize=(4,3))
        ax.hist(weights.flatten(), bins=30, color="skyblue", edgecolor="black")
        ax.set_title(f"{init_type} Weight Distribution")
        st.pyplot(fig)

    # Variance propagation in col2
    with col2:
        variances = propagate_variance(init_type,layers,fan_in,fan_out)
        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax2.plot(range(1, len(variances)+1), variances, marker="o", color="orange")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Activation Variance")
        ax2.set_title(f"{init_type} Variance Propagation")
        st.pyplot(fig2)

    # Explanation below
    st.subheader("Explanation")
    st.markdown(details[init_type])
    st.subheader("Why Variance Matters")
    st.markdown("""
    - Variance shows how activation magnitudes change across layers.
    - If variance shrinks → gradients vanish, network stops learning.
    - If variance explodes → gradients blow up, training diverges.
    - Xavier and He are designed to keep variance stable as they use input/output dims, making deep networks trainable.
    """)
    st.info("Xavier and He keep variance stable across layers, while Uniform/Normal may cause shrinkage or explosion.")
    



    
    
    

# --- Tab 2: Image Normalization ---
with tabs[1]:

    st.header("Image Normalization")

    st.markdown("""
    **Why normalize images?**
    - Keeps pixel values stable i.e. (0-1) or (-1-1) or zero centered for training.
    - Prevents exploding/vanishing gradients.
    - Matches expectations of pretrained models (e.g., VGG, ResNet).
    - Typically images are normalized such that they have zero mean and 1 standard deviation. This is accomplished by computing dataset mean and std. I=(I-mean)/std
    """)

    uploaded = st.file_uploader("Upload an image", type=["png","jpg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Original")

        choice = st.radio("Choose normalization", ["Scale [0,1]", "Scale [-1,1]", "ImageNet (VGG/ResNet)"])
        if choice == "Scale [0,1]":
            arr = np.array(img).astype(np.float32) / 255.0
            st.image(arr, caption="Normalized to [0,1]")
            st.code("arr = np.array(img).astype(np.float32) / 255.0", language="python")
            st.info("If the Image is 0-255 just convert to a 0-1 range without any other changes. ")
        elif choice == "Scale [-1,1]":
            arr = (np.array(img).astype(np.float32) / 255.0) * 2 - 1
            #st.image((arr+1)/2, caption="Normalized to [-1,1] (visualized back to [0,1])")
            st.image(np.clip(arr,0,1), caption="Normalized to [-1,1] ")
            st.code("import torchvision.transforms as T \n T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])", language="python")
            st.info("This is the normalization used for training restoration (image denoising, dehazing) models.")
        else:
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            arr = np.array(img).astype(np.float32)
            arr_norm = (arr/255.0 - mean) / std

            # For visualization, rescale back to [0,1]
            arr_vis = np.clip((arr_norm * std + mean), 0, 1)
            #st.image(arr_vis, caption="ImageNet Normalization (visualized back to [0,1])")
            st.image(np.clip(arr_norm,0,1) , caption="ImageNet Normalization ")

            st.code("import torchvision.transforms as T \n T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])", language="python")
            st.info("This is the normalization used in VGG and ResNet (ImageNet pretrained).")


# --- Tab 3: Feature Normalization ---
with tabs[2]:
    st.header("Feature Normalization")

    choice = st.radio("Choose normalization", ["BatchNorm", "InstanceNorm", "LayerNorm", "GroupNorm"])
    batch = st.slider("Select batch size (batch)", min_value=4, max_value=128, value=8, step=4)
    channels = st.slider("Number of channels (channels)", min_value=16, max_value=512, value=64, step=16)
    spatial = st.slider("Spatial resolution (S * S)", min_value=32, max_value=128, value=32, step=16)

    # Dummy input tensor
    x = torch.randn(batch, channels, spatial, spatial)  # (N=8, C=64, H=32, W=32)

    if choice == "BatchNorm":
        norm_layer = nn.BatchNorm2d(channels)
        explanation = """
        ### Batch Normalization
        - **What it does**: Normalizes activations across the batch and spatial dimensions.
        - **Running statistics**: Maintains a *running mean* and *running variance* during training.
          - During training: uses batch statistics.
          - During inference: uses the stored running mean/variance.
        - **Why it's used**: Reduces internal covariate shift, stabilizes training, speeds up convergence.
        - **When to use**: CNNs and feedforward networks with large batch sizes.
        - **Shape details**:
          - Input: (N, C, H, W)
          - Normalized per channel across N, H, W
          - Output: Same shape as input
        - **PyTorch code**:
          ```python
          nn.BatchNorm2d(num_features=C)
          ```
        """

    elif choice == "InstanceNorm":
        norm_layer = nn.InstanceNorm2d(channels)
        explanation = """
        ### Instance Normalization
        - **What it does**: Normalizes each sample independently, per channel.
        - **Statistics**: Uses per-instance mean and variance, no running stats.
        - **Why it's used**: Preserves style information, avoids batch statistics interfering with artistic features.
        - **When to use**: Style transfer, GANs, generative models.
        - **Shape details**:
          - Input: (N, C, H, W)
          - Normalized per channel per image
          - Output: Same shape as input
        - **PyTorch code**:
          ```python
          nn.InstanceNorm2d(num_features=C)
          ```
        """

    elif choice == "LayerNorm":
        x = torch.randn(batch, spatial**2)  # (N=8, Features=512)
        norm_layer = nn.LayerNorm(spatial**2)
        explanation = """
        ### Layer Normalization
        - **What it does**: Normalizes across all features of a single sample.
        - **Statistics**: Computed per sample, no running stats.
        - **Why it's used**: Works well with small batch sizes, stabilizes sequential models.
        - **When to use**: RNNs, Transformers, NLP tasks.
        - **Shape details**:
          - Input: (N, Features)
          - Normalized across feature dimension
          - Output: Same shape as input
        - **PyTorch code**:
          ```python
          nn.LayerNorm(normalized_shape=feature_dim)
          ```
        """

    else:
        norm_layer = nn.GroupNorm(num_groups=channels//4, num_channels=channels)
        explanation = """
        ### Group Normalization
        - **What it does**: Splits channels into groups and normalizes within each group.
        - **Statistics**: Computed per sample, per group. No running stats.
        - **Why it's used**: More stable than BatchNorm when batch sizes are small.
        - **When to use**: Small-batch CNN training, medical imaging, or resource-constrained setups.
        - **Shape details**:
          - Input: (N, C, H, W)
          - Normalized per group of channels, per sample
          - Output: Same shape as input
        - **PyTorch code**:
          ```python
          nn.GroupNorm(num_groups=32, num_channels=C)
          ```
        """

    # Apply normalization
    y = norm_layer(x)

    

    # Plot distributions before and after normalization
    st.subheader("Distribution Before vs After Normalization")
    fig, ax = plt.subplots(1, 2, figsize=(8,3))
    ax[0].hist(x.detach().numpy().flatten(), bins=50, color="skyblue")
    ax[0].set_title("Before Normalization")
    ax[1].hist(y.detach().numpy().flatten(), bins=50, color="orange")
    ax[1].set_title("After Normalization")
    st.pyplot(fig)

    # Show explanation
    st.markdown(explanation)

    # Show input/output shapes
    st.subheader("Input vs Output Shapes")
    st.write("Input shape:", tuple(x.shape))
    st.write("Output shape:", tuple(y.shape))

    # Plot distributions before and after normalization
    st.subheader("Statistics Before vs After Normalization")
    st.write("Before: mean =", x.mean().item(), ", std =", x.std().item())
    st.write("After: mean =", y.mean().item(), ", std =", y.std().item())

# --- Tab 4: Feature Augmentation ---
with tabs[3]:
    st.header("Feature Augmentation on ResNet18 Features")

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
        model_layer1 = torch.nn.Sequential(
    resnet.conv1,
    resnet.bn1,
    resnet.relu,
    resnet.maxpool,
    resnet.layer1
)

        # Forward pass
        feat = model_layer1(x)  # shape (1, C, H, W)

        st.write("Extracted feature shape:", feat.shape)
    # else:
    #     feat = torch.randn(1, 64, 160, 160)
        

        choice = st.radio("Choose augmentation", ["Dropout", "Noise Injection", "Feature Mixing", "Patch Dropout"])
        pval = st.slider("Select dropout (p)", min_value=0.1, max_value=0.6, value=0.2, step=0.1)
        nval = st.slider("Select noise level (sigma)", min_value=0.02, max_value=0.2, value=0.04, step=0.02)
        blend = st.slider("Select blend level (blend)", min_value=0.7, max_value=1.0, value=0.7, step=0.1)
        mixs = st.slider("Select blockdrop start(bstart) within the feature shape", min_value=0, max_value=50, value=10, step=1)
        mixe = st.slider("Select blockdrop end(bend) within the feature shape", min_value=0, max_value=50, value=15, step=1)
        
        

        if choice == "Dropout":
            layer = nn.Dropout(p=pval)
            y=layer(feat)
            explanation = """
            ### Dropout
            - Randomly sets ~50% of features to zero.
            - Prevents co-adaptation of neurons.
            - Common in fully-connected layers and CNNs.
            - Input/Output shape preserved.
            - PyTorch: `nn.Dropout(p=0.5)`
            """

        elif choice == "Noise Injection":
            noise = torch.randn_like(feat) * nval
            y = feat + noise
            explanation = """
            ### Noise Injection
            - Adds Gaussian noise to features.
            - Encourages robustness to perturbations.
            - Useful in adversarial defense and generalization.
            - Input/Output shape preserved.
            - PyTorch: `feat_noisy = feat + torch.randn_like(feat)*sigma`
            """

        elif choice == "Feature Mixing":
            x2 = feat+torch.randn_like(feat)
            lam = blend
            y = lam * feat + (1 - lam) * x2
            explanation = """
            ### Feature Mixing
            - Blends features from two samples.
            - Encourages smoother decision boundaries.
            - Used in Mixup / Manifold Mixup.
            - Input/Output shape preserved.
            - PyTorch: `feat_mix = blend*feat1 + (1-blend)*feat2`
            """

        else:  # Patch Dropout
            y = feat.clone()
            y[:, :, mixs:mixe, mixs:mixe] = 0  # zero out a patch
            explanation = """
            ### Patch Dropout (DropBlock style)
            - Zeros out contiguous patches of feature maps.
            - Forces CNNs to rely on multiple regions.
            - Useful for spatial regularization.
            - Input/Output shape preserved.
            - PyTorch (manual example): `feat[:, :, bstart:bstart+bend, bstart:bstart+bend] = 0`
            """

        st.markdown(explanation)

        # Show input/output stats
        # st.subheader("Statistics Before vs After Augmentation")
        # st.write("Before: mean =", round(x.mean().item(), 4), ", std =", round(x.std().item(), 4))
        # st.write("After: mean =", round(y.mean().item(), 4), ", std =", round(y.std().item(), 4))

        # Visualize one channel as heatmap
        
        st.subheader("Feature Map Visualization (Channel 0)")

        fig, axes = plt.subplots(8, 2, figsize=(8, 16))  # 8 rows, 2 columns

        for i in range(8):  # show 8 channels
            # Original feature map
            fmap = feat[0,i].detach().cpu().numpy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
            axes[i,0].imshow(fmap, cmap="viridis")
            axes[i,0].set_title(f"Channel {i} - Original")
            axes[i,0].axis("off")

            # Augmented feature map
            fmap1 = y[0,i].detach().cpu().numpy()
            fmap1 = (fmap1 - fmap1.min()) / (fmap1.max() - fmap1.min() + 1e-5)
            axes[i,1].imshow(fmap1, cmap="viridis")
            axes[i,1].set_title(f"Channel {i} - Augmented")
            axes[i,1].axis("off")

        st.pyplot(fig) # <-- Streamlit display
        # fig, ax = plt.subplots(1,2,figsize=(8,4))
        # ax[0].imshow(feat[0,0].detach().numpy(), cmap="viridis")
        # ax[0].set_title("Before Augmentation")
        # ax[1].imshow(y[0,0].detach().numpy(), cmap="viridis")
        # ax[1].set_title("After Augmentation")
        # st.pyplot(fig)

# --- Tab 5: Regularization ---
with tabs[4]:
    st.header("Regularization")
    

    with st.expander("What is regularization?"):
        st.write("Regularization techniques are methods used to prevent overfitting in machine learning by controlling model complexity")

    with st.expander("What are some regularization techniques"):
        st.write("Wait you covered most of them. \n\n"
                 "Feature augmentations like dropout,dropblock,noise injection,feature mixing.\n\n"
                 "Normalization techniques: Batch,Instance,Layer and Group.\n\n "
                 "Data Aumentation.\n\n"
                 "Others include L1/L2 penalties, early stopping, ensemble models, stochastic depths")



    # Define table data
    data = {
        "Technique": [
            "L1 Regularization", "L2 Regularization", "Dropout", "Noise Injection",
            "Data Augmentation", "BatchNorm", "LayerNorm", "GroupNorm",
            "Early Stopping", "Stochastic Depth", "DropBlock"
        ],
        "What it does": [
            "Adds absolute value penalty to weights → encourages sparsity.",
            "Adds squared penalty to weights → discourages large values.",
            "Randomly zeros activations during training.",
            "Adds Gaussian noise to inputs/features.",
            "Expands dataset via flips, rotations, crops.",
            "Normalizes activations per batch, uses running stats.",
            "Normalizes across features per sample.",
            "Normalizes per group of channels, per sample.",
            "Stops training when validation loss stops improving.",
            "Randomly skips residual blocks.",
            "Zeros contiguous patches in feature maps."
        ],
        "When to use": [
            "Sparse models, feature selection.",
            "Most deep nets, stabilizes training.",
            "CNNs, RNNs, Transformers.",
            "Robustness, adversarial defense.",
            "Computer vision, speech, NLP.",
            "Deep CNNs, large batches.",
            "Transformers, RNNs.",
            "Small-batch CNNs.",
            "Any deep net with risk of overfitting.",
            "Very deep ResNets, EfficientNets.",
            "CNNs, spatial regularization."
        ],
        "PyTorch Example": [
            "loss += lambda * torch.sum(torch.abs(weights))",
            "loss += lambda * torch.sum(weights**2)",
            "nn.Dropout(p=0.5)",
            "x_noisy = x + torch.randn_like(x)*sigma",
            "T.RandomHorizontalFlip()",
            "nn.BatchNorm2d(num_features)",
            "nn.LayerNorm(normalized_shape)",
            "nn.GroupNorm(num_groups, num_channels)",
            "Implement via training loop check.",
            "Custom forward logic.",
            "DropBlock2D(block_size=7, p=0.3)"
        ]
    }

    df = pd.DataFrame(data)

    # Checkbox to show table
    if st.checkbox("So which one should I use?"):
        st.dataframe(df, use_container_width=True)