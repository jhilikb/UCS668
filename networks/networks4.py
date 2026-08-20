import streamlit as st

st.set_page_config(layout="wide", page_title="Neural Architecture Interactive Lab")

st.title("🔬 Deep Learning Block Architecture Lab")
st.caption("Investigate mathematical mechanics, conceptual advantages, and dynamic execution shapes across classic layers.")

# ----------------------------------------------------------------
# Sidebar Strategy: Contextual Dimension Filtering
# ----------------------------------------------------------------
st.sidebar.header("🕹️ Network Selection & Profile")

selected_block = st.sidebar.selectbox(
    "Choose Active Architectural Block",
    [
        "Channel Attention (Fully Conv)", 
        "Spatial Attention", 
        "Residual Block", 
        "Inception Block",
        "MultiheadAttention",
        "Transformer Encoder Block",
        "LSTM",
        "GRU Block"
    ]
)

# Identify if the chosen block treats tensors as 4D Vision images or 3D Sequential data
is_sequence_block = selected_block in ["MultiheadAttention","Transformer Encoder Block", "LSTM", "GRU Block"]

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Input Tensor Control Room")
b = st.sidebar.slider("Batch Size (B)", 1, 16, 4)

if not is_sequence_block:
    # 4D Image Tensor Sliders [B, C, H, W]
    c = st.sidebar.slider("Input Channels (C)", 16, 256, 64, step=16)
    h = st.sidebar.slider("Height (H)", 14, 112, 32, step=2)
    w = st.sidebar.slider("Width (W)", 14, 112, 32, step=2)
else:
    # 3D Sequence Tensor Sliders [B, T, D]
    t = st.sidebar.slider("Sequence Length / Time Steps (T)", 10, 500, 50, step=10)
    d = st.sidebar.slider("Embedding Dimension (D)", 32, 1024, 256, step=32)
    h = st.sidebar.slider("Heads/Hidden Dim (h)", 8, 1024, 8, step=8)



# Helper function to render a structural block node
def block_node(step_num, title, input_shape, output_shape, behavior):
    st.html(
        f"""
        <div style="background-color: #1A1C24; border-radius: 8px; padding: 18px; border: 1px solid #343A40; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="background-color: #FF4B4B; color: white; padding: 2px 8px; border-radius: 20px; font-size: 0.75rem; font-weight: bold;">Step {step_num}</span>
                <span style="color: #8E94A5; font-family: monospace; font-size: 0.85rem; font-weight: bold;">{title}</span>
            </div>
            <div style="margin-top: 12px; display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <div style="font-size: 0.75rem; color: #ADB5BD;">IN TENSOR</div>
                    <strong style="font-family: monospace; color: #E9ECEF; font-size: 1.05rem;">{input_shape}</strong>
                </div>
                <div style="color: #FF4B4B; align-self: center; font-size: 1.1rem;">➡️</div>
                <div>
                    <div style="font-size: 0.75rem; color: #ADB5BD;">OUT TENSOR</div>
                    <strong style="font-family: monospace; color: #E9ECEF; font-size: 1.05rem;">{output_shape}</strong>
                </div>
            </div>
            <p style="color: #7A8294; margin-top: 12px; font-size: 0.88rem; border-top: 1px solid #2D313E; padding-top: 8px; line-height: 1.4;">{behavior}</p>
        </div>
        """
    )

# ----------------------------------------------------------------
# Modular Architectural Components
# ----------------------------------------------------------------
st.subheader(f"Current Structure: {selected_block}")

if selected_block == "Channel Attention (Fully Conv)":
    st.markdown("""
    ### Why this block is used:
    In deep CNNs, feature channels represent distinct high-level visual concepts (e.g., edges, textures, object parts). Traditional convolutions treat every channel with equal importance. 
    **Channel Attention** acts as a dynamic 'volume knob' that scores channel feature relevance. By using a fully convolutional layer style (`nn.Conv2d`), we preserve 2D coordinate spaces during compression and use fewer network parameters than standard MLPs.
    """)
    
    st.markdown("### Complete Python Module Code")
    st.code(f"""import torch
import torch.nn as nn

class FullyConvChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        mid_channels = in_channels // reduction_ratio
        
        # Fully Convolutional Squeeze-and-Excitation
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Squeeze spatial dimensions via pooling
        avg_out = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        max_out = torch.nn.functional.adaptive_max_pool2d(x, 1)
        
        # 2. Process descriptors through Conv2d filters
        scores = self.conv_block(avg_out) + self.conv_block(max_out)
        
        # 3. Apply attention weights scaling
        return x * self.sigmoid(scores)""", language="python")

    st.markdown("### Substep Execution Traces")
    t1, t2, t3 = st.tabs(["1. Squeeze Pooling", "2. Conv2d Excitation Mapping", "3. Element-wise Scaler"])
    with t1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(1, "Adaptive Pool 2D", f"[{b}, {c}, {h}, {w}]", f"[{b}, {c}, 1, 1]", "Squashes spatial geometry dimensions down into discrete 1x1 statistical vectors.")
        with col2:
            st.code(f"avg_out = torch.nn.functional.adaptive_avg_pool2d(x, 1)  # [{b}, {c}, 1, 1]\nmax_out = torch.nn.functional.adaptive_max_pool2d(x, 1)  # [{b}, {c}, 1, 1]")
    with t2:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(2, "1x1 Receptive Conv2d", f"[{b}, {c}, 1, 1]", f"[{b}, {c}, 1, 1]", f"Runs a 1x1 Bottleneck Conv kernel mapping channels through a compressed projection matrix layer ({c} ➡️ {c//16} ➡️ {c}).")
        with col2:
            st.code(f"scores = self.conv_block(avg_out) + self.conv_block(max_out)  # [{b}, {c}, 1, 1]\nscale = torch.sigmoid(scores)  # [{b}, {c}, 1, 1]")
    with t3:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(3, "Matrix Broadcast Multiplier", f"[{b}, {c}, {h}, {w}] × [{b}, {c}, 1, 1]", f"[{b}, {c}, {h}, {w}]", "Broadcards the channel weights along the spatial dimensions of the input tensor.")
        with col2:
            st.code(f"output_tensor = x * scale  # Final dimensions: [{b}, {c}, {h}, {w}]")

elif selected_block == "MultiheadAttention":
    st.markdown("""
    ### Why this block is used:
    **Multihead Attention** allows a model to jointly attend to information from different representation subspaces. 
    Instead of a single attention mechanism, multiple heads learn diverse relationships (e.g., long-range dependencies, local context).
    This improves expressiveness and stabilizes training compared to single-head attention.
    """)

    st.markdown("### Complete Python Module Code")
    st.code(f"""import torch
import torch.nn as nn

class SimpleMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # Self-attention: query, key, value are the same
        out, weights = self.attn(x, x, x)
        return out, weights""", language="python")

    st.markdown("### Substep Execution Traces")
    
    t1, t2, t3 ,t4= st.tabs(["1. Linear Projections", "2. Scaled Dot-Product", "3.Weighted Value Mix","4. Head Concatenation"])
    with t1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(1, "Q/K/V Linear Maps", f"[{b}, {t}, {d}]", f"[{b}, {h} , {t}, {d//h}]", "Projects input embeddings into query, key, and value spaces for each head.")
        with col2:
            st.code(f"Q = Wq @ x  # [{b},{h}, {t}, {d/h}]\nK = Wk @ x\nV = Wv @ x")
    with t2:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(2, "Attention Scores", f"[{b}, {h}, {t}, {d//h}]", f"[{b},{h}, {t}, {t}]", "Computes scaled dot-product attention weights across sequence positions.")
        with col2:
            st.code(f"attn_scores = (Q @ K.T) / sqrt(d/h)\nweights = softmax(attn_scores)")
    with t3:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(3, "Weighted Value Mix", f"[{b}, {h}, {t}, {t}] × [{b}, {h}, {t}, {d//h}]", f"[{b}, {h}, {t}, {d//h}]", "Applies attention weights to value vectors to produce context-aware representations.")
        with col2:
            st.code(f"context = weights @ V  # [{b}, {h}, {t}, {d//h}]")
    with t4:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(4, "Concat + Linear", f"[{b}, {h}, {t}, {d//h}]", f"[{b}, {t}, {d}]", "Concatenates all heads and maps back to embedding dimension.")
        with col2:
            st.code(f"output = concat(context_heads)# [{b}, {t}, {d}] \n output=self.Linear(output)  # [{b}, {t}, {d}]")


elif selected_block == "Spatial Attention":
    st.markdown("""
    ### Why this block is used:
    While channel attention tracks *what* concepts are present, **Spatial Attention** maps *where* those features matter across spatial coordinates. It isolates and highlights relevant structural boundaries while ignoring background elements.
    """)
    
    st.markdown("### Complete Python Module Code")
    st.code("""import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_summary = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(spatial_summary))
        return x * attention_map""", language="python")

    st.markdown("### Substep Execution Traces")
    t1, t2, t3 = st.tabs(["1. Descriptor Extraction", "2. Convolution Projection", "3. Calibration Multiply"])
    with t1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(1, "Axis Feature Reduction", f"[{b}, {c}, {h}, {w}]", f"[{b}, 2, {h}, {w}]", "Collapses all active feature maps into statistical representations across the channel dimension axis.")
        with col2:
            st.code(f"avg_out = torch.mean(x, dim=1, keepdim=True)  # [{b}, 1, {h}, {w}]\nmax_out, _ = torch.max(x, dim=1, keepdim=True)  # [{b}, 1, {h}, {w}]\nspatial_summary = torch.cat([avg_out, max_out], dim=1)  # [{b}, 2, {h}, {w}]")
    with t2:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(2, "2D Kernel Map Projection", f"[{b}, 2, {h}, {w}]", f"[{b}, 1, {h}, {w}]", "Convolves the descriptive summary layers down into a 1-channel spatial layout map.")
        with col2:
            st.code(f"attention_map = torch.sigmoid(self.conv(spatial_summary))  # Output dimensions: [{b}, 1, {h}, {w}]")
    with t3:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(3, "Mask Spatial Multiplication", f"[{b}, {c}, {h}, {w}] × [{b}, 1, {h}, {w}]", f"[{b}, {c}, {h}, {w}]", "Applies the single-channel spatial mask down across all feature channels.")
        with col2:
            st.code(f"output_tensor = x * attention_map  # Final dimensions: [{b}, {c}, {h}, {w}]")

elif selected_block == "Residual Block":
    st.markdown("""
    ### Why this block is used:
    As convolutional networks grow deeper, their accuracy can saturate and quickly degrade (known as the *vanishing gradient problem*). 
    **Residual Blocks** introduce a skip connection pathway ($x + F(x)$). This architectural shortcut provides a clear route for information and training gradients to flow backward through the network without dampening.
    """)
    
    st.markdown("### Complete Python Module Code")
    st.code("""import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)""", language="python")

    st.markdown("### Substep Execution Traces")
    t1, t2 = st.tabs(["1. Processing Transform Route", "2. Vector Addition Layer"])
    with t1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(1, "Weight Block Evaluation", f"[{b}, {c}, {h}, {w}]", f"[{b}, {c}, {h}, {w}]", "Processes incoming layer values down through the deep weight transformation stack.")
        with col2:
            st.code(f"identity = x  # Preserved: [{b}, {c}, {h}, {w}]\nout = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))  # [{b}, {c}, {h}, {w}]")
    with t2:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(2, "Matrix Residual Blending", f"[{b}, {c}, {h}, {w}] + [{b}, {c}, {h}, {w}]", f"[{b}, {c}, {h}, {w}]", "Combines the raw input features directly with the processed output via element-wise addition.")
        with col2:
            st.code(f"out += identity  # Matrix addition combines matching elements\nout = self.relu(out)  # Output: [{b}, {c}, {h}, {w}]")

elif selected_block == "Inception Block":
    st.markdown("""
    ### Why this block is used:
    Choosing an optimal convolution filter size (e.g., 1x1, 3x3, or 5x5) is difficult because distinct visual details exist at different scales across an image. 
    **Inception Blocks** solve this by executing multiple convolution filter operations *in parallel*. The network captures broad global structures and fine local details simultaneously, concatenating the results into a single multi-scale layer.
    """)
    
    st.markdown("### Complete Python Module Code")
    st.code("""import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_split):
        super().__init__()
        # Simplified structural demonstration allocating equal filter distributions
        self.p1 = nn.Conv2d(in_channels, out_split, kernel_size=1)
        self.p2 = nn.Conv2d(in_channels, out_split, kernel_size=3, padding=1)
        self.p3 = nn.Conv2d(in_channels, out_split, kernel_size=5, padding=2)
        
    def forward(self, x):
        branch1 = self.p1(x)
        branch2 = self.p2(x)
        branch3 = self.p3(x)
        return torch.cat([branch1, branch2, branch3], dim=1)""", language="python")

    st.markdown("### Substep Execution Traces")
    t1, t2 = st.tabs(["1. Multi-Scale Path Computation", "2. Channel Concatenation"])
    split_out = c // 4
    with t1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(1, "Parallel Filtering", f"[{b}, {c}, {h}, {w}]", f"3 × [{b}, {split_out}, {h}, {w}]", f"Processes the input tensor through parallel branches at the same time, allocating {split_out} channels to each branch.")
        with col2:
            st.code(f"branch1 = self.p1(x)  # Shape: [{b}, {split_out}, {h}, {w}]\nbranch2 = self.p2(x)  # Shape: [{b}, {split_out}, {h}, {w}]\nbranch3 = self.p3(x)  # Shape: [{b}, {split_out}, {h}, {w}]")
    with t2:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(2, "Depth Concat Fusion", f"3 × [{b}, {split_out}, {h}, {w}]", f"[{b}, {split_out * 3}, {h}, {w}]", f"Glues the parallel output vectors back together along the channel dimension axis (dim=1).")
        with col2:
            st.code(f"output = torch.cat([branch1, branch2, branch3], dim=1)  # Target Concat Shape: [{b}, {split_out * 3}, {h}, {w}]")

elif selected_block == "Transformer Encoder Block":
    st.markdown("""
    ### Why this block is used:
    Unlike RNNs that process tokens sequentially step-by-step, **Transformer Encoder Blocks** process entire sequences simultaneously. 
    They leverage **Multi-Head Self-Attention** to calculate global dependencies between all tokens in a sequence at once, regardless of their distance from one another. This enables highly parallel training and effectively models long-range dependencies.
    """)
    
    st.markdown("### Complete Python Module Code")
    st.code("""import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, src):
        # Multi-Head Attention Sub-Layer with Residual Connection
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # Feed-Forward Network Sub-Layer with Residual Connection
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout(src2)
        return self.norm2(src)""", language="python")

    st.markdown("### Substep Execution Traces")
    t1, t2 = st.tabs(["1. Multi-Head Self-Attention", "2. Feed-Forward Projection Network"])
    with t1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(1, "Self-Attention Mechanism", f"[{b}, {t}, {d}]", f"[{b}, {t}, {d}]", "Computes the relational context scores matching every sequence index location item to all other sequence positions.")
        with col2:
            st.code(f"# Computes dynamic Query, Key, and Value matrix interactions\nattn_output, _ = self.self_attn(src, src, src)  # [{b}, {t}, {d}]\nx = src + attn_output  # Skip Connection\nx = self.norm1(x)       # Shape: [{b}, {t}, {d}]")
    with t2:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(2, "FFN Expansion Bottleneck", f"[{b}, {t}, {d}]", f"[{b}, {t}, {d}]", f"Projects token attributes into an expanded intermediate layer space (typically {d} ➡️ 2048 ➡️ {d}) to extract deeper non-linear relationships.")
        with col2:
            st.code(f"ffn_out = self.linear1(x)  # Expanded representation space: [{b}, {t}, 2048]\nffn_out = self.linear2(self.activation(ffn_out))  # Projected back: [{b}, {t}, {d}]\noutput = self.norm2(x + ffn_out)  # Final Output Shape: [{b}, {t}, {d}]")

elif selected_block == "LSTM":
    st.markdown("""
    ### Why this block is used:
    **LSTMs (Long Short-Term Memory networks)** are designed to capture long-range dependencies in sequential data. 
    They use gating mechanisms (input, forget, output) to regulate information flow, mitigating vanishing/exploding gradient issues common in vanilla RNNs.
    """)

    st.markdown("### Complete Python Module Code")
    st.code(f"""import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x) #out: hidden states for every time step
        return out, (h_n, c_n)""", language="python")

    st.markdown("### Substep Execution Traces")
    t1, t2, t3 = st.tabs(["1. Gate Computations", "2. Cell State Update", "3. Hidden State Output"])
    with t1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(1, "Input/Forget/Output Gates", f"[{b}, {t}, {d}]", f"[{b}, {t}, {h}]", "Each gate decides how much new info enters, how much past info is forgotten, and how much is exposed.")
        with col2:
            st.code(f"i_t = σ(W_i x_t + U_i h_{t-1})\nf_t = σ(W_f x_t + U_f h_{t-1})\no_t = σ(W_o x_t + U_o h_{t-1})")
    with t2:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(2, "Cell State Update", f"[{b}, {t}, {h}]", f"[{b}, {t}, {h}]", "Combines gated input with previous cell state to maintain long-term memory.")
        with col2:
            st.code(f"c_t = f_t * c_{t-1} + i_t * tanh(W_c x_t + U_c h_{t-1})")
    with t3:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(3, "Hidden State Output", f"[{b}, {t}, {h}]", f"[{b}, {t}, {h}]", "Applies output gate to cell state to produce hidden representation.")
        with col2:
            st.code(f"h_t = o_t * tanh(c_t)")

if selected_block == "GRU Block":
    st.markdown("""
    ### Why this block is used:
    **GRUs (Gated Recurrent Units)** are a streamlined alternative to LSTMs. 
    They combine the forget and input gates into a single **update gate**, and use a **reset gate** to control how much past information is mixed with new input. 
    This makes GRUs computationally lighter while still effective at capturing long-term dependencies in sequential data.
    """)

    st.markdown("### Complete Python Module Code")
    st.code(f"""import torch
import torch.nn as nn

class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        out, h_n = self.gru(x) # out: hidden states for every time step,h_n: final hidden state(s) for each layer
        return out, h_n""", language="python")

    st.markdown("### Substep Execution Traces")
    t1, t2, t3 = st.tabs(["1. Gate Computations", "2. Candidate State", "3. Hidden State Update"])
    with t1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(1, "Update & Reset Gates", f"[{b}, {t}, {d}]", f"[{b}, {t}, {h}]", "Update gate decides how much of the past state to keep; reset gate decides how much past info to forget.")
        with col2:
            st.code(f"z_t = σ(W_z x_t + U_z h_{t-1})  # update gate\nr_t = σ(W_r x_t + U_r h_{t-1})  # reset gate")
    with t2:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(2, "Candidate Hidden State", f"[{b}, {t}, {h}]", f"[{b}, {t}, {h}]", "Mixes current input with reset-modulated past hidden state to form candidate representation.")
        with col2:
            st.code(f"h~_t = tanh(W_h x_t + U_h (r_t * h_{t-1}))")
    with t3:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            block_node(3, "Final Hidden State Update", f"[{b}, {t}, {h}]", f"[{b}, {t}, {h}]", "Blends old hidden state with candidate state using update gate.")
        with col2:
            st.code(f"h_t = (1 - z_t) * h_{t-1} + z_t * h~_t")
