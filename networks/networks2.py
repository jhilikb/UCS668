import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper functions for visualization ---
def show_block_diagram(concept_name):
    st.image(f"diagrams/{concept_name}.png", caption=f"{concept_name} Block Design")

def show_code(code_str):
    st.code(code_str, language="python")

# --- Concept tabs ---
def skip_connection_tab():
    st.header("Skip Connections (ResNet-style)")
    show_block_diagram("skip_connection")
    st.markdown("Skip connections allow gradients to flow directly, solving vanishing gradient issues.")
    code = """
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)
"""
    show_code(code)

def attention_tab():
    st.header("Channel & Spatial Attention")
    show_block_diagram("spatialattention")
    show_block_diagram("channelattention")
    st.markdown("Attention highlights important features across channels or spatial dimensions.")
    code = """
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn
"""
    show_code(code)
    code1= """
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 1x1 convs act like FC layers but keep spatial structure
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        # Global Average Pooling → squeeze spatial dimensions
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)

        # Pass through 1x1 conv bottleneck
        out = self.conv1(avg_out)
        out = F.relu(out)
        out = self.conv2(out)

        # Sigmoid gate for channel scaling
        scale = torch.sigmoid(out)
        return x * scale
"""
    show_code(code1)

def inception_tab():
    st.header("Inception Module")
    show_block_diagram("inception")
    st.markdown("Inception uses multiple kernel sizes in parallel to capture multi-scale features.")
    code = """
class Inception(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch3(x),
            self.branch5(x)
        ], dim=1)
"""
    show_code(code)

def depthwise_tab():
    st.header("Depthwise Separable Convolution")
    show_block_diagram("depthwise")
    st.markdown("Depthwise convolution reduces computation by separating spatial and channel mixing.")
    code = """
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
"""
    show_code(code)

def transformer_tab():
    st.header("Transformer Encoder Block")
    show_block_diagram("transformer")
    st.markdown("Transformers use self-attention to model long-range dependencies.")
    code = """
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)
"""
    show_code(code)

def lstm_gru_tab():
    st.header("LSTM & GRU")
    show_block_diagram("lstm")
    show_block_diagram("gru")
    st.markdown("LSTMs and GRUs are recurrent units designed to capture sequential dependencies.")
    code = """
class RNNModels(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)
        return lstm_out, gru_out
"""
    show_code(code)

# --- Streamlit App Layout ---
st.title("Neural Network Design Concepts")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Skip Connections", "Attention", "Inception", "Depthwise Conv", "Transformer", "LSTM/GRU"
])

with tab1: skip_connection_tab()
with tab2: attention_tab()
with tab3: inception_tab()
with tab4: depthwise_tab()
with tab5: transformer_tab()
with tab6: lstm_gru_tab()

