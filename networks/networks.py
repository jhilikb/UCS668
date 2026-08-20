import streamlit as st
import torch
import torch.nn as nn

# Set page configuration
st.set_page_config(page_title="NN Architecture Builder", layout="wide")

st.title("🧠 Interactive Neural Network Builder")
st.markdown(
    "Design your custom network layer by layer. Watch the PyTorch code, "
    "layer shapes, and calculation formulas update in real-time!"
)

# ------------------------------------------------------------------------------
# 1. INITIALIZE SESSION STATE
# ------------------------------------------------------------------------------
if "layers" not in st.session_state:
    st.session_state.layers = []

if "input_channels" not in st.session_state:
    st.session_state.input_channels = 3  # Default to RGB image

if "input_size" not in st.session_state:
    st.session_state.input_size = 224  # Default height/width

# ------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS TO GENERATE PYTORCH CODE & ANALYZE SHAPES
# ------------------------------------------------------------------------------
def generate_pytorch_code(layers, input_channels):
    """Generates a clean, copy-pasteable PyTorch class definition string."""
    code = "import torch\nimport torch.nn as nn\n\n"
    code += "class CustomNetwork(nn.Module):\n"
    code += "    def __init__(self):\n"
    code += "        super(CustomNetwork, self).__init__()\n"
    code += "        self.features = nn.Sequential(\n"
    
    current_channels = input_channels
    for layer in layers:
        l_type = layer["type"]
        if l_type == "Conv2d":
            out_ch = layer["out_channels"]
            k = layer["kernel_size"]
            s = layer["stride"]
            p = layer["padding"]
            d = layer["dilation"]
            code += f"            nn.Conv2d(in_channels={current_channels}, out_channels={out_ch}, kernel_size={k}, stride={s}, padding={p}, dilation={d}),\n"
            current_channels = out_ch
        elif l_type == "MaxPool2d":
            k = layer["kernel_size"]
            s = layer["stride"]
            p = layer["padding"]
            code += f"            nn.MaxPool2d(kernel_size={k}, stride={s}, padding={p}),\n"
        elif l_type == "Linear":
            out_features = layer["out_features"]
            in_features = layer.get("in_features", "DYNAMIC")
            code += f"            nn.Linear(in_features={in_features}, out_features={out_features}),\n"
        elif l_type in ["ReLU", "Sigmoid", "Tanh"]:
            code += f"            nn.{l_type}(),\n"
            
    code += "        )\n\n"
    code += "    def forward(self, x):\n"
    code += "        return self.features(x)\n"
    return code


def build_and_analyze_network(layers, in_channels, in_size):
    """Dynamically builds PyTorch modules and extracts formula math strings."""
    layer_info = []
    current_channels = in_channels
    
    try:
        x = torch.zeros(1, in_channels, in_size, in_size)
    except Exception:
        x = torch.zeros(1, in_channels)
        
    is_flattened = False

    for idx, layer in enumerate(layers):
        l_type = layer["type"]
        info = {
            "index": idx + 1, 
            "type": l_type, 
            "shape_in": list(x.shape)[1:], 
            "params": 0, 
            "formula": "0",
            "error": None
        }
        
        try:
            if l_type == "Conv2d":
                if is_flattened:
                    raise ValueError("Cannot pass flattened data into Conv2d layer without reshaping!")
                
                module = nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=layer["out_channels"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    dilation=layer["dilation"]
                )
                
                # Formula break down: (In_channels * Out_channels * K * K) + Out_channels (for biases)
                in_ch = current_channels
                out_ch = layer["out_channels"]
                k = layer["kernel_size"]
                weights = in_ch * out_ch * k * k
                biases = out_ch
                
                info["formula"] = f"({in_ch} × {out_ch} × {k} × {k}) + {biases} bias"
                info["params"] = weights + biases
                
                x = module(x)
                current_channels = out_ch
                
            elif l_type == "MaxPool2d":
                if is_flattened:
                    raise ValueError("Cannot pass flattened data into MaxPool2d layer!")
                module = nn.MaxPool2d(
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"]
                )
                x = module(x)
                info["formula"] = "Non-parametric layer"
                info["params"] = 0
                
            elif l_type == "Linear":
                if not is_flattened and len(x.shape) > 2:
                    x = x.view(x.size(0), -1)
                    is_flattened = True
                
                in_features = x.shape[1]
                layer["in_features"] = in_features 
                
                module = nn.Linear(in_features=in_features, out_features=layer["out_features"])
                
                # Formula break down: (In_features * Out_features) + Out_features (for biases)
                out_f = layer["out_features"]
                weights = in_features * out_f
                biases = out_f
                
                info["formula"] = f"({in_features} × {out_f}) + {biases} bias"
                info["params"] = weights + biases
                
                x = module(x)
                
            elif l_type in ["ReLU", "Sigmoid", "Tanh"]:
                if l_type == "ReLU": module = nn.ReLU()
                elif l_type == "Sigmoid": module = nn.Sigmoid()
                else: module = nn.Tanh()
                x = module(x)
                info["formula"] = "Activation function"
                info["params"] = 0

            info["shape_out"] = list(x.shape)[1:]
            
        except Exception as e:
            info["error"] = str(e)
            info["shape_out"] = "N/A"
            layer_info.append(info)
            break
            
        layer_info.append(info)
        
    return layer_info

# ------------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Input Configuration")
    st.session_state.input_channels = st.number_input("Input Channels", min_value=1, value=st.session_state.input_channels)
    st.session_state.input_size = st.number_input("Input Size (W/H)", min_value=1, value=st.session_state.input_size)
    
    st.markdown("---")
    st.header("2. Add a Layer")
    
    layer_type = st.selectbox(
        "Choose Layer Type",
        ["Conv2d", "MaxPool2d", "Linear", "ReLU", "Sigmoid", "Tanh"]
    )
    
    config = {"type": layer_type}
    
    if layer_type == "Conv2d":
        config["out_channels"] = st.number_input("Filters (Out Channels)", min_value=1, value=16)
        config["kernel_size"] = st.number_input("Kernel Size", min_value=1, value=3)
        config["stride"] = st.number_input("Stride", min_value=1, value=1)
        config["padding"] = st.number_input("Padding", min_value=0, value=0)
        config["dilation"] = st.number_input("Dilation", min_value=1, value=1)
        
    elif layer_type == "MaxPool2d":
        config["kernel_size"] = st.number_input("Kernel Size", min_value=1, value=2)
        config["stride"] = st.number_input("Stride", min_value=1, value=2)
        config["padding"] = st.number_input("Padding", min_value=0, value=0)
        
    elif layer_type == "Linear":
        config["out_features"] = st.number_input("Output Features (Neurons)", min_value=1, value=10)

    if st.button("➕ Add Layer to Architecture", use_container_width=True):
        st.session_state.layers.append(config)
        st.rerun()

    if st.button("🗑️ Clear All Layers", type="secondary", use_container_width=True):
        st.session_state.layers = []
        st.rerun()

# ------------------------------------------------------------------------------
# 4. MAIN LAYOUT
# ------------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🛠️ Current Network Pipeline")
    
    if not st.session_state.layers:
        st.info("Your network is empty. Use the sidebar to add your first layer!")
    else:
        analysis = build_and_analyze_network(
            st.session_state.layers, 
            st.session_state.input_channels, 
            st.session_state.input_size
        )
        
        total_params = sum(item["params"] for item in analysis if item["error"] is None)
        
        m1, m2 = st.columns(2)
        m1.metric("Total Layers", len(st.session_state.layers))
        m2.metric("Total Trainable Params", f"{total_params:,}")
        
        st.markdown("### Layer-by-Layer Diagnostics")
        
        for idx, (layer, info) in enumerate(zip(st.session_state.layers, analysis)):
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 4, 1])
                
                with c1:
                    st.markdown(f"**Layer {info['index']}**")
                    st.caption(f"`{info['type']}`")
                    
                with c2:
                    if info["error"]:
                        st.error(f"❌ **Dimension Mismatch:** {info['error']}")
                    else:
                        in_s = " × ".join(map(str, info['shape_in']))
                        out_s = " × ".join(map(str, info['shape_out']))
                        
                        st.markdown(f"**Tensor:** `[{in_s}]` ➔ `[{out_s}]`")
                        
                        # Display structural mathematical breakdown string
                        st.markdown(f"🔢 Parameters: `{info['formula']}` = **{info['params']:,}**")
                        
                with c3:
                    if st.button("🗑️", key=f"del_{idx}"):
                        st.session_state.layers.pop(idx)
                        st.rerun()

with col_right:
    st.subheader("💻 Generated PyTorch Code")
    if not st.session_state.layers:
        st.code("# Code will populate here once you add layers.", language="python")
    else:
        generated_code = generate_pytorch_code(st.session_state.layers, st.session_state.input_channels)
        st.code(generated_code, language="python")
