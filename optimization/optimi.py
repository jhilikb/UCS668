import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

torch.manual_seed(42)

st.set_page_config(
    page_title="Gradient Flow Explorer",
    layout="wide"
)

st.title("🧠 Gradient Flow Explorer")
st.markdown(
"""
Observe how gradients propagate through a neural network after **one backward pass**.

Experiment with:

- Activation Functions
- Batch Normalization
- Residual Connections
- Initialization
- Optimizers
- Gradient Clipping

The plots below are computed from an actual PyTorch network.
"""
)

########################################################################
# Sidebar
########################################################################

st.sidebar.header("Network")

depth = st.sidebar.slider(
    "Hidden Layers",
    2,
    12,
    6
)

filters = st.sidebar.slider(
    "No. of filters per layer",
    16,
    256,
    64,
    step=16
)

activation_name = st.sidebar.selectbox(
    "Activation",
    [
        "ReLU",
        "LeakyReLU",
        "Sigmoid",
        "Tanh",
        "GELU"
    ]
)

batchnorm = st.sidebar.checkbox("BatchNorm", True)

residual = st.sidebar.checkbox(
    "Residual Connections",
    False
)

########################################################################

st.sidebar.header("Initialization")

init_name = st.sidebar.selectbox(
    "Weight Initialization",
    [
        "He",
        "Xavier",
        "Normal"
    ]
)

########################################################################

st.sidebar.header("Optimizer")

optimizer_name = st.sidebar.selectbox(
    "Optimizer",
    [
        "SGD",
        "Momentum",
        "RMSProp",
        "Adam"
    ]
)

lr = st.sidebar.slider(
    "Learning Rate",
    0.0001,
    0.1,
    0.001,
    format="%.4f"
)

########################################################################

st.sidebar.header("Gradient")

clip = st.sidebar.checkbox(
    "Gradient Clipping",
    False
)

clip_value = st.sidebar.slider(
    "Clip Value",
    0.1,
    5.0,
    1.0
)

loss_scale = st.sidebar.slider(
    "Loss Scale",
    1,
    50,
    1,
    help="Increase to intentionally create exploding gradients."
)

run = st.sidebar.button("Run One Backward Pass")

########################################################################
# Activation
########################################################################

def get_activation(name):

    if name=="ReLU":
        return nn.ReLU()

    if name=="LeakyReLU":
        return nn.LeakyReLU(0.1)

    if name=="Sigmoid":
        return nn.Sigmoid()

    if name=="Tanh":
        return nn.Tanh()

    if name=="GELU":
        return nn.GELU()

########################################################################
# Model
########################################################################

class SimpleMLP(nn.Module):

    def __init__(
        self,
        depth,
        hidden,
        activation,
        batchnorm,
        residual
    ):

        super().__init__()

        self.residual = residual

        self.layers = nn.ModuleList()

        in_dim = 64

        for i in range(depth):

            block = []

            block.append(
                nn.Conv2d(
                    in_dim,
                    filters,3,1,1
                )
            )

            if batchnorm:
                block.append(
                    nn.BatchNorm2d(filters)
                )

            block.append(
                activation
            )

            self.layers.append(
                nn.Sequential(*block)
            )

            in_dim = filters
        in_dim = filters*28*28
        self.flatten_layer = nn.Flatten()
        self.out = nn.Linear(in_dim,10)

    def forward(self,x):

        for layer in self.layers:
            # st.write(x.shape,layer)

            previous = x

            x = layer(x)

            if self.residual:

                if previous.shape==x.shape:
                    x = x + previous

        return self.out(self.flatten_layer(x))

########################################################################
# Initialization
########################################################################

def initialize(model,name):

    for m in model.modules():

        if isinstance(m,nn.Linear):

            if name=="He":

                nn.init.kaiming_normal_(m.weight)

            elif name=="Xavier":

                nn.init.xavier_normal_(m.weight)

            elif name=="Normal":

                nn.init.normal_(
                    m.weight,
                    std=0.05
                )

            nn.init.zeros_(m.bias)

########################################################################
# Optimizer
########################################################################

def get_optimizer(name,model):

    if name=="SGD":

        return optim.SGD(
            model.parameters(),
            lr=lr
        )

    if name=="Momentum":

        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )

    if name=="RMSProp":

        return optim.RMSprop(
            model.parameters(),
            lr=lr
        )

    if name=="Adam":

        return optim.Adam(
            model.parameters(),
            lr=lr
        )

########################################################################
# Hooks
########################################################################

activations = {}

def save_activation(name):

    def hook(module,input,output):

        activations[name]=output.detach()

    return hook

########################################################################
# Run
########################################################################

if run:

    activation = get_activation(
        activation_name
    )

    model = SimpleMLP(
        depth,
        filters,
        activation,
        batchnorm,
        residual
    )

    initialize(
        model,
        init_name
    )

    idx=1

    for layer in model.layers:

        layer.register_forward_hook(
            save_activation(
                f"Layer {idx}"
            )
        )

        idx+=1

    optimizer = get_optimizer(
        optimizer_name,
        model
    )

    x = torch.randn(64,64,
        28,
        28
    )

    target = torch.randint(
        0,
        10,
        (64,)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    prediction = model(x)

    loss = criterion(
        prediction,
        target
    )

    loss = loss * loss_scale

    loss.backward()

    if clip:

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            clip_value
        )

    ####################################################################
    # Collect statistics
    ####################################################################

    gradient_norms = []

    parameter_updates = []

    weight_means = []

    weight_stds = []

    layer_names = []

    grad_values = []

    for name,param in model.named_parameters():

        if "weight" in name:

            if param.grad is None:
                continue

            grad_norm = (
                param.grad.norm()
                .item()
            )

            update = lr * grad_norm

            gradient_norms.append(
                grad_norm
            )

            parameter_updates.append(
                update
            )

            weight_means.append(
                param.data.mean().item()
            )

            weight_stds.append(
                param.data.std().item()
            )

            grad_values.extend(
                param.grad.detach()
                .cpu()
                .numpy()
                .flatten()
            )

            layer_names.append(name)

    activation_stats = []

    activation_names = []

    activation_distributions = {}

    for name,act in activations.items():

        activation_names.append(name)

        activation_stats.append(
            act.abs().mean().item()
        )

        activation_distributions[name]=(
            act.detach()
            .cpu()
            .numpy()
            .flatten()
        )

    ####################################################################
    # Determine gradient health
    ####################################################################

    mean_grad = np.mean(
        gradient_norms
    )

    if mean_grad < 1e-3:

        health = "Vanishing Gradients"

        color = "red"

    elif mean_grad > 5:

        health = "Exploding Gradients"

        color = "orange"

    else:

        health = "Healthy Gradient Flow"

        color = "green"

    ####################################################################
    # Layout
    ####################################################################

    left,right = st.columns([1,2])

    with left:

        st.subheader("Network")

        st.markdown(
        f"""
        **Input**

        ↓

        {'↓\n\n'.join([f'Conv {i+1}' for i in range(depth)])}

        ↓

        Output
        """
        )

        st.metric(
            "Loss",
            f"{loss.item():.3f}"
        )

        st.markdown(
            f"### :{color}[{health}]"
        )

    with right:

        st.subheader("Gradient Magnitude Through Layers")

                ###############################################################
        # Gradient Norm Bar Chart
        ###############################################################

        grad_df = pd.DataFrame({
            "Layer": layer_names,
            "Gradient Norm": gradient_norms
        })

        fig = px.bar(
            grad_df,
            x="Gradient Norm",
            y="Layer",
            orientation="h",
            color="Gradient Norm",
            color_continuous_scale="Viridis",
            title="Gradient Magnitude"
        )

        fig.update_layout(height=500)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    ###############################################################
    # Second Row
    ###############################################################

    c1, c2 = st.columns(2)

    ###############################################################
    # Activation Statistics
    ###############################################################

    with c1:

        st.subheader("Average Activation Magnitude")

        act_df = pd.DataFrame({
            "Layer": activation_names,
            "Activation": activation_stats
        })

        fig = px.bar(
            act_df,
            x="Layer",
            y="Activation",
            color="Activation",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    ###############################################################
    # Parameter Updates
    ###############################################################

    with c2:

        st.subheader("Estimated Parameter Update")

        update_df = pd.DataFrame({

            "Layer": layer_names,
            "Update": parameter_updates

        })

        fig = px.bar(
            update_df,
            x="Layer",
            y="Update",
            color="Update",
            color_continuous_scale="Oranges"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    ###############################################################
    # Weight Statistics
    ###############################################################

    st.subheader("Weight Statistics")

    stat_df = pd.DataFrame({

        "Layer": layer_names,
        "Mean": weight_means,
        "Std": weight_stds

    })

    st.dataframe(
        stat_df,
        use_container_width=True
    )

    ###############################################################
    # Gradient Histogram
    ###############################################################

    st.subheader("Gradient Distribution")

    fig = px.histogram(

        x=grad_values,
        nbins=80

    )

    fig.update_layout(

        xaxis_title="Gradient Value",
        yaxis_title="Count"

    )

    st.plotly_chart(

        fig,
        use_container_width=True

    )

    ###############################################################
    # Activation Histograms
    ###############################################################

    st.subheader("Activation Distributions")

    cols = st.columns(2)

    i = 0

    for name, values in activation_distributions.items():

        with cols[i % 2]:

            fig = px.histogram(

                x=values,
                nbins=60,
                title=name

            )

            st.plotly_chart(

                fig,
                use_container_width=True

            )

        i += 1

    ###############################################################
    # Network Visualization
    ###############################################################

    st.subheader("Gradient Flow Visualization")

    max_grad = max(gradient_norms)

    for layer, grad in zip(layer_names, gradient_norms):

        ratio = grad / max_grad

        if ratio > 0.75:
            color = "🟩"

        elif ratio > 0.40:
            color = "🟨"

        elif ratio > 0.15:
            color = "🟧"

        else:
            color = "🟥"

        bar = "█" * max(1, int(ratio * 25))

        st.markdown(

            f"**{layer}**  \n"
            f"{color} `{bar}` ({grad:.5f})"

        )

    ###############################################################
    # Automatic Interpretation
    ###############################################################

    st.divider()

    st.header("Interpretation")

    if mean_grad < 1e-3:

        st.error(
            """
**Vanishing Gradients Detected**

The gradients reaching earlier layers are extremely small.

Possible reasons:

- Sigmoid or Tanh saturation
- Deep network
- Poor initialization

Try:

- ReLU
- GELU
- He Initialization
- Residual Connections
- Batch Normalization
"""
        )

    elif mean_grad > 5:

        st.warning(
            """
**Exploding Gradients Detected**

Gradient norms are becoming excessively large.

Try:

- Gradient Clipping
- Xavier Initialization
- Smaller Learning Rate
- BatchNorm
"""
        )

    else:

        st.success(
            """
**Healthy Gradient Flow**

Gradients propagate through the network without major decay or explosion.

This usually leads to stable training.
"""
        )

    ###############################################################
    # Explain Current Choices
    ###############################################################

    st.subheader("Current Configuration")

    explanation = []

    if activation_name == "Sigmoid":

        explanation.append(
            "Sigmoid compresses outputs between 0 and 1 and can cause vanishing gradients."
        )

    if activation_name == "Tanh":

        explanation.append(
            "Tanh is zero-centered but can still saturate."
        )

    if activation_name == "ReLU":

        explanation.append(
            "ReLU generally maintains stronger gradients."
        )

    if activation_name == "GELU":

        explanation.append(
            "GELU provides smooth activations and often improves optimization."
        )

    if batchnorm:

        explanation.append(
            "BatchNorm normalizes activations and stabilizes gradient flow."
        )

    if residual:

        explanation.append(
            "Residual connections create shortcut paths that help gradients reach earlier layers."
        )

    if clip:

        explanation.append(
            f"Gradient clipping limits the norm to {clip_value:.2f}."
        )

    if optimizer_name == "Adam":

        st.markdown("## Adam (Adaptive Moment Estimation)")

        st.markdown("### PyTorch Function")

        st.code(
    """optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9,0.999),
        eps=1e-8
    )""",
    language="python"
        )

        st.markdown("""
    Adam is the **most widely used optimizer in deep learning** because it combines the strengths of **Momentum** and **RMSProp**.

    Adam keeps track of two quantities for every parameter:

    1. **First Moment (Mean of Gradients)** – remembers the direction of previous gradients (Momentum).
    2. **Second Moment (Mean of Squared Gradients)** – adapts the learning rate for each parameter (RMSProp).

    Because of these two mechanisms, Adam usually converges much faster than SGD while requiring very little hyperparameter tuning.

    ### Advantages
    - Fast convergence
    - Adaptive learning rate for every parameter
    - Combines Momentum and RMSProp
    - Works well on most deep learning problems
    - Excellent default optimizer for beginners

    ### Disadvantages
    - Requires more memory than SGD
    - More computationally expensive
    - Sometimes achieves slightly poorer final generalization than SGD on image classification tasks
    """)

        st.markdown("### Mathematical Formulation")

        st.markdown("Current Gradient")

        st.latex(r"g_t=\nabla L(w_t)")

        st.markdown("First Moment (Momentum Estimate)")

        st.latex(r"m_t=\beta_1m_{t-1}+(1-\beta_1)g_t")

        st.markdown("Second Moment (Variance Estimate)")

        st.latex(r"v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2")

        st.markdown("Bias Correction")

        st.latex(r"\hat{m}_t=\frac{m_t}{1-\beta_1^t}")

        st.latex(r"\hat{v}_t=\frac{v_t}{1-\beta_2^t}")

        st.markdown("Parameter Update")

        st.latex(r"w_{t+1}=w_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}")

        st.info(
            "💡 Intuition: Adam remembers the direction in which it has been moving (Momentum) while also adjusting the step size for every parameter independently (RMSProp). This combination makes it one of the fastest and most robust optimization algorithms for deep neural networks."
        )

    elif optimizer_name == "Momentum":

        st.markdown("## SGD with Momentum")

        st.markdown("### PyTorch Function")

        st.code(
    """optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9
    )""",
    language="python"
        )

        st.markdown("""
    Momentum remembers previous gradients.

    Instead of using only the current slope, it builds up velocity over time.

    ### Advantages
    - Faster convergence
    - Less oscillation
    - Escapes shallow minima

    ### Disadvantages
    - Extra hyperparameter
    - Can overshoot
    """)

        st.markdown("### Mathematical Formulation")

        st.latex(r"g_t=\nabla L(w_t)")
        st.latex(r"v_t=\mu v_{t-1}+g_t")
        st.latex(r"w_{t+1}=w_t-\eta v_t")

        st.info(
            "Imagine rolling a heavy ball downhill. The ball keeps moving due to momentum."
        )

    elif optimizer_name == "RMSProp":

        st.markdown("## RMSProp (Root Mean Square Propagation)")

        st.markdown("### PyTorch Function")

        st.code(
    """optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        alpha=0.99,
        eps=1e-8
    )""",
    language="python"
        )

        st.markdown("""
    RMSProp is an **adaptive learning rate optimizer**.

    Instead of using the same learning rate for every parameter, RMSProp keeps track of the recent history of gradients. Parameters that consistently receive **large gradients** automatically receive **smaller updates**, while parameters with **small gradients** receive relatively **larger updates**.

    This allows the optimizer to move quickly in flat regions while remaining stable in steep regions of the loss surface.

    ### Advantages
    - Adaptive learning rate for every parameter
    - Faster convergence than SGD on many problems
    - Handles noisy gradients well
    - Works well for recurrent neural networks

    ### Disadvantages
    - Requires additional memory to store gradient statistics
    - More hyperparameters than SGD
    - May not generalize as well as SGD on some vision tasks
    """)

        st.markdown("### Mathematical Formulation")

        st.markdown("Current Gradient")

        st.latex(r"g_t=\nabla L(w_t)")

        st.markdown("Running Average of Squared Gradients")

        st.latex(r"s_t=\beta s_{t-1}+(1-\beta)g_t^2")

        st.markdown("Parameter Update")

        st.latex(r"w_{t+1}=w_t-\eta\frac{g_t}{\sqrt{s_t+\epsilon}}")

        st.info(
            "💡 Intuition: Imagine driving on rough terrain. RMSProp automatically slows down in directions where the road is steep and speeds up where the terrain is relatively flat."
        )

    elif optimizer_name == "SGD":

        st.markdown("## SGD (Stochastic Gradient Descent)")

        st.markdown("### PyTorch Function")

        st.code(
    """optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )""",
    language="python"
        )

        st.markdown("""
    SGD updates every parameter using only the **current gradient**.

    It does not remember previous gradients or adapt the learning rate.

    ### Advantages
    - Simple
    - Low memory usage
    - Easy to understand
    - Often gives excellent final accuracy

    ### Disadvantages
    - Slow convergence
    - Can oscillate
    - Sensitive to learning rate
    """)

        st.markdown("### Mathematical Formulation")

        st.latex(r"g_t=\nabla L(w_t)")

        st.latex(r"w_{t+1}=w_t-\eta g_t")

        st.info(
            "Think of SGD as taking one step downhill using only the current slope."
        )

    for item in explanation:

        st.write("•", item)

    ###############################################################
    # Suggested Experiments
    ###############################################################

    st.divider()

    st.header("Suggested Classroom Experiments")

    st.markdown("""
### 1. Vanishing Gradients
- Activation = **Sigmoid**
- Hidden Layers = **10–12**
- BatchNorm = ❌
- Residual = ❌
- Initialization = Normal

Observe how early-layer gradients almost disappear.

---

### 2. Healthy Training
- Activation = **ReLU**
- He Initialization
- BatchNorm = ✅
- Adam

Notice relatively uniform gradient magnitudes.

---

### 3. Residual Connections
Keep everything fixed.

Toggle **Residual Connections**.

Observe how gradients become stronger in earlier layers.

---

### 4. Effect of BatchNorm

Run once with BatchNorm OFF.

Run again with BatchNorm ON.

Compare activation distributions and gradient norms.

---

### 5. Exploding Gradients

Increase:

- Learning Rate
- Loss Scale

Observe exploding gradient norms.

Now enable **Gradient Clipping**.

See how the gradients become bounded.
""")

else:

    st.info(
        """
Select your architecture from the sidebar and press

**Run One Backward Pass**

to compute gradient flow through the network.
"""
    )
