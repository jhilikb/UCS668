import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Extended dictionary of activation functions

# --- Gamified Learning Tab ---
def gamified_learning():
    st.header("🎮 Gamified Learning")

    # Example 1: Guess the Function
    st.subheader("Guess the Activation Function")
    st.write("Look at the curve below. Which activation function is it?")

    # Generate a random function to quiz
    funcs = {
        "ReLU": lambda x: np.maximum(0, x),
        "Sigmoid": lambda x: 1/(1+np.exp(-x)),
        "Tanh": lambda x: np.tanh(x),
        "Leaky ReLU": lambda x: np.where(x>=0, x, 0.1*x),
    }
    choice = np.random.choice(list(funcs.keys()))
    x = np.linspace(-5, 5, 200)
    y = funcs[choice](x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    st.pyplot(fig)

    guess = st.radio("Which function is this?", list(funcs.keys()))
    if st.button("Check Answer"):
        if guess == choice:
            st.success("✅ Correct! It was " + choice)
        else:
            st.error("❌ Nope, it was " + choice)

    # Example 2: Parameter Challenge
    st.subheader("Leaky ReLU Parameter Challenge")
    alpha = st.slider("Adjust α (negative slope)", 0.0, 1.0, 0.1)
    y_alpha = np.where(x>=0, x, alpha*x)
    fig2, ax2 = plt.subplots()
    ax2.plot(x, y_alpha)
    ax2.set_title(f"Leaky ReLU with α={alpha}")
    st.pyplot(fig2)
    st.write("Try different α values and see how the curve changes!")

    # Example 3: Input Playground
    st.subheader("Input Playground")
    val = st.slider("Choose an input x", -5.0, 5.0, 0.0)
    st.write("ReLU:", np.maximum(0, val))
    st.write("Sigmoid:", 1/(1+np.exp(-val)))
    st.write("Tanh:", np.tanh(val))
    st.write("SiLU:", val/(1+np.exp(-val)))

tab1, tab2 = st.tabs(["Learn", "Gamified Learning"])
activations = {
    "ReLU": {
        "desc": "Rectified Linear Unit. Best for deep networks, avoids vanishing gradients.",
        "torch": "torch.nn.ReLU() or F.relu(x)",
        "math": r"f(x) = \max(0, x)",
        "explain": "Outputs zero for negative inputs, identity for positive inputs."
    },
    "Leaky ReLU": {
        "desc": "Allows small gradient for negative inputs, avoids dying ReLU problem.",
        "torch": "torch.nn.LeakyReLU(negative_slope=0.01)",
        "math": r"f(x) = \begin{cases} x & x \ge 0 \\ \alpha x & x < 0 \end{cases}",
        "explain": "Negative inputs scaled by α instead of zero."
    },
    "Parametric ReLU": {
        "desc": "Like Leaky ReLU but slope α is learned during training.",
        "torch": "torch.nn.PReLU()",
        "math": r"f(x) = \begin{cases} x & x \ge 0 \\ \alpha x & x < 0 \end{cases}",
        "explain": "Model learns optimal slope for negative side."
    },
    "ELU": {
        "desc": "Exponential Linear Unit. Smooth negative side, avoids dead neurons.",
        "torch": "torch.nn.ELU(alpha=1.0)",
        "math": r"f(x) = \begin{cases} x & x \ge 0 \\ \alpha(e^x - 1) & x < 0 \end{cases}",
        "explain": "Negative inputs map smoothly, reducing bias shift."
    },
    "SELU": {
        "desc": "Scaled ELU. Self-normalizing networks.",
        "torch": "torch.nn.SELU()",
        "math": r"f(x) = \lambda \begin{cases} x & x \ge 0 \\ \alpha(e^x - 1) & x < 0 \end{cases}",
        "explain": "Keeps mean and variance stable across layers."
    },
    "Tanh": {
        "desc": "Squashes input to [-1,1]. Useful for centered outputs.",
        "torch": "torch.nn.Tanh() or torch.tanh(x)",
        "math": r"f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}",
        "explain": "Smooth, symmetric, but can saturate for large |x|."
    },
    "Sigmoid": {
        "desc": "Classic squashing function to [0,1].",
        "torch": "torch.nn.Sigmoid() or torch.sigmoid(x)",
        "math": r"f(x) = \sigma(x) = \frac{1}{1+e^{-x}}",
        "explain": "Good for probabilities, but saturates easily."
    },
    "SiLU / Swish": {
        "desc": "Smooth nonlinearity, combines ReLU and sigmoid behavior.",
        "torch": "torch.nn.SiLU() or F.silu(x)",
        "math": r"f(x) = x \cdot \sigma(x)",
        "explain": "Sigmoid gate scales input, avoids hard cutoff."
    },
    "GELU": {
        "desc": "Gaussian Error Linear Unit. Used in Transformers.",
        "torch": "torch.nn.GELU() or F.gelu(x)",
        "math": r"f(x) = x \cdot \Phi(x)",
        "explain": "Smoothly weights input by probability it’s positive."
    },
    "Softplus": {
        "desc": "Smooth approximation of ReLU.",
        "torch": "torch.nn.Softplus()",
        "math": r"f(x) = \ln(1+e^x)",
        "explain": "Always positive, differentiable everywhere."
    },
    "Hard Sigmoid": {
        "desc": "Piecewise linear approximation of sigmoid.",
        "torch": "torch.nn.Hardsigmoid()",
        "math": r"f(x) = \max(0, \min(1, (x+1)/2))",
        "explain": "Computationally cheaper than sigmoid."
    }
}

with tab1: 
    # Sidebar selection
    choice = st.sidebar.selectbox("Choose Activation Function", list(activations.keys()))

    # Display info
    info = activations[choice]
    st.header(choice)
    st.write("**Description:**", info["desc"])
    st.write("**PyTorch Code:**", info["torch"])
    st.latex(info["math"])
    st.write("**Explanation:**", info["explain"])

    # Plot graph
    x = np.linspace(-5, 5, 400)
    if choice == "ReLU":
        y = np.maximum(0, x)
    elif choice == "Leaky ReLU":
        y = np.where(x >= 0, x, 0.01*x)
    elif choice == "Parametric ReLU":
        alpha = 0.2
        y = np.where(x >= 0, x, alpha*x)
    elif choice == "ELU":
        alpha = 1.0
        y = np.where(x >= 0, x, alpha*(np.exp(x)-1))
    elif choice == "SELU":
        alpha, lam = 1.67326, 1.0507
        y = np.where(x >= 0, lam*x, lam*alpha*(np.exp(x)-1))
    elif choice == "Tanh":
        y = np.tanh(x)
    elif choice == "Sigmoid":
        y = 1 / (1 + np.exp(-x))
    elif choice == "SiLU / Swish":
        y = x / (1 + np.exp(-x))
    elif choice == "GELU":
        y = x * norm.cdf(x)
    elif choice == "Softplus":
        y = np.log(1 + np.exp(x))
    elif choice == "Hard Sigmoid":
        y = np.clip((x+1)/2, 0, 1)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(f"{choice} Activation")
    st.pyplot(fig)

with tab2:
    gamified_learning()