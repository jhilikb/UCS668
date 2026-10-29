import streamlit as st
import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.utils.prune as prune
import torch.quantization as quant
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import json
import requests
import time
import torchvision.models.quantization as qmodels
import psutil
import matplotlib.pyplot as plt
#from torch.autograd import profiler
import torch.profiler as profiler

st.set_page_config(page_title="PyTorch Tricks Playground", layout="wide")

st.title("🎮 PyTorch Tricks Playground")
st.markdown("Learn **Pruning**, **Quantization**, and **Gradient Checkpointing** with interactive stories and code!")

# --- Tab Setup ---
tab1, tab2, tab3 = st.tabs(["✂️ Pruning", "⚖️ Quantization", "🧠 Gradient Checkpointing"])
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = requests.get(url).text.splitlines()
# --- Pruning Tab ---
with tab1:
    st.header("✂️ Pruning: The Garden Story")
    st.markdown("""
    Imagine your neural network as a **garden of neurons** 🌱. Some branches are weak and don’t contribute much.
    Pruning is like trimming the garden to make it healthier and more efficient.
    """)

    # Choose pruning type
    
    # Demo model
    #model = nn.Linear(10, 5)
    model = models.resnet18(pretrained=True)
    model.eval()
    # Load ImageNet class labels
    

    uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg","JPEG"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Input Image")

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        input_tensor = transform(img).unsqueeze(0)

        pruning_type = st.radio("Choose pruning type:", 
                            ["L1 Unstructured", "Random Unstructured", "Structured (Neuron)", "Global Unstructured"])


        if pruning_type == "L1 Unstructured":
            st.subheader("🌿 L1 Unstructured Pruning")
            st.markdown("""
            This trims weights with the **smallest L1 norm** (absolute values).  
            Like cutting off the weakest twigs in the garden.
            """)
            amount = st.slider("Pruning Amount (%)", 0.0, 0.9, 0.4)
            prune.l1_unstructured(model.fc, name="weight", amount=amount)
            st.code("""
            prune.l1_unstructured(model, name="weight", amount=0.4)
            """, language="python")
            #st.write("Pruned Weights:", model.weight)

        elif pruning_type == "Random Unstructured":
            st.subheader("🎲 Random Unstructured Pruning")
            st.markdown("""
            This randomly removes connections, like a gardener tossing dice 🎲 to decide which branches to cut.
            """)
            amount = st.slider("Random Pruning Amount (%)", 0.0, 0.9, 0.3)
            prune.random_unstructured(model.fc, name="weight", amount=amount)
            st.code("""
            prune.random_unstructured(model, name="weight", amount=0.3)
            """, language="python")
            #st.write("Pruned Weights:", model.weight)

        elif pruning_type == "Structured (Neuron)":
            st.subheader("🌳 Structured Pruning (Neuron)")
            st.markdown("""
            Instead of cutting individual twigs, this removes **entire branches** (neurons).  
            It’s like chopping off whole stems to reshape the garden. For example remove 50%(amount=0.5) of output neurons(dim=0) using smallest l2 norm values (n=2)
            """)
            amount = st.slider("Pruning Amount (%)", 0.0, 0.9, 0.4)
            prune.ln_structured(model.fc, name="weight", amount=amount, n=2, dim=1)
            st.code("""
            prune.ln_structured(model, name="weight", amount=0.5, n=2, dim=1)
            """, language="python")
            #st.write("Pruned Weights:", model.weight)

        elif pruning_type == "Global Unstructured":
            st.subheader("🌍 Global Unstructured Pruning")
            st.markdown("""
            Instead of pruning layer by layer, this looks at the **whole garden** and trims the weakest twigs globally.
            """)
            #modules = [(model, 'weight')]
            parameters_to_prune = [
    (model.layer1[0].conv1, 'weight'),
    (model.layer1[0].conv2, 'weight'),
    (model.fc, 'weight'),
]
            amount = st.slider("Pruning Amount (%)", 0.0, 0.9, 0.4)
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
            st.code("""
            modules = [(model, 'weight')]
            prune.global_unstructured(modules, pruning_method=prune.L1Unstructured, amount=0.4)
            """, language="python")
            #st.write("Pruned Weights:", model.weight)

   

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top5 = torch.topk(probs, 5)

        st.write("Top-5 Predictions after pruning:")
        for idx, prob in zip(top5.indices, top5.values):
            classname = imagenet_classes[idx.item()]
            st.write(f"{classname} — {prob.item():.4f}")
        st.success("Gamified Challenge: Try each pruning style and see which keeps the garden healthy while reducing size!")


# --- Quantization Tab ---
with tab2:
    st.header("⚖️ Quantization: The Backpack Story")
    st.markdown("""
    Imagine carrying a **backpack full of weights** 🎒. If the weights are super precise (32-bit floats),
    your backpack is heavy. Quantization compresses them into lighter integers (like 8-bit), making deployment faster.
    """)

    st.title("⚖️ ResNet18 Precision Playground")
    st.markdown("Compare **FP32**, **FP16**, and **INT8 quantized** inference on ResNet‑18.")

    # Upload image
    uploaded_file = st.file_uploader("Upload another image", type=["jpg","png","jpeg","JPEG"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Input Image")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        input_tensor = transform(img).unsqueeze(0)

        # FP32 model
        model_fp32 = models.resnet18(pretrained=True).eval()

        # FP16 model (half precision, works best on GPU)
        model_fp16 = models.resnet18(pretrained=True).eval().half()
        input_fp16 = input_tensor.half()

        # INT8 quantized model (CPU only)
        # 1. Load pretrained ResNet18
        model = qmodels.resnet18(pretrained=True, quantize=True)
        model.eval()
        

        # # 2. Fuse common layers (Conv+BN+ReLU)
        # model_fused = torch.quantization.fuse_modules(
        #     model,
        #     [['conv1', 'bn1', 'relu']],  # fuse first block
        #     inplace=True
        # )

        # # 3. Set quantization config
        # model_fused.qconfig = quant.get_default_qconfig('fbgemm')

        # # 4. Prepare (insert observers)
        # quant.prepare(model_fused, inplace=True)

        # # 5. Calibrate (run representative data through model)
        # # for images, _ in dataloader:
        # #     model_fused(images)

        # # 6. Convert to quantized model
        # quant.convert(model_fused, inplace=True)
        # Run inference and measure time
        def run_inference(model, inp, label):
            start = time.time()
            with torch.no_grad():
                out = model(inp)
            elapsed = time.time() - start
            probs = torch.nn.functional.softmax(out[0], dim=0)
            top5 = torch.topk(probs, 5)
            st.subheader(f"{label} Inference")
            st.write(f"Time: {elapsed:.4f} sec")
            for idx, prob in zip(top5.indices, top5.values):
                st.write(f"{imagenet_classes[idx.item()]} — {prob.item():.4f}")

        

        col1, col2 = st.columns(2)

    # Results column
        with col1:
            st.subheader("Results")
            run_inference(model_fp32, input_tensor, "FP32")
            run_inference(model_fp16, input_fp16, "FP16")
            run_inference(model, input_tensor, "INT8")

        # Code column
        with col2:
            st.subheader("Code Snippets")
            st.code("""
input_tensor = transform(img).unsqueeze(0)
def run_inference(model, inp, label):
    start = time.time()
    with torch.no_grad():
        out = model(inp)
    elapsed = time.time() - start
    probs = torch.nn.functional.softmax(out[0], dim=0)
    top5 = torch.topk(probs, 5)
    
    print(f\"Time: {elapsed:.4f} sec\")
    for idx, prob in zip(top5.indices, top5.values):
        print(f\"{imagenet_classes[idx.item()]} — {prob.item():.4f}\")
""", language="python")
            st.code("model_fp32 = models.resnet18(pretrained=True).eval() \n  run_inference(model_fp32, input_tensor, \"FP32\")", language="python")
            st.code("model_fp16 = models.resnet18(pretrained=True).eval().half() \n input_fp16 = input_tensor.half() \n run_inference(model_fp16, input_fp16, \"FP16\")", language="python")
            st.code("model_int8 = torchvision.models.quantization.resnet18(pretrained=True, quantize=True).eval() \n run_inference(model_int8,input_tensor, \"INT8\")", language="python")


# --- Gradient Checkpointing Tab ---
with tab3:
    st.header("🧠 Gradient Checkpointing: The Memory Adventure")
    st.markdown("""
    Imagine your brain 🧠 trying to remember every detail of a long book.
    Gradient checkpointing says: *Don't remember everything, just re-read when needed!*  
    This saves memory at the cost of extra computation.
    """)

        # Dummy dataset
    X = torch.randn(32, 3, 64, 64)  # batch of 32 images
    y = torch.randint(0, 10, (32,)) # 10 classes

    # Model with optional checkpointing
    class BigModel(nn.Module):
        def __init__(self, use_checkpoint=False):
            super().__init__()
            self.use_checkpoint = use_checkpoint
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv4a = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv4b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv4c = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv5a = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv5b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv5c = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(512*64*64, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            def block1(x):
                return torch.relu(self.conv1(x))
            def block2(x):
                return torch.relu(self.conv2(x))
            def block3(x):
                return torch.relu(self.conv3(x))
            def block4(x):
                return torch.relu(self.conv4c(self.conv4b(self.conv4a(x))))
            def block5(x):
                return torch.relu(self.conv5c(self.conv5b(self.conv5a(x))))
            def block6(x):
                return torch.relu(self.conv6(x))

            if self.use_checkpoint:
                # ✅ Adding checkpoints here
                x = checkpoint.checkpoint(block1, x)
                x = checkpoint.checkpoint(block2, x)
                x = checkpoint.checkpoint(block3, x)
                x = checkpoint.checkpoint(block4, x)
                x = checkpoint.checkpoint(block5, x)
                x = checkpoint.checkpoint(block6, x)
            else:
                # 🚀 Normal forward without checkpoints
                x = block1(x)
                x = block2(x)
                x = block3(x)
                x = block4(x)
                x = block5(x)
                x = block6(x)

            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    


    # Training loop
    def train_one_epoch(model, X, y):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        start = time.time()
        optimizer.zero_grad()
        with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],  # add CUDA if you have GPU
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=profiler._ExperimentalConfig(verbose=True)
    ) as prof:
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Show aggregated table
        if X.shape[0]==8:
            st.text(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Optionally export Chrome trace for timeline visualization
        prof.export_chrome_trace("trace.json")
        elapsed = time.time() - start
        mem_used = psutil.Process().memory_info().rss / (1024**2)  # MB
        return elapsed, mem_used

    # Streamlit layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Results")
        # model_no_ckpt = BigModel(use_checkpoint=False)
        # t1, m1 = train_one_epoch(model_no_ckpt, X, y)
        # st.write(f"Without checkpointing: {t1:.4f} sec, {m1:.2f} MB")

        # model_ckpt = BigModel(use_checkpoint=True)
        # t2, m2 = train_one_epoch(model_ckpt, X, y)
        # st.write(f"With checkpointing: {t2:.4f} sec, {m2:.2f} MB")

        batch_sizes = [8, 16, 32, 64]
        results_no_ckpt = []
        results_ckpt = []

        for bs in batch_sizes:
            #X = torch.randn(bs, 3, 64, 64)
            X = torch.randn(bs, 3, 64, 64, requires_grad=True)

            y = torch.randint(0, 10, (bs,))
            model_no = BigModel(use_checkpoint=False)
            t1, m1 = train_one_epoch(model_no, X, y)
            st.write(f"Without checkpointing: {bs}- {t1:.4f} sec, {m1:.2f} MB")
            results_no_ckpt.append((t1, m1))

            model_ckpt = BigModel(use_checkpoint=True)
            t2, m2 = train_one_epoch(model_ckpt, X, y)
            st.write(f"With checkpointing:{bs}- {t2:.4f} sec, {m2:.2f} MB")
            results_ckpt.append((t2, m2))

        # Plot
        st.subheader("Memory vs Batch Size")
        fig, ax = plt.subplots()
        ax.plot(batch_sizes,[m for _,m in results_no_ckpt],'r-o',label="No CKPT")
        ax.plot(batch_sizes,[m for _,m in results_ckpt],'b-o',label="CKPT")
        ax.set_xlabel("Batch Size"); ax.set_ylabel("Memory (MB)")
        ax.legend(); st.pyplot(fig)

    
        st.subheader("Time vs Batch Size")
        fig, ax = plt.subplots()
        ax.plot(batch_sizes,[t for t,_ in results_no_ckpt],'r--o',label="No CKPT")
        ax.plot(batch_sizes,[t for t,_ in results_ckpt],'b--o',label="CKPT")
        ax.set_xlabel("Batch Size"); ax.set_ylabel("Time (sec)")
        ax.legend(); st.pyplot(fig)

    with col2:
        st.subheader("Code Snippet")
        st.code("""
    def __init__(self, use_checkpoint=False):
            super().__init__()
            self.use_checkpoint = use_checkpoint
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128*64*64, 256)
            self.fc2 = nn.Linear(256, 10)
    # Forward pass with and without checkpoints
    def forward(self, x):
        def block1(x):
            return torch.relu(self.conv1(x))
        def block2(x):
            return torch.relu(self.conv2(x))

        if self.use_checkpoint:
            # ✅ Adding checkpoints
            x = checkpoint.checkpoint(block1, x)
            x = checkpoint.checkpoint(block2, x)
        else:
            # 🚀 Normal forward
            x = block1(x)
            x = block2(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    """, language="python")
        st.code("""
def train_one_epoch(model, X, y):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    elapsed = time.time() - start
    mem_used = psutil.Process().memory_info().rss / (1024**2)  # MB
    return elapsed, mem_used
model_no_ckpt = BigModel(use_checkpoint=False)
t1, m1 = train_one_epoch(model_no_ckpt, X, y)
model_no_ckpt = BigModel(use_checkpoint=True)
t2, m2 = train_one_epoch(model_no_ckpt, X, y)
""", language="python")
