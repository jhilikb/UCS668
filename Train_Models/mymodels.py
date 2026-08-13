
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()])
transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])
# --- Dummy Models ---
class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Conv2d(3,32,3),nn.MaxPool2d(2),nn.Conv2d(32,64,3),nn.Flatten(),nn.Linear(10816, 1024),nn.ReLU(),nn.ReLU(),nn.Linear(1024, 128),nn.Linear(128, 10))  

    def forward(self, x):
        return self.fc(x)



# --- Streamlit UI ---
st.title("🖼️ Fun CV Playground")

tab1, tab2, tab3 = st.tabs(["Classification from Scratch", "Transfer Learning","Common Concepts"])

# --- Classification Tab ---
with tab1:
    st.header("Classification Demo")
    st.subheader("Let us learn to classify flowers")
    if "stage" not in st.session_state:
    	st.session_state.stage = 0
    if st.session_state.stage == 0:
        st.subheader("📂 Step 1: Load Data")
        if st.button("Collect Training Scrolls"):
        
            dataset = ImageFolder(root="Flower Classification Dataset/train", transform=transform)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            st.session_state.dataset = dataset
            st.session_state.dataloader = dataloader
            datasett = ImageFolder(root="Flower Classification Dataset/test", transform=transform)
            testloader = DataLoader(datasett, batch_size=4, shuffle=True)
            st.session_state.datasett = datasett
            st.session_state.testloader = testloader
            st.success("🎉 Data basket filled with flowers!")
            st.session_state.stage = 1
            images, labels = next(iter(dataloader))
            st.image(images.permute(0,2,3,1).numpy(),
                     caption=[dataset.classes[l] for l in labels],
                     width=64)
            time.sleep(1)
            st.rerun()

	# --- Stage 2: Create Model ---
    elif st.session_state.stage == 1:
        st.subheader("🤖 Step 2: Create Model")
        if st.button("Summon Tiny Wizard"):
            model=TinyClassifier()
            st.session_state.model = model
            st.success("✨ A tiny neural wizard appears, ready to learn!")
            st.session_state.stage = 2
            st.rerun()

        

        

        
        



	# --- Stage 3: Train Model ---
    elif st.session_state.stage == 2:
        st.subheader("⚙️ Step 3: Train Model")
        if "num_epochs" not in st.session_state:
            st.session_state.num_epochs = 3
        st.session_state.num_epochs = st.number_input(
        "Enter number of epochs:",
        min_value=1, max_value=20,
        value=st.session_state.num_epochs,
        step=1
    )
        if st.button("Teach Wizard Spells"):
            model = st.session_state.model
            dataloader = st.session_state.dataloader
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            loss_values = []
            # Training loop
            for epoch in range(st.session_state.num_epochs):
                running_loss = 0.0
                for images, labels in dataloader:
                # Flatten images if needed
                    images = images.to(torch.float32)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                avg_loss = running_loss / len(dataloader)
                loss_values.append(avg_loss)
                st.write(f"Epoch {epoch+1}/{st.session_state.num_epochs} - Loss: {avg_loss:.4f}")
            
            st.success("📉 Wizard mastered recognition spells!")
            st.session_state.trained_model = model
            time.sleep(1)
            st.session_state.stage = 3
            st.rerun()

	# --- Stage 4: Prediction ---
    elif st.session_state.stage == 3:
        st.subheader("🔮 Step 4: Prediction")
        if st.button("Wizard Makes a Prophecy"):
            #prediction = random.choice(["🍎 Apple", "🍌 Banana", "🐱 Cat"])
            model = st.session_state.trained_model
            testloader = st.session_state.testloader   # assume you stored a test DataLoader
            images, labels = next(iter(testloader))
            outputs = model(images.to(torch.float32))
            _, preds = torch.max(outputs, 1)
            class_names = st.session_state.dataset.classes
            captions = [
            f"Pred: {class_names[p]} | Actual: {class_names[l]}"
            for p, l in zip(preds, labels)
        ]
            st.image(images.permute(0,2,3,1).numpy(), caption=captions, width=64)
            #st.success(f"Wizard predicts: {prediction}")
            st.balloons()
            time.sleep(1)
            st.session_state.stage = 4  # reset for replay
            st.rerun()
    elif st.session_state.stage == 4:
        st.success("🎉 Wizard completed all stages!")

        mode = st.radio("What would you like to do next?",
                    ["Choose","See Explanations","Replay Adventure"])

        if mode == "Replay Adventure":
            st.session_state.stage = 0  # reset for replay
            st.rerun()
        elif mode == "See Explanations":
            st.subheader("📖 Learn What Each Module Does")

        # --- Stage 1: Data Loading ---
            with st.expander("Stage 1: Data Loading"):
                st.code("""
dataset = ImageFolder(root="Flower Classification Dataset/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
images, labels = next(iter(dataloader))
""", language="python")

                st.caption("Line 1: `ImageFolder` reads images from folders you provide, using folder names as class labels.")
                st.caption("Line 2: `DataLoader` batches and shuffles the dataset for training.")
                st.caption("Line 3: `next(iter(dataloader))` grabs one batch of images and labels.")

        # --- Stage 2: Model Creation ---
            with st.expander("Stage 2: Model Creation"):
                st.code("""
class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3),
            nn.Flatten(),
            nn.Linear(10816, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(x)
""", language="python")

                st.caption("Conv2d layers extract features from the image.")
                st.caption("MaxPool2d reduces spatial size, keeping important features.")
                st.caption("Flatten converts the 2D feature maps into a vector.")
                st.caption("Linear layers map features to class scores (here 10 flower classes).")

        # --- Stage 3: Training ---
            with st.expander("Stage 3: Training Loop"):
                st.code("""
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
""", language="python")

                st.caption("CrossEntropyLoss measures how far predictions are from actual labels.")
                st.caption("Adam optimizer updates weights to minimize loss.")
                st.caption("Loop runs for chosen number of epochs.")
                st.caption("Each batch: forward pass i.e. get output using model→ compute loss actual vs predicted → backward pass i.e. compute gradients→ update weights using gradients.")

        # --- Stage 4: Prediction ---
            with st.expander("Stage 4: Prediction"):
                st.code("""
images, labels = next(iter(testloader))
outputs = model(images)
_, preds = torch.max(outputs, 1)

captions = [f"Pred: {class_names[p]} | Actual: {class_names[l]}"
            for p, l in zip(preds, labels)]
st.image(images.permute(0,2,3,1).numpy(), caption=captions)
""", language="python")

                st.caption("Grab one batch from the test set.")
                st.caption("Run the model to get outputs (logits).")
                st.caption("Take argmax to get predicted class index.")
                st.caption("Map predictions and actual labels to flower classes.")
                st.caption("Display images with both predicted and actual captions.")

        else:
            st.info("Select an option")
# --- Detection Tab ---
with tab2:
    st.header("🐱 Classify with transfer learning")
    st.write("Use pretrained models!")
    st.header("Classification Demo")
    st.subheader("Let us learn to classify flowers")
    if "staget" not in st.session_state:
        st.session_state.staget = 0
    if st.session_state.staget == 0:
        st.subheader("📂 Step 1: Load Data")
        if st.button("Load Data for Transfer Learning"):
        
            dataset = ImageFolder(root="Flower Classification Dataset/train", transform=transform1)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            st.session_state.dataset = dataset
            st.session_state.dataloader = dataloader
            datasett = ImageFolder(root="Flower Classification Dataset/test", transform=transform1)
            testloader = DataLoader(datasett, batch_size=2, shuffle=True)
            st.session_state.datasett = datasett
            st.session_state.testloader = testloader
            st.success("🎉 Data basket filled with flowers!")
            st.session_state.staget = 1
            images, labels = next(iter(dataloader))
            st.image(images.permute(0,2,3,1).numpy(),
                     caption=[dataset.classes[l] for l in labels],
                     width=64)
            time.sleep(1)
            st.rerun()

    # --- Stage 2: Create Model ---
    elif st.session_state.staget == 1:
        st.subheader("🤖 Step 2: Create Model")
        model_choice = st.selectbox(
        "Choose a model architecture:",
        ["Previous model","ResNet18", "AlexNet", "VGG16"]
    )
        if st.button("Confirm Model Choice"):
            if model_choice == "ResNet18":
                model = models.resnet18(pretrained=True)
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, len(st.session_state.dataset.classes))

            elif model_choice == "AlexNet":
                model = models.alexnet(pretrained=True)
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(model.classifier[6].in_features,
                                            len(st.session_state.dataset.classes))
            elif model_choice == "VGG16":
                model = models.vgg16(pretrained=True)
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(model.classifier[6].in_features,
                                            len(st.session_state.dataset.classes))
            else:
                model=TinyClassifier()
            st.session_state.modelt = model
            st.success("✨ A tiny neural wizard appears, ready to learn!")
            st.session_state.staget = 2
            st.rerun()

        

        

        
        



    # --- Stage 3: Train Model ---
    elif st.session_state.staget == 2:
        st.subheader("⚙️ Step 3: Train after choosing model for Transfer Learning")
        if "num_epochs" not in st.session_state:
            st.session_state.num_epochs = 3
        st.session_state.num_epochs = st.number_input(
        "Enter number of epochs:",
        min_value=1, max_value=20,
        value=st.session_state.num_epochs,
        step=1
    )
        if st.button("Adapt Wizard Spells"):
            model = st.session_state.modelt
            dataloader = st.session_state.dataloader
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            loss_values = []
            # Training loop
            for epoch in range(st.session_state.num_epochs):
                running_loss = 0.0
                for images, labels in dataloader:
                # Flatten images if needed
                    images = images.to(torch.float32)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                avg_loss = running_loss / len(dataloader)
                loss_values.append(avg_loss)
                st.write(f"Epoch {epoch+1}/{st.session_state.num_epochs} - Loss: {avg_loss:.4f}")
            
            st.success("📉 Wizard mastered recognition spells!")
            st.session_state.trained_modelt = model
            time.sleep(1)
            st.session_state.staget = 3
            st.rerun()

    # --- Stage 4: Prediction ---
    elif st.session_state.staget == 3:
        st.subheader("🔮 Step 4: Prediction post transfer learning")
        if st.button("Check your results"):
            #prediction = random.choice(["🍎 Apple", "🍌 Banana", "🐱 Cat"])
            model = st.session_state.trained_modelt
            testloader = st.session_state.testloader   # assume you stored a test DataLoader
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in testloader:
                    outputs = model(images.to(torch.float32))
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                accuracy = 100 * correct / total
            st.success(f"✨ Wizard’s accuracy on test set: {accuracy:.2f}%")
            st.balloons()
            time.sleep(1)
            st.session_state.staget = 0  # reset for replay
            st.rerun()
    

# --- Segmentation Tab ---
with tab3:
    st.header("📚 Wizard’s Knowledge Scroll")

    with st.expander("Why resize to 224 for AlexNet?"):
        st.write("AlexNet (and many torchvision models) were trained on ImageNet, "
                 "where all images are resized to 224×224. The network expects this input size "
                 "because the convolution and fully connected layers are dimensioned accordingly.")

    with st.expander("Transfer Learning vs Fine-tuning"):
        st.write("**Transfer Learning**: Freeze pretrained layers and train only the final classifier. "
                 "This is fast and works well when your dataset is small.\n\n"
                 "**Fine-tuning**: Unfreeze some or all layers and retrain them on your dataset. "
                 "This adapts the feature extractor more deeply but requires more data and compute.")

    with st.expander("Why set `requires_grad=False`?"):
        st.write("This freezes the parameters of pretrained layers so gradients are not computed. "
                 "It saves memory and ensures only the last layer’s weights are updated during training.")

    with st.expander("Matching dimension of last FC layer"):
        st.write("The final fully connected layer must output the same number of units as your dataset’s classes. "
                 "For example, if you have 10 classes, set `nn.Linear(in_features, 4)`.")

    with st.expander("Which network is better?"):
        st.write("It depends:\n"
                 "- **ResNet18**: Lightweight, residual connections help deeper learning.\n"
                 "- **AlexNet**: Historically important, simpler, but less accurate today.\n"
                 "- **VGG16**: Very deep, good accuracy, but heavier in parameters.\n"
                 "For small datasets, ResNet18 is often a good balance.")
    with st.expander("When to use Transfer Learning"):
        st.write("Transfer learning is best when your dataset is small or similar to ImageNet but have different classes. "
                 "You freeze pretrained layers (`requires_grad=False`) and train only the last classifier. "
                 "This is fast, avoids overfitting, and leverages learned features like edges and textures.")

    with st.expander("When to use Fine‑tuning"):
        st.write("Fine‑tuning is useful when your dataset is large or very different from ImageNet. "
                 "You unfreeze some or all layers and retrain them. This adapts the feature extractor "
                 "to your domain (e.g., medical images, satellite images) but requires more compute and careful regularization.")


    with st.expander("Basic differences between the networks"):
        st.write("- **AlexNet**: Early CNN, 8 layers, large kernels.\n"
                 "- **VGG16**: 16 layers, uses many small 3×3 convolutions stacked.\n"
                 "- **ResNet18**: 18 layers, introduces residual skip connections to avoid vanishing gradients.")

    


