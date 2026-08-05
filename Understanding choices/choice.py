import streamlit as st
st.title("A Vision Learning Adventure")
# Create tabs
tab1, tab2 = st.tabs(["🌸 Garden Mystery", "🌱 Farm Doctor"])

with tab1:

    # --- Scene 1: Introduction ---
    st.header("Scene 1: The Garden at Peace")
    st.image("garden.jpeg", caption="A peaceful garden...")
    st.write("All is well in the garden… until intruders arrive. As a young student you want to develop a solution to find the wrong doer?")

    choice1 = st.radio(
        "Choose your starting strategy:",
        ["Using a camera to record events in your absence", "Using a motion sensor to judge movement in your garden", "Using other sensors like laser"]
    )

    if choice1 == "Using a camera to record events in your absence":
        st.success("✅ Smart choice! Camera will be able to detect the state of the garden at different instances along with the intruder.")
    elif choice1 == "Using a motion sensor to judge movement in your garden":
        st.info("While motion sensor can map changes and sound alarms, it can be triggered by a cat or a bird just minding their own business. Moreover you wont understand if the movement results in any damage, or who the damage doer was.")
    else:
        st.info("Lasers or Lidars do capture data points in the scene, but again will not be informative without a RGB camera which can check the garden as well as the intruder if any.")

    # --- Scene 2: Model Choice ---
    st.header("Scene 2: The Intruder Appears")
    st.image("messed.png", caption="An intruder sneaks in...")
    st.write("Someone is near the flowers. What’s the best way to detect them?")

    choice2 = st.selectbox(
        "Decide the type of vision task required:",
        ["Classification", "Detection", "Segmentation"]
    )
    #b  
    if choice2 == "Classification":
        st.warning(" 📊Sure you may feel like you want to classify neat garden vs messy garden, but this technique has its loopholes. For one you need to collect a lot of images clean/messy to train your model. Also messy is a relative term, just destroying a pot may feel like messy for the owner but clean in terms of the image. ")
    elif choice2 == "Detection":
        st.success("✅Good choice. Just detecting a person with a object detector in the frame may save you the trouble. If person is present, you may  save the frame. This will later be helpful for the owner.")
    else:
        st.warning("🌀 Segmentation is powerful but may be overkill for simple person detection. You dont need to create masks in the whole image just to detect a person in the frame. A general detector will be more handy.")
    # --- Scene 3: Model Choice ---
    st.header("Scene 3: Catching the  Intruder")
    st.image("intruder.png", caption="An intruder detected...")
    st.write("What’s the best way to detect the intruder?")

    choice3 = st.selectbox(
        "Pick a detection model:",
        ["Pretrained YOLO", "Train YOLO","Faster R-CNN", "DETR"]
    )

    if choice3 == "Pretrained YOLO":
        st.success("✅ YOLO is fast and accurate for real-time detection. Remember its already trained on COCO dataset which has person class, so you can use it right away. ")
    elif choice3 == "Train YOLO":
        st.error("This will be a waste completely. Pretrained YOLO is available with good accuracy and has person class. Training is only required if the intended class(ex person in this case) is not present in the pretrained model. Good training requires a lot of time, huge amount of labeled data and should be avoided if available models serve the purpose.")
    elif choice3 == "Faster R-CNN":
        st.info("📊 Good accuracy, but slower than YOLO.")
    else:
        st.warning("🌀 DETR is powerful but may be too much resource consuming for simple person detection.")

    # --- Scene 4: Data Dilemma ---
    st.header("Scene 4: Data Dilemma")
    st.write("The garden is dark at night. Your detector struggles. What do you do?")

    choice4 = st.radio(
        "Choose your solution:",
        ["Augment dataset and finetune detector", "Collect more night images and finetune detector", "Use a preprocessing to enhance images without training detector","Ignore the problem"]
    )

    if choice4 == "Augment dataset and finetune detector":
        st.success("✅ Correct! Augmentation (add brightness, contrast, etc. to existing images) helps models generalize. However use light finetuning only, dont retrain")
    elif choice4 == "Collect more night images and finetune detector":
        st.info("📸 More data helps, but augmentation is faster and cheaper.")
    elif choice4 == "Use a preprocessing to enhance images without training detector":
        st.info("📸 You can preprocess your image to make it brighter before passing it to the detector. However you need to check that your preprocessing is helping.")
    else:
        st.error("⚠️ Ignoring the problem means your detector fails at night.")

    # --- Scene 5: Timeline ---
    st.header("Scene 5: Logging Events")
    st.write("What is the best way to maintain logs?")

    choice5 = st.selectbox(
        "Pick your analysis method:",
        ["Save frame on detecting person", "Save detection", "Save all frames","Do nothing"]
    )

    if choice5 == "Save frame on detecting person":
        st.success("✅ Great! Saved data is minimal and will require less space as well as less time for viewing.")
    elif choice5 == "Save detection":
        st.info("📊 If you save only detection results, you will know whether people entered your garden, but will not be able to figure out when the mess happened or who was responsible.")
    elif choice5 == "Save all frames":
        st.info("📊 Too much data, works like a basic surveillance camera without any automation (person detection) help.")
    else:
        st.error("⚠️ Doing nothing means you miss valuable insights.")

    # --- Wrap-up ---
    st.header("Scene 6: Wrap-up")
    st.write("🎉 Congratulations! You’ve protected the flowers.")
    st.write("Optimal pipeline: Camera sensor → Detection Problem → Pretrained YOLO → Dataset augmentation → Timeline visualization.")

# --- Tab 2: Farm Doctor ---
with tab2:
    st.header("Scene 1: The Farm Problem")
    st.image("leaf.jpeg", caption="A peaceful garden...")
    st.write("Farmers are worried — some leaves show disease, can AI solve this task for an early warning?")
    choice0 = st.radio("What’s your first step?",
                       ["Use RGB camera based analysis", "Use Multispectral Camera based Analysis", "Use satellite data"])
    if choice0 == "Use RGB camera based analysis":
        st.success("✅ This is the simplest and cheapest. You can analyse images and report decision or early warning.")
    elif choice0 == "Use Multispectral Camera based Analysis":
        st.info("This means that you have a drone mounted multispectral camera, you can fly over the crops and analyse")
   
    else:
        st.info("⚠️ Satellite data has very poor resolution and will not be able to offer insights into individual crop leaves.")

    st.header("Scene 2: What task will you pose this as?")
    
    st.write("Will you solve it as a detection/ classification or any other kind of problem")
    choice1 = st.radio("Chooes the task",
                       ["Classification", "Detection", "Segmentation"])
    
     
    
    #b  
    if choice1 == "Classification":
        st.info(" Sure you can label the whole image as a healthy or diseased region, but this requires a very zoomed in image. ")
    elif choice1 == "Detection":
        st.info("In this case it is assumed that you are able to take an image and mark parts which are diseased.So first you should have a dataset which is labeled for detection. Second the images are taken from a height. Multispectral images may be used in this case")
    else:
        st.info("Segmentation is similar to detection, however every pixel will be mapped to a class. Hence an image will have diseased pixels, healty pixels, may be branch pixels,soil pixels. It makes the labeling complicated and should be avoided")
    
    st.header("Scene 3: How will you proceed with the model and data?")
    
    st.write("Now you need to decide about pretrained/finetuning/training from scratch and data collection for the later 2")
    
    choice2 = st.radio("Strategy",
                       ["Use pretrained detector", "Collect dataset and finetune", "Train from scratch"])
    if choice2 == "Collect dataset and finetune":
        st.success("✅ Correct! You need a dataset of healthy and diseased leaves. You can now keep a part of the network fixed (whether classification or detection) and finetune the rest. This is fast, more appropriate even with small datasets.")
    elif choice2 == "Train from scratch":
        st.warning("You need a very big dataset of healthy and diseased leaves for prominent generalizable results. Unlike finetuning it is slow and more prone to error.")
    
    else:
        st.error("⚠️ Pretrained detectors don’t know leaf diseases. Dataset collection is essential.")

    st.header("Scene 4: Dataset Collection")
    choice3 = st.selectbox("How do you collect data?",
                           ["Collect balanced dataset", "Collect only diseased leaves", "Use random internet images"])
    if choice3 == "Collect balanced dataset":
        st.success("✅ Balanced datasets prevent bias and improve training.")
    elif choice3 == "Collect only diseased leaves":
        st.error("⚠️ Without healthy samples, the model can’t distinguish properly.")
    else:
        st.warning("⚠️ Internet images may be noisy and inconsistent.")

    
    st.header("Scene 4: Network Choice")
    choice4 = st.selectbox("Pick a network:", ["ResNet", "MobileNet", "YOLO","Custom CNN"])
    if choice4 == "ResNet":
        st.success("✅ ResNet is strong and reliable for classification tasks.")
    elif choice4 == "MobileNet":
        st.info("📱 MobileNet is lightweight and efficient — good for deployment of classification tasks.")
    elif choice4 == "YOLO":
        st.info("YOLO is ok if you choose detection.")
    
    else:
        st.warning("⚠️ Custom CNNs can work, but require large datasets and careful design.")

    st.header("Scene 5: Wrap-up")
    st.write("🎉 You’ve trained a leaf doctor model that helps farmers protect crops.")
    st.write("Optimal pipeline: Balanced dataset → Transfer learning /Finetuning → ResNet/MobileNet.")