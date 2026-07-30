import streamlit as st
# --- Static descriptions and illustrative images ---
cv_tasks = {
 "Classification": {
        "file": "classi.jpg",  # replace with your own image path
        "description": "Classification assigns a label to an entire image. For example, classifying whether an image contains a cat or a dog or the breed of the dog. Popular models used for this task are Resnet, VGGNet, Inception-Net. These are trained on benchmark dataset Imagenet containing millions of images grouped/labelled into classes/categories."
    },
    "Detection": {
        "file": "det.jpeg",  # replace with your own image path
        "description": "Detection is about locating and identifying objects in an image. For example, detecting cars in a traffic scene using bounding boxes. Each image may have multiple objects and they can be detected. Popular models used for this task are YOLO, Faster-RCNN, RT-DETR. These are pretrained models which are trained on benchmark datasets like COCO. COCO dataset has hundred thousands of images annotated/labeled with bounding box of objects."
    },
   
    "Segmentation": {
        "file": "seg.jpeg",  # replace with your own image path
        "description": "Segmentation divides an image into regions corresponding to different objects or classes. For example, separating pedestrians from the background.It is very similar to detection, but detection can choose to identify 5 or 6 objects in the images, whereas segmentation has to classify every point (called pixel) in the image to a particular category. Popular models are UNet, Mask-RCNN trained on Cityscapes datasets."
    }
}

# --- Streamlit UI ---
#st.title("Interactive Image Task Explorer")

st.sidebar.header("Main Computer Vision Tasks")
st.sidebar.write("Select a task to learn more:")

# Sidebar checkboxes
selected_tasks = []
for task in cv_tasks.keys():
    if st.sidebar.checkbox(task):
        selected_tasks.append(task)

# Display selected tasks
for task in selected_tasks:
    st.subheader(task)
    st.image(cv_tasks[task]["file"], caption=task, width=500)
    st.write(cv_tasks[task]["description"])

# --- Static data: images and tasks ---
images = {
    "Traffic Scene": {
        "file": "traffic.jpeg",  # replace with your actual image path
        "tasks": {
            "Count Cars": "Counting cars is a detection problem focused on the 'car' class. You would typically use an object detection model like YOLO or Faster R-CNN trained to detect cars.",
            "Count Pedestrians": "Counting pedestrians is similar to counting cars, but the detection class is 'person'. Models trained on COCO dataset already include 'person' as a class.",
            "Remove Pedestrians": "Removing pedestrians can be achieved using segmentation models (like Mask R-CNN) to isolate pedestrian pixels, followed by inpainting techniques to fill the removed regions.",
            "Blur Pedestrians": "Detect pedestrian using any detection model like yolo or faster r-cnn and then apply cv2 blur filter on it."
        }
    },
    "Park Scene": {
        "file": "park.jpeg",  # replace with your actual image path
        "tasks": {
            "Count Trees": "Tree counting can be done using semantic segmentation or detection models trained to identify trees in aerial or ground-level images.",
            "Identify Benches": "Bench detection is an object detection problem. You would need a dataset with labeled benches to train or fine-tune a detection model.",
            "Remove Trash": "Trash removal involves detecting trash items and then applying inpainting or generative fill to remove them from the scene."
        }
    },
    "Shopping Mall": {
        "file": "mall.jpeg",  # replace with your actual image path
        "tasks": {
            "Count Shoppers": "Shoppers can be counted using pedestrian detection models. Crowd counting techniques (like density estimation) may also be useful in dense scenes.",
            "Detect Shops": "Shop detection can be achieved by identifying storefronts using detection or segmentation models trained on retail environments.",
            "Remove Advertisements": "Removing advertisements involves detecting ad boards or posters and then applying inpainting to replace them with background textures."
        }
    }
}

# --- Streamlit UI ---
st.title("Interactive Image Task Explorer")

# Step 1: Choose an image
selected_image = st.selectbox("Your first exercise is to understand how to select which CV task to perform given a goal.  Choose an image:", list(images.keys()))

# Display the chosen image
#st.image(images[selected_image]["file"], caption=selected_image, use_column_width=True)
st.image(images[selected_image]["file"], caption=selected_image, width=600)

# Step 2: Show tasks as radio buttons
selected_task = st.radio("Choose a task:", list(images[selected_image]["tasks"].keys()))

# Step 3: Display details of the chosen task
st.subheader(f"Details for: {selected_task}")
st.write(images[selected_image]["tasks"][selected_task])



# --- Static creative CV scenarios ---
examples = {
    "Automatic Camera Effects": {
        "description": "Your friend is filming a short movie and wants automatic camera effects. If the main character enters the scene, the video should zoom in; if the character leaves, it should zoom out.",
        "solution": "Formulate this as a face detection + face recognition problem. Use a face detector like Viola Jones or MTCNN. Pass the detected faces for recognition to understand if it belongs to the main character. Trigger zoom-in when recognized, zoom-out when not. Zooming an image is like resizing it, making it bigger./smaller This is supported by cv2 toolkit."
    },
    "Grayscale Vision Effect": {
        "description": "A movie character sees everything in grayscale except one character or objects he touches, which remain in color.",
        "solution": "Use semantic segmentation to isolate the specific character or object. Apply a grayscale filter to the rest of the frame, while keeping the segmented region in color."
    },
    "Video Call Background Change": {
        "description": "Your sibling wants an app that changes the background of a video call in real time according to user actions.",
        "solution": "Use semantic segmentation (e.g., Mask-RCNN, Deeplab) to separate foreground (person) from background. Classify user actions and map a background based on the action class. The classification task can be done first on the whole image or on the segmented foreground i.e. user."
    },
    "Wildlife Monitoring": {
        "description": "A researcher wants to monitor animals visiting a water source using a  camera system.",
        "solution": "This is an animal detection problem using a standard object detector like YOLO"
    },
    "Highlighting Objects": {
        "description": "A character can highlight objects he thinks about — they glow while the rest of the scene stays normal.",
        "solution": "Use object detection to identify the said object. Apply post-processing to add glow effects on the object using cv2 library."
    },
    "Silhouette Effect": {
        "description": "In a thriller movie, only silhouettes of people are visible while actual bodies are invisible.",
        "solution": "Use  segmentation to get person regions. Replace their regions with silhouette masks (black fill with edge glow) while removing texture details."
    },
    "Two-Character Montage": {
        "description": "In a montage, whenever two characters appear, the camera zooms in and background darkens.",
        "solution": "Use segmentation for person class. If count == 2, trigger zoom-in (resizing) and apply background darkening via segmentation."
    },
    "Mirror Reflection Delay": {
        "description": "Reflections in mirrors show a delayed version of movements.",
        "solution": "Detect mirrors using scene segmentation. Apply a time-shifted buffer to the mirrored region, replaying past frames with a delay."
    },
    "Game with Glowing Clues": {
        "description": "Clues appear as glowing symbols only when the user points at the correct object.",
        "solution": "Use object detection to store the location of all object bounding boxes. If the user clicks on the correct one, then clues are overlayed on the image ."},
        "Dynamic Spotlight": {
        "description": "In a concert film, whenever the singer raises their hand, a spotlight follows them automatically.",
        "solution": "Use pose estimation (body skeleton detection, another CV task beyond the basic ones, use models like OpenPose, MediaPipe) to detect hand-raising gestures. Trigger a spotlight overlay effect that tracks the performer’s position."
    },
        "Emotion-Based Filters": {
        "description": "In a drama, the camera applies different filters depending on the actor’s facial expression.",
        "solution": "Use facial emotion recognition (a kind of classification into expressions happy/sad). Map detected emotions to cinematic filters (warm tones for joy, desaturation for sadness)."
    },
        "Time Freeze Effect": {
        "description": "In a fantasy scene, everything freezes except one character who keeps moving.",
        "solution": "Segment the main character and keep their regions alive every frame, while holding background and other people static using first frame repetition."
    },
           "Object-to-Text Transformation": {
        "description": "In an experimental film, objects touched by the character transform into floating text labels.",
        "solution": "Detect objects in the scene and hand, if they overlap (contact detection), overlay animated text labels at the object’s location."
    },
    "Scene Mood Detection": {
        "description": "In a montage, the system automatically detects if a scene is calm or chaotic and adjusts music accordingly.",
        "solution": "You can simply apply frame to frame subtraction or optical flow (another basic cv2 operation). Calm scenes have low motion vectors; chaotic scenes have high activity. This can be used to soundtrack changes."
    }
    }


# --- Streamlit UI ---
st.title("Creative Computer Vision Scenarios")

selected_example = st.selectbox("Choose a scenario:", list(examples.keys()))

st.subheader(selected_example)
st.write(examples[selected_example]["description"])
st.markdown("**Solution Approach:**")
st.write(examples[selected_example]["solution"])

