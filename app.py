import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


# Import the Network class (needs to match your model definition)
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(3, 8, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.1),

            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(8, 16, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.2),

            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(16, 16, kernel_size=5),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.2),

            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(16, 8, kernel_size=3),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(102400 // 200, 256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 6),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(-1, 102400 // 200)
        x = self.classifier(x)
        return x


# Set page config
st.set_page_config(
    page_title="Weld Defect Classifier",
    page_icon="🔍",
    layout="wide"
)

# Define the defect labels
defect_labels = {
    0: 'Good Weld',
    1: 'Burn Through',
    2: 'Contamination',
    3: 'Lack of Fusion',
    4: 'Misalignment',
    5: 'Lack of Penetration'
}

# Define binary labels
binary_labels = {
    0: 'Good Weld',
    1: 'Defective Weld'
}

# Color scheme for different classes
colors = {
    0: '#2ECC71',  # Good weld - green
    1: '#E74C3C',  # Burn through - red
    2: '#9B59B6',  # Contamination - purple
    3: '#F39C12',  # Lack of fusion - orange
    4: '#3498DB',  # Misalignment - blue
    5: '#F1C40F'  # Lack of penetration - yellow
}


@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Network()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image):
    """Preprocess image for the model"""
    # Apply the same transforms used during testing
    transform = transforms.Compose([
        transforms.CenterCrop((800, 800)),
        transforms.ToTensor(),
    ])

    return transform(image).unsqueeze(0)  # Add batch dimension


def predict_image(model, image, device):
    """Process the image and return prediction"""
    with torch.no_grad():
        # Make prediction
        output = model(image.to(device))

        # Get prediction class and confidence
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        return probabilities


def main():
    st.title("🔍 Weld Defect Classifier")

    # Sidebar for model selection
    st.sidebar.title("Model Settings")

    # Model path selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value="./data/modelv5.pt",
        help="Path to your trained PyTorch model (.pt file)"
    )

    if not os.path.exists(model_path):
        st.sidebar.error(f"Model not found at {model_path}")
        st.warning(f"⚠️ Model file not found at {model_path}. Please check the path.")
        return

    # Load model
    try:
        model, device = load_model(model_path)
        st.sidebar.success(f"Model loaded successfully! Using device: {device}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.error(f"⚠️ Error loading model: {e}")
        return

    # Display some info about the model
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown("This model classifies welds into the following categories:")
    for idx, label in defect_labels.items():
        st.sidebar.markdown(f"- **{label}** (Class {idx})")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Image")

        # Option to use sample images or upload
        option = st.radio(
            "Choose input method:",
            ["Upload my own image", "Use a URL"]
        )

        uploaded_image = None

        if option == "Upload my own image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                try:
                    uploaded_image = Image.open(uploaded_file).convert('RGB')
                    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error opening image: {e}")
        else:
            url = st.text_input("Enter image URL:")
            if url:
                try:
                    import requests
                    from io import BytesIO

                    response = requests.get(url)
                    uploaded_image = Image.open(BytesIO(response.content)).convert('RGB')
                    st.image(uploaded_image, caption="Image from URL", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image from URL: {e}")

    with col2:
        st.subheader("Prediction Results")

        if uploaded_image is not None:
            with st.spinner("Analyzing image..."):
                # Preprocess image
                try:
                    preprocessed = preprocess_image(uploaded_image)

                    # Get prediction
                    probabilities = predict_image(model, preprocessed, device)

                    # Get top prediction
                    pred_class = torch.argmax(probabilities).item()
                    confidence = probabilities[pred_class].item() * 100

                    # Binary classification
                    binary_class = 0 if pred_class == 0 else 1

                    # Display prediction
                    st.markdown(f"### Prediction")

                    # Show primary prediction with colored box
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: {colors[pred_class]}; color: white;">
                            <h3 style="margin: 0;">{defect_labels[pred_class]}</h3>
                            <p style="margin: 0; font-size: 20px; font-weight: bold;">{confidence:.2f}% Confidence</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Show binary classification
                    binary_color = colors[0] if binary_class == 0 else "#E74C3C"
                    st.markdown(
                        f"""
                        <div style="margin-top: 10px; padding: 10px; border-radius: 5px; background-color: {binary_color}; color: white;">
                            <h4 style="margin: 0;">Binary Classification: {binary_labels[binary_class]}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Display probabilities as a bar chart
                    st.markdown("### Class Probabilities")

                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 5))

                    # Get all probabilities
                    probs = probabilities.cpu().numpy() * 100

                    # Create bars
                    bars = ax.bar(
                        range(len(defect_labels)),
                        probs,
                        color=[colors[i] for i in range(len(defect_labels))]
                    )

                    # Add percentage labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.,
                            height,
                            f'{height:.1f}%',
                            ha='center', va='bottom', rotation=0
                        )

                    # Customize chart
                    ax.set_xticks(range(len(defect_labels)))
                    ax.set_xticklabels([defect_labels[i] for i in range(len(defect_labels))], rotation=45, ha='right')
                    ax.set_ylabel('Probability (%)')
                    ax.set_ylim(0, 100)
                    ax.set_title('Prediction Probabilities by Class')
                    plt.tight_layout()

                    # Display the chart
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error processing image: {e}")
        else:
            st.info("Please upload an image or provide a URL to get predictions")

    # Add explanation at the bottom
    with st.expander("About this app"):
        st.markdown("""
        ### Weld Defect Classifier

        This application uses a trained deep learning model to classify welds into 6 categories:

        1. **Good Weld** - A properly formed weld with no defects
        2. **Burn Through** - Excessive heat causing the base metal to melt through
        3. **Contamination** - Foreign materials in the weld
        4. **Lack of Fusion** - Insufficient bonding between weld and base metal
        5. **Misalignment** - Poor fit-up between the parts being welded
        6. **Lack of Penetration** - Insufficient weld penetration into the base metal

        The model was trained on images of various welds and their defects.

        For best results, upload clear images of welds similar to those used in training.
        """)


if __name__ == "__main__":
    main()