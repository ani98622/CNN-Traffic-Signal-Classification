import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('img_model.h5')

# Class names
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing vehicles over 3.5 tons', 11: 'Right-of-way at intersection', 
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 tons prohibited', 
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right', 
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End of all speed and passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 
    42: 'End no passing vehicles > 3.5 tons'
}

# Constants
IMG_HEIGHT, IMG_WIDTH = 30, 30

# Title and description
st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout='centered')
st.title("German Traffic Sign Recognition")

# Use HTML/CSS to increase font size
st.markdown("""
    <h3 style='font-size: 24px;'>Upload an image of a traffic sign, and the model will predict its class</h3>
    """, unsafe_allow_html=True)

# File uploader

# predicted_prob = 0

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="visible")

# Initialize session state for image and prediction
if 'image' not in st.session_state:
    st.session_state.image = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None

if uploaded_file:
    # Open and convert the image
    st.session_state.image = Image.open(uploaded_file)
    st.session_state.image = st.session_state.image.convert('RGB') if st.session_state.image.mode == 'RGBA' else st.session_state.image
    st.session_state.image = st.session_state.image.resize((IMG_WIDTH, IMG_HEIGHT))

# Layout for side-by-side display
col1, col2 = st.columns([1, 1])

with col1:
    if st.session_state.image and uploaded_file:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True, channels="RGB")
    # else:
    #     st.warning("Please upload an image first.")

with col2:
    if st.button("Predict the Signal"):
        if st.session_state.image:
            # Convert image to numpy array and normalize
            image_array = np.array(st.session_state.image)
            image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 30, 30, 3)
            image_array = image_array / 255.0

            with st.spinner('Analyzing the image...'):
                # Make prediction
                prediction = model.predict(image_array)
                predicted_class = np.argmax(prediction, axis=-1)[0]
                predicted_prob = np.max(prediction, axis=-1)[0]
                st.session_state.prediction = classes[predicted_class]
                st.session_state.probabilities = prediction[0]
        else:
            st.warning("Please upload an image and click 'Predict the Signal'.")


    if st.session_state.prediction:
        st.markdown(f"""
        <style>
            .result {{
                color: #ffffff;
                background-color: #4caf50;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }}
            .prediction {{
                color: #ffffff;
                background-color: #2196f3;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                font-size: 20px;
                font-weight: bold;
            }}
            .probabilities {{
                color: #ffffff;
                background-color: #f44336;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                font-size: 16px;
                font-weight: bold;
            }}
        </style>
        <div class="result">
            <h2>Prediction Result:</h2>
            <p class="prediction">{st.session_state.prediction}</p>
            <p class="probabilities">Probability: {predicted_prob:.2f}</p>
        </div>
        """, unsafe_allow_html=True)