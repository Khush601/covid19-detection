import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model('covid_model.h5')
        #st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_trained_model()


CLASS_NAMES = ['BacterialPneumonia','COVID-19','Normal','ViralPneumonia']


def preprocess_image(image):
    try:
        image = image.convert("RGB")  # Convert to RGB
        image = image.resize((150, 150))  # Resize to model input size
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return tf.convert_to_tensor(image_array, dtype=tf.float32)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Streamlit UI
st.title("COVID-19 Detection")
st.write("Upload a chest X-ray image to detect the condition.")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])


if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    if processed_image is not None and model is not None:
            
        
        try:
            predictions = model.predict(processed_image)[0]  
            confidence_scores = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
            print(predictions)
            # Get the top predicted class
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = max(predictions)  # Highest probability

            # Display the result
            st.subheader(f"Prediction: ➡️ {predicted_class}")
            st.write(f"Confidence Score: {confidence:.4f}")

            # Show confidence for all categories
            st.write("### Confidence Scores for All Categories:")
            for label, score in confidence_scores.items():
                st.write(f"- {label}: {score:.4f}")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
