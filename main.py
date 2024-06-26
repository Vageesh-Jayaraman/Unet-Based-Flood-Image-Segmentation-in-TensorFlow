import streamlit as st
import gdown
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

@st.cache_data
def download_model():
    url = 'https://drive.google.com/uc?id=1oCZFRhuZ9j6t-FdPpg1E6SUX_XkJ4nRW'
    output = 'floodSegmentationModel.h5'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return output

@st.cache_resource
def load_model():
    model_path = download_model()
    model = tf.keras.models.load_model(model_path)
    return model

def segment_image(image, model):
    input_image = cv2.resize(image, (128, 128))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image / 255.0

    segmentation_mask = model.predict(input_image)
    segmentation_mask = segmentation_mask[0]
    segmentation_mask = cv2.resize(segmentation_mask, (image.shape[1], image.shape[0]))
    segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8)

    return segmentation_mask

def calculate_water_coverage(mask):
    total_pixels = mask.shape[0] * mask.shape[1]
    water_pixels = np.sum(mask)
    water_coverage_percentage = (water_pixels / total_pixels) * 100
    return water_coverage_percentage

def main():
    st.title('Flood Image Segmentation')
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Vageesh-Jayaraman/Unet-Based-Flood-Image-Segmentation-in-TensorFlow/tree/main)")
    st.markdown("[![Google Drive](https://img.shields.io/badge/Download%20Model-Google%20Drive-blue?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1oCZFRhuZ9j6t-FdPpg1E6SUX_XkJ4nRW/view?usp=sharing)")

    uploaded_file = st.file_uploader('Upload an image file', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        with st.spinner('Downloading and loading the model...'):
            model = load_model()

        image = Image.open(uploaded_file)
        image_np = np.array(image)

        segmented_mask = segment_image(image_np, model)
        water_coverage = calculate_water_coverage(segmented_mask)

        colored_mask = np.zeros_like(image_np)
        colored_mask[segmented_mask == 1] = [0, 255, 0]  # Green mask

        output_image = cv2.addWeighted(image_np, 0.7, colored_mask, 0.3, 0)

        st.subheader('Original Image')
        st.image(image, use_column_width=True)

        st.subheader('Segmented Image')
        st.image(output_image, use_column_width=True)

        st.subheader('Water Coverage')
        st.write(f'The percentage of the area covered with water is: {water_coverage:.2f}%')

        st.markdown('<h2 style="font-size: 24px; font-family: Arial, sans-serif;">Model Accuracy</h2>', unsafe_allow_html=True)
        st.write('<p style="font-size: 20px; font-family: Arial, sans-serif;">The model accuracy is: 86%</p>', unsafe_allow_html=True)

    st.subheader('Tech Stack Used')
    tech_stack_images = ['icons/tensorflow.png', 'icons/opencv.png', 'icons/streamlit.png']
    st.image(tech_stack_images, width=60)

if __name__ == "__main__":
    main()
