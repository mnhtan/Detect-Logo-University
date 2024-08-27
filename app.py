import streamlit as st
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import ImageDraw
import os
import tensorflow as tf
import pandas as pd



st.set_page_config(
                page_title="Objects localization",
                page_icon="🧊",
            )

def inject_custom_css():
    st.markdown("""
        <style>
            /* Bỏ background gradient, sử dụng màu đơn sắc */
            

            /* Custom font and color for title */
            h1 {
                font-family: 'Montserrat', sans-serif;
                color: #FF4B4B; /* Màu đỏ cho tiêu đề */
                font-weight: bold;
            }
                
            


            /* Custom style for buttons */
            .stButton > button {
                font-family: 'Montserrat', sans-serif;
                font-weight: bold; /* In đậm chữ */
                color: #FFFFFF;
                background-color: #4CAF50; /* Màu xanh lá */
                padding: 10px 24px;
                border-radius: 8px;
                border: none;
                transition: background-color 0.3s, transform 0.3s;
            }

            /* Button hover effect */
            .stButton > button:hover {
                background-color: #45a049; /* Màu xanh đậm hơn khi hover */
                transform: translateY(-2px);
            }

            /* Custom style for file uploader */
            .stFileUploader > div > div > button {
                background-color: #4CAF50; /* Màu xanh lá */
                color: #FFFFFF;
                border-radius: 8px;
                border: none;
            }

            /* Box shadow for input widgets */
            .stTextInput, .stSelectbox, .stFileUploader {
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
                border-radius: 50px;
               
            }

            /* Custom tooltip styling */
            .stTooltip > div {
                background-color: #4CAF50; /* Màu xanh lá */
                color: #FFFFFF;
            }

            /* Custom style for radio buttons */
            .stRadio > label {
                display: inline-block;
                padding: 8px 15px;
                font-family: 'Montserrat', sans-serif;
                border: 1px solid #4CAF50; /* Màu xanh lá */
                border-radius: 20px;
                background-color: transparent;
                transition: background-color 0.3s;
            }
            .stRadio > label:hover {
                background-color: #4CAF50; /* Màu xanh lá */
                color: white;
            }
            .stRadio > label > input[type="radio"]:checked + div {
                background-color: #4CAF50; /* Màu xanh lá */
                color: white;
            }

            /* Custom style for progress bar */
            .stProgress > div > div > div > div {
                background-color: #4CAF50; /* Màu xanh lá */
            }
                
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()


# Thay thế đường dẫn đến model 
st.title('Ứng Dụng Xác Định Vị Trí')
BASE_MODEL_PATH = r'D:/Nam3/hk2/DeepLearning/DoAn_HocSau/'


# Load model
MODEL_PATHS = {
    'VGG16': r'VGG16_b_OD.h5',
    'ResNet50': r'Resnet_OD.h5',
    'CNN': r'CNN_OD.h5',
    'MLP': r'MLP_OD.h5',
}

selected_model = st.selectbox('Chọn mô hình:', list(MODEL_PATHS.keys()))

model_path = MODEL_PATHS[selected_model]
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error(f"File mô hình không tồn tại tại đường dẫn: {model_path}")

class_labels = ['HoaSen', 'Huflit', 'Hutech', 'Rmit', 'UFM']

uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đầu vào', use_column_width=True)

    if st.button('Dự đoán'):
        image_resized = image.resize((224, 224))  
        image_array = img_to_array(image_resized) / 255.0 
        image_array = np.expand_dims(image_array, axis=0)  

        # Dự đoán
        predictions = model.predict(image_array)
        class_predictions = predictions[0]
        bbox_predictions = predictions[1]

       
        predicted_class_index = np.argmax(class_predictions, axis=-1)
        predicted_label = class_labels[predicted_class_index[0]]
        

      
        original_width, original_height = image.size
        x_min = int(bbox_predictions[0][0] * original_width)
        y_min = int(bbox_predictions[0][1] * original_height)
        x_max = int(bbox_predictions[0][2] * original_width)
        y_max = int(bbox_predictions[0][3] * original_height)

       
        draw = ImageDraw.Draw(image)
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)
       
        
        st.image(image, caption='Ảnh với bounding box', use_column_width=True)
        st.write(f'Kết quả dự đoán lớp: {predicted_label}')
        st.write(f'Kết quả dự đoán bounding box: {(x_min, y_min, x_max, y_max)}')

