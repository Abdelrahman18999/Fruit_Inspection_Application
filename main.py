import streamlit as st
import base64
import gdown
import numpy as np
import pandas as pd
from PIL import Image

from keras.models import load_model
from util import set_background, classify


file_id = '1kr6YPUxyVXVMQQ1wcdprgmqePRKnyNhL'
output = 'fruits_inspection_model.h5'

gdown.download(f'https://drive.google.com/file/d/1kr6YPUxyVXVMQQ1wcdprgmqePRKnyNhL/view?usp=sharing={file_id}', output, quiet=False)

st.title("Fruit Inspection Application")
st.markdown("This web application about classify fruits if it is fresh or spoild")
st.header("Upload a fruit image")

# Set a background image
#set_background("fruit_background.jpeg")

# upload an image file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load the saved model
model = load_model("fruits_inspection_model.h5")

#classify(image, model)

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    index, conf_score = classify(image, model)
    
    # write classification
    st.write("## {}".format(index))
    st.write("### score: {}%".format(int(conf_score * 100) / 100))

