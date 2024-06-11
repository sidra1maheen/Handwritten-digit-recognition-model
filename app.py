import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('/content/project1.keras')

st.header('Hand digit recognition model')

img = st.text_input('Enter Image Name')

model = tf.keras.models.load_model('/content/project1.keras')

image = cv2.imread(img)[:,:,0]
image = np.invert(np.array([image]))

output = model.predict(image)
stn = 'Digit in the image is '+ str( np.argmax(output))
st.markdown(stn)
