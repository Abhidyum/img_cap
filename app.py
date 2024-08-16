
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

from PIL import Image



# Load the tokenizer
with open('saved_files/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the trained model
model = load_model('saved_files/imgcap_model3.h5')

# Load the VGG16 model
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load the image features
with open('saved_files/features.pkl', 'rb') as f:
    features = pickle.load(f)

max_length = 35  # Update this to the max length used during training


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


# Streamlit UI
st.title("CaptionGenie")
st.write("Upload an image to generate its caption.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")


    # Preprocess the uploaded image
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)

    # Generate caption
    caption = predict_caption(model, feature, tokenizer, max_length)


    text_to_display = caption[8:-6]

    # Use Markdown to format the text
    st.markdown(f"**<span style='font-size:20px;'>{text_to_display}</span>**", unsafe_allow_html=True)