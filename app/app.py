import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
model = tf.keras.models.load_model("app/model_herbal.h5")
df = pd.read_csv("app/leaf_data.csv")
header=st.container()


confidence_score=[0]


def preprocess_image(_image_data):

    img_array = tf.image.resize(_image_data, (224, 224))
    img_array = img_array / 255.0  
    img_array = tf.expand_dims(img_array, axis=0)

    return img_array

def predict_label(_image_data, _model):

    img_array = preprocess_image(_image_data)
    prediction = _model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence_score[0] = np.max(prediction)

    return predicted_class



def display_leaf_info(predicted_class, df):

    leaf_info = df[df['Name'] == predicted_class]

    if not leaf_info.empty:

        sample_image_path = f"app/leaf/{predicted_class}.jpeg"
        st.image(sample_image_path, caption=f"Sample {predicted_class} Leaf", width=300,use_column_width="never")
        st.subheader(f"Uses of {predicted_class} Leaf:")
        st.write(leaf_info['Uses'].values[0])
        st.write("")
        if st.button("Total Info"):
            st.subheader(f"Total Description of {predicted_class} Leaf:")
            st.write(leaf_info['Description'].values[0])
    else:
        st.warning("Leaf information not found.")


with header:
    st.title("Herbal Leaf Classification")
    st.write("Our Project classifies the herbal image")

    uploaded_image = st.file_uploader("upload image", type=["jpg", "png", "jpeg"])


    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image",width=150,use_column_width="never")
        image = Image.open(uploaded_image)
        predicted_class = predict_label(image, model)
        class_labels = {}
        classes=['Alpinia Galanga (Rasna)', 'Amaranthus Viridis (Arive-Dantu)', 'Artocarpus Heterophyllus (Jackfruit)', 'Azadirachta Indica (Neem)', 'Basella Alba (Basale)', 'Brassica Juncea (Indian Mustard)', 'Carissa Carandas (Karanda)', 'Citrus Limon (Lemon)', 'Ficus Auriculata (Roxburgh fig)', 'Ficus Religiosa (Peepal Tree)', 'Hibiscus Rosa-sinensis', 'Jasminum (Jasmine)', 'Mangifera Indica (Mango)', 'Mentha (Mint)', 'Moringa Oleifera (Drumstick)', 'Muntingia Calabura (Jamaica Cherry-Gasagase)', 'Murraya Koenigii (Curry)', 'Nerium Oleander (Oleander)', 'Nyctanthes Arbor-tristis (Parijata)', 'Ocimum Tenuiflorum (Tulsi)', 'Piper Betle (Betel)', 'Plectranthus Amboinicus (Mexican Mint)', 'Pongamia Pinnata (Indian Beech)', 'Psidium Guajava (Guava)', 'Punica Granatum (Pomegranate)', 'Santalum Album (Sandalwood)', 'Syzygium Cumini (Jamun)', 'Syzygium Jambos (Rose Apple)', 'Tabernaemontana Divaricata (Crape Jasmine)', 'Trigonella Foenum-graecum (Fenugreek)']

        for i in range(30):
            class_labels[i]=classes[i]

        predicted_label = class_labels.get(predicted_class, "No Class with that Number")

        if (confidence_score[0] == 1):
            st.header(f"Predicted Leaf Name: {predicted_label}")
            display_leaf_info(predicted_label, df)
        else:
            st.write("Not a herbal Leaf ")

