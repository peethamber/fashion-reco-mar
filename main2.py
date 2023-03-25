import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from requests import request
import re
import sys

from urllib.parse import urlparse



codes_array = [
        "ROD",
        "RCB",
        "REL",
        "RCT",
        "RET",
        "RCD",
        "RED",
        "RKS",
        "RFT",
        "RFS",
        "RES",
        "RCS",
        "RCK",
        "RCP",
        "REP",
        "REG",
        "ROK",
        "RGB"

    ]
codes_array_lower = [
        "rod",
        "rcb",
        "rel",
        "rct",
        "ret",
        "rcd",
        "red",
        "rks",
        "rft",
        "rfs",
        "res",
        "rcs",
        "rck",
        "rcp",
        "rep",
        "reg",
        "rok",
        "rgb"

    ]
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
url = request.META.get('HTTP_REFERER')
url = urlparse(url)
st.write(url)

filenames = pickle.load(open('filenames.pkl','rb'))


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,GlobalMaxPooling2D()
])


st.title("Designs Recommender")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("./",uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result/norm(result)

    return normalized_result

def check_sub(sub,filename):
    
    if sub == "RKFCW" or sub == "rkfcw":
        if sub in os.path.basename(filename):
                ind = os.path.basename(filename).find(sub)
                sub1 = os.path.basename(filename)[ind:ind+8]
                
                sub1 = re.sub('-','',sub1)
                url = "https://www.rebblebee.com/AllProducts?"+sub1
                
                st.markdown(f'''
                <a href={url}><button style="background-color:GreenYellow;">{sub1}</button></a>
                ''',
                unsafe_allow_html=True)

    if sub == "RKFCTT" or sub == "rkfctt":
        if sub in os.path.basename(filename):
                ind = os.path.basename(filename).find(sub)
                sub1 = os.path.basename(filename)[ind:ind+9]
                
                sub1 = re.sub('-','',sub1)
                url = "https://www.rebblebee.com/AllProducts?"+sub1
                
                st.markdown(f'''
                <a href={url}><button style="background-color:GreenYellow;">{sub1}</button></a>
                ''',
                unsafe_allow_html=True)

    if sub == "REJS" or sub == "rejs":
        if sub in os.path.basename(filename):
                ind = os.path.basename(filename).find(sub)
                sub1 = os.path.basename(filename)[ind:ind+7]
                
                sub1 = re.sub('-','',sub1)
                url = "https://www.rebblebee.com/AllProducts?"+sub1
                
                st.markdown(f'''
                <a href={url}><button style="background-color:GreenYellow;">{sub1}</button></a>
                ''',
                unsafe_allow_html=True)
                
    if sub == "RRBMS" or sub == "rrbms":
        if sub in os.path.basename(filename):
                ind = os.path.basename(filename).find(sub)
                sub1 = os.path.basename(filename)[ind:ind+8]
                
                sub1 = re.sub('-','',sub1)
                url = "https://www.rebblebee.com/AllProducts?"+sub1
                
                st.markdown(f'''
                <a href={url}><button style="background-color:GreenYellow;">{sub1}</button></a>
                ''',
                unsafe_allow_html=True)

    if sub in codes_array:
         if sub in os.path.basename(filename):
                ind = os.path.basename(filename).find(sub)
                sub1 = os.path.basename(filename)[ind:ind+6]
                
                sub1 = re.sub('-','',sub1)
                url = "https://www.rebblebee.com/AllProducts?"+sub1
                
                st.markdown(f'''
                <a href={url}><button style="background-color:GreenYellow;">{sub1}</button></a>
                ''',
                unsafe_allow_html=True)

    if sub in codes_array_lower:
         if sub in os.path.basename(filename):
                ind = os.path.basename(filename).find(sub)
                sub1 = os.path.basename(filename)[ind:ind+6]
                
                sub1 = re.sub('-','',sub1)
                url = "https://www.rebblebee.com/AllProducts?"+sub1
                
                st.markdown(f'''
                <a href={url}><button style="background-color:GreenYellow;">{sub1}</button></a>
                ''',
                unsafe_allow_html=True)
         

def do_all_code_checks(fname):
    check_sub("RKFCW",fname)
    check_sub("rkfcw",fname)

    check_sub("RKFCTT",fname)
    check_sub("rkfctt",fname)

    check_sub("REJS",fname)
    check_sub("rejs",fname)

    check_sub("RRBMS",fname)
    check_sub("rrbms",fname)

    for code in codes_array:
         check_sub(code,fname)

    # for code in codes_array_lower:
    #      check_sub(code,fname)
         

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)

    distances,indices = neighbors.kneighbors([features])
    return indices


# uploaded_file = st.file_uploader("choose an image")


# st.write(sys.argv[1])
img_width = 230

display_image = Image.open(sys.argv[1])
st.image(display_image,width=450)
features = feature_extraction(os.path.join('./',sys.argv[1]),model)
indices = recommend(features,feature_list)

st.subheader("Design recommendations from us")

col1,col2,col3,col4,col5 = st.columns(5)

with col1:
    st.image(filenames[indices[0][0]],width = img_width)
    st.write(os.path.basename(filenames[indices[0][0]]))
    do_all_code_checks(filenames[indices[0][0]])
    
    
with col2:
    st.image(filenames[indices[0][1]],width = img_width)
    st.write(os.path.basename(filenames[indices[0][1]]))
    do_all_code_checks(filenames[indices[0][1]])
    
with col3:
    st.image(filenames[indices[0][2]],width = img_width)
    st.write(os.path.basename(filenames[indices[0][2]]))
    do_all_code_checks(filenames[indices[0][2]])

with col4:
    st.image(filenames[indices[0][3]],width = img_width)
    st.write(os.path.basename(filenames[indices[0][3]]))
    do_all_code_checks(filenames[indices[0][3]])

with col5:
    st.image(filenames[indices[0][4]],width = img_width)
    st.write(os.path.basename(filenames[indices[0][4]]))
    do_all_code_checks(filenames[indices[0][4]])
    

