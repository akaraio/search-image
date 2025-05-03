from create_database import *
from transformers import AutoImageProcessor, AutoModel
import streamlit as st
from itertools import cycle

# States
if "search_disabled" not in st.session_state:
    st.session_state["search_disabled"] = False

if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = None

if 'search_status' not in st.session_state:
    st.session_state['search_status'] = False

if 'search_folder' not in st.session_state:
    st.session_state['search_folder'] = None

if 'search_path' not in st.session_state:
    st.session_state['search_path'] = None

if 'load_status' not in st.session_state:
    st.session_state['load_status'] = False

if 'can_search' not in st.session_state:
    st.session_state['can_search'] = False

def load_status_change():
    st.session_state['load_status'] == True

# Model
MODEL_PATH = PATH + r'data'
st.session_state['image_processor'] = AutoImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
st.session_state['model'] = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)

image_processor = st.session_state['image_processor']
model = st.session_state['model']

# Interface
search_path = st.sidebar.text_input('Search folder path')
search_path = search_path.replace('"', '')

if st.session_state['search_status'] == False:
    with st.sidebar:
        with st.spinner():
            if search_path:
                st.session_state['search_status'] = True
                st.session_state['can_search'] = True
                st.session_state['search_path'] = search_path
                st.session_state['dataframe'] = create_dataframe(search_path)
                embeddings = create_batch(st.session_state['dataframe'], image_processor, model)
                st.session_state['index'] = create_index(embeddings) 
                save_index(st.session_state['index'], st.session_state['dataframe'])
                st.sidebar.success(f'{len(st.session_state['dataframe'])} images uploaded')

load_form = st.sidebar.form('load')
load_index_path = load_form.text_input('Index path')
load_index_path = load_index_path.replace('"', '')
load_df_path = load_form.text_input('Dataframe path')
load_df_path = load_df_path.replace('"', '')  
submitted = load_form.form_submit_button("Submit")
if submitted:
    with st.spinner():
        st.session_state['index'] = load_index(load_index_path)
        st.session_state['dataframe'] = load_df(load_df_path)
        st.sidebar.success(f'{len(st.session_state['dataframe'])} images uploaded')
        st.session_state['can_search'] = True

if st.session_state['can_search'] == True:
    search_image_path = st.file_uploader('Choose an image to search', type=['png', 'jpg', 'jpeg'])
    if search_image_path:
        st.divider()
        st.session_state['search_path'] = search_image_path
        embeds = extract_features(search_image_path, image_processor, model)
        D, I = st.session_state['index'].search(embeds, k=12)
        
        cols = cycle(st.columns(4))
        for images in st.session_state['dataframe'].loc[I[0], 'image_path'].values:
            next(cols).image(images, use_container_width=True)

        st.sidebar.image(st.session_state['search_path'])             