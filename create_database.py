import pandas as pd
import numpy as np
import torch
from PIL import Image
import faiss
from faiss import write_index, read_index
import os
from datetime import datetime
from dotenv import load_dotenv

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
PATH = os.path.dirname(__file__)

def create_dataframe(path):
    roots = set()
    images_list = list()

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.png', 'jpg', 'jpeg')):
                roots.add(root)
                images_list.append(root + '\\' + file)
    df = pd.DataFrame(images_list, columns=['image_path'])
    return df

def load_image(image):
    image = Image.open(image).convert('RGB')
    if image.mode != 'RGB':
            image = image.convert('RGB')
    return image

def extract_features(path, image_processor, model):  
    img = load_image(path)
    inputs = image_processor(img, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state
    embedding = embedding[:, 0, :].squeeze(1)
    return embedding.numpy()

def create_batch(dataframe, image_processor, model):
    return np.vstack([extract_features(path, image_processor, model) for path in dataframe['image_path']])

def create_index(embeddings):
    vector_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    index.add(embeddings)
    return index

def save_index(index, df):
    df.to_csv(PATH + r'data'+ datetime.now().strftime("%d-%m-%y_%H-%M") + '.csv')
    write_index(index, PATH + r'data'+ datetime.now().strftime("%d-%m%y_%H-%M") + '.index')

def load_index(path):
    return read_index(path)

def load_df(path):
    return pd.read_csv(path)