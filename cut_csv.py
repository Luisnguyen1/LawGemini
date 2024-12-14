import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from tqdm import tqdm
import csv

file_path = 'crawl-data/demo_data.csv'
index_file_path = 'faiss_index.bin'
vectors_file_path = 'vectors.npy'

# Read file csv using csv module
with open(file_path, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    data = [row[0].split(',', 2) for row in reader]
    df = pd.DataFrame(data, columns=['so_hieu', 'dieu', 'truncated_text'])
    
print(df.head())