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


# Load data in chunks to handle large CSV file
chunk_size = 1000  # Adjust the chunk size as needed
chunks = pd.read_csv(file_path, header=None, names=["raw_data"], skiprows=2, encoding='utf-8', chunksize=chunk_size)

# Process each chunk and concatenate the results
df_list = []
for chunk in chunks:
    chunk['raw_data'] = chunk['raw_data'].astype(str)
    chunk = chunk['raw_data'].str.split(',', n=2, expand=True)
    chunk.columns = ['so_hieu', 'dieu', 'truncated_text']
    df_list.append(chunk)

df = pd.concat(df_list, ignore_index=True)
    


# Use the correct column name
column_name = 'truncated_text'

# Load the model
model = SentenceTransformer('intfloat/multilingual-e5-small')

# Check if the index and vectors files exist
if os.path.exists(index_file_path) and os.path.exists(vectors_file_path):
    # Load the FAISS index and vectors from files
    index = faiss.read_index(index_file_path)
    vectors = np.load(vectors_file_path)
else:
    # Vectorize the text data with progress display
    vectors = []
    for text in tqdm(df[column_name].tolist(), desc="Embedding vectors"):
        vector = model.encode([text], convert_to_tensor=True).cpu().numpy()
        vectors.append(vector)
    vectors = np.vstack(vectors)

    # Create FAISS index
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save the FAISS index and vectors to files
    faiss.write_index(index, index_file_path)
    np.save(vectors_file_path, vectors)

def retrieve_documents(query, k=5, threshold=0.7):
    query_vector = model.encode([query], convert_to_tensor=True).cpu().numpy()
    D, I = index.search(query_vector, k)  # Retrieve top k documents
    
    # Convert distances to similarity scores (assuming L2 distance)
    similarities = 1 / (1 + D[0])
    
    # Filter documents based on the similarity threshold
    filtered_documents = []
    for i, similarity in enumerate(similarities):
        if similarity >= threshold:
            filtered_documents.append(df.iloc[I[0][i]][column_name])
    
    return filtered_documents
