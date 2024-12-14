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
