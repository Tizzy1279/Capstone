import faiss
import os
import numpy as np
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd

# Define paths for saving/loading chunked data and FAISS index
chunked_data_path = '/workspaces/Capstone/chunked_data.pkl'
faiss_index_path = '/workspaces/Capstone/faiss_index.bin'
embeddings_path = '/workspaces/Capstone/embeddings.npy'

# Load PDF documents
documents = []
pdf_folder_path = '/workspaces/Capstone/PDF Folder/'
for f in os.listdir(pdf_folder_path):
    if f.endswith('.pdf'):
        loader = PyPDFLoader(os.path.join(pdf_folder_path, f))
        documents.extend(loader.load())
        print(f'Loaded {f}')

# Define the chunk size and chunk overlap
chunk_size = 500  # Adjusted chunk size
chunk_overlap = 0  # No chunk overlap

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = text_splitter.split_documents(documents)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Check if chunked data and FAISS index already exist
if os.path.exists(chunked_data_path) and os.path.exists(faiss_index_path) and os.path.exists(embeddings_path):
    # Load chunked data and FAISS index from disk
    with open(chunked_data_path, 'rb') as f:
        chunks = pickle.load(f)
    embedding_matrix = np.load(embeddings_path)
    index = faiss.read_index(faiss_index_path)
    print('Loaded chunked data and FAISS index from disk.')
else:
    # Embed all chunks
    embeddings = model.encode([c.page_content for c in chunks])

    # Convert embeddings to numpy array
    embedding_matrix = np.array(embeddings)

    # Create FAISS index
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    # Save chunked data and FAISS index to disk
    with open(chunked_data_path, 'wb') as f:
        pickle.dump(chunks, f)
    np.save(embeddings_path, embedding_matrix)
    faiss.write_index(index, faiss_index_path)
    print('Saved chunked data and FAISS index to disk.')

# Implement a search function using FAISS
def search_faiss(query, index, chunks, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = [chunks[i] for i in indices[0]]
    return results

# Function to chunk data from a DataFrame
def chunk_data(data, chunk_size=500):
    chunks = []
    for start in range(0, len(data), chunk_size):
        end = start + chunk_size
        chunk = data.iloc[start:end]
        chunks.append(chunk)
    return chunks

# Print the number of chunks created and total documents loaded
print(f'Total PDF documents loaded: {len(documents)}')
print(f'Total chunks created: {len(chunks)}')