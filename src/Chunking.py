import faiss
import os
import re
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.evaluation.qa import QAEvalChain
from sentence_transformers import SentenceTransformer
import numpy as np

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

# Embedding function using OpenAI embeddings
def embed_text(text):
    response = openai.embeddings.create(
        input=[text],  # Ensure input is a list
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Initialize SentenceTransformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Embed all chunks
embeddings = model.encode([c.page_content for c in chunks])

# Convert embeddings to numpy array
embedding_matrix = np.array(embeddings)

# Create FAISS index
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# Implement a search function using FAISS
def search_faiss(query, index, chunks, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = [chunks[i] for i in indices[0]]
    return results

# Print the number of chunks created and total documents loaded
print(f'Total PDF documents loaded: {len(documents)}')
print(f'Total chunks created: {len(chunks)}')