import sys
import os
import streamlit as st
import matplotlib.pyplot as plt

# Add the directory containing Business_Intelligence.py to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Print the Python path for debugging
print("Python Path:", sys.path)

# Print the contents of the parent directory for debugging
print("Contents of parent directory:", os.listdir(parent_dir))
