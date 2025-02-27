import streamlit as st
import pandas as pd
from src.Business_Intelligence import (
    show_monthly_sales,
    show_product_sales,
    show_product_customer_age,
    show_product_satisfaction,
    show_regional_sales,
    show_regional_customer_age,
    show_regional_satisfaction,
    show_gender_analysis,
    show_age_analysis,
    show_regional_demographics,
    show_regional_age_analysis,
    show_satisfaction_correlation,
    show_age_distribution
)

# Load Data
@st.cache
def load_data():
    file_path = '/workspaces/Capstone/sales_data_capstone.csv'
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()

# Streamlit App
st.title("Business Intelligence Dashboard")

menu_options = [
    "Sales Performance",
    "Product Analysis",
    "Regional Analysis",
    "Demographics"
]

choice = st.sidebar.selectbox("Choose Analysis Category", menu_options)

if choice == "Sales Performance":
    st.header("Sales Performance")
    show_monthly_sales()

elif choice == "Product Analysis":
    st.header("Product Analysis")
    product_options = [
        "Show me product sales",
        "Show me product customer age",
        "Show me product satisfaction"
    ]
    product_choice = st.selectbox("Choose Product Analysis", product_options)
    if product_choice == "Show me product sales":
        show_product_sales()
    elif product_choice == "Show me product customer age":
        show_product_customer_age()
    elif product_choice == "Show me product satisfaction":
        show_product_satisfaction()

elif choice == "Regional Analysis":
    st.header("Regional Analysis")
    regional_options = [
        "Show me regional sales",
        "Show me regional customer age",
        "Show me regional satisfaction"
    ]
    regional_choice = st.selectbox("Choose Regional Analysis", regional_options)
    if regional_choice == "Show me regional sales":
        show_regional_sales()
    elif regional_choice == "Show me regional customer age":
        show_regional_customer_age()
    elif regional_choice == "Show me regional satisfaction":
        show_regional_satisfaction()

elif choice == "Demographics":
    st.header("Demographics")
    demographics_options = [
        "Show me gender analysis",
        "Show me age analysis",
        "Show me regional demographics",
        "Show me regional age analysis",
        "Show me satisfaction correlation",
        "Show me age distribution"
    ]
    demographics_choice = st.selectbox("Choose Demographics Analysis", demographics_options)
    if demographics_choice == "Show me gender analysis":
        show_gender_analysis()
    elif demographics_choice == "Show me age analysis":
        show_age_analysis()
    elif demographics_choice == "Show me regional demographics":
        show_regional_demographics()
    elif demographics_choice == "Show me regional age analysis":
        show_regional_age_analysis()
    elif demographics_choice == "Show me satisfaction correlation":
        show_satisfaction_correlation()
    elif demographics_choice == "Show me age distribution":
        show_age_distribution()
