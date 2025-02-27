import streamlit as st
import matplotlib.pyplot as plt
from AI_Powered_Business_Intelligence import (
    load_data,
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

def main():
    st.title("AI-Powered Business Intelligence App")

    # Load the data
    data = load_data()
    st.write("Data Loaded Successfully")

    # Sidebar navigation for different analysis sections
    analysis_type = st.sidebar.selectbox("Choose Analysis Type", [
        "Monthly Sales",
        "Product Sales",
        "Product Customer Age",
        "Product Satisfaction",
        "Regional Sales",
        "Regional Customer Age",
        "Regional Satisfaction",
        "Gender Analysis",
        "Age Analysis",
        "Regional Demographics",
        "Regional Age Analysis",
        "Satisfaction Correlation",
        "Age Distribution"
    ])

    # Show selected analysis
    if analysis_type == "Monthly Sales":
        show_monthly_sales()
    elif analysis_type == "Product Sales":
        show_product_sales()
    elif analysis_type == "Product Customer Age":
        show_product_customer_age()
    elif analysis_type == "Product Satisfaction":
        show_product_satisfaction()
    elif analysis_type == "Regional Sales":
        show_regional_sales()
    elif analysis_type == "Regional Customer Age":
        show_regional_customer_age()
    elif analysis_type == "Regional Satisfaction":
        show_regional_satisfaction()
    elif analysis_type == "Gender Analysis":
        show_gender_analysis()
    elif analysis_type == "Age Analysis":
        show_age_analysis()
    elif analysis_type == "Regional Demographics":
        show_regional_demographics()
    elif analysis_type == "Regional Age Analysis":
        show_regional_age_analysis()
    elif analysis_type == "Satisfaction Correlation":
        show_satisfaction_correlation()
    elif analysis_type == "Age Distribution":
        show_age_distribution()

if __name__ == "__main__":
    main()
