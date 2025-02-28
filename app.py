import streamlit as st
import pandas as pd
import os
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
from src.Q_A import answer_chain, evaluation_chain, load_and_summarize_data

# Path to the CSV file
file_path = '/workspaces/Capstone/sales_data_capstone.csv'

# Load and summarize the data
data_summary = load_and_summarize_data(file_path)

def main():
    st.title("Interactive Business Intelligence and Q&A App")

    st.sidebar.title("Navigation")
    options = ["Business Intelligence", "Q&A"]
    choice = st.sidebar.radio("Go to", options)

    if choice == "Business Intelligence":
        st.header("Business Intelligence")
        analysis_options = [
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
        ]
        analysis_choice = st.selectbox("Choose Analysis", analysis_options)

        if analysis_choice == "Monthly Sales":
            show_monthly_sales()
        elif analysis_choice == "Product Sales":
            show_product_sales()
        elif analysis_choice == "Product Customer Age":
            show_product_customer_age()
        elif analysis_choice == "Product Satisfaction":
            show_product_satisfaction()
        elif analysis_choice == "Regional Sales":
            show_regional_sales()
        elif analysis_choice == "Regional Customer Age":
            show_regional_customer_age()
        elif analysis_choice == "Regional Satisfaction":
            show_regional_satisfaction()
        elif analysis_choice == "Gender Analysis":
            show_gender_analysis()
        elif analysis_choice == "Age Analysis":
            show_age_analysis()
        elif analysis_choice == "Regional Demographics":
            show_regional_demographics()
        elif analysis_choice == "Regional Age Analysis":
            show_regional_age_analysis()
        elif analysis_choice == "Satisfaction Correlation":
            show_satisfaction_correlation()
        elif analysis_choice == "Age Distribution":
            show_age_distribution()

    elif choice == "Q&A":
        st.header("Q&A")
        user_question = st.text_input("Please enter your question:")
        if user_question:
            input_pair = {
                "question": user_question,
                "data_summary": data_summary
            }
            predicted_response = answer_chain(input_pair)
            predicted_answer = predicted_response['text'].strip()

            evaluation_input = {
                "query": user_question,
                "answer": predicted_answer,
                "data_summary": data_summary
            }
            evaluation_result = evaluation_chain(evaluation_input)

            st.write(f"Question: {user_question}")
            st.write(f"Predicted Answer: {predicted_answer}")
            st.write(f"Evaluation: {evaluation_result['text'].strip()}")

if __name__ == "__main__":
    main()

#streamlit run /workspaces/Capstone/app.py
#./run_streamlit.sh