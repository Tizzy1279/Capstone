import openai
import os
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
import sys

# Add the root directory to the Python path
sys.path.append('/workspaces/Capstone')

from config import OPENAI_API_KEY

# Set the API key
openai.api_key = OPENAI_API_KEY

# Initialize the language model
llm = OpenAI(api_key=OPENAI_API_KEY)

# Define the template for the LLM to generate answers
answer_template = """
You are an expert AI assistant.
Given the following question and the data summary, provide a detailed answer:

Question: {question}
Data Summary: {data_summary}
Answer:
"""

# Define the template for the evaluation
evaluation_template = """
Given the question: {query}
Is the predicted answer: {answer}
correct based on the data summary: {data_summary}?
"""

# Create PromptTemplate objects
answer_prompt = PromptTemplate(input_variables=["question", "data_summary"], template=answer_template)
evaluation_prompt = PromptTemplate(input_variables=["query", "answer", "data_summary"], template=evaluation_template)

# Initialize the LLMChains for generating answers and evaluating them
answer_chain = LLMChain(prompt=answer_prompt, llm=llm)
evaluation_chain = LLMChain(prompt=evaluation_prompt, llm=llm)

# Function to load and summarize data from the CSV file
def load_and_summarize_data(file_path):
    data = pd.read_csv(file_path)

    # Create summaries for each relevant column
    region_summary = data.groupby('Region')['Sales'].sum().reset_index()
    product_summary = data.groupby('Product')['Sales'].sum().reset_index()
    age_summary = data['Customer_Age'].describe()
    gender_summary = data.groupby('Customer_Gender')['Sales'].sum().reset_index()
    satisfaction_summary = data['Customer_Satisfaction'].describe()
    year_summary = data.groupby(data['Date'].str[:4])['Sales'].sum().reset_index() # Summarize sales by year

    # Find the year with the highest sales
    highest_sales_year = year_summary.loc[year_summary['Sales'].idxmax(), 'Date']

    summary = {
        'data_description': data.describe().to_string(),
        'region_sales_summary': region_summary.sort_values(by='Sales', ascending=False).to_string(index=False),
        'product_sales_summary': product_summary.sort_values(by='Sales', ascending=False).to_string(index=False),
        'age_summary': age_summary.to_string(),
        'gender_sales_summary': gender_summary.sort_values(by='Sales', ascending=False).to_string(index=False),
        'satisfaction_summary': satisfaction_summary.to_string(),
        'year_sales_summary': year_summary.sort_values(by='Sales', ascending=False).to_string(index=False),
        'highest_sales_year': highest_sales_year
    }
    return summary

# Path to the CSV file
file_path = '/workspaces/Capstone/sales_data_capstone.csv'

# Load and summarize the data
data_summary = load_and_summarize_data(file_path)

def main():
    while True:
        # Ask the user to input a question
        user_question = input("Please enter your question (or type 'quit' to exit): ")

        if user_question.lower() == 'quit':
            print("Exiting...")
            break

        # Generate prediction using the LLM
        input_pair = {
            "question": user_question,
            "data_summary": data_summary
        }
        predicted_response = answer_chain(input_pair)
        predicted_answer = predicted_response['text'].strip()

        # Evaluate prediction
        evaluation_input = {
            "query": user_question,
            "answer": predicted_answer,
            "data_summary": data_summary
        }
        evaluation_result = evaluation_chain(evaluation_input)

        # Output the generated prediction and evaluation result
        print(f"Question: {user_question}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Evaluation: {evaluation_result['text'].strip()}\n")

if __name__ == "__main__":
    main()