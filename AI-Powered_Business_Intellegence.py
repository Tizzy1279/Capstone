import openai
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import PromptTemplate
from config import OPENAI_API_KEY
from config import HF_Key


# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=Warning)

# Define an evaluation function or use existing metrics
def naive_evaluator(predicted_answer, correct_answer):
    # Simple comparison
    return predicted_answer.strip() == correct_answer.strip()  # Return True if the answers match exactly

def get_regional_customer_age_stats(data):
    """Calculate mean, median, and std deviation of customer ages per region."""
    return data.groupby('Region')['Age'].agg(['mean', 'median', 'std']).reset_index()

#memory
class SimpleMemory:
    def __init__(self):
        self.memory = []

    def add_interaction(self, question, response):
        self.memory.append({'question': question, 'response': response})

    def retrieve_memory(self):
        # Return the memory as a string for context in follow-up questions
        memory_text = ""
        for entry in self.memory:
            memory_text += f"Q: {entry['question']}\nA: {entry['response']}\n"
        return memory_text

def main():
  # Set your OpenAI API key as an environment variable
  openai.api_key = OPENAI_API_KEY
  hf_api_key = HF_Key

# Part 1: AI-Powered Business Intelligence Assistant
from google.colab import drive
drive.mount('/content/drive')

# Step 1: Data Preparation
def load_data():
  file_path = '/workspaces/Capstone/sales_data_capstone v2.csv'
  data = pd.read_csv(file_path)
  data['Date'] = pd.to_datetime(data['Date'])
  return data

# Call the load_data function to load the data
data = load_data()

 # Display a summary of the data
print("Data summary:")
print(data.info())

print("\nFirst few rows of data:")
print(data.head())

print("\nStatistical summary of numerical columns:")
print(data.describe())  # Provides a statistical summary of numerical columns

# Read the CSV file
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])

# Function to interpret and execute questions dynamically
def interpret_question(question, df, memory, summary_stats): #added df
    question = question.lower()

    if "total sales" in question:
        total_sales = summary_stats['total_sales']
        response = f"Total sales for all products: ${total_sales:,.2f}"

    elif "average sales" in question:
        avg_sales = summary_stats['avg_sales']
        response = f"Average sales for all products: ${avg_sales:,.2f}"

    elif "standard deviation" in question:
        std_sales = summary_stats['std_sales']
        response = f"Standard deviation of sales: ${std_sales:,.2f}"

    elif "median sales" in question:
        med_sales = summary_stats['med_sales']
        response = f"Median sales: ${med_sales:,.2f}"

    else:
        response = "Sorry, I don't understand the question."

    return response

def interpret_product_question(question, df, memory):
    question = question.lower()

    if "total sales" in question:
        product_match = re.search(r'widget \w+', question)
        if product_match:
            product = product_match.group(0)
            if product in df['Product'].values:
                total_sales = df[df['Product'] == product]['Sales'].sum()
                response = f"Total sales for {product.capitalize()}: ${total_sales:,.2f}"
            else:
                response = f"No sales data found for {product.capitalize()}. Please check the product name."
        else:
            response = f"Total sales for all products: ${df['Sales'].sum():,.2f}"

    elif "total sales" in question:
        product_match = re.search(r'widget \w+', question)
        if product_match:
            product = product_match.group(0)
            if product in df['Product'].values:
                avg_sales = df[df['Product'] == product]['Sales'].mean()
                response = f"Average sales for {product.capitalize()}: ${avg_sales:,.2f}"
            else:
                response = f"No sales data found for {product.capitalize()}. Please check the product name."
        else:
            response = f"Average sales for all products: ${df['Sales'].mean():,.2f}"

    else:
        response = "Sorry, I don't understand the question."

    return response

def interpret_gender_question(question, df, memory):
    question = question.lower()

    if "total sales" in question:
        gender_match = re.search(r'female|male', question)
        if gender_match:
            gender = gender_match.group(0)
            if gender in df['Customer_Gender'].values:
                total_sales = df[df['Customer_Gender'] == gender.capitalize()]['Sales'].sum()
                response = f"Total sales for {gender.capitalize()} customers: ${total_sales:,.2f}"
            else:
                response = f"No sales data found for {gender.capitalize()} customers."
        else:
            response = f"Total sales for all customers: ${df['Sales'].sum():,.2f}"

    elif "total sales" in question:
        gender_match = re.search(r'female|male', question)
        if gender_match:
            gender = gender_match.group(0)
            if gender in df['Customer_Gender'].values:
                avg_sales = df[df['Customer_Gender'] == gender.capitalize()]['Sales'].mean()
                response = f"Average sales for {gender.capitalize()} customers: ${avg_sales:,.2f}"
            else:
                response = f"No sales data found for {gender.capitalize()} customers."
        else:
            response = f"Average sales for all customers: ${df['Sales'].mean():,.2f}"

    else:
        response = "Sorry, I don't understand the question."

    return response

def interpret_region_question(question, df, memory):
    question = question.lower()

    if "total sales" in question:
        region_match = re.search(r'\b(north|south|east|west)\b', question)
        if region_match:
            region = region_match.group(0).capitalize()
            if region in df['Region'].values:
                total_sales = df[df['Region'] == region]['Sales'].sum()
                response = f"Total sales in the {region} region: ${total_sales:,.2f}"
            else:
                response = f"No sales data found for the {region} region."
        else:
            response = f"Total sales for all regions: ${df['Sales'].sum():,.2f}"

    elif "total sales" in question:
        region_match = re.search(r'\b(north|south|east|west)\b', question)
        if region_match:
            region = region_match.group(0).capitalize()
            if region in df['Region'].values:
                avg_sales = df[df['Region'] == region]['Sales'].mean()
                response = f"Average sales in the {region} region: ${avg_sales:,.2f}"
            else:
                response = f"No sales data found for the {region} region."
        else:
            response = f"Average sales for all regions: ${df['Sales'].mean():,.2f}"

    else:
        response = "Sorry, I don't understand the question."

    return response

    # Handle question type and check for previous responses
    conversation_history = memory.retrieve_memory()
    question_with_context = conversation_history + "Q: " + question.lower()

    # Add the current interaction to memory
    memory.add_interaction(question, response)

# Function to annotate statistical data on bar plot
def annotate_stats(ax, series, is_sales=False):
    mean_val = series.mean()
    median_val = series.median()
    std_dev_val = series.std()

    if is_sales:
        stats_text = (
            f'Mean: ${mean_val:.2f}\n'
            f'Median: ${median_val:.2f}\n'
            f'Standard Deviation: ${std_dev_val:.2f}'
        )
    else:
        stats_text = (
            f'Mean: {mean_val:.2f}\n'
            f'Median: {median_val:.2f}\n'
            f'Standard Deviation: {std_dev_val:.2f}'
        )

    ax.annotate(stats_text, xy=(0.75, 0.9), xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))

    print(f'Statistical Data:\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStandard Deviation: {std_dev_val:.2f}')

# Sales Performance Analysis Functions
def show_monthly_sales():
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' column is in datetime format
    df.set_index('Date', inplace=True)  # Set 'Date' column as the index
    monthly_sales = df.resample('MS')['Sales'].sum().reset_index()
    monthly_sales['YearMonth'] = monthly_sales['Date'].dt.strftime('%Y-%m')
    ax = monthly_sales.plot(x='YearMonth', y='Sales', kind='line', title='Monthly Sales')
    annotate_stats(ax, monthly_sales['Sales'], is_sales=True)
    plt.xlabel('YearMonth')
    plt.ylabel('Sales ($)')
    plt.title('Monthly Sales')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()


# Product Analysis Functions
def show_product_sales():
    product_sales = df.groupby('Product')['Sales'].sum().reset_index()
    ax = product_sales.plot(x='Product', y='Sales', kind='bar', title='Product Sales')
    annotate_stats(ax, product_sales['Sales'], is_sales=True)
    plt.xlabel('Product')
    plt.ylabel('Sales ($)')
    plt.title('Product Sales')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

def show_product_customer_age():
    product_age = df.groupby('Product')['Customer_Age'].mean().reset_index()
    ax = product_age.plot(x='Product', y='Customer_Age', kind='bar', title='Average Customer Age by Product')
    annotate_stats(ax, product_age['Customer_Age'])
    plt.xlabel('Product')
    plt.ylabel('Average Customer Age')
    plt.title('Average Customer Age by Product')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

def show_product_satisfaction():
    product_satisfaction = df.groupby('Product')['Customer_Satisfaction'].mean().reset_index()
    ax = product_satisfaction.plot(x='Product', y='Customer_Satisfaction', kind='bar', title='Customer Satisfaction by Product')
    annotate_stats(ax, product_satisfaction['Customer_Satisfaction'])
    plt.xlabel('Product')
    plt.ylabel('Customer Satisfaction')
    plt.title('Customer Satisfaction by Product')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

# Regional Analysis Functions
def show_regional_sales():
    regional_sales = df.groupby('Region')['Sales'].sum().reset_index()
    ax = regional_sales.plot(x='Region', y='Sales', kind='bar', title='Regional Sales')
    annotate_stats(ax, regional_sales['Sales'], is_sales=True)
    plt.xlabel('Region')
    plt.ylabel('Sales ($)')
    plt.title('Regional Sales')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

def show_regional_customer_age():
    regional_age = df.groupby('Region')['Customer_Age'].mean().reset_index()
    ax = regional_age.plot(x='Region', y='Customer_Age', kind='bar', title='Average Customer Age by Region')
    annotate_stats(ax, regional_age['Customer_Age'])
    plt.xlabel('Region')
    plt.ylabel('Average Customer Age')
    plt.title('Average Customer Age by Region')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

def show_regional_satisfaction():
    regional_satisfaction = df.groupby('Region')['Customer_Satisfaction'].mean().reset_index()
    ax = regional_satisfaction.plot(x='Region', y='Customer_Satisfaction', kind='bar', title='Customer Satisfaction by Region')
    annotate_stats(ax, regional_satisfaction['Customer_Satisfaction'])
    plt.xlabel('Region')
    plt.ylabel('Customer Satisfaction')
    plt.title('Customer Satisfaction by Region')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

# Demographic Analysis Functions
def show_gender_analysis():
    gender_sales = df.groupby('Customer_Gender')['Sales'].sum().reset_index()
    ax = gender_sales.plot(x='Customer_Gender', y='Sales', kind='bar', title='Sales by Gender')
    annotate_stats(ax, gender_sales['Sales'], is_sales=True)
    plt.xlabel('Customer Gender')
    plt.ylabel('Sales ($)')
    plt.title('Sales by Gender')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

def show_age_analysis():
    age_sales = df.groupby('Customer_Age')['Sales'].sum().reset_index()
    ax = age_sales.plot(x='Customer_Age', y='Sales', kind='bar', title='Sales by Age')
    annotate_stats(ax, age_sales['Sales'], is_sales=True)
    plt.xlabel('Customer Age')
    plt.ylabel('Sales ($)')
    plt.title('Sales by Age')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

def show_regional_demographics():
    regional_demographics = df.groupby('Region')['Customer_Gender'].value_counts(normalize=True).unstack().reset_index()
    ax = regional_demographics.plot(x='Region', kind='bar', title='Regional Demographics')
    plt.xlabel('Region')
    plt.ylabel('Proportion')
    plt.title('Regional Demographics')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

def show_regional_age_analysis():
    regional_age = df.groupby('Region')['Customer_Age'].mean().reset_index()
    ax = regional_age.plot(x='Region', y='Customer_Age', kind='bar', title='Average Customer Age by Region')
    annotate_stats(ax, regional_age['Customer_Age'])
    plt.xlabel('Region')
    plt.ylabel('Average Customer Age')
    plt.title('Average Customer Age by Region')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

def show_satisfaction_correlation():
    numeric_df = df.select_dtypes(include=[float, int])
    satisfaction_correlation = numeric_df.corr()['Customer_Satisfaction'].sort_values(ascending=False).reset_index()
    ax = satisfaction_correlation.plot(x='index', y='Customer_Satisfaction', kind='bar', title='Satisfaction Correlation')
    annotate_stats(ax, satisfaction_correlation['Customer_Satisfaction'])
    plt.xlabel('Variables')
    plt.ylabel('Correlation with Customer Satisfaction')
    plt.title('Satisfaction Correlation')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

def show_age_distribution():
    age_distribution = df['Customer_Age'].value_counts().reset_index()
    ax = age_distribution.plot(x='index', y='Customer_Age', kind='bar', title='Age Distribution')
    annotate_stats(ax, age_distribution['Customer_Age'])
    plt.xlabel('Customer Age')
    plt.ylabel('Count')
    plt.title('Age Distribution')
    plt.show()   # Show the plot
    plt.pause(0.1)  # Pause to ensure it displays correctly
    plt.close()

# Menu to select plot option
# Function to format statistical data without dollar signs
def print_statistical_data(mean, median, std_dev):
        print(f"Mean: {mean:.2f}")
        print(f"Median: {median:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")

# Example usage in your menu
def menu():
    while True:
        print("\nChoose Analysis Category:")
        print("1. Sales Performance")
        print("2. Product Analysis")
        print("3. Regional Analysis")
        print("4. Demographics")
        print("5. Quit")
        choice = input("Enter choice: ")

        if choice == '1':
            show_monthly_sales()

        elif choice == '2':
            while True:
                print("Product Analysis:")
                print("1. Show me product sales")
                print("2. Show me product customer age")
                print("3. Show me product satisfaction")
                print("4. Back to main menu")
                option = input("Enter option: ")
                if option == '1':
                    show_product_sales()
                elif option == '2':
                    show_product_customer_age()
                elif option == '3':
                    show_product_satisfaction()
                elif option == '4':
                    break
                else:
                    print("Invalid option. Please try again.")

        elif choice == '3':
            while True:
                print("Regional Analysis:")
                print("1. Show me regional sales")
                print("2. Show me regional customer age")
                print("3. Show me regional satisfaction")
                print("4. Back to main menu")
                option = input("Enter option: ")
                if option == '1':
                    show_regional_sales()
                elif option == '2':
                    # Assuming the following functions return the mean, median, and std_dev for customer age
                    mean_age, median_age, std_dev_age = get_regional_customer_age_stats(data)
                    print("Statistical Data:")
                    print_statistical_data(mean_age, median_age, std_dev_age, data_type="age")
                elif option == '3':
                    show_regional_satisfaction()
                elif option == '4':
                    break
                else:
                    print("Invalid option. Please try again.")

        elif choice == '4':
            while True:
                print("Demographics:")
                print("1. Show me gender analysis")
                print("2. Show me age analysis")
                print("3. Show me regional demographics")
                print("4. Show me regional age analysis")
                print("5. Show me satisfaction correlation")
                print("6. Back to main menu")
                option = input("Enter option: ")
                if option == '1':
                    show_gender_analysis()
                elif option == '2':
                    show_age_analysis()
                elif option == '3':
                    show_regional_demographics()
                elif option == '4':
                    show_regional_age_analysis()
                elif option == '5':
                    show_satisfaction_correlation()
                elif option == '6':
                    break
                else:
                    print("Invalid option. Please try again.")

        elif choice == '5':
            print("Exiting menu.")
            break

        else:
            print("Invalid choice. Please try again.")

# Call the menu
menu()

# Initialize memory
memory = SimpleMemory()

def process_user_question(llm_chain, summary, question):
    # Generate response using LLM chain
    response = llm_chain({'summary': summary, 'question': question})

    # Store interaction in memory
    memory.add_interaction(question, response['text'])
    return response['text']

# After processing a question, you can access past interactions like this:
print("Past interactions:")
for entry in memory.retrieve_memory():
    print(f"Q: {entry['question']}, A: {entry['response']}")

#Part 2 and 3
# Define the language model
llm = OpenAI(openai.api_key=OPENAI_API_KEY)

template = '''
You are an expert AI sales analyst
look into the summary below
{summary}

and answer user questions

question: {question}
detailed analysis and recommendation:
'''
# Calculate sales statistics from your data
total_sales = data['Sales'].sum()
avg_sales = data['Sales'].mean()
std_sales = data['Sales'].std()
med_sales = data['Sales'].median()

summary = 'The overall summary goes as follows:\n' \
          'Total sales: ${total_sales}\n' \
          'Average sales: ${avg_sales}\n' \
          'Standard deviation: ${std_sales}\n' \
          'Median sales: ${med_sales}'.format(total_sales=total_sales, std_sales=std_sales, med_sales=med_sales, avg_sales=avg_sales)

# Group data by 'Product' and calculate total sales, average sales, standard deviation, and median sales
product_sales_stats = data.groupby('Product')['Sales'].agg(
    total_sales='sum',
    avg_sales='mean',
    std_sales='std',
    med_sales='median'
).reset_index()

# Display statistics for each product
for index, row in product_sales_stats.iterrows():
    product_name = row['Product']
    total_sales = row['total_sales']
    avg_sales = row['avg_sales']
    std_sales = row['std_sales']
    med_sales = row['med_sales']

    print(f"Product: {product_name}")
    print(f"  Total sales: ${total_sales}")
    print(f"  Average sales: ${avg_sales:.2f}")
    print(f"  Standard deviation: ${std_sales:.2f}")
    print(f"  Median sales: ${med_sales:.2f}")
    print("-" * 40)

# Group data by 'Gender' and calculate total sales, average sales, standard deviation, and median sales
product_sales_stats = data.groupby('Customer_Gender')['Sales'].agg(
    total_sales='sum',
    avg_sales='mean',
    std_sales='std',
    med_sales='median'
).reset_index()

# Display statistics for each product
for index, row in product_sales_stats.iterrows():
    gender_name = row['Customer_Gender']
    total_sales = row['total_sales']
    avg_sales = row['avg_sales']
    std_sales = row['std_sales']
    med_sales = row['med_sales']

    print(f"Gender: {gender_name}")
    print(f"  Total sales: ${total_sales}")
    print(f"  Average sales: ${avg_sales:.2f}")
    print(f"  Standard deviation: ${std_sales:.2f}")
    print(f"  Median sales: ${med_sales:.2f}")
    print("-" * 40)

# Group data by 'Region' and calculate total sales, average sales, standard deviation, and median sales
product_sales_stats = data.groupby('Region')['Sales'].agg(
    total_sales='sum',
    avg_sales='mean',
    std_sales='std',
    med_sales='median'
).reset_index()

# Display statistics for each product
for index, row in product_sales_stats.iterrows():
    region_name = row['Region']
    total_sales = row['total_sales']
    avg_sales = row['avg_sales']
    std_sales = row['std_sales']
    med_sales = row['med_sales']

    print(f"Region: {region_name}")
    print(f"  Total sales: ${total_sales}")
    print(f"  Average sales: ${avg_sales:.2f}")
    print(f"  Standard deviation: ${std_sales:.2f}")
    print(f"  Median sales: ${med_sales:.2f}")
    print("-" * 40)

# Ask the user for their question
question = input("Please enter your question: ")

# Create the PromptTemplate
pt = PromptTemplate(template=template, input_variables=['summary', 'question'])

# Create the chain
llm_chain = LLMChain(prompt=pt, llm=llm)

# Generate the response using the invoke method
analysis = llm_chain({'summary': summary, 'question': question})

# Store the generated response
predicted_answers.append(analysis['text'])

print(analysis['text'])

recommendation_template = '''
Based on the following analysis:
{response}

Provide specific recommendations according to the question: {question}

Recommendation:
'''

analysis_template = '''
Analysis summary as follows:
{summary}
'''

analysis_prompt = PromptTemplate(template=analysis_template, input_variables=['summary', 'question'])
recommendation_prompt = PromptTemplate(template=recommendation_template, input_variables=['response', 'question'])

analysis_chain = LLMChain(prompt=analysis_prompt, llm=llm, output_key="response")
recommendation_chain = LLMChain(prompt=recommendation_prompt, llm=llm, output_key="recommendation")

sequential_chain = SequentialChain(
    chains=[analysis_chain, recommendation_chain],
    input_variables=['summary', 'question'],
    output_variables=['response', 'recommendation']
)

# Function to evaluate the predictions and provide feedback
def print_evaluation_results(evaluation_results):
    print("Evaluation Results:")
    for i, res in enumerate(evaluation_results):
        if isinstance(res, list):
            for r in res:
                if isinstance(r, dict) and 'results' in r:
                    print(f"\nEvaluation {i + 1}:")
                    print(f"  {r['results'].strip()}")
                else:
                    print(f"\nEvaluation {i + 1}: Invalid result structure.")
        else:
            print(f"\nEvaluation {i + 1}: Invalid result structure.")

def evaluate_predictions(df, memory): #added df, summary_stats inside ()
    evaluation_results = []
    for i in range(len(questions)):
        input_pair = {
            "query": questions[i],
            "answer": predicted_answers[i],
            #"result": correct_answer[i]  # Adding the correct answer for evaluation
        }
        if "product" in questions[i].lower():
            correct_answer_dynamic = interpret_product_question(questions[i], df)
        elif "gender" in questions[i].lower():
            correct_answer_dynamic = interpret_gender_question(questions[i], df)
        elif "region" in questions[i].lower():
            correct_answer_dynamic = interpret_region_question(questions[i], df)
        else:
            correct_answer_dynamic = interpret_question(questions[i], df, memory, summary_stats)

        input_pair["result"] = correct_answer_dynamic

        # Evaluating prediction accuracy
        evaluation_result = qa_eval_chain.evaluate(
            [input_pair],  # Passing input as a list containing a single dictionary
        )
        if isinstance(evaluation_result, list) and len(evaluation_result) > 0:
            for res in evaluation_result:
                if isinstance(res, dict) and 'results' in res:
                    evaluation_results.append(res)  # Add the dictionary to the evaluation results
                else:
                    print(f"Unexpected structure: {res}")
        else:
            print(f"Unexpected structure: {evaluation_result}")

    # Output the evaluation results
    print_evaluation_results(evaluation_results)

# Function to ask user questions and provide responses and recommendations
def ask_user_questions():
    while True:
        question = input("Please enter your question: ")

        # Create the prompt template for the question
        template = "Please provide an analysis summary for the following question: {question}"
        pt = PromptTemplate(template=template, input_variables=['question'])
        llm_chain = LLMChain(prompt=pt, llm=llm)

        # Generate the response using the question
        analysis = llm_chain({'question': question})

        # Generate recommendation using sequential chain
        response = sequential_chain({'summary': analysis['text'], 'question': question})

        print("Analysis:", response['response'])
        print("Recommendation:", response['recommendation'])

        # Ask if the user wants to ask a follow-up question
        while True:
            follow_up = input("Would you like to ask a follow-up question? (y/n): ").strip().lower()
            if follow_up == 'y':
                break  # Continue the loop and ask for a new question
            elif follow_up == 'n':
                print("Exiting question loop.")
                return  # Exit the function entirely
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")

# Calling the function to test
ask_user_questions()

# Call the main function to run the application
if __name__ == "__main__":
    main()
