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
import streamlit as st

# Load Data
def load_data():
    file_path = '/workspaces/Capstone/sales_data_capstone.csv'
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()

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
    monthly_sales = data.resample('MS', on='Date')['Sales'].sum().reset_index()
    monthly_sales['YearMonth'] = monthly_sales['Date'].dt.strftime('%Y-%m')
    ax = monthly_sales.plot(x='YearMonth', y='Sales', kind='line', title='Monthly Sales')
    annotate_stats(ax, monthly_sales['Sales'], is_sales=True)
    plt.xlabel('YearMonth')
    plt.ylabel('Sales ($)')
    plt.title('Monthly Sales')
    st.pyplot(plt)
    plt.close()

# Product Analysis Functions
def show_product_sales():
    product_sales = data.groupby('Product')['Sales'].sum().reset_index()
    ax = product_sales.plot(x='Product', y='Sales', kind='bar', title='Product Sales')
    annotate_stats(ax, product_sales['Sales'], is_sales=True)
    plt.xlabel('Product')
    plt.ylabel('Sales ($)')
    plt.title('Product Sales')
    st.pyplot(plt)
    plt.close()

def show_product_customer_age():
    product_age = data.groupby('Product')['Customer_Age'].mean().reset_index()
    ax = product_age.plot(x='Product', y='Customer_Age', kind='bar', title='Average Customer Age by Product')
    annotate_stats(ax, product_age['Customer_Age'])
    plt.xlabel('Product')
    plt.ylabel('Average Customer Age')
    plt.title('Average Customer Age by Product')
    st.pyplot(plt)
    plt.close()

def show_product_satisfaction():
    product_satisfaction = data.groupby('Product')['Customer_Satisfaction'].mean().reset_index()
    ax = product_satisfaction.plot(x='Product', y='Customer_Satisfaction', kind='bar', title='Customer Satisfaction by Product')
    annotate_stats(ax, product_satisfaction['Customer_Satisfaction'])
    plt.xlabel('Product')
    plt.ylabel('Customer Satisfaction')
    plt.title('Customer Satisfaction by Product')
    st.pyplot(plt)
    plt.close()

# Regional Analysis Functions
def show_regional_sales():
    regional_sales = data.groupby('Region')['Sales'].sum().reset_index()
    ax = regional_sales.plot(x='Region', y='Sales', kind='bar', title='Regional Sales')
    annotate_stats(ax, regional_sales['Sales'], is_sales=True)
    plt.xlabel('Region')
    plt.ylabel('Sales ($)')
    plt.title('Regional Sales')
    st.pyplot(plt)
    plt.close()

def show_regional_customer_age():
    regional_age = data.groupby('Region')['Customer_Age'].mean().reset_index()
    ax = regional_age.plot(x='Region', y='Customer_Age', kind='bar', title='Average Customer Age by Region')
    annotate_stats(ax, regional_age['Customer_Age'])
    plt.xlabel('Region')
    plt.ylabel('Average Customer Age')
    plt.title('Average Customer Age by Region')
    st.pyplot(plt)
    plt.close()

def show_regional_satisfaction():
    regional_satisfaction = data.groupby('Region')['Customer_Satisfaction'].mean().reset_index()
    ax = regional_satisfaction.plot(x='Region', y='Customer_Satisfaction', kind='bar', title='Customer Satisfaction by Region')
    annotate_stats(ax, regional_satisfaction['Customer_Satisfaction'])
    plt.xlabel('Region')
    plt.ylabel('Customer Satisfaction')
    plt.title('Customer Satisfaction by Region')
    st.pyplot(plt)
    plt.close()

# Demographic Analysis Functions
def show_gender_analysis():
    gender_sales = data.groupby('Customer_Gender')['Sales'].sum().reset_index()
    ax = gender_sales.plot(x='Customer_Gender', y='Sales', kind='bar', title='Sales by Gender')
    annotate_stats(ax, gender_sales['Sales'], is_sales=True)
    plt.xlabel('Customer Gender')
    plt.ylabel('Sales ($)')
    plt.title('Sales by Gender')
    st.pyplot(plt)
    plt.close()

def show_age_analysis():
    age_sales = data.groupby('Customer_Age')['Sales'].sum().reset_index()
    ax = age_sales.plot(x='Customer_Age', y='Sales', kind='bar', title='Sales by Age')
    annotate_stats(ax, age_sales['Sales'], is_sales=True)
    plt.xlabel('Customer Age')
    plt.ylabel('Sales ($)')
    plt.title('Sales by Age')
    st.pyplot(plt)
    plt.close()

def show_regional_demographics():
    regional_demographics = data.groupby('Region')['Customer_Gender'].value_counts(normalize=True).unstack().reset_index()
    ax = regional_demographics.plot(x='Region', kind='bar', title='Regional Demographics')
    plt.xlabel('Region')
    plt.ylabel('Proportion')
    plt.title('Regional Demographics')
    st.pyplot(plt)
    plt.close()

def show_regional_age_analysis():
    regional_age = data.groupby('Region')['Customer_Age'].mean().reset_index()
    ax = regional_age.plot(x='Region', y='Customer_Age', kind='bar', title='Average Customer Age by Region')
    annotate_stats(ax, regional_age['Customer_Age'])
    plt.xlabel('Region')
    plt.ylabel('Average Customer Age')
    plt.title('Average Customer Age by Region')
    st.pyplot(plt)
    plt.close()

def show_satisfaction_correlation():
    numeric_df = data.select_dtypes(include=[float, int])
    satisfaction_correlation = numeric_df.corr()['Customer_Satisfaction'].sort_values(ascending=False).reset_index()
    ax = satisfaction_correlation.plot(x='index', y='Customer_Satisfaction', kind='bar', title='Satisfaction Correlation')
    annotate_stats(ax, satisfaction_correlation['Customer_Satisfaction'])
    plt.xlabel('Variables')
    plt.ylabel('Correlation with Customer Satisfaction')
    plt.title('Satisfaction Correlation')
    st.pyplot(plt)
    plt.close()

def show_age_distribution():
    age_distribution = data['Customer_Age'].value_counts().reset_index()
    age_distribution.columns = ['Customer_Age', 'Count']
    ax = age_distribution.plot(x='Customer_Age', y='Count', kind='bar', title='Age Distribution')
    annotate_stats(ax, age_distribution['Count'])
    plt.xlabel('Customer Age')
    plt.ylabel('Count')
    plt.title('Age Distribution')
    st.pyplot(plt)
    plt.close()

