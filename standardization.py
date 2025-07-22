import pandas as pd
import re
import json
from datetime import datetime
import phonenumbers
from typing import Any

exchange_rates = {
    'USD': 1,         # USD to USD (no conversion needed)
    'INR': 0.012,     # Indian Rupee to USD
    'EUR': 1.18,      # Euro to USD
    'GBP': 1.38,      # British Pound to USD
    'JPY': 0.0091,    # Japanese Yen to USD
    'AUD': 0.74,      # Australian Dollar to USD
}
# Load the master file containing regex patterns and formats
def load_master_file() -> dict:
    with open('master_file.json', 'r') as f:
        return json.load(f)


# Utility function to standardize dates based on pandas' to_datetime

def standardize_dates(df: pd.DataFrame, date_columns: list, date_format: str = "%Y-%m-%d") -> pd.DataFrame:
    for col in date_columns:
        # Using dayfirst=True to handle European date format (DD-MM-YYYY)
        df[col] = pd.to_datetime(df[col],format='mixed', infer_datetime_format=True)
        
        # Convert to the specified standard date format
        df[col] = df[col].dt.strftime(date_format)
    return df 



# Utility function to standardize phone numbers using string operations
# Function to standardize phone numbers in DataFrame
def standardize_phone_numbers(df: pd.DataFrame, phone_columns: list) -> pd.DataFrame:
    """
    This function standardizes phone numbers in the specified columns of a DataFrame.

    :param df: DataFrame with phone number columns to standardize
    :param phone_columns: List of column names that contain phone numbers
    :return: DataFrame with standardized phone numbers
    """
    
    # Function to standardize a single phone number
    def standardize_phone_number(phone: Any) -> str:
        if isinstance(phone, str):
            # Remove all non-numeric characters
            phone = ''.join(filter(str.isdigit, phone))
            if len(phone) == 10:  # Assuming 10 digits for North America
                return f"+1 ({phone[:3]}) {phone[3:6]}-{phone[6:]}"
            elif len(phone) == 11 and phone.startswith('1'):  # If it's 11 digits and starts with '1'
                return f"+1 ({phone[1:4]}) {phone[4:7]}-{phone[7:]}"  # Keep as is but format properly
        
            else:
                return '0000000000'
        return phone  # In case phone is None, NaN or other non-string types
    
    # Loop over the specified columns
    for col in phone_columns:
        for index in range(df.shape[0]):  # Use df.shape[0] for the number of rows
            value = df.at[index, col]  # Access the value at the specific row and column
            if pd.isna(value):  # Skip if the value is NaN or None
                continue
            df.at[index, col] = standardize_phone_number(value)  # Standardize the phone number
    
    return df

# Hardcoded exchange rates (relative to USD)
exchange_rates = {
    'USD': 1,         # USD to USD (no conversion needed)
    'INR': 0.012,     # Indian Rupee to USD
    'EUR': 1.18,      # Euro to USD
    'GBP': 1.38,      # British Pound to USD
    'JPY': 0.0091,    # Japanese Yen to USD
    'AUD': 0.74,      # Australian Dollar to USD
}

# Function to get the exchange rate for a specific currency
def get_exchange_rate(currency: str) -> float:
    return exchange_rates.get(currency, 1)  # Default to 1 (USD) if not found


# Function to extract the currency from the column name
def extract_currency_from_column_name(column_name: str) -> str:
    # Extract the currency from the column name (e.g., 'INR' from 'Salary (INR)')
    if '(' in column_name and ')' in column_name:
        return column_name.split('(')[-1].split(')')[0].strip()
    return 'USD'  # Default to USD if no currency is found


# Function to standardize numeric columns (currency to USD)
def standardize_numerics(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    for col in numeric_columns:
        # value_type = match_column_type(df, col, master)
        
        
        # Extract the currency from the column name (e.g., 'INR', 'EUR')
        currency = extract_currency_from_column_name(col)
        
        # If the currency is not USD, convert the whole column to USD
        if currency != 'USD':
            # Get the exchange rate for the currency
            exchange_rate = get_exchange_rate(currency)
            # Convert the whole column to USD
            df[col] = df[col].apply(lambda x: f"USD {float(x.replace(',', '').replace('$', '').replace('€', '').replace('₹', '').strip()) * exchange_rate:,.2f}" if isinstance(x, (int, float)) or x.isnumeric() else 'Invalid Value')
        else:
            # If the column is in USD, just add 'USD' in front
            df[col] = 'USD ' + df[col].astype(str)  # Simply add 'USD' in front of the value
    
    return df
    #     # Apply the standardization function for each row in the column
    #     df[col] = df[col].apply(lambda x: standardize_numeric(x, col))
    # return col



# Utility function to standardize categorical text columns (like names) by title case
def standardize_text(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    for col in text_columns:
        df[col] = df[col].apply(lambda x: standardize_string(x))
    return df

# Function to standardize text (capitalize properly)
def standardize_string(text: Any) -> str:
    if isinstance(text, str):
        return text.strip().title()  # Title case for better consistency
    return text


# Function to match column name against synonyms in the master file
def match_column_type(df: pd.DataFrame, column_name: str, master: dict) -> str:
    column_name_lower = column_name.lower()  # Case insensitive matching
    
    # Loop through each type of data in master file to match synonyms
    for data_type, data_info in master.items():
        for synonym in data_info["synonyms"]:
            if synonym.lower() in column_name_lower:
                return data_type
    
    return "unknown"


# Main function to dynamically apply standardization based on data type
def auto_standardize(df: pd.DataFrame, master: dict) -> pd.DataFrame:
    # Create empty lists to hold column names by type
    date_columns = []
    phone_columns = []
    numeric_columns = []
    text_columns = []
    
    # Loop over columns and detect their types using master file
    for col in df.columns:
        detected_type = match_column_type(df, col, master)
        if detected_type == "dates":
            date_columns.append(col)
        elif detected_type == "phone_numbers":
            phone_columns.append(col)
        elif detected_type == "currency" :
            numeric_columns.append(col)
        elif detected_type == "name":
            text_columns.append(col)
   
    # Apply standardization based on detected types
    df = standardize_dates(df, date_columns)
    df = standardize_phone_numbers(df, phone_columns)
    # df = standardize_numerics(df, numeric_columns)
    df = standardize_text(df, text_columns)
    
    return df
# Sample DataFrame with mixed data
# data = {
#     'Name': ['John Doe', 'jane smith', 'alice Johnson'],
#     'Phone': ['+1 (555) 123-4567', '5552345678', '555-345-6789'],
#     'Salary': ['USD 1000.50', '$1,500.00', '1500.45'],
#     'Join Date': ['12/15/2021', '23 July 2025', '15-12-2021'],
#     'Status': ['active', 'inactive', 'Pending'],
#     'Percentage': ['20%', '30%', '50%'],
# }

# # Creating DataFrame
# df = pd.DataFrame(data)

# Load the master file containing regex patterns
master = load_master_file()

# Automatically standardizing the DataFrame using regex patterns from master file

# df=pd.read_csv("random_employee_data_mixed_names.csv")
# Display the standardized DataFrame
# print(standardized_df)
# import time

# start_time = time.time()  # Record start time
# standardized_df = auto_standardize(df, master)         # Call your function
# end_time = time.time()    # Record end time

# elapsed_time = end_time - start_time
# standardized_df.to_csv('processed_employee_data.csv', index=False)

# print(f"Function took {elapsed_time:.4f} seconds to run.")