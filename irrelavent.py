import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from langdetect import detect, DetectorFactory, LangDetectException
import pandas as pd
import ftfy
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os
import re
import ast
os.environ['OPENAI_API_KEY'] = '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'  # Replace with your actual key or use st.secrets

def detect_and_remove_outliers(df, method='iqr', cardinality_threshold=100, avg_digit_threshold=2.5,sample_size=10):
    """
    Detects and removes outliers from numerical columns using IQR, Z-Score, or Isolation Forest.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        method (str): 'iqr', 'zscore', or 'isolation_forest'.
        columns (list): List of columns to check. If None, all numeric columns are used.
        contamination (float): Proportion of outliers for Isolation Forest.
        z_thresh (float): Threshold for Z-Score method.

    Returns:
        cleaned_df (pd.DataFrame): DataFrame with outliers removed.
        report (dict): Report of how many rows were removed per column.
    """

    df = df.copy()
    report = {}
    
    # numeric_candidates = []
    # for col in df.columns:
    #     if df[col].dtype == 'object':
    #         # Strip commas and other non-numeric characters, then try casting
    #         cleaned_sample = df[col].dropna().astype(str).str.replace(',', '', regex=False).head(10)
    #         try:
    #             sample = cleaned_sample.astype(float)
    #             if sample.count() == 10:
    #                 numeric_candidates.append(col)
    #         except:
    #             continue
    #     elif pd.api.types.is_numeric_dtype(df[col]):
    #         numeric_candidates.append(col)
    # for col in numeric_candidates:
    #     if df[col].dtype == 'object':
    #         # Remove commas and strip whitespace
    #         df[col] = df[col].dropna().astype(str).str.replace(',', '', regex=False).str.strip()
    #         try:
    #             df[col] = df[col].astype(float)
    #         except ValueError:
    #             pass 
    # Step 3: Define Prompt
    prompt = PromptTemplate.from_template("""
    You are given a sample DataFrame:
    {sample_dataframe}

    Identify the column names that are truly numeric and suitable for statistical analysis like outlier detection.
    Exclude columns that are identifiers or categorical codes (e.g. postal_code, employee_id).
    Return only the column names as a Python list.
    """)

    # Step 4: Create Chain
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")  # or use AzureChatOpenAI if needed
    chain = LLMChain(llm=llm, prompt=prompt)

    # Step 5: Run Chain
    response = chain.run({"sample_dataframe": df.head()})
    pattern = r"\[\s*['\"].*?['\"](?:\s*,\s*['\"].*?['\"])*\s*\]"
    print("****************************8")
    print (response)
    matches = re.findall(pattern, response, re.DOTALL)

    for match in matches:
       
        parsed = ast.literal_eval(match.strip())
    # print(column_list) 
    true_numeric=set(parsed)
    if method == 'iqr':
        for col in true_numeric:
          
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.strip()
                .astype('Float64')
            )
            # IQR method for outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Replace outliers with pd.NA
            # original_count = df[col].notna().sum()
            df[col] = df[col].where(
                (df[col] >= lower_bound) & (df[col] <= upper_bound),
                pd.NA
            )
            # new_count = df[col].notna().sum()
    # elif method == 'zscore':
    #     for col in columns:
    #         z_scores = np.abs(stats.zscore(df[col]))
    #         initial_rows = df.shape[0]
    #         df = df[z_scores < z_thresh]
    #         report[col] = initial_rows - df.shape[0]

    # elif method == 'isolation_forest':
    #     iso = IsolationForest(contamination=contamination, random_state=42)
    #     X = df[columns]
    #     preds = iso.fit_predict(X)
    #     initial_rows = df.shape[0]
    #     df = df[preds == 1]
    #     report['isolation_forest'] = initial_rows - df.shape[0]

    else:
        raise ValueError("Method must be 'iqr', 'zscore', or 'isolation_forest'")
    
    return df


def handle_outliers_dynamic_smart(df, columns=None, contamination=0.01, threshold_factor=3.0):
    """
    Detects and handles outliers using Isolation Forest.
    - Drops clearly invalid (extreme) outliers
    - Imputes slightly off values with median

    Parameters:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to process (default = all numerics)
        contamination (float): Proportion of expected outliers
        threshold_factor (float): Threshold to define "clearly invalid"

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = df.copy()

    # if columns is None:
    # columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_candidates = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Strip commas and other non-numeric characters, then try casting
            cleaned_sample = df[col].dropna().astype(str).str.replace(',', '', regex=False).head(10)
            try:
                sample = cleaned_sample.astype(float)
                if sample.count() == 10:
                    numeric_candidates.append(col)
            except:
                continue
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_candidates.append(col)
    for col in numeric_candidates:
        if df[col].dtype == 'object':
            # Remove commas and strip whitespace
            df[col] = df[col].dropna().astype(str).str.replace(',', '', regex=False).str.strip()
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass  # Skip if conversion still fails

    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df[numeric_candidates])
    df['outlier_flag'] = preds

    for col in numeric_candidates:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        mean = df[col].mean()
        std = df[col].std()
        upper_bound = mean + threshold_factor * std
        lower_bound = mean - threshold_factor * std

        is_outlier = df['outlier_flag'] == -1
        extreme_outliers = is_outlier & ((df[col] > upper_bound) | (df[col] < lower_bound))
        slight_outliers = is_outlier & ~extreme_outliers

        # Drop clearly invalid
        df = df[~extreme_outliers]

        # Impute mild outliers
        median_val = df.loc[~is_outlier, col].median()
        df.loc[slight_outliers, col] = median_val

    df = df.drop(columns='outlier_flag')
    return df.reset_index(drop=True)


def clean_irrelevant_rows(df, garbage_values=None):
    """
    Removes rows with all empty values or garbage/irrelevant markers like '###', '???', 'N/A', etc.

    Parameters:
        df (pd.DataFrame): Input DataFrame to clean.
        garbage_values (list): List of strings considered as irrelevant. Default includes common patterns.

    Returns:
        cleaned_df (pd.DataFrame): Cleaned DataFrame.
    """

    if garbage_values is None:
        garbage_values = ['###', '???', 'N/A', 'n/a', 'NULL', 'null', '', '-', ' ']

    # Replace garbage values with NaN
    # df_cleaned = df.replace(garbage_values, pd.NA)
    junk_pattern = r'^[^a-zA-Z0-9\s]+$'  # only special characters

    # Apply replacement across entire DataFrame
    df_cleaned = df.applymap(lambda x: pd.NA if isinstance(x, str) and pd.Series([x]).str.match(junk_pattern).iloc[0] else x)    # df_cleaned = df.dropna(how='all')

   

    return df_cleaned



DetectorFactory.seed = 42  # for consistent results

def clean_foreign_and_encoding_issues(df, text_columns=None, expected_lang='en'):
    """
    Detects and removes foreign-language rows and fixes encoding issues.

    Parameters:
        df (pd.DataFrame): Input dataframe
        text_columns (list): Columns to check for language and encoding. If None, all object (string) columns are used.
        expected_lang (str): ISO 639-1 language code (e.g. 'en' for English)

    Returns:
        pd.DataFrame: Cleaned dataframe
        dict: Report of rows removed or fixed
    """

    if text_columns is None:
        text_columns = df.select_dtypes(include=['object']).columns.tolist()

    cleaned_df = df.copy()
    report = {"foreign_lang_rows_removed": 0, "encoding_fixes_applied": 0}
    indices_to_drop = set()

    for col in text_columns:
        for idx, val in cleaned_df[col].items():
            if pd.isna(val) or not isinstance(val, str):
                continue

            # Fix encoding
            fixed_val = ftfy.fix_text(val)
            if fixed_val != val:
                cleaned_df.at[idx, col] = fixed_val
                report["encoding_fixes_applied"] += 1

            # Language detection
            try:
                lang = detect(fixed_val)
                if lang != expected_lang:
                    indices_to_drop.add(idx)
            except LangDetectException:
                indices_to_drop.add(idx)

    # Drop rows with out-of-scope language
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop(index=list(indices_to_drop))
    report["foreign_lang_rows_removed"] = initial_rows - len(cleaned_df)

    return cleaned_df.reset_index(drop=True), report


def main_dqm_pipeline(df, outlier_method='isolation_forest', expected_lang='en', garbage_values=None, contamination=0.01):
    """
    Full Data Quality Management (DQM) pipeline:
    - Cleans irrelevant rows
    - Fixes encoding and removes non-English text rows
    - Detects and handles outliers (drop/impute)
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        outlier_method (str): 'iqr', 'zscore', or 'isolation_forest'
        expected_lang (str): ISO code for expected language (default 'en')
        garbage_values (list): List of values considered garbage
        contamination (float): Used for Isolation Forest

    Returns:
        cleaned_df (pd.DataFrame): Cleaned DataFrame
        reports (dict): Reports from each cleaning step
    """
    reports = {}

    # Step 1: Clean irrelevant/garbage rows
    df_cleaned = clean_irrelevant_rows(df, garbage_values)
    reports['irrelevant_rows_cleaned'] = len(df) - len(df_cleaned)

    # Step 2: Fix encoding & remove foreign language rows
    # df_cleaned, lang_report = clean_foreign_and_encoding_issues(df_cleaned, expected_lang=expected_lang)
    # reports.update(lang_report)

    # Step 3: Handle outliers using smart logic (drop or impute)
    df_cleaned = detect_and_remove_outliers(df_cleaned)
    reports['outliers_handled'] = 'drop (extreme) + impute (slight)'
    # print(df_cleaned.dtypes)
    # print(df_cleaned.head())
    return df_cleaned

# df=pd.read_csv("C:/Users/mihir.sinha/Downloads/employee_data 3.csv")
# answer=main_dqm_pipeline(df)
# print(answer)