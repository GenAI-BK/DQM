import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import fuzz, process
from dateutil.parser import parse
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from sklearn.impute import SimpleImputer
from difflib import SequenceMatcher
import time 
import re
import requests
from sklearn.ensemble import IsolationForest
import streamlit as st
import json
# from langchain_community.agent_toolkits import create_pandas_dataframe_agent
# from langchain.agents import create_pandas_dataframe_agent
import re
import ast
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
from standardization import auto_standardize
from irrelavent import main_dqm_pipeline
import plotly.graph_objects as go
### DATA PROFILING

def check_missing_duplicates_outliers(df, contamination=0.01):
    """
    Check for missing values, duplicate rows, and number of outliers in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The dataset to check.
    - contamination (float): Proportion of outliers expected in the data.

    Returns:
    - result (dict): Summary of missing values, duplicates, and outliers.
    """
    result = {}

    # Missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()

    # Duplicate rows
    total_duplicates = df.duplicated().sum()

    # Outlier detection using Isolation Forest (only for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number])
    outlier_count = 0

    if not numeric_cols.empty:
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(numeric_cols)
        outlier_count = (preds == -1).sum()

    # Compile results
    result["missing_values_per_column"] = missing_values.to_dict()
    result["total_missing_values"] = int(total_missing)
    result["total_duplicate_rows"] = int(total_duplicates)
    result["total_outlier_rows"] = int(outlier_count)

    return result







###  handling missing Data
def detect_unique_columns(df, threshold_min=70):
    """Detect columns with high uniqueness (likely identifiers)."""
    total_rows = len(df)
    return [
        col for col in df.columns
        if df[col].nunique(dropna=True) / total_rows * 100 >= threshold_min
        and df[col].dtype == 'object'
    ]

def fill_missing_ids_dynamic(df, threshold=0.9, uniqueness_threshold=70):
    # df_filled = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # Step 1: Try to detect ID column
    id_candidates = [col for col in df.columns if 'id' in col and df[col].notna().sum() > 0]

    if not id_candidates:
        print("⚠️ No ID column explicitly found — fallback to unique columns.")
        id_candidates = detect_unique_columns(df, threshold_min=uniqueness_threshold)
        if not id_candidates:
            print("❌ No unique columns available to fill missing IDs.")
            return df
    id_col = id_candidates[0]

    # Step 2: Use other identity-like or high-uniqueness columns to compare similarity
    reference_cols = detect_unique_columns(df.drop(columns=[id_col]), threshold_min=uniqueness_threshold)
    if not reference_cols:
        print("⚠️ No suitable reference columns found to compute similarity.")
        return df

    def similarity(val1, val2):
        if pd.isna(val1) or pd.isna(val2):
            return 0.0
        return SequenceMatcher(None, str(val1).lower(), str(val2).lower()).ratio()

    # Step 3: Fill missing values in the detected ID column using similarity
    for idx, row in df[df[id_col].isna()].iterrows():
        for prev_idx, prev_row in df.loc[:idx-1].iterrows():
            if pd.notna(prev_row[id_col]):
                sim_scores = [similarity(row[col], prev_row[col]) for col in reference_cols if col in df.columns]
                if sim_scores:
                    avg_sim = np.mean(sim_scores)
                    if avg_sim >= threshold:
                        df.at[idx, id_col] = prev_row[id_col]
                        break

    return df


def fill_from_previous_records_autoid(df):
    """
    Automatically detects an ID column (like empid, id, etc.)
    and fills missing values in repeated records using previous data.

    Parameters:
    - df (pd.DataFrame): Input DataFrame

    Returns:
    - DataFrame with enriched records based on previous entries
    """

    # df_filled = df.copy()

    # Step 1: Try to detect the ID column
    possible_id_columns = [col for col in df.columns if 'id' in col.lower()]
    if not possible_id_columns:
        raise ValueError("No ID column found. Ensure there's a column with 'id' in the name.")

    id_column = possible_id_columns[0]  # Use the first matched ID column

    # Step 2: Initialize memory for each ID
    last_seen = {}

    # Step 3: Iterate over rows
    for i, row in df.iterrows():
        current_id = row[id_column]

        if pd.isna(current_id):
            continue  # skip rows without ID

        if current_id in last_seen:
            for col in df.columns:
                val = df.at[i, col]
                if pd.isna(val) or str(val).strip().lower() in ['', 'nan', 'none']:
                    df.at[i, col] = last_seen[current_id].get(col, val)
        else:
            last_seen[current_id] = {}

        # Update last_seen with current row’s non-missing values
        last_seen[current_id].update({
            col: row[col] for col in df.columns
            if not pd.isna(row[col]) and str(row[col]).strip().lower() not in ['', 'nan', 'none']
        })

    return df



def fill_missing_numeric_statistical(df, uniqueness_threshold=80):
    # df = df.copy()  # Work on a copy to avoid changing the original df directly
    
    df=df.reset_index()
    total_rows = len(df)
    fill_report = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        
        missing_count = df[col].isna().sum()
        if missing_count == 0:
            continue

        skewness = df[col].dropna().skew()
        num_unique = df[col].nunique(dropna=True)

        if num_unique <= 5:
            
            fill_value = df[col].mode().iloc[0]
            method = "mode"
        elif abs(skewness) < 0.5:
            fill_value = df[col].mean()
            method = "mean"
        else:
            fill_value = df[col].median()
            method = "median"
        
        
        # Correctly fill and update the DataFrame
        df[col] = df[col].fillna(fill_value)
        fill_report[col] = {
            "method": method,
            "value_used": fill_value,
            "missing_filled": missing_count
        }

    # Return only the filled DataFrame
    return df


# --- TITLE INFERENCE ---

def dynamic_title_handler(df):
    # df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    first_name_col = next((col for col in ['firstname', 'first_name', 'name'] if col in df.columns), None)
    last_name_col = next((col for col in ['lastname', 'last_name', 'surname'] if col in df.columns), None)
    df['__title'] = np.nan

    def infer_title_rule(row):
        gender = str(row.get('gender', '') or row.get('gendercode', '')).lower()
        marital = str(row.get('maritaldesc', '')).lower()
        profession = str(row.get('profession', '') or row.get('jobfunctiondescription', '')).lower()
        if 'doctor' in profession or 'dr' in profession: return 'Dr.'
        elif 'prof' in profession: return 'Prof.'
        elif gender == 'male': return 'Mr.'
        elif gender == 'female': return 'Mrs.' if 'married' in marital else 'Ms.'
        return None

    feature_cols = [col for col in ['gender', 'gendercode', 'maritaldesc', 'profession', 'jobfunctiondescription'] if col in df.columns]
    if first_name_col and len(feature_cols) >= 2:
        try:
            df['__temp_title'] = df[first_name_col].apply(lambda x: x.split()[0] if isinstance(x, str) and any(x.lower().startswith(t) for t in ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'mx.', 'miss']) else np.nan)
            train_df = df[df['__temp_title'].notna()]
            X = train_df[feature_cols].fillna('Unknown').astype(str)
            y = train_df['__temp_title']
            le = LabelEncoder(); y_encoded = le.fit_transform(y)
            X_encoded = X.apply(LabelEncoder().fit_transform)
            model = DecisionTreeClassifier(max_depth=4, random_state=42)
            model.fit(X_encoded, y_encoded)
            predict_df = df[df['__temp_title'].isna()]
            X_new = predict_df[feature_cols].fillna('Unknown').astype(str)
            X_new_encoded = X_new.apply(LabelEncoder().fit_transform)
            df.loc[df['__temp_title'].isna(), '__title'] = le.inverse_transform(model.predict(X_new_encoded))
            df.loc[df['__temp_title'].notna(), '__title'] = df['__temp_title']
        except:
            df['__title'] = df.apply(infer_title_rule, axis=1)
    else:
        df['__title'] = df.apply(infer_title_rule, axis=1)

    if first_name_col:
        df[first_name_col] = df['__title'].fillna('') + ' ' + df[first_name_col].fillna('')
        df[first_name_col] = df[first_name_col].str.strip()
    df.drop(columns=['__title', '__temp_title'], inplace=True, errors='ignore')
    return df


def hybrid_text_imputer(df, model_name='all-MiniLM-L6-v2', threshold=0.8):
    model = SentenceTransformer(model_name)
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    object_cols = df.select_dtypes(include='object').columns

    for col in object_cols:
        missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == '')
        if not missing_mask.any():
            continue

        context_cols = [c for c in object_cols if c != col]

        # Build context sentences
        def build_context(row):
            return ' | '.join([f"{c}: {str(row[c])}" for c in context_cols if pd.notna(row[c]) and str(row[c]).strip() != ''])

        context_data = df.apply(build_context, axis=1)

        # Embed all rows
        embeddings = model.encode(context_data.tolist(), convert_to_tensor=True)

        for idx in df[missing_mask].index:
            row_emb = embeddings[idx]

            non_missing_idx = df[~missing_mask].index
            sims = util.pytorch_cos_sim(row_emb, embeddings[non_missing_idx])[0]
            best_match_idx = sims.argmax().item()
            real_index = non_missing_idx[best_match_idx]
            df.at[idx, col] = df.at[real_index, col]

    return df


def handle_missing_data(df):
    # st.info("🔍 Handling Missing Data...")

    # Step 1: Fill missing ID based on similarity
    df = fill_missing_ids_dynamic(df)

    # Step 2: Auto-fill previous repeated records based on ID
    df = fill_from_previous_records_autoid(df)

    # Step 3: Fill numeric columns smartly (mean, median, mode)
    df = fill_missing_numeric_statistical(df)

    # Step 4: Infer titles from profession, gender, etc.
    df = dynamic_title_handler(df)

    # Step 5: Use sentence-transformer-based imputer for object/text columns
    df = hybrid_text_imputer(df)

    st.success("✅ Missing data handled successfully.")
    return df


### duplicate  data
from duplicate import find_duplicates_and_remove_with_count

##irrlavant data


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

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Apply Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df[columns])
    df['outlier_flag'] = preds

    extreme_outlier_indices = set()

    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        mean = df[col].mean()
        std = df[col].std()
        upper_bound = mean + threshold_factor * std
        lower_bound = mean - threshold_factor * std

        is_outlier = df['outlier_flag'] == -1
        extreme = is_outlier & ((df[col] > upper_bound) | (df[col] < lower_bound))
        slight = is_outlier & ~extreme

        # Track extreme outliers to drop
        extreme_outlier_indices.update(df[extreme].index)

        # Impute slight outliers with median of non-outliers
        median_val = df.loc[~is_outlier, col].median()
        df.loc[slight, col] = median_val

    # Drop extreme outliers (after loop)
    df = df.drop(index=extreme_outlier_indices)
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
    df= df.replace(garbage_values, pd.NA)

    # Drop rows where all columns are NaN
    df = df.dropna(how='all')

    # Optionally drop rows where any column is NaN (if you want stricter cleaning)
    # df_cleaned = df_cleaned.dropna(how='any')

    return df






# Load the master file containing regex patterns and formats
def load_master_file() -> dict:
    with open('master_file.json', 'r') as f:
        return json.load(f)

master = load_master_file()
# -------------------------
# Main Streamlit App
# -------------------------


# from missingvalue import apply_custom_css





# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'  # Replace with your actual key or use st.secrets




#
import base64
image_path="bk-logo.png"
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
   
icon_base64 = get_base64_image(image_path)  
icon_url = f"data:image/png;base64,{icon_base64}"

st.markdown(f"""
    <div class="title-box">
        <img src="{icon_url}" alt="icon">
        <div class="title-text"></div>  <!-- Optional title text -->
    </div>
""", unsafe_allow_html=True)

# ========== Gray UI Styling ==========

st.markdown("""
<style>
           .stMainBlockContainer {
            padding-top:9rem ;
            }
            .title-box {
    width: 300px;
    margin: 0 auto;
    position: absolute;
    top: -4rem;
    left: 0;
    right: 0;
}
           
/* Gray background for sidebar */
section[data-testid="stSidebar"] > div:first-child {
  background-color: #477293;
  padding: 20px;
  box-shadow: 0 0 8px rgba(0, 0, 0, 0.1);
}
.stSidebar button.st-emotion-cache-qm7g72.eacrzsi2 {
  background: transparent!important;
  width: 100%;
  justify-content: left !important;
  padding: 0;
  padding-left: 15px;
  min-height: auto;
  position: relative;
            color:#fff !important;
}
            .stSidebar button.st-emotion-cache-qm7g72.eacrzsi2:hover{
            color:#ccc !important;
            }
.stSidebar button.st-emotion-cache-qm7g72.eacrzsi2:after {
  position: absolute;
  content: "";
  width: 0;
  left: 0;
  height: 0;
  border-top: 6px solid transparent;
  border-bottom: 6px solid transparent;
  border-left: 5px solid #fff;
}
  .stSidebar button.st-emotion-cache-qm7g72.eacrzsi2:hover:after{
            border-left: 5px solid #ccc;
            }        
           
            .stFileUploader p {
    font-size: 26px;
    margin-bottom: 5px;color:#477293;
}
         
/* Import Alasasy font */
@import url("https://fonts.googleapis.com/css2?family=Alasasy&display=swap");
 
/* Apply font to the profiling title */
.profiling-title {
  font-family: "Alasasy", sans-serif;
  font-size: 32px;
  font-weight: 600;
  color: #333;
  padding-top: 10px;
}
 
/* Gray main app background */
[data-testid="stAppViewContainer"] {
  background-color: #fff;
}
 
/* Sidebar buttons uniform style */
button[kind="secondary"] {
  background-color: #cccccc !important;
  color: black !important;
  border: none !important;
}
.st-emotion-cache-102y9h7 h2 {
  font-size: 32px;
  margin-top: 0;
  padding-top: 0;
            color: #fff;
    font-weight: 600;
}
/* Table background and header styling */
thead tr th {
  background-color: #dcdcdc !important;
  color: black !important;
}
tbody tr td {
  background-color: #f0f0f0 !important;
  color: black !important;
}
table {
  background-color: #e6e6e6 !important;
}
 
</style>
""", unsafe_allow_html=True)

# ========== Init Session State ==========
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = None

def update_clean_data(result, step_name=""):
    if isinstance(result, pd.DataFrame):
        st.session_state.clean_data = result
    elif isinstance(result, tuple) and isinstance(result[0], pd.DataFrame):
        st.session_state.clean_data = result[0]
        # st.warning(f"⚠️ '{step_name}' returned a tuple. Using only the DataFrame.")
    else:
        st.error(f"❌ '{step_name}' did not return a valid DataFrame.")

# ========== Data Profiling ==========
def check_missing_duplicates_outliers(df, contamination=0.01):
    result = {}
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    total_duplicates = df.duplicated().sum()

    # numeric_cols = df.select_dtypes(include=[np.number])
    total_outliers = 0
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
    matches = re.findall(pattern, response, re.DOTALL)

    for match in matches:
       
        parsed = ast.literal_eval(match.strip())
    true_numeric=set(parsed)
    for col in true_numeric:
        df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.strip()
                .astype('Float64')
            )
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR


        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outliers.sum()

        # outlier_summary[col] = int(outlier_count)
        total_outliers += outlier_count

    result["total_missing_values"] = int(total_missing)
    result["total_duplicate_rows"] = int(total_duplicates)
    result["total_outlier_rows"] = int(total_outliers)
    return result

def plot_clean_anomaly_chart(profile, total_rows, title):
    missing = profile["total_missing_values"]
    duplicates = profile["total_duplicate_rows"]
    outliers = profile["total_outlier_rows"]
    anomalies = missing + duplicates + outliers
    normal = total_rows - anomalies
    labels = ["Normal", "Missing", "Duplicates", "Outliers"]
    values = [normal, missing, duplicates, outliers]
    colors = ["#8fd9a8", "#f7c59f", "#f78da7", "#c1c8e4"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.45,
        textinfo='none'
    )])

    fig.update_layout(
        title=title,
        legend_title="Data Breakdown",
        legend=dict(orientation="v", x=1, y=0.5),
        margin=dict(t=40, b=20, l=20, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)
def clean_data(df):
    result = main_dqm_pipeline(df)
    update_clean_data(result, "Remove Junk Values")
    df = st.session_state.clean_data
    # st.dataframe(df.head())
    
    
    result = handle_missing_data(df)
    update_clean_data(result, "Handle Missing Data")
    df = st.session_state.clean_data
    
    
    df.drop('index', axis=1, inplace=True)
    
    result = find_duplicates_and_remove_with_count(df)
    update_clean_data(result, "Remove Duplicates")
    df = st.session_state.clean_data

    result = auto_standardize(df, master=master)
    update_clean_data(result, "Standardize Data")
    df = st.session_state.clean_data
    
    return df
    
    
# ========== Upload ==========
# st.title("🧼 Data Quality Management System")
# apply_custom_css()
uploaded_file = st.file_uploader("📁 Upload your CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.raw_data = pd.read_csv(uploaded_file)
        else:
            st.session_state.raw_data = pd.read_excel(uploaded_file)
        st.session_state.clean_data = st.session_state.raw_data.copy()
        st.success("✅ File uploaded successfully.")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

# ========== Sidebar Buttons ==========
with st.sidebar:
    st.header("Cleaning Steps")
    step_selected = None

    if st.button("Data Profiling"):
        step_selected = "profile"
    if st.button("Handle Missing Data"):
        step_selected = "missing"
    if st.button("Remove Duplicates"):
        step_selected = "duplicates"
    if st.button("Standardize Data"):
        step_selected = "standardize"
    if st.button("Remove Junk Values"):
        step_selected = "junk"
    if st.button("Data Visualization"):
        step_selected = "data_visualization"
    if st.button("Download Full Cleaned Data"):
        step_selected = "download_all"

# ========== Main Logic ==========
if st.session_state.clean_data is not None:
    df = st.session_state.clean_data.copy()
    
    df.reset_index(drop=True, inplace=True)
    df_old=df.copy()
    if step_selected == "profile":
        st.subheader("Data Profiling Report")
        

    elif step_selected == "missing":
        result = handle_missing_data(df)
        update_clean_data(result, "Handle Missing Data")
        st.download_button("Download Missing Cleaned",
                           st.session_state.clean_data.to_csv(index=False).encode('utf-8'),
                           "missing_cleaned.csv", "text/csv", key="missing_download")

    elif step_selected == "duplicates":

        result = find_duplicates_and_remove_with_count(df)
        update_clean_data(result, "Remove Duplicates")
        remaining_duplicates = st.session_state.clean_data.duplicated().sum()
        st.success("✅ Duplicates removed.")
        st.info(f"🔍 Remaining duplicate rows: **{remaining_duplicates}**")
        # st.dataframe(st.session_state.clean_data.head())
        st.download_button("Download Duplicate Cleaned",
                           st.session_state.clean_data.to_csv(index=False).encode('utf-8'),
                           "duplicates_cleaned.csv", "text/csv", key="duplicate_download")

    elif step_selected == "standardize":
         
        result = auto_standardize(df, master=master)
        update_clean_data(result, "Standardize Data")
        st.success("✅ Data standardized.")
        # st.dataframe(st.session_state.clean_data.head())
        st.download_button("Download Standardized Data",
                           st.session_state.clean_data.to_csv(index=False).encode('utf-8'),
                           "standardized_data.csv", "text/csv", key="standardize_download")

    elif step_selected == "junk":
        result = main_dqm_pipeline(df)
        update_clean_data(result, "Remove Junk Values")
        # remaining_rows = len(st.session_state.clean_data)
        st.success("✅ Junk values removed.")
        # st.info(f"Remaining rows: **{remaining_rows}**")
        # st.dataframe(st.session_state.clean_data.head())
        st.download_button("📥 Download Junk Cleaned",
                           st.session_state.clean_data.to_csv(index=False).encode('utf-8'),
                           "junk_cleaned.csv", "text/csv", key="junk_download")


         
    elif step_selected == "download_all":
        # if st.button("🚀 Run Full Cleaning Pipeline"):
            df = st.session_state.raw_data.copy()
            result = main_dqm_pipeline(df)
            update_clean_data(result, "Remove Junk Values")
            df = st.session_state.clean_data
           
            
            result = handle_missing_data(df)
            update_clean_data(result, "Handle Missing Data")
            df = st.session_state.clean_data
          
            
            df.drop('index', axis=1, inplace=True)
           
            result = find_duplicates_and_remove_with_count(df)
            update_clean_data(result, "Remove Duplicates")
            df = st.session_state.clean_data

            result = auto_standardize(df, master=master)
            update_clean_data(result, "Standardize Data")
            df = st.session_state.clean_data
            

           

            st.success("✅ Full pipeline complete!")
            st.dataframe(df.head())

            final_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Final Cleaned Data", final_csv,
                               "final_cleaned_data.csv", "text/csv", key="final_download")
            
    elif step_selected == "data_visualization":
        profile_original = check_missing_duplicates_outliers(df_old)

        # Clean data
        df_cleaned = clean_data(df_old)
        profile_cleaned = check_missing_duplicates_outliers(df_cleaned)

        # Side-by-side charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔍 Before Cleaning")
            plot_clean_anomaly_chart(profile_original, len(df_old), "Original Data")

        with col2:
            st.markdown("### ✨ After Cleaning")
            plot_clean_anomaly_chart(profile_cleaned, len(df_cleaned), "Cleaned Data")
        total_rows = len(df_old)

        summary_df = pd.DataFrame([
        {
            "Content": "Total Rows",
            "Count": total_rows,
            "Percentage": "100%"
        },
        {
            "Content": "Missing Values",
            "Count": profile_original["total_missing_values"],
            "Percentage": f'{round((profile_original["total_missing_values"] / total_rows) * 100, 2)}%'
        },
        {
            "Content": "Duplicate Rows",
            "Count": profile_original["total_duplicate_rows"],
            "Percentage": f'{round((profile_original["total_duplicate_rows"] / total_rows) * 100, 2)}%'
        },
        {
            "Content": "Outlier Rows",
            "Count": profile_original["total_outlier_rows"],
            "Percentage": f'{round((profile_original["total_outlier_rows"] / total_rows) * 100, 2)}%'
        }
    ])

        st.markdown("### 📊 Profiling Summary (Original Data)")
        st.table(summary_df)