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
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
from irrelavent import main_dqm_pipeline
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
    print(len(df.columns))
    total_rows = len(df)
    fill_report = {}
    print(df.dtypes)
    for col in df.select_dtypes(include=[np.number]).columns:
        print(col)
        
        missing_count = df[col].isna().sum()
        print(missing_count)
        if missing_count == 0:
            continue

        # unique_percent = (df[col].nunique(dropna=True) / total_rows) * 100
        # if unique_percent >= uniqueness_threshold:
        #     continue

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
        print(df[col])
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

    # st.success("✅ Missing data handled successfully.")
    return df
# df=pd.read_csv("C:/Users/mihir.sinha/Downloads/junk_cleaned (1).csv")
# answer=handle_missing_data(df.head(10))
# print(answer)