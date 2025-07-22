# data_cleaning_utils.py

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

# --- MISSING VALUE HANDLING ---

def missing_value_summary(df):
    total_rows = len(df)
    summary = pd.DataFrame({
        'Missing Count': df.isna().sum(),
        'Missing %': (df.isna().sum() / total_rows) * 100
    })
    summary = summary[summary['Missing Count'] > 0]
    return summary.sort_values(by='Missing Count', ascending=False)

def drop_high_missing_columns(df, threshold=90):
    total_rows = len(df)
    missing_percent = (df.isna().sum() / total_rows) * 100
    cols_to_drop = missing_percent[missing_percent >= threshold].index.tolist()
    df_cleaned = df.drop(columns=cols_to_drop)
    return df_cleaned, cols_to_drop

# --- DATA TYPE DETECTION/CORRECTION ---

def identify_column_data_types(df):
    return pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values
    })




def correct_column_data_types(df):
    # df = df.copy()
    # print(type(df))

    for col in df.columns:
        col_lower = col.lower()
        if 'date' in col_lower or 'dob' in col_lower:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except: pass
        elif df[col].dtype == 'object':
            try:
                if df[col].str.replace('.', '', 1).str.isnumeric().all():
                    if df[col].str.contains('.').any():
                        df[col] = df[col].astype(float)
                    else:
                        df[col] = df[col].astype(int)
            except: pass
    return df

# --- UNIQUENESS DETECTION ---

def identify_unique_columns(df, threshold_min=70, threshold_max=100):
    total_rows = len(df)
    result = []
    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        percent_unique = round((unique_count / total_rows) * 100, 2)
        result.append({
            'Column': col,
            'Unique Values': unique_count,
            'Uniqueness (%)': percent_unique,
            'Likely Unique Column': threshold_min <= percent_unique <= threshold_max
        })
    return pd.DataFrame(result)



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



def detect_id_column(df):
    """Auto-detect a likely ID column if one exists."""
    possible_id_cols = [col for col in df.columns if 'id' in col.lower()]
    if possible_id_cols:
        return possible_id_cols[0]
    return None


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


# --- FUZZY LOGIC CLEANING ---

def correct_fuzzy_column(df, threshold=90):
    # df_corrected = df.copy()
    for col in df.select_dtypes(include='object').columns:
        unique_values = df[col].dropna().unique().tolist()
        correction_dict = {}
        for val in unique_values:
            if val in correction_dict:
                continue
            match, score = process.extractOne(val, unique_values, scorer=fuzz.token_sort_ratio)
            if score >= threshold and val != match:
                correction_dict[val] = match
        df[col] = df[col].replace(correction_dict)
    return df

# --- CLEAN & MASK UNIQUE COLUMNS ---

def clean_and_fill_unique_columns(df, unique_threshold=0.7):
    # df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    possible_ids = [col for col in df.columns if 'id' in col]
    id_column = possible_ids[0] if possible_ids else None
    unique_cols = [col for col in df.columns
                   if df[col].nunique(dropna=True) / len(df) >= unique_threshold
                   and col != id_column]
    if not unique_cols:
        raise ValueError("No unique columns found to operate on.")

    def clean_email(email):
        if pd.isna(email) or not isinstance(email, str): return np.nan
        if '@' not in email: return np.nan
        name, domain = email.split('@', 1)
        return f"{name}@{domain.strip()}" if '.' in domain else f"{name}@gmail.com"

    def clean_phone(phone):
        digits = re.sub(r'\D', '', str(phone))
        return digits if len(digits) == 10 else digits + 'x' * (10 - len(digits)) if 5 <= len(digits) < 10 else 'x' * 10

    def mask_id_like(value, total_len=12):
        digits = re.sub(r'\D', '', str(value))
        return digits if len(digits) == total_len else digits + 'x' * (total_len - len(digits)) if 5 <= len(digits) < total_len else 'x' * total_len

    last_seen_map = {}
    for i, row in df.iterrows():
        identity = row[id_column] if id_column else tuple(row[col] for col in unique_cols if pd.notna(row[col]))
        if not identity: continue
        if identity in last_seen_map:
            for col in unique_cols:
                if pd.isna(row[col]) or str(row[col]).strip().lower() in ['', 'nan', 'none']:
                    df.at[i, col] = last_seen_map[identity].get(col)
        for col in unique_cols:
            val = df.at[i, col]
            if 'email' in col:
                df.at[i, col] = clean_email(val)
            elif 'phone' in col or 'mobile' in col:
                df.at[i, col] = clean_phone(val)
            elif 'adhar' in col or 'pan' in col or 'ssn' in col or 'number' in col:
                df.at[i, col] = mask_id_like(val)
        if identity not in last_seen_map:
            last_seen_map[identity] = {}
        for col in unique_cols:
            if pd.notna(df.at[i, col]) and str(df.at[i, col]).strip().lower() not in ['', 'nan', 'none']:
                last_seen_map[identity][col] = df.at[i, col]
    return df

# ---  Name TITLE INFERENCE ---

def dynamic_title_handler(df):
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

# --- DATE HANDLING ---

def infer_partial_date(val):
    try:
        return parse(str(val), default=datetime(2000, 1, 1), fuzzy=True)
    except:
        return np.nan

def detect_date_columns(df):
    date_cols = []
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20)
        parsed = sample.apply(lambda x: infer_partial_date(x))
        if parsed.notna().sum() / len(sample) > 0.7:
            date_cols.append(col)
    return date_cols

def handle_dates_dqm(df):
    report = {}; df = df.copy()
    date_cols = detect_date_columns(df)
    for col in date_cols:
        original_col = df[col].copy()
        df[col] = df[col].apply(lambda x: infer_partial_date(x))
        df[col] = pd.to_datetime(df[col], errors='coerce')
        non_null = df[col].dropna(); imputed_with = 'not_imputed'
        mode_val = non_null.mode()
        if mode_val.size > 0 and (non_null == mode_val[0]).sum() / len(non_null) > 0.5:
            df[col] = df[col].fillna(mode_val[0]); imputed_with = 'mode'
        elif non_null.is_monotonic_increasing or non_null.is_monotonic_decreasing:
            df[col] = df[col].fillna(method='ffill'); imputed_with = 'forward_fill'
        elif len(non_null) >= 3:
            df[col] = df[col].fillna(non_null.mean()); imputed_with = 'mean'
        df[col] = df[col].dt.strftime("%d/%m/%y")
        report[col] = {
            'original_missing': original_col.isna().sum(),
            'final_missing': df[col].isna().sum(),
            'impute_strategy': imputed_with,
            'date_format': "%d/%m/%y"
        }
    return df

#Handling Texual data

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

# --- Data.gov.in API details ---
API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
RESOURCE_ID = "6176ee09-3d56-4a3b-8115-21841576b2f6"

# --- Helper: Detect PIN/ZIP column ---
def detect_pincode_column(df):
    pin_keywords = ['pin', 'zipcode', 'zip code', 'zip']
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(keyword in col_lower for keyword in pin_keywords):
            return col
    return 'pincode'  # Default fallback

# --- Fetch address details from Pincode ---
def fetch_address_from_pincode(pincode):
    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?api-key={API_KEY}&format=json&filters[pincode]={pincode}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            records = response.json().get('records', [])
            if records:
                return {
                    'district': records[0].get('districtname'),
                    'state': records[0].get('statename')
                }
    except Exception as e:
        print(f"Error fetching address for PIN {pincode}: {e}")
    return {'district': None, 'state': None}

# --- Fetch pincode from district/state ---
def fetch_pincode_from_address(district=None, state=None):
    query = []
    if district:
        query.append(f"filters[districtname]={district}")
    if state:
        query.append(f"filters[statename]={state}")
    if not query:
        return None
    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?api-key={API_KEY}&format=json&" + "&".join(query)
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json().get('records', [])
            if data:
                return data[0].get('pincode')
    except Exception as e:
        print(f"Error fetching PIN from address: {e}")
    return None

# --- Main function for  addresss data filling--
def fill_missing_full_address_india(df):
    # df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # Detect or create columns
    pin_col = detect_pincode_column(df)
    if pin_col not in df.columns:
        df[pin_col] = None
    if 'district' not in df.columns:
        df['district'] = None
    if 'state' not in df.columns:
        df['state'] = None
    if 'address' not in df.columns:
        df['address'] = None
    if 'country' not in df.columns:
        df['country'] = 'India'

    for idx, row in df.iterrows():
        pin = str(row[pin_col]).strip() if pd.notna(row[pin_col]) else ''
        district = str(row['district']).strip() if pd.notna(row['district']) else ''
        state = str(row['state']).strip() if pd.notna(row['state']) else ''
        address = str(row['address']).strip() if pd.notna(row['address']) else ''

        # --- Case 1: Pincode available, fill missing district/state ---
        if pin.isdigit() and len(pin) == 6:
            if not district or not state:
                fetched = fetch_address_from_pincode(pin)
                if not district:
                    df.at[idx, 'district'] = fetched['district']
                if not state:
                    df.at[idx, 'state'] = fetched['state']

        # --- Case 2: Pincode missing, use district/state to fill ---
        elif (district or state) and (not pin or pin == 'nan'):
            new_pin = fetch_pincode_from_address(district, state)
            if new_pin:
                df.at[idx, pin_col] = new_pin

        # --- Case 3: Fill address column if empty ---
        if not address:
            components = [row.get('district'), row.get('state'), df.at[idx, pin_col], 'India']
            clean_components = [str(c).strip() for c in components if c and str(c).strip() != '' and str(c).strip().lower() != 'nan']
            full_address = ', '.join(clean_components)
            if full_address:
                df.at[idx, 'address'] = full_address

    return df

def fill_location_from_zip(df, country_code='us', delay=1.0):
    """
    Dynamically fills missing city or state using ZIP code via Zippopotam.us API.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - country_code (str): Country code (e.g., 'us', 'ca', 'de')
    - delay (float): Seconds to wait between API calls (Zippopotam has rate limits)

    Returns:
    - pd.DataFrame with filled city/state
    """

    # df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # Detect relevant columns
    zip_col = next((col for col in df.columns if 'zip' in col or 'postal' in col), None)
    city_col = next((col for col in df.columns if re.search(r'\bcity\b', col)), None)
    state_col = next((col for col in df.columns if re.search(r'\bstate\b', col)), None)

    if not zip_col:
        raise ValueError("No ZIP/postal code column found.")

    def fetch_city_state(zip_code):
        try:
            url = f"http://api.zippopotam.us/{country_code}/{zip_code}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                places = data.get('places', [{}])[0]
                return places.get('place name'), places.get('state')
        except Exception as e:
            return None, None
        return None, None

    # Iterate over missing city/state rows
    for idx, row in df.iterrows():
        if (city_col and (pd.isna(row[city_col]) or row[city_col] == '')) or \
           (state_col and (pd.isna(row[state_col]) or row[state_col] == '')):
            zip_code = str(row[zip_col]).strip()
            if zip_code and zip_code.isdigit():
                city, state = fetch_city_state(zip_code)
                if city_col and (pd.isna(row[city_col]) or row[city_col] == ''):
                    df.at[idx, city_col] = city
                if state_col and (pd.isna(row[state_col]) or row[state_col] == ''):
                    df.at[idx, state_col] = state
                time.sleep(delay)  # Respect API rate limit

    return df


