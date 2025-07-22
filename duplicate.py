
import pandas as pd
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import time
from word2number import w2n  # For converting word-based numbers to numeric

# --- 1. Data-driven detection functions ---

def is_date_column(series, sample_size=20, threshold=0.8):
    sample = series.dropna().astype(str).sample(min(sample_size, len(series)), random_state=1)
    success = 0
    for val in sample:
        try:
            pd.to_datetime(val)
            success += 1
        except:
            pass
    return (success / len(sample)) >= threshold if len(sample) > 0 else False


def detect_tag(series, col_name=None):
    # Check for numeric columns first
    if pd.api.types.is_numeric_dtype(series):
    # Handle numeric IDs based on uniqueness, length, and type
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        avg_len = series.dropna().astype(str).map(len).mean() if len(series.dropna()) > 0 else 0

        # If numeric column has high uniqueness and short length, treat as 'id'
        if unique_ratio > 0.8 and avg_len < 15:
            return 'id'
       # Check if the column is numeric or contains mostly numeric data
    numeric_values = pd.to_numeric(series, errors='coerce')  # Convert to numeric, non-numeric becomes NaN
    numeric_count = numeric_values.notna().sum()  # Count how many values are numeric
    if numeric_count > len(series) / 2:  # If more than half of the values are numeric
        return 'numeric'
    

      

    # Date detection
    if is_date_column(series):
        return 'date'

    # Check for multi-word columns
    non_null_series = series.dropna().astype(str)
    sample_size = min(100, len(non_null_series))
    sample = non_null_series.sample(sample_size, random_state=1) if sample_size > 0 else pd.Series(dtype=str)
    multi_word_ratio = sample.map(lambda x: ' ' in x).mean() if len(sample) > 0 else 0

    if multi_word_ratio > 0.5:
        return 'multi-word'

    return 'single-word'


# --- 2. Similarity functions by tag ---

def similarity_score(val1, val2, tag):
    if pd.isna(val1) and pd.isna(val2):
        return 100
    if pd.isna(val1) or pd.isna(val2):
        return 0

       # Convert word-based numbers to numeric if the values are not numeric
    if tag == 'numeric':
        try:
            # Try converting text-based numbers to actual numeric values
            val1 = w2n.word_to_num(str(val1)) if not isinstance(val1, (int, float)) else val1
            val2 = w2n.word_to_num(str(val2)) if not isinstance(val2, (int, float)) else val2
        except ValueError:
            # If conversion fails, return 0 similarity for non-numeric text values
            return 0
        
        # Now compare the numeric values
        return 100 if val1 == val2 else 0

    val1 = str(val1).strip().lower()
    val2 = str(val2).strip().lower()

    if tag == 'id':
        return 100 if val1 == val2 else JaroWinkler.similarity(val1, val2) * 100
    elif tag == 'single-word':
        return JaroWinkler.similarity(val1, val2) * 100
    elif tag == 'multi-word':
        return fuzz.token_sort_ratio(val1, val2)
    elif tag == 'date':
        try:
            d1 = pd.to_datetime(val1)
            d2 = pd.to_datetime(val2)
            return 100 if d1 == d2 else 0
        except:
            return 0
    else:
        return JaroWinkler.similarity(val1, val2) * 100

# --- 3. Weights for tags ---

tag_weights = {
    'id': 3,
    'numeric': 2,
    'single-word': 1.5,
    'multi-word': 1,
    'date': 2
}

# --- 4. Composite similarity between two rows ---

def composite_similarity(row1, row2, columns, tags, weights):
    total_score = 0
    total_weight = 0
    for col in columns:
        tag = tags[col]
        score = similarity_score(row1[col], row2[col], tag)
        weight = weights.get(tag, 1)
        total_score += score * weight
        total_weight += weight
    return total_score / total_weight if total_weight != 0 else 0

# --- 5. Blocking to reduce comparisons ---

def choose_blocking_keys(df, tags, top_n=1):
    # Calculate the uniqueness of each column
    uniqueness = {col: df[col].nunique() / len(df) for col in df.columns}
    
    # Separate columns tagged as 'id' and 'single-word'
    id_cols = [col for col in df.columns if tags.get(col) == 'id']
    single_word_cols = [col for col in df.columns if tags.get(col) == 'single-word']
    
    # First, prioritize 'id' columns
    candidate_cols = id_cols  # Start with ID columns first
    
    # Sort the columns based on uniqueness in descending order
    candidate_cols = sorted(candidate_cols, key=lambda c: uniqueness[c], reverse=True)
    
    # If there are any candidate columns, return the top N
    if candidate_cols:
        return candidate_cols[:top_n]
    
    # If no matching columns, return the top N columns from the DataFrame
    return df.columns[:top_n].tolist()


def block_data(df, blocking_cols):
    blocks = defaultdict(list)
    for idx, row in df.iterrows():
        key = '||'.join(str(row[col]).strip().lower() if pd.notna(row[col]) else '' for col in blocking_cols)
        blocks[key].append(idx)
    return blocks

# --- 6. Find duplicates using blocking ---



def find_duplicates_and_remove_with_count(df, threshold=70, max_block_size=1000):
    # print("********************************************")
    tags = {col: detect_tag(df[col]) for col in df.columns}
    columns = df.columns.tolist()
    weights = tag_weights
    blocking_cols = choose_blocking_keys(df, tags)
    # print(blocking_cols)

    blocks = block_data(df, blocking_cols)
    # print(blocks)
    visited = set()  # Set to track already visited (duplicate) rows
    rows_to_keep = []  # List to store indices of rows to keep
    duplicate_count = defaultdict(int)  # Dictionary to track duplicate counts

    # Iterate over the blocks and detect duplicates
    for block_key, indices in blocks.items():
        for i_idx in range(len(indices)):
            i = indices[i_idx]
            if i in visited:
                continue
            # Add the first occurrence (i) to the rows_to_keep list
            rows_to_keep.append(i)
            group = {i}
            for j_idx in range(i_idx + 1, len(indices)):
                j = indices[j_idx]
                if j in visited:
                    continue
                score = composite_similarity(df.loc[i], df.loc[j], columns, tags, weights)
                if score >= threshold:
                    # Mark the duplicate row as visited
                    visited.add(j)
                    duplicate_count[i] += 1  # Increase the count for the first occurrence

            # If group has more than one element, remove the duplicate indices
            if len(group) > 1:
                visited.update(group)

    # Return the DataFrame with duplicates removed (keeping only first occurrences)
    df_no_duplicates = df.iloc[rows_to_keep].reset_index(drop=True)

    # Add the duplicate count as a new column to the cleaned DataFrame
    # df_no_duplicates['Duplicate Count'] = df_no_duplicates['Index'].map(duplicate_count)
    # print(len(duplicate_count))
    return df_no_duplicates, len(duplicate_count)


# df=pd.read_csv("C:/Users/mihir.sinha/Downloads/employee_data 3.csv")
# answer=find_duplicates_and_remove_with_count(df)
# print(answer)