import pandas as pd 
import numpy as np 



def process_categorical(df, cols): 
    
    for col in cols: 
        col_dummies = pd.get_dummies(df[col], drop_first = True, dtype = int) 
        df = pd.concat([df, col_dummies], axis = 1)
        df = df.drop(columns = [col], inplace = False) 

    return df


def data_cleaning_pipeline(df, cols=None, types=None, cat_cols=None):

    if cols is not None:
        df = df.loc[:, cols]

    size_before = df.shape[0]
    df = df.drop_duplicates(keep='first')
    size_after = df.shape[0]
    print(f'Duplicates Removed: {size_before - size_after}')

    excluded_chunks = [] 

    df['ChoiceValue'] = pd.to_numeric(df['ChoiceValue'], errors='coerce')

    filter_questionkey = df['QuestionKey'].isin(['0-3', 'Y/N'])
    filter_choicevalue = df['ChoiceValue'].notna()
    valid_mask = filter_questionkey & filter_choicevalue

    excluded_chunks.append(df.loc[~valid_mask].copy())  # FIX
    df = df.loc[valid_mask]

    group_cols = [
        'OptionsNumber',
        'DateCompleted',
        'CANSCategoryName',
        'ItemTitle',
        'QuestionKey',
    ]

    df['max_ChoiceValue'] = df.groupby(group_cols)['ChoiceValue'].transform('max')
    is_max = df['ChoiceValue'] == df['max_ChoiceValue']

    df_dropped = df.loc[~is_max].drop(columns='max_ChoiceValue')
    df = df.loc[is_max].drop(columns='max_ChoiceValue')

    # excluded = pd.concat([excluded, df_dropped], ignore_index=True)  # REMOVED: same issue
    excluded_chunks.append(df_dropped.copy())  # FIX

    # FIX: single concat at the end — no empty DataFrame involved
    excluded = pd.concat(excluded_chunks, ignore_index=True) if excluded_chunks else pd.DataFrame(columns=df.columns)

    print(
        f'Excluded {excluded.shape[0]} rows: invalid QuestionKey, '
        f'missing ChoiceValue, or non-max ChoiceValue within group'
    )

    if types:
        for col, dtype in types.items():
            if col not in df.columns:
                continue
            if dtype.startswith('datetime'):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                df[col] = df[col].astype(dtype)

    return df, excluded


def category_impute(df, item_to_cat_dict):
    """
    Fills NaNs using the median of other items in the same CANS category 
    for the SAME assessment (row-wise).
    """
    df_imputed = df.copy()
    
    cat_to_items = {}
    for item, cat in item_to_cat_dict.items():
        if item in df.columns:
            cat_to_items.setdefault(cat, []).append(item)

    for cat, items in cat_to_items.items():
        if len(items) > 1:

            row_peer_median = df_imputed[items].median(axis=1)
            
            for item in items:
                df_imputed[item] = df_imputed[item].fillna(row_peer_median)
                
    return df_imputed

def impute_historical_median(df):
    """
    Fills remaining NaNs using the expanding median of that specific 
    OptionNumber's history. Uses .shift() to prevent data leakage.
    """
    df_sorted = df.sort_values(['OptionsNumber', 'DateCompleted'])
    cols_to_fill = df_sorted.columns.difference(['OptionsNumber', 'DateCompleted'])
    
    for col in cols_to_fill:

        past_series = (
            df_sorted.groupby('OptionsNumber')[col]
            .expanding()
            .median()
            .shift()
        )
    
        past_series = past_series.reset_index(level=0, drop=True)
        
        df_sorted[col] = df_sorted[col].fillna(past_series)
    
    return df_sorted

def row_median_impute(df, cans_cols):
    """
    Final row-wise fallback: fills remaining NaNs using the median of 
    ALL available CANS scores for that specific assessment row.
    """
    df_imputed = df.copy()
    global_row_median = df_imputed[cans_cols].median(axis=1)
    
    for col in cans_cols:
        df_imputed[col] = df_imputed[col].fillna(global_row_median)
        
    return df_imputed

def impute_missing_values(df, item_title_mapping, cans_cols): 
   
    df = impute_historical_median(df)
    df = category_impute(df, item_title_mapping)
    df = row_median_impute(df, cans_cols)

 
    return df

def drop_degenerate_row(df, cans_cols, min_valid=1): 
    """
    Excludes rows that have fewer than 'min_valid' non-null values 
    in the specified CANS item columns.
    """
    mask = df[cans_cols].count(axis=1) >= min_valid


    df_clean = df[mask].copy()
    excluded_rows = df[~mask].copy()
    
    return df_clean, excluded_rows



def consolidate_ethnicity(ethnicity_series):
    ethnicity_map = {
        'white': 'White',
        'white - central american': 'White',
        'white - middle eastern': 'White',
        'white - armenian': 'White',
        'white - european': 'White',
        'spanish': 'White',
        'hispanic/latino': 'Latino',
        'mexican': 'Latino',
        'central american': 'Latino',
        'black': 'Black',
        'filipino': 'Asian',
        'japanese': 'Asian',
        'chinese': 'Asian',
        'asian indian': 'Asian',
        'other asian': 'Asian',
        'no preference': 'No_Preference',
        'american indian': 'Other',
        'other pacific islander': 'Other',
        '[not entered]': 'Other'
    }
    
    consolidated = ethnicity_series.astype(str).str.strip().str.lower().map(ethnicity_map)
    return consolidated.fillna('Other')

def model_ready_df(df, item_title_mapping, top_n_lowest_nulls=45):
    
    pivot = df.pivot_table(
            index=['OptionsNumber', 'DateCompleted'], 
            columns='ItemTitle', 
            values='ChoiceValue'
    ).reset_index()

    lowest_null_n_cans = pivot.isna().sum().sort_values().head(top_n_lowest_nulls).index.tolist()
    print(f"1. Initial top_n count: {len(lowest_null_n_cans)}")

    lowest_null_n_cans = [c for c in lowest_null_n_cans if c not in ['OptionsNumber', 'DateCompleted']]
    print(f"2. Count after removing Index columns: {len(lowest_null_n_cans)}")

    pivot = pivot[['OptionsNumber', 'DateCompleted'] + lowest_null_n_cans]
    pivot, degen_row = drop_degenerate_row(pivot, lowest_null_n_cans)
    print(f'3. Degenerate rows excluded: {degen_row.shape[0]}')
        
    pivot = impute_missing_values(pivot, item_title_mapping, lowest_null_n_cans)
    df = pivot
    
    variances = df[lowest_null_n_cans].var()
    zero_var_cols = variances[variances == 0].index.tolist()
        
    if zero_var_cols:
        print(f"5. WARNING: {len(zero_var_cols)} columns have ZERO variance and may not appear in heatmap: {zero_var_cols}")
    else:
        print("5. All features have variance.")

     
       
    
    return df, lowest_null_n_cans




def cans_pipeline(cans_scores, cols, types, top_n_lowest, pre_demo_cols, cat_cols, item_title_mapping):
    
    df, excluded = data_cleaning_pipeline(cans_scores, cols = cols, types = types, cat_cols = cat_cols) 
    pivoted_df, lowest_null_n_cans = model_ready_df(df, item_title_mapping= item_title_mapping, top_n_lowest_nulls = top_n_lowest)
    pivoted_df['OptionsNumber'] = pivoted_df['OptionsNumber'].astype(str)

    df = df.copy()
    df['Ethnicity'] = consolidate_ethnicity(df[pre_demo_cols]['Ethnicity'])

    eth_dummies = pd.get_dummies(df['Ethnicity']).astype(int).drop(columns=['White'])
    language_dummies = pd.get_dummies(df['PrimaryLanguage']).astype(int).drop(columns=['English'])
    gender_dummies = pd.get_dummies(df['Gender']).astype(int).drop(columns=['F'])


    demo_df = pd.concat([df[['OptionsNumber', 'DateCompleted', 'AgeWhenAssessed']], eth_dummies, language_dummies, gender_dummies], axis=1)

    merged_df = pivoted_df.merge(
        demo_df.drop_duplicates(subset=['OptionsNumber', 'DateCompleted']),
        on=['OptionsNumber', 'DateCompleted'],
        how='left'
    )

    
    return merged_df, lowest_null_n_cans