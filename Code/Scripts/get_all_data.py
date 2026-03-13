from .cans_data_cleaning_pipeline import cans_pipeline
from .build_hazard_table import build_full_hazard_table, make_hazard_table
import pandas as pd 
import numpy as np 
import os 

types = {
            'OptionsNumber': 'string',
            'DateCompleted': 'datetime64[ns]',
            'CANSCategoryName': 'string',
            'QuestionKey': 'string',
            'ItemTitle': 'string',
            'ChoiceValue': 'Int64'
             }
    
CANS_cols = ['OptionsNumber', 'DateCompleted', 'CANSCategoryName', 'QuestionKey', 'ItemTitle','ChoiceValue', 'AgeWhenAssessed', 'Gender', 'Ethnicity', 'PrimaryLanguage', 'County']
cat_cols = ['Gender', 'Ethnicity', 'PrimaryLanguage']
pre_demo_calls = ['AgeWhenAssessed', 'Gender', 'Ethnicity', 'PrimaryLanguage'] 
post_demo_columns = ["Black","Latino","No_Preference","Other","White","Spanish","M", 'AgeWhenAssessed']
    
       
folder_path = '/Users/daniellancet/Desktop/Spring_2026/CDSS_170/CANS_INCIDENTS_PROJECT/Aspiranet_Data'
incident_category_path = '/Users/daniellancet/Desktop/Spring_2026/CDSS_170/CANS_INCIDENTS_PROJECT/Novel_Data/UPDATED Incident Matching - Exact_Column_Name_Table.csv'


def read_dfs(folder_path): 

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found. Please ensure it exists in your Google Drive.")
    else:
        data_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

        if not data_files:
            print(f"No XLSX files found in '{folder_path}'.")
        else:
            
            print(f"Found {len(data_files)} data files. Loading...")
            dfs = {}
            for data_file in data_files:
                file_path = os.path.join(folder_path, data_file)
                try:
                    df = pd.read_excel(file_path)
                    
                    if 'CANS' in file_path: 
                        dfs['CANS'] = df 
                    if 'Incidents' in file_path: 
                        dfs['Incidents'] = df 
                    if 'Demographics' in file_path: 
                        dfs['Demographics'] = df 
            
                except Exception as e:
                    print(f"Error loading {data_file}: {e}")

            return dfs
        
def get_full_hazard(dfs, incident_category_path = incident_category_path, filter_amount = 200):
    incident_categories = pd.read_csv(incident_category_path)
    cans_cleaned = dfs['CLEANED_CANS'] 
    incidents_df = dfs['Incidents']
        
    def incident_cat_cols(incident_category, incident_path = incident_category_path): 
        df = pd.read_csv(incident_path) 
        return df[incident_category].dropna().values

    suicide_related_incidents = incident_cat_cols('Suicide Related Incidents', incident_path= incident_category_path)

    base_hazard_table = make_hazard_table(cans_df=cans_cleaned, incidents_df=incidents_df, incident_list=suicide_related_incidents, filter_amount=filter_amount) # could be any incident list
    incident_cat_dict = {inc_cat.replace('/', '_').replace(' ', '_').replace('__', '_'): incident_cat_cols(incident_category=inc_cat, incident_path = incident_category_path) for inc_cat in incident_categories.columns}
    incident_cols = [
    'event_date',
    'event',
    'stop_days',
    'days_since_last_incident'
    ]
    keep_cols = [c for c in base_hazard_table.columns if c not in incident_cols]
    base_table = base_hazard_table[keep_cols]
    full_hazard_table, base_cols, incident_cols = build_full_hazard_table(incident_cat_dict, cans_df=cans_cleaned, incidents_df=incidents_df, base_table = base_table, filter_interval=False)


    return full_hazard_table, base_cols, incident_cols
        


def build_new_dfs(dfs, types = types, CANS_cols = CANS_cols, cat_cols = cat_cols, pre_demo_calls = pre_demo_calls, lowest_n_cols = 45, filter_amount = 2000): 
    cans = dfs['CANS'] 

    category_mapping = cans.groupby('ItemTitle')['CANSCategoryName'].unique().to_dict()
    item_title_mapping = {k: v[0] for k, v in category_mapping.items()}
    cleaned_cans, lowest_n_cols = cans_pipeline(cans_scores=cans, cols=CANS_cols, types = types, cat_cols= cat_cols,top_n_lowest=lowest_n_cols, pre_demo_cols=pre_demo_calls, item_title_mapping = item_title_mapping);

    dfs['CLEANED_CANS'] = cleaned_cans
    full_hazard_table, base_cols, incident_cols = get_full_hazard(dfs, filter_amount=filter_amount)
    dfs['FULL_HAZARD_TABLE'] = full_hazard_table 

    return dfs, lowest_n_cols, base_cols, incident_cols




# data_df = read_dfs(folder_path)
# dfs, top_n_cols = build_new_dfs(data_df)