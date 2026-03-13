
import pandas as pd 
import numpy as np 

def make_hazard_table(cans_df, incidents_df, incident_list,
                      admin_censor_date="2025-12-31", max_gap_days=180, filter_interval= True, filter_amount = 200):
    """
    Construct a hazard table for survival analysis suitable for lifelines CoxPHFitter.
    Durations are numeric floats (days since first assessment per person).
    
    Parameters
    ----------
    cans_df : pd.DataFrame
        CANS assessment data with at least ['OptionsNumber', 'DateCompleted'].
    incidents_df : pd.DataFrame
        Incident data with ['OptionsNumber', 'IncidentType', 'IncidentDate'].
    incident_list : list of str
        Incident types to include as the "event".
    admin_censor_date : str or pd.Timestamp
        Administrative censor date (used to fill last assessment interval).
    max_gap_days : int
        Maximum days allowed between assessments before considering censoring.
    
    Returns
    -------
    pd.DataFrame
        Hazard-table-ready DataFrame with numeric durations and binary events:
        ['OptionsNumber', 'DateCompleted', 'start', 'stop', 'start_days', 'stop_days',
         'event', 'days_since_last_assessment', 'days_since_last_incident']
    """
    
    DAYS_CONVERT = 86400

    incidents_sub = incidents_df[incidents_df['IncidentType'].isin(incident_list)].copy()
    incidents_sub['IncidentDate'] = pd.to_datetime(incidents_sub['IncidentDate'])
    incident_dict = incidents_sub.groupby('OptionsNumber')['IncidentDate'].apply(list).to_dict()
    
    cans = cans_df.copy()
    cans['DateCompleted'] = pd.to_datetime(cans['DateCompleted'])
    cans = cans.sort_values(['OptionsNumber', 'DateCompleted'])
    
    cans['Next_Assessment'] = cans.groupby('OptionsNumber')['DateCompleted'].shift(-1)
    
    admin_censor_date = pd.Timestamp(admin_censor_date)
    max_followup = cans['DateCompleted'] + pd.Timedelta(days=max_gap_days)
    
    cans['Next_Assessment'] = cans['Next_Assessment'].fillna(
        pd.Series(np.minimum(max_followup.values.astype('int64'), admin_censor_date.value))
        .astype('datetime64[ns]')
    )
    
    cans['start'] = cans['DateCompleted']
    cans['stop']  = cans['Next_Assessment']

    def get_event(row):
        """Return first incident date in interval and binary event."""
        incidents = incident_dict.get(row['OptionsNumber'], [])
        interval_incidents = [d for d in incidents if row['start'] < d <= row['stop']]
        if not interval_incidents:
            return pd.Series([pd.NaT, 0])
        return pd.Series([min(interval_incidents), 1])
    
    cans[['event_date', 'event']] = cans.apply(get_event, axis=1)

    origin = cans.groupby('OptionsNumber')['start'].transform('min')

    event_date = pd.to_datetime(cans['event_date']).fillna(pd.to_datetime(cans['stop']))
    cans['start_days'] = (cans['start'] - origin).dt.total_seconds() / DAYS_CONVERT
    cans['stop_days']  = (event_date - origin).dt.total_seconds() / DAYS_CONVERT
    
    cans['start_days'] = cans['start_days'].astype(float)
    cans['stop_days']  = cans['stop_days'].astype(float)
    cans['event']      = cans['event'].astype(int)
    
    cans['days_since_last_assessment'] = cans.groupby('OptionsNumber')['DateCompleted']\
                                             .diff().dt.days.fillna(0)
    
    def days_since_last_incident(row):
        incidents = incident_dict.get(row['OptionsNumber'], [])
        past_incidents = [d for d in incidents if d <= row['start']]
        if len(past_incidents) == 0:
            return np.nan
        return (row['start'] - past_incidents[-1]).days
    
    cans['days_since_last_incident'] = cans.apply(days_since_last_incident, axis=1)

    hazard_table = cans

    if filter_interval:
        hazard_table = hazard_table[hazard_table['stop_days'] - hazard_table['start_days'] <= filter_amount]
    
    return hazard_table


def build_full_hazard_table(
        incident_cat_dict,
        cans_df,
        incidents_df,
        base_table,
        admin_censor_date="2025-12-31",
        max_gap_days=180,
        filter_interval=True,
        filter_amount=200):
    """ 
    Adds incident event_date, event indicator, stop_days, 
    and days since last incident for all incident catgories to one master table
    which can then be used to apply Cox Proportional Hazards model to each incident category seperately 
    """
    event_cols = [
    'event',
    'event_date',
    'stop_days',
    'days_since_last_incident'
 ]

    base_table_no_incidents = base_table.drop(columns=event_cols, errors='ignore')
    base_cols = base_table_no_incidents.columns
    
    incident_cols_dict = {}

    for incident_cat, incident_cat_lst in incident_cat_dict.items():

        hazard_table = make_hazard_table(
            cans_df=cans_df,
            incidents_df=incidents_df,
            incident_list=incident_cat_lst,
            admin_censor_date=admin_censor_date,
            max_gap_days=max_gap_days,
            filter_interval=filter_interval,
            filter_amount=filter_amount
        )

        incident_h_table = hazard_table[
            ['OptionsNumber','DateCompleted','event_date','event','stop_days', 'days_since_last_incident']
        ]

        rename_cols = {
            c: f"{incident_cat}_{c}"
            for c in [
                        'event_date',
                        'event',
                        'stop_days',
                        'days_since_last_incident'
                        ]
        }

        incident_h_table = incident_h_table.rename(columns=rename_cols)
        
        incident_cols_dict[incident_cat] = list(rename_cols.values())


        base_table = base_table.merge(
            incident_h_table,
            on=['OptionsNumber','DateCompleted'],
            how='left'
        )
    incident_cols = [c for c in base_table.columns if c not in base_cols]

 
    return base_table, base_cols, incident_cols_dict
